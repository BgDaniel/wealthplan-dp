import logging
import datetime as dt
import math
from numba import njit, prange
from typing import Dict, Optional, List, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


from wealthplan.cashflows.base import Cashflow

from wealthplan.optimizer.stochastic.bellmann.binomial_tree.result_cache import (
    ResultCache,
)


logger = logging.getLogger(__name__)


@njit(parallel=True)
def compute_optimal_policy(
    wealth_grid, u, d, p, beta, v_t_next_j, v_t_next_jp1, q_t, a_grid
):
    n_w = wealth_grid.shape[0]

    v_opt = np.zeros(n_w, dtype=np.float32)
    consumption_opt = np.zeros(n_w, dtype=np.float32)

    for i in prange(n_w):
        W = wealth_grid[i]

        a_vals = np.minimum(a_grid[0, :], W)         # vectorized
        W_up_arr = (W - a_vals) * u
        W_down_arr = (W - a_vals) * d
        V_up_arr = np.interp(W_up_arr, wealth_grid, v_t_next_jp1)
        V_down_arr = np.interp(W_down_arr, wealth_grid, v_t_next_j)
        instant_util_arr = -(a_vals ** 2)
        total_val_arr = instant_util_arr + beta * q_t * (p * V_up_arr + (1-p) * V_down_arr)

        idx_max = np.argmax(total_val_arr)
        v_opt[i] = total_val_arr[idx_max]
        consumption_opt[i] = a_vals[idx_max]

    return v_opt, consumption_opt


class BinTreeBellmanOptimizer:
    def __init__(
        self,
        run_id: str,
        start_date: dt.date,
        end_date: dt.date,
        current_age: int,
        wealth_0: float,
        cashflows: List[Cashflow],
        sigma: float,
        r: float,
        b: float,
        c: float,
        instant_utility: Callable[[np.ndarray], np.ndarray],
        terminal_penalty: Callable[[np.ndarray], np.ndarray],
        max_wealth: float = 500000.0,
        a_steps: int = 1000,
        delta: float = 500.0,
        beta: float = 1.0,
        save: bool = True,
        parallelize: bool = False,
    ) -> None:
        """
        Initialize the stochastic Bellman optimizer.

        Parameters
        ----------
        gbm_returns : object
            Stochastic returns simulator
        survival_process : object
            Survival / mortality process
        *args, **kwargs : forwarded to BaseConsumptionOptimizer
        """
        self.start_date: dt.date = start_date
        self.end_date: dt.date = end_date

        self.current_age = current_age

        self.wealth_0 = wealth_0
        self.cashflows: List[Cashflow] = cashflows

        self.sigma = sigma
        self.r = r
        self.b = b
        self.c = c

        self.instant_utility = instant_utility
        self.terminal_penalty = terminal_penalty

        self.max_wealth = max_wealth
        self.a_steps = a_steps

        self.delta = delta

        self.beta: float = beta

        self.months = [
            d.date()
            for d in pd.date_range(start=self.start_date, end=self.end_date, freq="MS")
        ]
        self.n_months = len(self.months)

        self.wealth_grid = np.arange(0.0, self.max_wealth, self.delta, dtype=np.float32)
        self.n_steps = len(self.wealth_grid)

        self.save = save

        self.parallelize = parallelize

        self.run_id = run_id

        self.cache = ResultCache(enabled=save, run_id=run_id)

        self.save: bool = save

        self.dt = 1.0 / 12.0
        self.sqrt_dt = math.sqrt(self.dt)

        self.u = math.exp(self.sigma * self.sqrt_dt)
        self.d = 1.0 / self.u

        self.p = (math.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.q = 1.0 - self.p

        self.time_grid = np.array(
            [(m - self.months[0]).days / 365.0 for m in self.months],
            dtype=np.float64,
        )

        age_t = self.current_age + self.time_grid

        # Integrated hazard over [t, t+dt]
        hazard_integral = (self.b / self.c) * (
            np.exp(self.c * (age_t + self.dt)) - np.exp(self.c * age_t)
        )

        # Conditional survival probabilities q_t
        self.survival_probs = np.exp(-hazard_integral)

        # outputs (to be populated by solve())
        self.value_function: Dict[dt.date, np.ndarray] = {}
        self.policy: Dict[dt.date, np.ndarray] = {}
        self.opt_wealth: pd.DataFrame = pd.DataFrame(dtype=float)
        self.opt_consumption: pd.DataFrame = pd.DataFrame(dtype=float)
        self.monthly_cashflows: pd.DataFrame = pd.DataFrame(dtype=float)

    def monthly_cashflow(self, date: dt.date) -> float:
        """Sum deterministic cashflows for the given month."""
        return sum(cf.cashflow(date) for cf in self.cashflows)

    def _compute_optimal_policy(
        self, j: int, v_t_next: list, q_t: float, a_grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute value function and policy for one stock/tree node (vectorized over wealth and consumption).

        Returns
        -------
        v_node : np.ndarray
            Optimal value function for this node (shape: n_wealth,)
        policy_node : np.ndarray
            Optimal consumption for this node (shape: n_wealth,)
        """
        n_w = len(self.wealth_grid)
        W_mat = self.wealth_grid[:, np.newaxis]  # (n_w, 1)
        a_mat = np.minimum(a_grid, W_mat)  # (n_w, n_a)

        # Wealth next period
        W_up = (W_mat - a_mat) * self.u
        W_down = (W_mat - a_mat) * self.d

        # Interpolate continuation value
        V_up = np.array(
            [
                np.interp(W_up[i, :], self.wealth_grid, v_t_next[j + 1])
                for i in range(n_w)
            ]
        )
        V_down = np.array(
            [np.interp(W_down[i, :], self.wealth_grid, v_t_next[j]) for i in range(n_w)]
        )

        continuation = q_t * (self.p * V_up + (1 - self.p) * V_down)

        # Instant utility
        instant_util = self.instant_utility(a_mat)  # vectorized

        total_values = instant_util + self.beta * continuation

        # Optimal consumption
        idx_opt = np.argmax(total_values, axis=1)
        v_opt = total_values[np.arange(n_w), idx_opt]
        comsumption_opt = a_mat[np.arange(n_w), idx_opt]

        return v_opt, comsumption_opt

    def _backward_induction(self) -> None:
        """
        Compute value function and policy by backward induction on discrete grids.
        Results are stored in self.value_function and self.policy keyed by date.
        Handles binomial tree structure for risky asset and survival probabilities.
        """
        n_w = len(self.wealth_grid)
        n_t = self.n_months

        # -------------------------
        # Initialize terminal value
        # -------------------------
        v_terminal = self.terminal_penalty(self.wealth_grid)
        v_t_next = np.array([v_terminal.copy() for _ in range(n_t)])

        # Consumption grid
        a_grid = np.linspace(0, self.max_wealth, self.a_steps, dtype=np.float32)[
            np.newaxis, :
        ]  # shape (1, n_a)

        # -------------------------
        # Backward induction
        # -------------------------
        for t_idx in tqdm(
            reversed(range(n_t - 1)),  # loop backwards
            total=n_t - 1,  # total steps for the progress bar
            desc="Backward Induction",  # description
        ):
            date_t = self.months[t_idx]
            q_t = self.survival_probs[t_idx]

            # Check cache
            if self.cache.has(date_t):
                logger.info("Cache hit for %s", date_t)

                v_t, policy_t = self.cache.load_date(date_t)

                self.value_function[date_t] = v_t
                self.policy[date_t] = policy_t

                v_t_next = v_t
                continue

            n_s = t_idx + 1  # number of stock nodes at this time
            v_t = np.zeros((n_s, n_w), dtype=np.float32)
            policy_t = np.zeros((n_s, n_w), dtype=np.float32)

            results = []

            if self.parallelize:
                # Parallel computation over stock nodes
                results = Parallel(n_jobs=-1)(
                    delayed(self._compute_optimal_policy)(j, v_t_next, q_t, a_grid)
                    for j in range(n_s)
                )
            else:
                for j in range(n_s):
                    results.append(
                        compute_optimal_policy(
                            self.wealth_grid,
                            self.u,
                            self.d,
                            self.p,
                            self.beta,
                            np.array(v_t_next[j], dtype=np.float32),  # down node
                            np.array(v_t_next[j + 1], dtype=np.float32),  # up node
                            q_t,
                            a_grid,
                        )
                    )

            # Collect results
            for j, (v_opt, comsumption_opt) in enumerate(results):
                v_t[j] = v_opt
                policy_t[j] = comsumption_opt

            # Store results
            self.value_function[date_t] = v_t.copy()
            self.policy[date_t] = policy_t.copy()

            self.cache.store_date(date_t=date_t, value_function=v_t, policy=policy_t)

            # Prepare for next iteration
            v_t_next = v_t

    def _roll_forward(self) -> None:
        """
        Given self.policy filled per date and self.wealth_grid, compute
        opt_wealth, opt_consumption and monthly_cashflows by forward simulation.
        """
        n_months = self.n_months
        wealth_paths = np.zeros((n_months, self.n_sims))
        consumption_paths = np.zeros((n_months, self.n_sims))
        cashflow_paths = np.zeros((n_months, self.n_sims))

        # month 0
        wealth_paths[0] = np.full(self.n_sims, self.wealth.initial_wealth)

        cf0 = self.monthly_cashflow(self.months[0])
        cashflow_paths[0] = np.full(self.n_sims, cf0)

        _, policy0, _ = self.cache.load_date(self.months[0])

        for i in range(self.n_sims):
            consumption_paths[0, i] = float(
                np.interp(wealth_paths[0][i], self.wealth_grid, policy0[:, i])
            )

        current_wealth_paths = (
            wealth_paths[0] + cf0 - consumption_paths[0]
        ) * self.monthly_returns[:, 0]

        # forward roll for months 1..n-1 (we only simulate up to n_months - 1)
        for t in range(1, n_months - 1):
            date_t = self.months[t]
            cf_t = self.monthly_cashflow(date_t)
            cashflow_paths[t] = np.full(self.n_sims, cf_t)

            wealth_paths[t] = current_wealth_paths

            _, policy_t, _ = self.cache.load_date(date_t)

            for i in range(self.n_sims):
                consumption_paths[t, i] = float(
                    np.interp(wealth_paths[t][i], self.wealth_grid, policy_t[:, i])
                )

            # wealth for next month (record current after return)
            current_wealth_paths = (
                current_wealth_paths + cf_t - consumption_paths[t, i]
            ) * self.monthly_returns[:, t]

        # Final index handling (set last month values)
        if n_months >= 2:
            # set last month's cashflow (could be zero) and consumption via interpolation
            final_idx = n_months - 1
            date_f = self.months[final_idx]
            cashflow_paths[final_idx] = np.full(
                self.n_sims, self.monthly_cashflow(date_f)
            )

            wealth_paths[final_idx] = current_wealth_paths

        # Save as pandas Dataframe indexed by months
        self.opt_wealth = pd.DataFrame(wealth_paths, index=self.months)
        self.opt_consumption = pd.DataFrame(consumption_paths, index=self.months)
        self.monthly_cashflows = pd.DataFrame(cashflow_paths, index=self.months)

    def solve(self) -> None:
        """Solve with backward induction (or load cache if available), then roll forward.
        Returns (value_function, policy).
        """
        logger.info("BellmanOptimizer.solve() started.")

        self._backward_induction()

        logger.info("Rolling forward to get paths.")
        self._roll_forward()

        logger.info("BellmanOptimizer.solve() finished.")

    def plot(
        self,
        percentiles: tuple[float, ...] = (5, 10),
        sample_sim: Optional[int] = None,
    ):
        """
        Plot stochastic results with mean + percentile bands, a sample path,
        and retirement line.

        Parameters
        ----------
        percentiles : tuple of float, default=(5, 10)
            Percentile levels (e.g. 5 -> 5–95%, 10 -> 10–90%)
        sample_sim : int, optional
            Simulation index to plot as a sample path.
            If None, a random simulation is chosen.
        """

        if self.opt_wealth.empty:
            raise RuntimeError("No solution available — call solve() first.")

        percentiles = sorted(percentiles)

        # --- Pick sample simulation ---
        if sample_sim is None:
            sample_sim = np.random.randint(self.n_sims)

        # --- Detect retirement date ---
        retirement_date: Optional[dt.date] = None
        for cf in getattr(self, "cashflows", []):
            if hasattr(cf, "retirement_date"):
                retirement_date = cf.retirement_date
                break

        months = self.months

        # ============================
        # Helper: mean + bands
        # ============================
        def mean_and_bands(df: pd.DataFrame):
            mean = df.mean(axis=1)
            bands = {
                p: (
                    df.quantile(p / 100, axis=1),
                    df.quantile(1 - p / 100, axis=1),
                )
                for p in percentiles
            }
            return mean, bands

        # ============================
        # Data prep
        # ============================
        cons_mean, cons_bands = mean_and_bands(self.opt_consumption)
        wealth_mean, wealth_bands = mean_and_bands(self.opt_wealth)

        surv_mean = pd.DataFrame(self.survival_paths.T, index=months).mean(axis=1)

        det_cf = (
            self.monthly_cashflows.iloc[:, 0]
            if isinstance(self.monthly_cashflows, pd.DataFrame)
            else self.monthly_cashflows
        )

        cons_sample = self.opt_consumption.iloc[:, sample_sim]
        wealth_sample = self.opt_wealth.iloc[:, sample_sim]

        # ============================
        # Plotting
        # ============================
        fig, axes = plt.subplots(3, 1, figsize=(18, 20), sharex=False)

        def thousands_formatter(x, pos):
            return f"{x:,.0f}"

        def plot_with_bands(ax, x, mean, bands, color, title, yfmt=False):
            ax.plot(x, mean, color=color, lw=2, label="Mean")

            for i, (p, (lo, hi)) in enumerate(bands.items()):
                ax.fill_between(
                    x,
                    lo,
                    hi,
                    color=color,
                    alpha=0.15 + 0.15 * i,
                    label=f"{p}–{100 - p}%",
                )

            if retirement_date:
                ax.axvline(retirement_date, color="red", linestyle="--")

            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
            if yfmt:
                ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

        # --- 1. Consumption ---
        plot_with_bands(
            axes[0],
            months,
            cons_mean,
            cons_bands,
            color="tab:green",
            title="Optimal Consumption Over Time",
            yfmt=True,
        )
        axes[0].plot(
            cons_sample.index,
            cons_sample.values,
            color="red",
            lw=1.0,
            alpha=0.8,
            label="Sample Path",
        )
        axes[0].plot(
            det_cf.index,
            det_cf.values,
            color="blue",
            linestyle="--",
            lw=1.0,
            label="Deterministic Cashflows",
        )
        axes[0].legend()

        # --- 2. Wealth ---
        plot_with_bands(
            axes[1],
            months,
            wealth_mean,
            wealth_bands,
            color="tab:blue",
            title="Wealth Over Time",
            yfmt=True,
        )
        axes[1].plot(
            wealth_sample.index,
            wealth_sample.values,
            color="red",
            lw=1.0,
            alpha=0.8,
            label="Sample Path",
        )
        axes[1].legend()

        # --- 3. Survival (mean only) ---
        axes[2].plot(months, surv_mean, color="tab:purple", lw=2)
        if retirement_date:
            axes[2].axvline(retirement_date, color="red", linestyle="--")
        axes[2].set_title("Survival Probability Over Time")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        axes[2].set_xlabel("Date")

        plt.tight_layout()
        plt.show()
