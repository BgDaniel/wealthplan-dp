import logging
import datetime as dt
import math
from numba import njit, prange
from typing import Optional, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

from wealthplan.cache.result_cache import VALUE_FUNCTION_KEY, POLICY_KEY
from wealthplan.cashflows.base import Cashflow


from wealthplan.optimizer.bellman_optimizer import BellmanOptimizer
from wealthplan.optimizer.math_tools.penality_functions import (
    PenalityFunction,
    square_penality,
)
from wealthplan.optimizer.math_tools.utility_functions import (
    UtilityFunction,
    crra_utility,
    crra_utility_numba,
)
from wealthplan.optimizer.stochastic.survival_process.survival_model import (
    SurvivalModel,
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

        a_vals = np.minimum(a_grid[0, :], W)  # vectorized
        W_up_arr = (W - a_vals) * u
        W_down_arr = (W - a_vals) * d
        V_up_arr = np.interp(W_up_arr, wealth_grid, v_t_next_jp1)
        V_down_arr = np.interp(W_down_arr, wealth_grid, v_t_next_j)
        instant_util_arr = crra_utility_numba(a_vals)
        total_val_arr = instant_util_arr + beta * q_t * (
            p * V_up_arr + (1 - p) * V_down_arr
        )

        idx_max = np.argmax(total_val_arr)
        v_opt[i] = total_val_arr[idx_max]
        consumption_opt[i] = a_vals[idx_max]

    return v_opt, consumption_opt


class BinTreeBellmanOptimizer(BellmanOptimizer):
    """
    Binomial tree-based Bellman optimizer.
    """

    def __init__(
        self,
        run_id: str,
        start_date: dt.date,
        end_date: dt.date,
        retirement_date: dt.date,
        initial_wealth: float,
        yearly_return: float,
        cashflows: List[Cashflow],
        sigma: float,
        survival_model: SurvivalModel,
        current_age: int,
        instant_utility: UtilityFunction = crra_utility,
        terminal_penalty: PenalityFunction = square_penality,
        beta: float = 1.0,
        w_max: float = 750_000.0,
        w_step: float = 500.0,
        c_step: float = 500.0,
        n_sims: int = 2500,
        seed: int = 42,
        save: bool = True,
        stochastic: bool = True,
    ) -> None:
        """
        Initialize the binomial tree Bellman optimizer.
        """
        # ---- Call base constructor ----
        super().__init__(
            run_id=run_id,
            start_date=start_date,
            end_date=end_date,
            retirement_date=retirement_date,
            initial_wealth=initial_wealth,
            yearly_return=yearly_return,
            cashflows=cashflows,
            instant_utility=instant_utility,
            terminal_penalty=terminal_penalty,
            beta=beta,
            w_max=w_max,
            w_step=w_step,
            c_step=c_step,
            save=save,
        )

        self.sigma = sigma

        self.survival_model = survival_model
        self.current_age = current_age

        self.age_grid = (
            np.array(
                [(m - self.months[0]).days / 365.0 for m in self.months],
                dtype=np.float64,
            )
            + self.current_age
        )

        # Compute conditional survival probabilities over one time step
        self.survival_probs = survival_model.conditional_survival_probabilities(
            self.age_grid, self.dt
        )

        self.n_sims = n_sims
        self.seed = seed

        self.stochastic = stochastic

        # Compute binomial parameters
        self._compute_binomial_params()

    def _compute_binomial_params(self) -> None:
        """
        Compute the binomial tree parameters u (up), d (down), and p (up probability)
        based on the risk-free rate, volatility, and time step.
        """
        self.sqrt_dt: float = math.sqrt(self.dt)

        if self.stochastic:
            self.u: float = math.exp(self.sigma * self.sqrt_dt)
            self.d: float = 1.0 / self.u

            self.p: float = (1.0 + self.monthly_return - self.d) / (self.u - self.d)
            self.q: float = 1.0 - self.p
        else:
            self.u: float = 1.0
            self.d: float = 1.0

            self.p: float = 0.5
            self.q: float = 0.5

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
        c_grid = np.arange(0, self.w_max, self.c_step, dtype=np.float32)[
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
                        c_grid,
                    )
                )

            # Collect results
            for j, (v_opt, comsumption_opt) in enumerate(results):
                v_t[j] = v_opt
                policy_t[j] = comsumption_opt

            # Store results
            self.value_function[date_t] = v_t.copy()
            self.policy[date_t] = policy_t.copy()

            self.cache.store_date(
                date_t=date_t, data={VALUE_FUNCTION_KEY: v_t, POLICY_KEY: policy_t}
            )

            # Prepare for next iteration
            v_t_next = v_t

    def _roll_single_path(self, policy_list, cashflow_list, updown_path: np.ndarray):
        """
        Roll out a single path along the binomial tree given a fixed sequence of up/down jumps.

        Parameters
        ----------
        policy_list : list of np.ndarray
            Optimal policies per month from cache.
        cashflow_list : list of float
            Deterministic cashflows per month.
        updown_path : np.ndarray of shape (n_months - 1,)
            Sequence of 0/1 for down/up moves.

        Returns
        -------
        wealth_path : np.ndarray
        consumption_path : np.ndarray
        cashflow_path : np.ndarray
        """
        wealth_path = np.zeros(self.n_months, dtype=np.float32)
        consumption_path = np.zeros(self.n_months, dtype=np.float32)
        cashflow_path = np.zeros(self.n_months, dtype=np.float32)

        # initial month
        wealth_path[0] = self.wealth_0
        cashflow_path[0] = cashflow_list[0]

        # node index at time t=0 is always 0
        node_idx = 0
        cons = np.interp(wealth_path[0], self.wealth_grid, policy_list[0][node_idx, :])
        consumption_path[0] = cons

        # forward roll
        for t in range(1, self.n_months - 1):
            W_prev = wealth_path[t - 1]

            node_idx = updown_path[:t].sum()
            cons = np.interp(W_prev, self.wealth_grid, policy_list[t][node_idx, :])
            consumption_path[t] = cons
            cashflow_path[t] = cashflow_list[t]

            W_next = W_prev + cashflow_list[t] - cons

            # use provided up/down jump: 1 = up, 0 = down
            wealth_path[t] = W_next * (self.u if updown_path[t - 1] == 1 else self.d)

        return wealth_path, consumption_path, cashflow_path

    def _roll_forward(self):
        """
        Roll out multiple Monte Carlo sample paths along the binomial tree.

        Parameters
        ----------
        n_sims : int
            Number of paths to simulate.
        seed : int
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(self.seed)

        # prepare policies and cashflows per month
        policy_list = []
        cashflow_list = []

        for t_idx, date_t in enumerate(self.months[:-1]):
            _, policy_t = self.cache.load_date(date_t)
            policy_list.append(policy_t)
            cashflow_list.append(self.monthly_cashflow(date_t))

        # generate random up/down paths
        updown_paths = rng.integers(0, 2, size=(self.n_sims, self.n_months - 1))

        # initialize storage
        wealth_paths = np.zeros((self.n_months, self.n_sims), dtype=np.float32)
        consumption_paths = np.zeros((self.n_months, self.n_sims), dtype=np.float32)
        cashflow_paths = np.zeros((self.n_months, self.n_sims), dtype=np.float32)

        # simulate each path
        for sim in range(self.n_sims):
            W_path, C_path, CF_path = self._roll_single_path(
                policy_list, cashflow_list, updown_paths[sim]
            )
            wealth_paths[:, sim] = W_path
            consumption_paths[:, sim] = C_path
            cashflow_paths[:, sim] = CF_path

        # store as DataFrames for plotting
        self.opt_wealth = pd.DataFrame(wealth_paths, index=self.months)
        self.opt_consumption = pd.DataFrame(consumption_paths, index=self.months)
        self.monthly_cashflows = pd.DataFrame(cashflow_paths, index=self.months)

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
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)

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
                ax.axvline(
                    retirement_date,
                    color="red",
                    linestyle="--",
                    lw=2,
                    label="Retirement",
                )

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
        axes[0].set_ylim((0.0, 20_000))

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

        # --- 3. Investment / Withdrawal ---
        inv_df = det_cf.values[:, None] - self.opt_consumption.values
        inv_df = pd.DataFrame(inv_df, index=self.opt_consumption.index)

        inv_mean, inv_bands = mean_and_bands(inv_df)
        inv_sample = inv_df.iloc[:, sample_sim]

        plot_with_bands(
            axes[2],
            months,
            inv_mean,
            inv_bands,
            color="tab:purple",
            title="Monthly Investment / Withdrawal in Portfolio",
            yfmt=True,
        )

        axes[2].plot(
            inv_sample.index,
            inv_sample.values,
            color="red",
            lw=1.0,
            alpha=0.8,
            label="Sample Path",
        )

        axes[2].axhline(0.0, color="black", lw=1.0, alpha=0.7)
        axes[2].legend()
        axes[2].set_ylim(-20_000, 20_000)

        # 4. Cumulative survival probability
        cumulative_survival = np.cumprod(self.survival_probs)
        age_grid = self.current_age + self.time_grid

        axes[3].plot(age_grid, cumulative_survival, color="tab:orange", lw=2)
        axes[3].set_title("Survival Probability")
        axes[3].set_xlabel("Age")
        axes[3].grid(True)
        axes[3].set_ylim(0, 1.05)

        plt.tight_layout()
        plt.show()
