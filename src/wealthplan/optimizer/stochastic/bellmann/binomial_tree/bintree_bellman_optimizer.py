import logging
import datetime as dt
import math
from typing import Dict, Tuple, Optional, List, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm
import numba as nb
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


from src.wealthplan.optimizer.deterministic.base_optimizer import (
    BaseConsumptionOptimizer,
)
from wealthplan.cashflows.base import Cashflow
from wealthplan.optimizer.stochastic.market_model.gbm_returns import GBM
from wealthplan.optimizer.stochastic.result_cache import ResultCache
from wealthplan.optimizer.stochastic.survival_process.survival_process import (
    SurvivalProcess,
)
from wealthplan.optimizer.stochastic.wealth_regression.wealth_regression import (
    WealthRegressor,
)

logger = logging.getLogger(__name__)





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
        delta: float = 500.0,
        beta: float = 1.0,
        save: bool = True,
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
        self.delta = delta

        self.beta: float = beta

        self.months = [
            d.date()
            for d in pd.date_range(start=self.start_date, end=self.end_date, freq="MS")
        ]
        self.n_months = len(self.months)

        self.wealth_grid = np.arange(0.0, self.max_wealth, self.delta, dtype=np.float32)

        self.save = save

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
                np.exp(self.c * (age_t + self.dt))
                - np.exp(self.c * age_t)
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

    def _regress_value_function(
        self,
        date_t: dt.date,
        v_next: np.ndarray,
        returns_paths: np.ndarray,
        survival_paths: np.ndarray,
        plot: bool = True,
    ) -> np.ndarray:
        """
        Regress the value function over wealth, returns, and survival paths.

        Parameters
        ----------
        v_next : np.ndarray, shape (n_w, n_sims)
            Value function or continuation values to regress.
        returns_paths : np.ndarray, shape (n_sims,)
            Simulated returns for the current time step.
        survival_paths : np.ndarray, shape (n_sims,)
            Survival indicator/probabilities for the current time step.

        Returns
        -------
        np.ndarray
            Regressed value function, same shape as v_next.

        Raises
        ------
        RuntimeError
            If the regression R-squared falls below self.r_squared_threshold.
        """
        # Setup the regressor
        regressor = WealthRegressor(
            wealth_grid=self.wealth_grid,
            returns=returns_paths,
            survival_paths=survival_paths,
        )

        # Perform regression
        try:
            v_regressed, r_squared = regressor.regress(v_next=v_next)
        except RuntimeError as e:
            regressor.plot_regression_fit_1d(v_next)
            regressor.plot_regression_fit_2d(v_next, self.n_sims)

            raise e

        self.r_squared[date_t] = r_squared

        # regressor.plot_regression_fit_1d(v_next)
        # regressor.plot_regression_fit_2d(v_next, self.n_sims)

        # Check threshold
        if r_squared < self.r_squared_threshold:
            # Raise error afterwards
            if plot:
                regressor.plot_regression_fit_1d(v_next)
                regressor.plot_regression_fit_2d(v_next, self.n_sims)

            raise RuntimeError(
                f"Fit R-squared = {r_squared:.4f} below threshold {self.r_squared_threshold}"
            )

        return v_regressed

    def _compute_available_wealth(self, cf_t: float):
        """Compute available wealth and candidate consumption grid."""
        available = self.wealth_grid[:, None] + cf_t
        available = np.repeat(available, self.n_sims, axis=1)  # (n_w, n_sims)

        n_c = int(np.ceil(np.max(available) / self.c_step)) + 1
        c_grid = np.arange(0.0, n_c * self.c_step, self.c_step, dtype=np.float32)

        return available, c_grid

    def _compute_value_candidates(
        self, c_grid: np.ndarray, v_next_interp: np.ndarray, t: int, beta: float
    ) -> np.ndarray:
        """Compute total value including instantaneous utility, vectorized over simulations."""
        instant_util = self.instant_utility(c_grid)  # shape (n_c,)

        # Repeat across simulations
        instant_util = np.tile(
            instant_util[:, None], (1, self.n_sims)
        )  # shape (n_c, n_sims)

        # Set utility to 0 where dead
        instant_util *= self.survival_paths[:, t][
            None, :
        ]  # broadcast survival_paths over n_c

        # Add continuation value (v_next_interp is shape (n_w, n_c, n_sims))
        value_candidates = (
            instant_util[None, :, :] + beta * v_next_interp
        )  # shape (n_w, n_c, n_sims)

        return value_candidates

    def _select_best_policy(
        self, value_candidates: np.ndarray, w_next: np.ndarray, c_grid: np.ndarray
    ):
        """Select best consumption for each (wealth, simulation) point."""
        feasible_mask = w_next >= 0
        feasible_values = np.where(feasible_mask, value_candidates, -np.inf)
        best_idx = np.argmax(feasible_values, axis=1)

        n_w = value_candidates.shape[0]
        s_idx = np.arange(self.n_sims)[None, :]
        w_idx = np.arange(n_w)[:, None]

        optimal_value = value_candidates[w_idx, best_idx, s_idx]
        optimal_policy = c_grid[best_idx]  # (n_w, n_sims)
        return optimal_value, optimal_policy

    def _backward_induction(self) -> None:
        """
        Compute value function and policy by backward induction on discrete grids.
        Results are stored in self.value_function and self.policy keyed by date.
        """
        # initialize terminal value
        v_next = np.array(
                [self.terminal_penalty(w) for w in self.wealth_grid], dtype=np.float32
            )  # (n_w, n_sims)

        self.monthly_returns = self.returns_paths[:, 1:] / self.returns_paths[:, :-1]

        beta = self.beta

        # iterate backwards (skip terminal period)
        for t in tqdm(
            reversed(range(self.n_months - 1)),
            total=max(0, self.n_months - 1),
            desc="Backward Induction",
        ):
            date_t = self.months[t]

            # Check cache
            if self.cache.has(date_t):
                logger.info("Cache hit for %s", date_t)

                optimal_value, optimal_policy, r_sq = self.cache.load_date(date_t)

                self.value_function[date_t] = optimal_value
                self.policy[date_t] = optimal_policy
                self.r_squared[date_t] = r_sq

                v_next = optimal_value
                continue

            # Regression of continuation value
            v_regressed = self._regress_value_function(
                date_t,
                v_next,
                self.returns_paths[:, t - 1],
                self.survival_paths[:, t - 1],
            )  # (n_w, n_sims)

            cf_t = self.monthly_cashflow(date_t)

            # Candidate consumption grid and available wealth
            available, c_grid = self._compute_available_wealth(
                cf_t
            )  #   # (n_w, n_sims), (n_c)

            # Next-period wealth for all (w, c, sims)
            w_next = (
                available[:, None, :] - c_grid[None, :, None]
            ) * self.monthly_returns[:, t][
                None, None, :
            ]  # (n_w, n_c, n_sims)

            # Interpolate continuation value over wealth
            v_next_interp = interp_2d(w_next, self.wealth_grid, v_regressed)

            # Compute total value including instant utility
            value_candidates = self._compute_value_candidates(
                c_grid, v_next_interp, t, beta
            )

            # Enforce feasibility and select best consumption
            optimal_value, optimal_policy = self._select_best_policy(
                value_candidates, w_next, c_grid
            )

            # Store results
            self.value_function[date_t] = optimal_value.copy()
            self.policy[date_t] = optimal_policy.copy()

            self.cache.store_date(
                date_t=date_t,
                value_function=optimal_value,
                policy=optimal_policy,
                r_squared=self.r_squared[date_t],
            )

            # Step backwards
            v_next = optimal_value

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
