import logging
import datetime as dt

from matplotlib.ticker import FuncFormatter

import math
from numba import njit, prange
from typing import Optional, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from wealthplan.cache.result_cache import VALUE_FUNCTION_KEY, POLICY_KEY
from wealthplan.cashflows.base import Cashflow


from wealthplan.optimizer.bellman_optimizer import BellmanOptimizer, create_grid
from wealthplan.optimizer.deterministic.deterministic_bellman_optimizer import (
    DeterministicBellmanOptimizer,
)
from wealthplan.optimizer.math_tools.dynamic_wealth_grid import DynamicGridBuilder
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
    wealth_grid,
    u,
    d,
    p,
    beta,
    v_t_next_j,
    v_t_next_jp1,
    q_t,
    cf_t,
    c_step,
    wealth_grid_next,
    r,
):
    n_w = wealth_grid.shape[0]

    v_opt = np.zeros(n_w, dtype=np.float32)
    consumption_opt = np.zeros(n_w, dtype=np.float32)

    for i in prange(n_w):
        W = wealth_grid[i]

        # make sure, you only consider admissible consumption levels in order to enter
        # the next admissible wealth region
        c_min = max(0.0, W + cf_t - wealth_grid_next[-1] / u)
        c_max = min(W + cf_t, W + cf_t - wealth_grid_next[0] / d)
        c_cands = create_grid(c_min, c_max, c_step)

        W_up_arr = (W + cf_t - c_cands) * u
        W_down_arr = (W + cf_t - c_cands) * d

        V_up_arr = np.interp(W_up_arr, wealth_grid_next, v_t_next_jp1)
        V_down_arr = np.interp(W_down_arr, wealth_grid_next, v_t_next_j)

        instant_util_arr = crra_utility_numba(c_cands)
        total_val_arr = instant_util_arr + beta * q_t * (
            p * V_up_arr + (1 - p) * V_down_arr
        )

        idx_max = np.argmax(total_val_arr)
        v_opt[i] = total_val_arr[idx_max]
        consumption_opt[i] = c_cands[idx_max]

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
        w_max: float = 750_000.0,
        w_step: float = 500.0,
        c_step: float = 500.0,
        n_sims: int = 2500,
        seed: int = 42,
        save: bool = True,
        stochastic: bool = True,
        use_dynamic_wealth_grid: bool = False,
        max_grid_iteration: int = 100,
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
            w_max=w_max,
            w_step=w_step,
            c_step=c_step,
            save=save,
        )

        self.sigma = sigma

        self.survival_model = survival_model
        self.current_age = current_age

        self.age_grid = self.time_grid + self.current_age

        # Compute conditional survival probabilities over one time step
        self.survival_probs = survival_model.conditional_survival_probabilities(
            self.age_grid, self.dt
        )

        self.n_sims = n_sims
        self.seed = seed

        self.stochastic = stochastic

        # Compute binomial parameters
        self._compute_binomial_params()

        self.use_dynamic_wealth_grid = use_dynamic_wealth_grid

        self.dynamic_grid_builder = None
        self.dynamic_wealth_grid = None
        self.opt_wealth_det = None

        self.max_grid_iteration = max_grid_iteration

        if self.use_dynamic_wealth_grid:
            self._roll_out_wealth_grid()

    def _roll_out_wealth_grid(self) -> None:
        deterministic_optimizer: DeterministicBellmanOptimizer = (
            DeterministicBellmanOptimizer(
                run_id=self.run_id + "_det",
                start_date=self.start_date,
                end_date=self.end_date,
                retirement_date=self.retirement_date,
                initial_wealth=self.initial_wealth,
                yearly_return=self.yearly_return,
                cashflows=self.cashflows,
                w_max=self.w_max,
                w_step=self.w_step,
                c_step=self.c_step,
                save=self.save,
            )
        )

        deterministic_optimizer.solve()
        self.opt_wealth_det = deterministic_optimizer.opt_wealth.values

        self.dynamic_grid_builder = DynamicGridBuilder(
            T=self.time_grid[-1],
            delta_w=self.w_step,
            w_max=self.w_max,
            n_steps=self.n_months,
            c_step=self.c_step,
            L_t=self.opt_wealth_det,
            cf=self.cf,
        )

        self.dynamic_wealth_grid = self.dynamic_grid_builder.build_initial_grid()

    def _compute_binomial_params(self) -> None:
        """
        Compute the binomial tree parameters u (up), d (down), and p (up probability)
        based on the risk-free rate, volatility, and time step.
        """
        self.sqrt_dt: float = math.sqrt(self.dt)

        if self.stochastic:
            self.u: float = math.exp(self.sigma * self.sqrt_dt)
            self.d: float = 1.0 / self.u

            assert (
                self.d < 1.0 + self.monthly_return < self.u
            ), "Binomial tree model is not arbitrage free!"

            self.p: float = (1.0 + self.monthly_return - self.d) / (self.u - self.d)
            self.q: float = 1.0 - self.p
        else:
            self.u: float = 1.0 + self.monthly_return
            self.d: float = 1.0 + self.monthly_return

            self.p: float = 0.5
            self.q: float = 0.5

    def _backward_induction(self) -> None:
        """
        Compute value function and policy by backward induction on discrete grids.
        Results are stored in self.value_function and self.policy keyed by date.
        Handles binomial tree structure for risky asset and survival probabilities.
        """
        n_t = self.n_months

        wealth_grid = (
            self.dynamic_wealth_grid[-1]
            if self.use_dynamic_wealth_grid
            else self.wealth_grid
        )

        # -------------------------
        # Initialize terminal value
        # -------------------------
        v_terminal = self.terminal_penalty(wealth_grid)
        v_t_next = np.array([v_terminal.copy() for _ in range(n_t)])

        # -------------------------
        # Backward induction
        # -------------------------
        for t in tqdm(
            reversed(range(n_t - 1)),  # loop backwards
            total=n_t - 1,  # total steps for the progress bar
            desc="Backward Induction",  # description
        ):
            date_t = self.months[t]
            cf_t = self.cf[t]
            q_t = self.survival_probs[t]

            # Check cache
            if self.cache.has(date_t):
                logger.info("Cache hit for %s", date_t)

                cached_data = self.cache.load_date(date_t)

                v_t = cached_data[VALUE_FUNCTION_KEY]
                policy_t = cached_data[POLICY_KEY]

                self.value_function[date_t] = v_t
                self.policy[date_t] = policy_t

                v_t_next = v_t
                continue

            n_s = t + 1  # number of stock nodes at this time

            wealth_grid = (
                self.dynamic_wealth_grid[t]
                if self.use_dynamic_wealth_grid
                else self.wealth_grid
            )

            n_w = len(wealth_grid)

            v_t = np.zeros((n_s, n_w), dtype=np.float32)
            policy_t = np.zeros((n_s, n_w), dtype=np.float32)

            results = []

            wealth_grid_next = (
                self.dynamic_wealth_grid[t + 1]
                if self.use_dynamic_wealth_grid
                else self.wealth_grid
            )

            for j in range(n_s):
                results.append(
                    compute_optimal_policy(
                        wealth_grid,
                        self.u,
                        self.d,
                        self.p,
                        self.beta,
                        np.array(v_t_next[j], dtype=np.float32),  # down node
                        np.array(v_t_next[j + 1], dtype=np.float32),  # up node
                        q_t,
                        cf_t,
                        self.c_step,
                        wealth_grid_next,
                        self.monthly_return,
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

    def _roll_forward(self):
        """
        Vectorized rollout of multiple Monte Carlo paths along the binomial tree.
        """
        rng = np.random.default_rng(self.seed)

        # generate random up/down paths: shape (n_sims, n_months-1)
        updown_paths = rng.integers(0, 2, size=(self.n_months - 1, self.n_sims))

        # simulate survival: shape (n_months, n_sims)
        survival_paths = self.survival_model.simulate_survival(
            self.age_grid, self.dt, self.n_sims
        )

        # initialize arrays: shape (n_months, n_sims)
        wealth_paths = np.zeros((self.n_months, self.n_sims), dtype=np.float32)
        consumption_paths = np.zeros((self.n_months, self.n_sims), dtype=np.float32)
        cashflow_paths = np.zeros((self.n_months, self.n_sims), dtype=np.float32)

        # initial month
        wealth_paths[0, :] = self.initial_wealth
        cashflow_paths[0, :] = self.cf[0]

        wealth_grid = (
            self.dynamic_wealth_grid[0]
            if self.use_dynamic_wealth_grid
            else self.wealth_grid
        )

        # consumption for t=0
        node_idx0 = 0
        consumption_paths[0, :] = np.interp(
            wealth_paths[0, :],
            wealth_grid,
            self.policy[self.months[0]][node_idx0, :],
        )

        # forward rollout vectorized
        for t in range(1, self.n_months):
            month = self.months[t]

            alive_mask = survival_paths[t, :] > 0  # alive sims
            W_prev = wealth_paths[t - 1, :].copy()

            # wealth update
            W_current = (W_prev + self.cf[t - 1] - consumption_paths[t - 1]) * np.where(
                updown_paths[t - 1, :] == 1, self.u, self.d
            )

            W_current = np.clip(
                W_current,
                wealth_grid[0],  # lower bound
                wealth_grid[-1],  # upper bound
            )

            wealth_paths[t, :] = W_current

            # compute node indices: cumulative sum of up moves along each path
            node_idx = updown_paths[:t, :].sum(axis=0)

            if t < self.n_months - 1:

                wealth_grid = (
                    self.dynamic_wealth_grid[t]
                    if self.use_dynamic_wealth_grid
                    else self.wealth_grid
                )

                for sim_idx in np.where(alive_mask)[0]:
                    consumption_paths[t, sim_idx] = np.interp(
                        W_current[sim_idx],
                        wealth_grid,
                        self.policy[month][node_idx[sim_idx], :],
                    )

            cashflow_paths[t, :] = self.cf[t]

            # zero out dead paths
            dead_mask = ~alive_mask

            wealth_paths[t, dead_mask] = 0.0
            consumption_paths[t, dead_mask] = 0.0
            cashflow_paths[t, dead_mask] = 0.0

        # store as DataFrames for plotting
        self.opt_wealth = pd.DataFrame(wealth_paths, index=self.months)
        self.opt_consumption = pd.DataFrame(consumption_paths, index=self.months)
        self.monthly_cashflows = pd.DataFrame(cashflow_paths, index=self.months)

    def _solve(self) -> None:
        """
        Generic solver that dynamically calls the child class implementation
        of backward induction and roll-forward to generate optimal paths.

        Stores results as instance attributes.

        Returns:
            None
        """
        # Backward induction step
        self._backward_induction()

        # Forward roll-out of paths
        logger.info("Rolling forward to compute optimal paths.")
        self._roll_forward()

    def _solve_with_dynamic_grid(self, tolerance: float = 1e-3) -> None:
        """
        Generic solver that dynamically calls the child class implementation
        of backward induction and roll-forward to generate optimal paths.

        Stores results as instance attributes.

        Logging is included to track the evolution of grid boundaries.

        Args:
            tolerance: float, tolerance to check if upper/lower bounds have stabilized
        """
        # Initial solver step
        self.cache.clear()

        self._backward_induction()
        self._roll_forward()

        self.plot()

        # Get current upper/lower bounds
        current_upper = self.dynamic_grid_builder.upper_bounds
        current_lower = self.dynamic_grid_builder.lower_bounds

        logging.info("Initial grid boundaries set.")

        for i in range(1, self.max_grid_iteration + 1):
            self.cache.clear()

            # Extend grid based on simulations
            self.dynamic_wealth_grid = self.dynamic_grid_builder.extend_grid(
                simulations=self.opt_wealth.values
            )

            next_upper = self.dynamic_grid_builder.upper_bounds
            next_lower = self.dynamic_grid_builder.lower_bounds

            # Logging boundaries for this iteration
            logging.debug("Iteration %d:", i)

            # Check if grid has stabilized
            upper_stable = np.allclose(current_upper, next_upper, atol=tolerance)
            lower_stable = np.allclose(current_lower, next_lower, atol=tolerance)

            if upper_stable and lower_stable:
                logging.info("Dynamic grid stabilized after %d iterations.", i)
                break  # stop iteration if boundaries are stable

            # Update for next iteration
            current_upper = next_upper.copy()
            current_lower = next_lower.copy()

            # Solve again with updated grid
            self._backward_induction()
            self._roll_forward()

            self.plot()

    def solve(self) -> None:
        """
        Generic solver that dynamically calls the child class implementation
        of backward induction and roll-forward to generate optimal paths.

        Stores results as instance attributes.

        Returns:
            None
        """
        logger.info("%s.solve() started.", self.__class__.__name__)

        if self.use_dynamic_wealth_grid:
            self._solve_with_dynamic_grid()
        else:
            self._solve()

        logger.info("%s.solve() finished.", self.__class__.__name__)

    def plot(
        self,
        percentiles: tuple[float, ...] = (5, 10),
        sample_sim: Optional[int] = None,
        *,
        title_size: int = 22,
        legend_size: int = 16,
        tick_size: int = 16,
    ) -> None:
        """
        Plot stochastic results with mean paths, percentile bands, sample paths,
        and survival probabilities.

        The figure contains four subplots:
            1. Optimal consumption over time
            2. Wealth over time
            3. Monthly investment / withdrawal
            4. Cumulative survival probability

        Parameters
        ----------
        percentiles : tuple[float, ...], default=(5, 10)
            Percentile levels (e.g. 5 → 5–95%, 10 → 10–90%).

        sample_sim : int, optional
            Index of simulation to plot as a sample path.
            If None, a random simulation is selected.

        title_size : int, default=18
            Font size for subplot titles.

        legend_size : int, default=14
            Font size for legend text.

        tick_size : int, default=14
            Font size for axis tick labels.

        Returns
        -------
        None
        """
        if self.opt_wealth.empty:
            raise RuntimeError("No solution available — call solve() first.")

        percentiles = sorted(percentiles)

        # --- Pick sample simulation ---
        if sample_sim is None:
            np.random.seed(self.seed)
            sample_sim = np.random.randint(self.n_sims)

        months = self.months

        # ============================
        # Helper: mean + percentile bands
        # ============================
        def mean_and_bands(
            df: pd.DataFrame,
        ) -> tuple[pd.Series, dict[float, tuple[pd.Series, pd.Series]]]:
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
        # Data preparation
        # ============================
        cons_mean, cons_bands = mean_and_bands(self.opt_consumption)
        wealth_mean, wealth_bands = mean_and_bands(self.opt_wealth)

        cons_sample = self.opt_consumption.iloc[:, sample_sim]
        wealth_sample = self.opt_wealth.iloc[:, sample_sim]

        # ============================
        # Plotting
        # ============================
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)

        def plot_with_bands(
            ax: plt.Axes,
            x: pd.Index | np.ndarray,
            mean: pd.Series,
            bands: dict[float, tuple[pd.Series, pd.Series]],
            color: str,
            title: str,
            yfmt: bool = False,
        ) -> None:
            all_values = [mean]

            for i, (p, (lo, hi)) in enumerate(bands.items()):
                ax.fill_between(
                    x,
                    lo,
                    hi,
                    color=color,
                    alpha=0.15 + 0.15 * i,
                    label=f"{p}–{100 - p}%",
                )
                all_values.extend([lo, hi])

            ax.plot(x, mean, color=color, lw=2, label="Mean", linestyle="--")

            if x[0] <= self.retirement_date <= x[-1]:
                ax.axvline(
                    self.retirement_date,
                    color="red",
                    linestyle="--",
                    lw=2,
                    label="Retirement",
                )

            # Dynamic y-limits
            all_values = np.concatenate([np.ravel(v) for v in all_values])
            ymin, ymax = np.min(all_values), np.max(all_values)

            # Adjust lower limit
            if ymin >= 0:
                ymin *= 0.9
            else:
                ymin *= 1.1  # expand downward for negative min

            # Adjust upper limit
            if ymax >= 0:
                ymax *= 1.1
            else:
                ymax *= 0.9  # expand upward for negative max

            ax.set_ylim(ymin, ymax)

            ax.set_title(title, fontsize=title_size)
            ax.legend(fontsize=legend_size)
            ax.grid(alpha=0.3)
            ax.tick_params(axis="both", labelsize=tick_size)

            if yfmt:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

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
            months,
            self.cf,
            color="blue",
            linestyle="--",
            lw=1.0,
            label="Deterministic Cashflows",
        )
        axes[0].legend(fontsize=legend_size)
        axes[0].tick_params(axis="both", labelsize=tick_size)

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

        if self.dynamic_grid_builder:
            axes[1].plot(
                months,
                self.opt_wealth_det,
                color="tab:cyan",
                lw=1.5,
                label="Optimal Wealth (det.)",
            )

            axes[1].plot(
                months, self.dynamic_grid_builder.upper_bounds, color="tab:grey", lw=2.0
            )

            axes[1].plot(
                months, self.dynamic_grid_builder.lower_bounds, color="tab:grey", lw=2.0
            )

        axes[1].legend(fontsize=legend_size)
        axes[1].tick_params(axis="both", labelsize=tick_size)

        # --- 3. Investment / Withdrawal ---
        inv_df = self.cf[:, np.newaxis] - self.opt_consumption.values
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
        axes[2].legend(fontsize=legend_size)
        axes[2].tick_params(axis="both", labelsize=tick_size)

        # --- 4. Survival probability ---
        cumulative_survival = np.cumprod(self.survival_probs)

        axes[3].plot(self.age_grid, cumulative_survival, color="tab:orange", lw=2)
        axes[3].set_title("Survival Probability", fontsize=title_size)
        axes[3].set_ylim(0.0, 1.05)
        axes[3].grid(True)
        axes[3].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        axes[3].tick_params(axis="both", labelsize=tick_size)

        plt.tight_layout()
        plt.show()
