import logging
import datetime as dt
from typing import List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import prange, njit

from tqdm import tqdm

from wealthplan.cache.result_cache import VALUE_FUNCTION_KEY, POLICY_KEY
from wealthplan.cashflows.base import Cashflow

from wealthplan.optimizer.bellman_optimizer import (
    BellmanOptimizer, create_grid,
)
from wealthplan.optimizer.math_tools.penality_functions import (
    PenalityFunction,
    square_penality,
)
from wealthplan.optimizer.math_tools.utility_functions import (
    UtilityFunction,
    crra_utility,
    crra_utility_numba,
)

logger = logging.getLogger(__name__)


@njit(parallel=True)
def compute_optimal_policy(wealth_grid, r, beta, v_t_next, cf_t, c_step):
    n_w = wealth_grid.shape[0]

    v_opt = np.zeros(n_w, dtype=np.float32)
    consumption_opt = np.zeros(n_w, dtype=np.float32)

    for i in prange(n_w):
        W = wealth_grid[i]

        # make sure, yu do not leave the wealth grid due to low consumption if we
        # are already at the upper bound
        c_min = max(0.0, W + cf_t - wealth_grid[-1] / (1.0 + r))
        c_cands = create_grid(c_min, W + cf_t, c_step)

        W_next = (W + cf_t - c_cands) * (1.0 + r)
        V_next = np.interp(W_next, wealth_grid, v_t_next)

        instant_util_arr = crra_utility_numba(c_cands)
        total_val_arr = instant_util_arr + beta * V_next

        idx_max = np.argmax(total_val_arr)
        v_opt[i] = total_val_arr[idx_max]
        consumption_opt[i] = c_cands[idx_max]

    return v_opt, consumption_opt


class DeterministicBellmanOptimizer(BellmanOptimizer):
    """
    Deterministic Bellman (backward induction) optimizer.
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
        instant_utility: UtilityFunction = crra_utility,
        terminal_penalty: PenalityFunction = square_penality,
        w_max: float = 750_000.0,
        w_step: float = 50.0,
        c_step: float = 50.0,
        save: bool = True,
    ) -> None:
        """
        Initialize the deterministic Bellman optimizer.

        All parameters are forwarded unchanged to BellmanOptimizer.
        """
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

    def _backward_induction(self) -> None:
        """
        Compute value function and policy by backward induction on discrete grids.
        Results are stored in self.value_function and self.policy keyed by date.
        """
        v_t_next = np.array([self.terminal_penalty(w) for w in self.wealth_grid])

        # iterate backwards (skip terminal period)
        for t in tqdm(
            reversed(range(self.n_months - 1)),
            total=max(0, self.n_months - 1),
            desc="Backward Induction",
        ):
            date_t = self.months[t]
            cf_t = self.cf[t]

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

            v_t, policy_t = compute_optimal_policy(
                self.wealth_grid,
                self.monthly_return,
                self.beta,
                v_t_next,
                cf_t,
                self.c_step,
            )

            # Store results
            self.value_function[date_t] = v_t.copy()
            self.policy[date_t] = policy_t.copy()

            self.cache.store_date(
                date_t=date_t, data={VALUE_FUNCTION_KEY: v_t, POLICY_KEY: policy_t}
            )

            v_t_next = v_t

    def _roll_forward(self) -> None:
        """
        Given self.policy filled per date and self.wealth_grid, compute
        opt_wealth, opt_consumption and monthly_cashflows by forward simulation.
        """
        # initialize arrays: shape (n_months)
        wealth_path = np.zeros(self.n_months)
        consumption_path = np.zeros(self.n_months)
        cashflow_path = np.zeros(self.n_months)

        # initial month
        wealth_path[0] = self.initial_wealth
        cashflow_path[0] = self.cf[0]

        consumption_path[0] = np.interp(
            wealth_path[0],
            self.wealth_grid,
            self.policy[self.months[0]],
        )

        # forward rollout vectorized
        for t in range(1, self.n_months):
            month = self.months[t]

            W_prev = wealth_path[t - 1].copy()

            # wealth update
            W_current = (W_prev + self.cf[t - 1] - consumption_path[t - 1]) * (
                1.0 + self.monthly_return
            )

            W_current = np.clip(
                W_current,
                self.wealth_grid[0],  # lower bound
                self.wealth_grid[-1],  # upper bound
            )

            wealth_path[t] = W_current

            if t < self.n_months - 1:

                consumption_path[t] = np.interp(
                    W_current,
                    self.wealth_grid,
                    self.policy[month],
                )

            cashflow_path[t] = self.cf[t]

        # Save as pandas Series indexed by months
        self.opt_wealth = pd.Series(wealth_path, index=self.months)
        self.opt_consumption = pd.Series(consumption_path, index=self.months)
        self.monthly_cashflows = pd.Series(cashflow_path, index=self.months)

    def plot(
        self,
        *,
        title_size: int = 20,
        legend_size: int = 16,
        tick_size: int = 16,
    ) -> None:
        """
        Plot deterministic results for consumption, wealth, and
        investment/withdrawal over time.

        The figure contains three subplots:
            1. Optimal consumption over time
            2. Wealth over time
            3. Monthly investment / withdrawal

        Each subplot includes a retirement date marker (if within range)
        and dynamically scaled y-axis limits.

        Parameters
        ----------
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

        months = self.months

        cons: np.ndarray = self.opt_consumption.values
        wealth: np.ndarray = self.opt_wealth.values
        inv: np.ndarray = self.cf - cons

        # ============================
        # Plotting
        # ============================
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=False)

        def plot_curve(
            ax: plt.Axes,
            x: pd.Index | np.ndarray,
            y: np.ndarray,
            color: str,
            title: str,
            yfmt: bool = False,
            show_retirement: bool = True,
        ) -> None:
            ax.plot(x, y, color=color, lw=2, label=title)

            if show_retirement and x[0] <= self.retirement_date <= x[-1]:
                ax.axvline(
                    self.retirement_date,
                    color="red",
                    linestyle="--",
                    lw=2,
                    label="Retirement",
                )

            ymin, ymax = np.min(y), np.max(y)

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
            ax.grid(True)
            ax.tick_params(axis="both", labelsize=tick_size)

            if yfmt:
                from matplotlib.ticker import FuncFormatter

                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

        # --- 1. Consumption ---
        plot_curve(
            axes[0],
            months,
            cons,
            color="tab:green",
            title="Optimal Consumption Over Time",
            yfmt=True,
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

        # --- 2. Wealth ---
        plot_curve(
            axes[1],
            months,
            wealth,
            color="tab:blue",
            title="Wealth Over Time",
            yfmt=True,
        )
        axes[1].legend(fontsize=legend_size)

        # --- 3. Investment / Withdrawal ---
        plot_curve(
            axes[2],
            months,
            inv,
            color="tab:purple",
            title="Monthly Investment / Withdrawal",
            yfmt=True,
        )
        axes[2].axhline(0.0, color="black", lw=1.0, alpha=0.7)
        axes[2].legend(fontsize=legend_size)

        plt.tight_layout()
        plt.show()
