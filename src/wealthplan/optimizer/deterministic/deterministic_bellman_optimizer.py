import logging
import datetime as dt
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from wealthplan.cache.result_cache import VALUE_FUNCTION_KEY, POLICY_KEY
from wealthplan.cashflows.base import Cashflow
from wealthplan.cashflows.salary import Salary
from wealthplan.optimizer.bellman_optimizer import (
    BellmanOptimizer,
)
from wealthplan.optimizer.math_tools.penality_functions import (
    PenalityFunction,
    square_penality,
)
from wealthplan.optimizer.math_tools.utility_functions import (
    UtilityFunction,
    crra_utility,
)

logger = logging.getLogger(__name__)


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
        beta: float = 1.0,
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
            beta=beta,
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
        n_w = len(self.wealth_grid)
        # initialize terminal value
        v_t_next = np.array([self.terminal_penalty(w) for w in self.wealth_grid])

        beta = self.beta
        c_step = self.c_step

        # iterate backwards (skip terminal period)
        for t in tqdm(
            reversed(range(self.n_months - 1)),
            total=max(0, self.n_months - 1),
            desc="Backward Induction",
        ):
            date_t = self.months[t]
            cf_t = self.monthly_cashflow(date_t)

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

            # candidate consumptions
            available = self.wealth_grid + cf_t  # shape (n_w,)
            n_c = int(np.ceil(np.max(available) / c_step)) + 1
            c_grid = np.arange(0.0, n_c * c_step, c_step)  # shape (n_c,)

            # next period wealth for each (w, c)
            w_next = (available[:, None] - c_grid[None, :]) * (
                1 + self.monthly_return
            )  # (n_w, n_c)

            # interpolate v_next for each row of w_next
            # np.interp is 1d so vectorize with comprehension (ok for moderate grid sizes)
            v_next_interp = np.array(
                [np.interp(w_next[i], self.wealth_grid, v_t_next) for i in range(n_w)]
            )

            # instant utility vectorized (use log with numerical floor)
            instant_util = np.log(np.maximum(c_grid, 1e-8))  # shape (n_c,)
            values = instant_util[None, :] + beta * v_next_interp  # (n_w, n_c)

            # enforce feasibility (next wealth >= 0)
            feasible_mask = w_next >= 0
            feasible_values = np.where(feasible_mask, values, -np.inf)

            # select best consumption index per starting wealth
            best_idx = np.argmax(feasible_values, axis=1)
            v_t = values[np.arange(n_w), best_idx]
            policy_t = c_grid[best_idx]

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
        n_months = self.n_months
        wealth_path = np.zeros(n_months)
        consumption_path = np.zeros(n_months)
        cashflow_path = np.zeros(n_months)

        # month 0
        wealth_path[0] = self.initial_wealth
        cf0 = self.monthly_cashflow(self.months[0])
        consumption_path[0] = float(
            np.interp(wealth_path[0], self.wealth_grid, self.policy[self.months[0]])
        )
        cashflow_path[0] = cf0
        current_wealth = (wealth_path[0] + cf0 - consumption_path[0]) * (
            1 + self.monthly_return
        )
        wealth_path[0] = wealth_path[0]  # keep initial wealth as-is in index 0

        # forward roll for months 1..n-1 (we only simulate up to n_months - 1)
        for t in range(1, n_months - 1):
            date_t = self.months[t]
            cf_t = self.monthly_cashflow(date_t)
            cashflow_path[t] = cf_t

            c_opt = float(
                np.interp(current_wealth, self.wealth_grid, self.policy[date_t])
            )
            consumption_path[t] = c_opt

            # wealth for next month (record current after return)
            current_wealth = (current_wealth + cf_t - c_opt) * (1 + self.monthly_return)
            wealth_path[t] = current_wealth

        # Final index handling (set last month values)
        if n_months >= 2:
            # set last month's cashflow (could be zero) and consumption via interpolation
            final_idx = n_months - 1
            date_f = self.months[final_idx]
            cashflow_path[final_idx] = self.monthly_cashflow(date_f)
            # For last month, if policy available, interpolate, otherwise 0
            if date_f in self.policy:
                consumption_path[final_idx] = float(
                    np.interp(current_wealth, self.wealth_grid, self.policy[date_f])
                )
            else:
                consumption_path[final_idx] = 0.0
            wealth_path[final_idx] = current_wealth

        # Save as pandas Series indexed by months
        self.opt_wealth = pd.Series(wealth_path, index=self.months)
        self.opt_consumption = pd.Series(consumption_path, index=self.months)
        self.monthly_cashflows = pd.Series(cashflow_path, index=self.months)

    def plot(self) -> None:
        """Plot the optimized wealth, consumption and cashflows (after solve())."""
        if self.opt_wealth.empty:
            raise RuntimeError("No solution available — call solve() first.")

        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Formatter for y-axis: thousand separators
        def thousands_formatter(x, pos):
            return f"{x:,.0f}"

        # --- Wealth plot ---
        axs[0].plot(
            self.opt_wealth.index,
            self.opt_wealth.values,
            label="Optimized Wealth",
            color="tab:blue",
            linewidth=2,
        )
        axs[0].axvline(
            self.retirement_date,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Retirement Date",
        )
        axs[0].set_ylabel("Wealth")
        axs[0].set_title("Optimized Wealth Over Time")
        axs[0].legend(loc="lower left")
        axs[0].grid(True, alpha=0.3)
        axs[0].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

        # --- Consumption + Cashflows plot ---
        axs[1].plot(
            self.opt_consumption.index,
            self.opt_consumption.values,
            label="Optimized Consumption",
            color="tab:green",
            linewidth=2,
        )
        axs[1].plot(
            self.monthly_cashflows.index,
            self.monthly_cashflows.values,
            label="Monthly Cashflows",
            color="tab:orange",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
        )
        axs[1].axvline(
            self.retirement_date,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Retirement Date",
        )
        axs[1].set_ylabel("Consumption / Cashflows")
        axs[1].set_title("Consumption and Cashflows Over Time")
        axs[1].legend(loc="lower left")
        axs[1].grid(True, alpha=0.3)
        axs[1].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()
