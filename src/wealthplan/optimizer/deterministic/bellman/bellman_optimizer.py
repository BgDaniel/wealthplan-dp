import logging
import datetime as dt
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


from src.wealthplan.optimizer.deterministic.base_optimizer import (
    BaseConsumptionOptimizer,
)


logger = logging.getLogger(__name__)


class BellmanOptimizer(BaseConsumptionOptimizer):
    """
    Deterministic Bellman (backward induction) optimizer.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        All parameters are forwarded to BaseConsumptionOptimizer.
        """
        super().__init__(*args, **kwargs)

    def _backward_induction(self) -> None:
        """
        Compute value function and policy by backward induction on discrete grids.
        Results are stored in self.value_function and self.policy keyed by date.
        """
        n_w = len(self.wealth_grid)
        # initialize terminal value
        v_next = np.array([self.terminal_penalty(w) for w in self.wealth_grid])

        r = self.wealth.monthly_return()
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

            # candidate consumptions
            available = self.wealth_grid + cf_t  # shape (n_w,)
            n_c = int(np.ceil(np.max(available) / c_step)) + 1
            c_grid = np.arange(0.0, n_c * c_step, c_step)  # shape (n_c,)

            # next period wealth for each (w, c)
            w_next = (available[:, None] - c_grid[None, :]) * (1 + r)  # (n_w, n_c)

            # interpolate v_next for each row of w_next
            # np.interp is 1d so vectorize with comprehension (ok for moderate grid sizes)
            v_next_interp = np.array(
                [np.interp(w_next[i], self.wealth_grid, v_next) for i in range(n_w)]
            )

            # instant utility vectorized (use log with numerical floor)
            instant_util = np.log(np.maximum(c_grid, 1e-8))  # shape (n_c,)
            values = instant_util[None, :] + beta * v_next_interp  # (n_w, n_c)

            # enforce feasibility (next wealth >= 0)
            feasible_mask = w_next >= 0
            feasible_values = np.where(feasible_mask, values, -np.inf)

            # select best consumption index per starting wealth
            best_idx = np.argmax(feasible_values, axis=1)
            v_curr = values[np.arange(n_w), best_idx]
            policy_curr = c_grid[best_idx]

            self.value_function[date_t] = v_curr.copy()
            self.policy[date_t] = policy_curr.copy()

            v_next = v_curr  # step backwards

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
        wealth_path[0] = self.wealth.initial_wealth
        cf0 = self.monthly_cashflow(self.months[0])
        consumption_path[0] = float(
            np.interp(wealth_path[0], self.wealth_grid, self.policy[self.months[0]])
        )
        cashflow_path[0] = cf0
        current_wealth = (wealth_path[0] + cf0 - consumption_path[0]) * (
            1 + self.wealth.monthly_return()
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
            current_wealth = (current_wealth + cf_t - c_opt) * (
                1 + self.wealth.monthly_return()
            )
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

    def solve(self) -> Tuple[Dict[dt.date, np.ndarray], Dict[dt.date, np.ndarray]]:
        """
        Solve with backward induction (or load cache if available), then roll forward.
        Returns (value_function, policy).
        """
        logger.info("BellmanOptimizer.solve() started.")
        cache_loaded = self._load_cache()
        if cache_loaded:
            logger.info("Cache loaded; skipping backward induction.")
        else:
            logger.info("Running backward induction.")
            self._backward_induction()
            if self.save:
                logger.info("Saving expansion to cache.")
                self._save_cache()

        logger.info("Rolling forward to get paths.")
        self._roll_forward()
        logger.info("BellmanOptimizer.solve() finished.")
        return self.value_function, self.policy
