import logging
import datetime as dt
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import numba as nb


from src.wealthplan.optimizer.deterministic.base_optimizer import (
    BaseConsumptionOptimizer,
)
from wealthplan.optimizer.stochastic.market_model.gbm_returns import GBM
from wealthplan.optimizer.stochastic.survival_process.survival_process import (
    SurvivalProcess,
)
from wealthplan.optimizer.stochastic.wealth_regression.wealth_regression import (
    WealthRegressor,
)

logger = logging.getLogger(__name__)


@nb.njit(parallel=True)
def interp_2d(
    w_next: np.ndarray, wealth_grid: np.ndarray, v_regressed: np.ndarray
) -> np.ndarray:
    """
    Perform 2D interpolation of regressed continuation values along wealth for each simulation.

    Parameters
    ----------
    w_next : np.ndarray, shape (n_w, n_c, n_sims)
        Wealth points to interpolate to.
    wealth_grid : np.ndarray, shape (n_w,)
        Grid of wealth points corresponding to v_regressed.
    v_regressed : np.ndarray, shape (n_w, n_sims)
        Regressed continuation values.

    Returns
    -------
    np.ndarray, shape (n_w, n_c, n_sims)
        Interpolated continuation values at `w_next`.
    """
    n_w, n_c, n_sims = w_next.shape
    v_next_interp = np.empty_like(w_next, dtype=np.float32)

    for s in nb.prange(n_sims):
        for i in range(n_w):
            # np.interp is not natively supported by Numba, so we need a workaround
            # But if using object mode or JIT with nopython=False, it will work
            v_next_interp[i, :, s] = np.interp(
                w_next[i, :, s], wealth_grid, v_regressed[:, s]
            )

    return v_next_interp


class StochasticBellmanOptimizer(BaseConsumptionOptimizer):
    def __init__(
        self,
        gbm_returns: GBM,
        survival_process: SurvivalProcess,
        n_sims: int,
        w_max: float = 500000.0,
        w_min: float = 0.0,
        w_step: float = 1000.0,
        c_step: float = 500.0,
        r_squared_threshold: float = 0.85,
        *args,
        **kwargs,
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
        super().__init__(
            w_max=w_max, w_min=w_min, w_step=w_step, c_step=c_step, *args, **kwargs
        )

        self.gbm_returns = gbm_returns
        self.survival_process = survival_process

        self.n_sims = n_sims

        self.returns_paths: np.ndarray = self.gbm_returns.simulate(
            n_sims=n_sims, dates=self.months
        )

        self.survival_paths: np.ndarray = self.survival_process.simulate(
            n_sims=n_sims, dates=self.months
        )

        self.wealth_grid = np.arange(
            self.w_min, self.w_max, self.w_step, dtype=np.float32
        )

        self.r_squared_threshold = r_squared_threshold

        self.r_squared = {}

    def _regress_value_function(
        self,
        date_t: dt.date,
        v_next: np.ndarray,
        returns_paths: np.ndarray,
        survival_paths: np.ndarray,
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
        v_regressed, r_squared = regressor.regress(v_next=v_next)

        self.r_squared[date_t] = r_squared

        # Check threshold
        if r_squared < self.r_squared_threshold:
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

    def _select_best_policy(self, value_candidates: np.ndarray, w_next: np.ndarray, c_grid: np.ndarray):
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
        n_w = len(self.wealth_grid)
        # initialize terminal value

        v_terminal = np.array(
            [self.terminal_penalty(w) for w in self.wealth_grid], dtype=np.float32
        )

        v_next = np.tile(v_terminal[:, np.newaxis], (1, self.n_sims))

        r = self.returns_paths[:, 1:] / self.returns_paths[:, :-1]

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

            # Candidate consumption grid and available wealth
            available, c_grid = self._compute_available_wealth(cf_t)

            # Next-period wealth for all (w, c, sims)
            w_next = (available[:, None, :] - c_grid[None, :, None]) * r[:, t][
                None, None, :
            ]  # (n_w, n_c, n_sims)

            # Regression of continuation value
            v_regressed = self._regress_value_function(
                date_t, v_next, self.returns_paths[:, t], self.survival_paths[:, t]
            )  # (n_w, n_sims)

            # Interpolate continuation value over wealth
            v_next_interp = interp_2d(w_next, self.wealth_grid, v_regressed)

            # Compute total value including instant utility
            value_candidates = self._compute_value_candidates(
                c_grid, v_next_interp, t, beta
            )

            # Enforce feasibility and select best consumption
            optimal_value, optimal_policy = self._select_best_policy(
                value_candidates, w_next,c_grid
            )

            # Store results
            self.value_function[date_t] = optimal_value.copy()
            self.policy[date_t] = optimal_policy.copy()

            # Step backwards
            v_next = optimal_value

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
        """Solve with backward induction (or load cache if available), then roll forward.
        Returns (value_function, policy).
        """
        logger.info("BellmanOptimizer.solve() started.")

        self._backward_induction()

        logger.info("Rolling forward to get paths.")
        self._roll_forward()

        logger.info("BellmanOptimizer.solve() finished.")

        return self.value_function, self.policy
