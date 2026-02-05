import logging
import datetime as dt
import math
from numba import njit, prange
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm


from result_cache.result_cache import VALUE_FUNCTION_KEY, POLICY_KEY, ResultCache
from wealthplan.cashflows.cashflow_base import CashflowBase

from wealthplan.optimizer.math_tools.penality_functions import (
    square_penalty,
)
from wealthplan.optimizer.math_tools.utility_functions import (
    crra_utility_numba
)
from wealthplan.optimizer.optimizer_base import create_grid
from wealthplan.optimizer.stochastic.stochastic_optimizer import StochasticOptimizerBase
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
    gamma,
    epsilon
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

        instant_util_arr = crra_utility_numba(c_cands, gamma, epsilon)
        total_val_arr = instant_util_arr + beta * q_t * (
            p * V_up_arr + (1 - p) * V_down_arr
        )

        idx_max = np.argmax(total_val_arr)
        v_opt[i] = total_val_arr[idx_max]
        consumption_opt[i] = c_cands[idx_max]

    return v_opt, consumption_opt


class BinTreeOptimizer(StochasticOptimizerBase):
    """
    Binomial tree-based Bellman optimizer.
    """

    def __init__(
        self,
        run_config_id: str,
        start_date: dt.date,
        end_date: dt.date,
        retirement_date: dt.date,
        initial_wealth: float,
        yearly_return: float,
        cashflows: List[CashflowBase],
        survival_model: SurvivalModel,
        current_age: int,
        beta: float,
        sigma: float,
        n_sims: int,
        seed: int,
        w_max: float,
        w_step: float,
        c_step: float,
        use_cache: bool,
        gamma: float,
        epsilon: float,
        stochastic: bool,
        use_dynamic_grid: bool = False
    ) -> None:
        """
        Initialize the binomial tree Bellman optimizer.
        """
        # ---- Call base constructor ----
        super().__init__(run_config_id=run_config_id, start_date=start_date, end_date=end_date,
                         retirement_date=retirement_date, initial_wealth=initial_wealth,
                         yearly_return=yearly_return, cashflows=cashflows, survival_model=survival_model, current_age=current_age,
                         stochastic=stochastic)

        self.beta = beta
        self.sigma = sigma
        self.n_sims = n_sims
        self.seed = seed

        self.w_max = w_max
        self.w_step = w_step

        self.wealth_grid = create_grid(min_val=0.0, max_val=self.w_max, delta=self.w_step)

        self.c_step = c_step

        self.use_cache = use_cache
        self.cache = ResultCache(run_id=run_config_id, enabled=self.use_cache)

        self.gamma = gamma
        self.epsilon = epsilon

        self.terminal_penalty = square_penalty

        self.use_dynamic_grid = use_dynamic_grid

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

    def backward_induction(self) -> None:
        """
        Compute value function and policy by backward induction on discrete grids.
        Results are stored in self.value_function and self.policy keyed by date.
        Handles binomial tree structure for risky asset and survival probabilities.
        """
        self.value_function = {}
        self.policy = {}

        n_t = self.n_months

        # -------------------------
        # Initialize terminal value
        # -------------------------
        wealth_grid = (
            self.dynamic_wealth_grid[-1]
            if self.use_dynamic_grid
            else self.wealth_grid
        )

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
            q_t = self.survival_probabilities[t]

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
                if self.use_dynamic_grid
                else self.wealth_grid
            )

            n_w = len(wealth_grid)

            v_t = np.zeros((n_s, n_w), dtype=np.float32)
            policy_t = np.zeros((n_s, n_w), dtype=np.float32)

            results = []

            wealth_grid_next = (
                self.dynamic_wealth_grid[t + 1]
                if self.use_dynamic_grid
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
                        self.gamma,
                        self.epsilon
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

    def roll_forward(self):
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
            if self.use_dynamic_grid
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
                    if self.use_dynamic_grid
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

