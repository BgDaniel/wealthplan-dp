import datetime as dt
import pickle
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


from src.wealthplan.cashflows.base import Cashflow
from src.wealthplan.wealth import Wealth


logger = logging.getLogger(__name__)


# Constants
ENV_CACHE_FOLDER = "WEALTHPLAN_CACHE"
CACHE_KEY_VALUE_FUNCTION = "value_function"
CACHE_KEY_POLICY = "policy"
CACHE_FILE_NAME = "bellman_cache.pkl"


class BellmanOptimizer:
    """
    Bellman optimizer for deterministic multi-period consumption problems.
    """

    def __init__(
        self,
        start_date: dt.date,
        end_date: dt.date,
        wealth: Wealth,
        cashflows: List[Cashflow],
        beta: float = 1.0,
        w_max: float = 250000.0,
        w_min: float = 0.0,
        w_step: float = 50.0,
        c_step: float = 50.0,
        instant_utility: Callable[[float], float] = None,
        terminal_penalty: Callable[[float], float] = None,
        save: bool = True,
    ) -> None:
        """
        Initialize the Bellman optimizer.

        Args:
            start_date: Simulation start date.
            end_date: Simulation end date.
            wealth: Wealth object.
            cashflows: List of deterministic cashflows.
            beta: Monthly discount factor.
            w_max: Maximum wealth in grid.
            w_min: Minimum wealth in grid.
            w_step: Wealth discretization step.
            c_step: Consumption discretization step.
            instant_utility: Function mapping consumption to utility.
            terminal_penalty: Function mapping terminal wealth to penalty.
            save: Whether to save backward induction results for reuse.
        """
        self.start_date: dt.date = start_date
        self.end_date: dt.date = end_date
        self.wealth: Wealth = wealth
        self.cashflows: List[Cashflow] = cashflows
        self.beta: float = beta
        self.w_max: float = w_max
        self.w_min: float = w_min
        self.w_step: float = w_step
        self.c_step: float = c_step
        self.save: bool = save  # store the flag

        self.instant_utility: Callable[[float], float] = instant_utility or (
            lambda c: np.log(c) if c > 0 else -1e10
        )
        self.terminal_penalty: Callable[[float], float] = terminal_penalty or (
            lambda w: -(w**2)
        )

        self.months: List[dt.date] = [
            d.date()
            for d in pd.date_range(start=self.start_date, end=self.end_date, freq="MS")
        ]
        self.n_months: int = len(self.months)

        self.value_function: Dict[dt.date, np.ndarray] = {}  # shape = (n_wealth,)
        self.policy: Dict[dt.date, np.ndarray] = {}  # shape = (n_wealth,)
        self.wealth_grid: np.ndarray = np.arange(
            self.w_min, self.w_max + self.w_step, self.w_step
        )

        self.opt_wealth: pd.Series = pd.Series(dtype=float)
        self.opt_consumption: pd.Series = pd.Series(dtype=float)
        self.monthly_cashflows: pd.Series = pd.Series(dtype=float)

    def monthly_cashflow(self, date: dt.date) -> float:
        """Sum of all deterministic cashflows for a given month."""
        return sum(cf.cashflow(date) for cf in self.cashflows)

    def _optimize_consumption(self, w: float, cf_t: float, v_next: np.ndarray) -> float:
        """
        Return optimal consumption for given wealth and cashflow.

        Args:
            w: Current wealth.
            cf_t: Cashflow this month.
            v_next: Next month's value function (array).

        Returns:
            Optimal consumption (float).
        """
        available = w + cf_t
        c_grid = np.arange(0.0, available + self.c_step, self.c_step)
        values = np.array(
            [
                self.instant_utility(c)
                + self.beta
                * np.interp(
                    (w + cf_t - c) * (1 + self.wealth.monthly_return()),
                    self.wealth_grid,
                    v_next,
                )
                for c in c_grid
            ]
        )
        best_idx = np.argmax(values)
        return c_grid[best_idx]

    def _backward_induction(self) -> None:
        """
        Perform backward induction to compute the value function and policy.
        Uses NumPy arrays for performance and a separate consumption optimization function.
        """
        # Initialize terminal value function
        n_w = len(self.wealth_grid)
        v_next = np.array([self.terminal_penalty(w) for w in self.wealth_grid])

        r = self.wealth.monthly_return()  # precompute monthly return
        beta = self.beta
        c_step = self.c_step

        # Skip last month since terminal is already initialized
        for t in tqdm(
            reversed(range(self.n_months - 1)),
            total=self.n_months - 1,
            desc="Backward Induction",
        ):
            date_t = self.months[t]
            cf_t = self.monthly_cashflow(date_t)

            # Create candidate consumption grid for all wealth states
            available = self.wealth_grid + cf_t  # shape (n_w,)
            n_c = int(np.ceil(np.max(available) / c_step)) + 1
            c_grid = np.arange(0.0, n_c * c_step, c_step)  # shape (n_c,)

            # Broadcast to create all (wealth x consumption) combinations
            w_next = (available[:, None] - c_grid[None, :]) * (
                1 + r
            )  # shape (n_w, n_c)

            # Interpolate next period value function
            v_next_interp = np.array(
                [np.interp(w_next[i], self.wealth_grid, v_next) for i in range(n_w)]
            )

            # Instant utility (vectorized)
            instant_util = np.log(np.maximum(c_grid, 1e-8))  # shape (n_c,)
            values = instant_util[None, :] + beta * v_next_interp  # shape (n_w, n_c)

            # Pick optimal consumption per wealth state
            best_idx = np.argmax(values, axis=1)
            v_curr = values[np.arange(n_w), best_idx]
            policy_curr = c_grid[best_idx]

            # Save dicts
            self.value_function[date_t] = (
                v_curr.copy()
            )  # NumPy array of shape (n_wealth,)
            self.policy[date_t] = policy_curr.copy()

            # Prepare for next iteration
            v_next = v_curr

    def _roll_forward(self) -> None:
        """
        Roll forward to compute optimized paths for wealth, consumption, and cashflows.
        Stores results as pandas Series.
        This version keeps the initial wealth unchanged for month 0.
        """
        n_months = self.n_months

        # Pre-allocate arrays
        wealth_path = np.zeros(n_months)
        consumption_path = np.zeros(n_months)
        cashflow_path = np.zeros(n_months)

        # Month 0: initial wealth
        wealth_path[0] = self.wealth.initial_wealth
        cf_t0 = self.monthly_cashflow(self.months[0])
        c_opt0 = np.interp(
            wealth_path[0], self.wealth_grid, self.policy[self.months[0]]
        )
        consumption_path[0] = c_opt0
        cashflow_path[0] = cf_t0

        # Update wealth for next month
        current_wealth = (wealth_path[0] + cf_t0 - c_opt0) * (
            1 + self.wealth.monthly_return()
        )

        # Forward roll for remaining months
        for t in range(1, n_months):
            date_t = self.months[t]
            cf_t = self.monthly_cashflow(date_t)
            cashflow_path[t] = cf_t

            # Optimal consumption for current wealth
            c_opt = np.interp(current_wealth, self.wealth_grid, self.policy[date_t])
            consumption_path[t] = c_opt

            # Update wealth for next month
            current_wealth = (current_wealth + cf_t - c_opt) * (
                1 + self.wealth.monthly_return()
            )
            wealth_path[t] = current_wealth

        # Convert to pandas Series
        self.opt_wealth = pd.Series(wealth_path, index=self.months)
        self.opt_consumption = pd.Series(consumption_path, index=self.months)
        self.monthly_cashflows = pd.Series(cashflow_path, index=self.months)

    def _get_cache_file(self) -> str:
        """Return the path to the fixed cache file."""
        cache_folder = os.environ.get(ENV_CACHE_FOLDER, "./cache")
        os.makedirs(cache_folder, exist_ok=True)
        return os.path.join(cache_folder, CACHE_FILE_NAME)

    def _load_cache(self) -> bool:
        """
        Try to load backward induction results from cache.

        Returns:
            True if cache was loaded, False otherwise.
        """
        if not self.save:
            return False

        cache_file = self._get_cache_file()

        if os.path.isfile(cache_file):
            print(f"Loading cached backward induction data from {cache_file}...")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                self.value_function = cache_data[CACHE_KEY_VALUE_FUNCTION]
                self.policy = cache_data[CACHE_KEY_POLICY]
            return True

        return False

    def _save_cache(self) -> None:
        """Save backward induction results to cache."""
        if not self.save:
            return

        cache_file = self._get_cache_file()

        with open(cache_file, "wb") as f:
            pickle.dump(
                {CACHE_KEY_VALUE_FUNCTION: self.value_function, CACHE_KEY_POLICY: self.policy},
                f
            )

    def solve(self) -> Tuple[Dict[dt.date, np.ndarray], Dict[dt.date, np.ndarray]]:
        """
        Solve the Bellman equation and record optimized paths.
        Uses cached backward induction if available.
        """
        logger.info("Starting BellmanOptimizer.solve()")

        # Try loading from cache
        cache_loaded = self._load_cache()

        if cache_loaded:
            logger.info("Cache loaded successfully. Skipping backward induction.")
        else:
            logger.info("No cache found or saving disabled. Running backward induction...")
            self._backward_induction()

            if self.save:
                logger.info("Saving backward induction results to cache...")
                self._save_cache()

        logger.info("Rolling forward to compute optimized paths...")
        self._roll_forward()
        logger.info("Finished solving Bellman optimization.")

        return self.value_function, self.policy

    def plot_paths(self) -> None:
        """Plot wealth, consumption, and cashflows over time."""
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1 = axs[0]
        ax2 = ax1.twinx()
        ax1.plot(
            self.opt_wealth.index,
            self.opt_wealth.values,
            color="blue",
            label="Optimized Wealth",
        )
        ax2.plot(
            self.opt_consumption.index,
            self.opt_consumption.values,
            color="green",
            linestyle="--",
            label="Optimized Consumption",
        )
        ax1.set_ylabel("Wealth")
        ax2.set_ylabel("Consumption")
        ax1.set_title("Optimized Wealth and Consumption")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        axs[1].bar(
            self.monthly_cashflows.index, self.monthly_cashflows.values, color="orange"
        )
        axs[1].set_ylabel("Cashflows")
        axs[1].set_title("Monthly Cashflows (Salary, Rent, Insurance, Pension, etc.)")

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()
