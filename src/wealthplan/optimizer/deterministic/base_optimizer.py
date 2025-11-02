import os
import pickle
import logging
import datetime as dt
from typing import List, Dict, Callable, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.wealthplan.cashflows.base import Cashflow
from src.wealthplan.wealth import Wealth

logger = logging.getLogger(__name__)

# Cache constants
ENV_CACHE_FOLDER = "WEALTHPLAN_CACHE"
CACHE_FILE_NAME = "bellman_cache.pkl"
CACHE_KEY_VALUE_FUNCTION = "value_function"
CACHE_KEY_POLICY = "policy"


# ---------------------------
# Base optimizer (common API)
# ---------------------------
class BaseConsumptionOptimizer:
    """
    Base class that stores problem data and provides a common interface for
    concrete solver implementations.

    Subclasses must implement `solve()` to populate:
        - self.value_function: Dict[dt.date, np.ndarray]
        - self.policy: Dict[dt.date, np.ndarray]
        - self.opt_wealth: pd.Series
        - self.opt_consumption: pd.Series
        - self.monthly_cashflows: pd.Series
    """

    def __init__(
        self,
        start_date: dt.date,
        end_date: dt.date,
        wealth: Wealth,
        cashflows: List[Cashflow],
        beta: float = 1.0,
        w_max: float = 250_000.0,
        w_min: float = 0.0,
        w_step: float = 50.0,
        c_step: float = 50.0,
        instant_utility: Optional[Callable[[float], float]] = None,
        terminal_penalty: Optional[Callable[[float], float]] = None,
        save: bool = True,
    ) -> None:
        """
        Initialize common problem data.

        Args:
            start_date: simulation start (inclusive).
            end_date: simulation end (inclusive).
            wealth: Wealth object providing .initial_wealth and monthly_return().
            cashflows: list of Cashflow objects with .cashflow(date) method.
            beta: monthly discount factor.
            w_max, w_min, w_step: wealth grid spec.
            c_step: consumption discretization step (used by Bellman).
            instant_utility: u(c) function (defaults to log utility).
            terminal_penalty: function penalizing terminal wealth (default -w^2).
            save: whether to allow caching (Bellman uses it).
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

        # utilities
        self.instant_utility: Callable[[float], float] = instant_utility or (
            lambda c: np.log(c) if c > 0 else -1e10
        )
        self.terminal_penalty: Callable[[float], float] = terminal_penalty or (
            lambda w: -(w**2)
        )

        self.save: bool = save

        # Derived / prepared attributes
        self.months: List[dt.date] = []
        self.n_months: int = 0
        self.wealth_grid: np.ndarray = np.array([])

        # outputs (to be populated by solve())
        self.value_function: Dict[dt.date, np.ndarray] = {}
        self.policy: Dict[dt.date, np.ndarray] = {}
        self.opt_wealth: pd.Series = pd.Series(dtype=float)
        self.opt_consumption: pd.Series = pd.Series(dtype=float)
        self.monthly_cashflows: pd.Series = pd.Series(dtype=float)

        # Prepare commonly-used structures
        self.prepare()

    def prepare(self) -> None:
        """
        Prepare time grid (months) and discrete wealth grid.
        Can be extended/overridden by subclasses if they need extra precomputation.
        """
        self.months = [
            d.date()
            for d in pd.date_range(start=self.start_date, end=self.end_date, freq="MS")
        ]
        self.n_months = len(self.months)
        self.wealth_grid = np.arange(self.w_min, self.w_max + self.w_step, self.w_step)

    def monthly_cashflow(self, date: dt.date) -> float:
        """Sum deterministic cashflows for the given month."""
        return sum(cf.cashflow(date) for cf in self.cashflows)

    # ----- Cache helpers (Bellman may use these) -----
    def _get_cache_file(self) -> str:
        """Return path to the cache file (creates folder if needed)."""
        cache_folder = os.environ.get(ENV_CACHE_FOLDER, "./cache")
        os.makedirs(cache_folder, exist_ok=True)
        return os.path.join(cache_folder, CACHE_FILE_NAME)

    def _load_cache(self) -> bool:
        """
        Try to load cached (value_function, policy). Returns True on success.
        Subclasses may call this; it's a convenience.
        """
        if not self.save:
            return False
        cache_file = self._get_cache_file()
        if os.path.isfile(cache_file):
            logger.info("Loading cache from %s", cache_file)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            self.value_function = data.get(CACHE_KEY_VALUE_FUNCTION, {})
            self.policy = data.get(CACHE_KEY_POLICY, {})
            return True
        return False

    def _save_cache(self) -> None:
        """Save current (value_function, policy) to the fixed cache file."""
        if not self.save:
            return
        cache_file = self._get_cache_file()
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    CACHE_KEY_VALUE_FUNCTION: self.value_function,
                    CACHE_KEY_POLICY: self.policy,
                },
                f,
            )

    # ----- Interface -----
    def solve(self) -> Tuple[Dict[dt.date, np.ndarray], Dict[dt.date, np.ndarray]]:
        """
        Solve the optimization problem.

        Must be implemented by subclasses. Should fill in:
            - self.value_function (per-date arrays) -- may be empty for open-loop solvers
            - self.policy (per-date arrays)
            - self.opt_wealth, self.opt_consumption, self.monthly_cashflows
        and return (value_function, policy).
        """
        raise NotImplementedError("Subclasses must implement solve().")

    def plot(self) -> None:
        """Plot the optimized wealth, consumption and cashflows (after solve())."""
        if self.opt_wealth.empty:
            raise RuntimeError("No solution available — call solve() first.")

        fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        axs[0].plot(
            self.opt_wealth.index, self.opt_wealth.values, label="Optimized Wealth"
        )
        axs[0].set_ylabel("Wealth")
        axs[0].set_title("Optimized Wealth Over Time")
        axs[0].legend(loc="upper left")
        axs[0].grid(True)

        axs[1].plot(
            self.opt_consumption.index,
            self.opt_consumption.values,
            label="Optimized Consumption",
            linewidth=2,
        )
        axs[1].set_ylabel("Consumption")
        axs[1].set_title("Optimized Consumption Over Time")
        axs[1].legend(loc="upper left")
        axs[1].grid(True)

        axs[2].bar(
            self.monthly_cashflows.index,
            self.monthly_cashflows.values,
            alpha=0.7,
            label="Monthly Cashflows",
        )
        axs[2].set_ylabel("Cashflows")
        axs[2].set_title("Monthly Cashflows")
        axs[2].legend(loc="upper left")
        axs[2].grid(True)

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()
