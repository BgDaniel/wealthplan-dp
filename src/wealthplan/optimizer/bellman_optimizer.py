from abc import ABC, abstractmethod
import logging
import datetime as dt
from typing import List, Dict
import numpy as np
import pandas as pd
from numba import njit

from src.wealthplan.cashflows.cashflow_base import CashflowBase
from wealthplan.cache.result_cache import ResultCache
from wealthplan.optimizer.math_tools.penality_functions import (
    square_penality,
    PenalityFunction,
)
from wealthplan.optimizer.math_tools.utility_functions import crra_utility

from wealthplan.optimizer.math_tools.utility_functions import UtilityFunction



logger = logging.getLogger(__name__)


@njit
def create_grid(min_val: float, max_val: float, delta: float):
    n = int(np.floor((max_val - min_val) / delta))

    size = n + 1
    if min_val + n * delta < max_val:
        size += 1

    grid = np.empty(size, dtype=np.float32)

    for i in range(n + 1):
        grid[i] = min_val + i * delta

    if size > n + 1:
        grid[-1] = max_val

    return grid



# ---------------------------
# Base optimizer (common API)
# ---------------------------
class BellmanOptimizer(ABC):
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
        run_id: str,
        start_date: dt.date,
        end_date: dt.date,
        retirement_date: dt.date,
        initial_wealth: float,
        yearly_return: float,
        beta: float,
        cashflows: List[CashflowBase],
        instant_utility: UtilityFunction = crra_utility,
        terminal_penalty: PenalityFunction = square_penality,
        dt: float = 1.0 / 12.0,
        w_max: float = 750000.0,
        w_step: float = 50.0,
        c_step: float = 50.0,
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
            dt: time step in years (default monthly = 1/12).
            save: whether to allow caching (Bellman uses it).
        """
        self.run_id: str = run_id
        self.start_date: dt.date = start_date
        self.end_date: dt.date = end_date
        self.retirement_date = retirement_date
        self.initial_wealth: float = initial_wealth
        self.yearly_return: float = yearly_return
        self.beta = beta
        self.monthly_return: float = (1 + self.yearly_return) ** (1/12) - 1
        self.cashflows: List[CashflowBase] = cashflows
        self.instant_utility: UtilityFunction = instant_utility
        self.terminal_penalty: PenalityFunction = terminal_penalty

        self.dt = dt
        self.w_max: float = w_max
        self.w_step: float = w_step
        self.c_step: float = c_step
        self.save: bool = save

        # Derived / prepared attributes
        self.months: List[dt.date] = []
        self.n_months: int = 0
        self.wealth_grid: np.ndarray = create_grid(0.0, self.w_max, self.w_step)

        self.months = [
            d.date()
            for d in pd.date_range(start=self.start_date, end=self.end_date, freq="MS")
        ]
        self.n_months = len(self.months)

        self.cf = np.array([self.monthly_cashflow(month) for month in self.months])

        self.time_grid = (
            np.array(
                [(m - self.months[0]).days / 365.0 for m in self.months],
                dtype=np.float64,
            )
        )

        self.cache = ResultCache(enabled=save, run_id=run_id)

        self.value_function: Dict[dt.date, np.ndarray()] = {}
        self.policy: Dict[dt.date, np.ndarray()] = {}

    def monthly_cashflow(self, date: dt.date) -> float:
        """Sum deterministic cashflows for the given month."""
        return sum(cf.cashflow(date) for cf in self.cashflows)

    def solve(self) -> None:
        """
        Generic solver that dynamically calls the child class implementation
        of backward induction and roll-forward to generate optimal paths.

        Stores results as instance attributes.

        Returns:
            None
        """
        logger.info("%s.solve() started.", self.__class__.__name__)

        # Backward induction step
        self._backward_induction()

        # Forward roll-out of paths
        logger.info("Rolling forward to compute optimal paths.")
        self._roll_forward()

        logger.info("%s.solve() finished.", self.__class__.__name__)

    @abstractmethod
    def _backward_induction(self) -> None:
        """
        Perform backward induction to compute value function and optimal policy.
        Must be implemented by subclasses.

        Returns:
            None
        """
        pass

    @abstractmethod
    def _roll_forward(self) -> None:
        """
        Use the computed policy to generate optimal wealth and consumption paths.
        Must be implemented by subclasses.

        Returns:
            None
        """
        pass

    @abstractmethod
    def plot(self) -> None:
        """
        Plot the results of the optimization (e.g. value function, policy).

        Returns:
            None
        """
        pass
