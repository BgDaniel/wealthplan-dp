from abc import ABC, abstractmethod
import logging
import datetime as dt
from typing import List, Dict
import numpy as np
import pandas as pd

from src.wealthplan.cashflows.base import Cashflow
from wealthplan.cache.result_cache import ResultCache
from wealthplan.optimizer.math_tools.penality_functions import (
    square_penality,
    PenalityFunction,
)
from wealthplan.optimizer.math_tools.utility_functions import crra_utility

from wealthplan.optimizer.math_tools.utility_functions import UtilityFunction



logger = logging.getLogger(__name__)


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
        cashflows: List[Cashflow],
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
        self.monthly_return: float = (1 + self.yearly_return) ** (1/12) - 1
        self.cashflows: List[Cashflow] = cashflows
        self.instant_utility: UtilityFunction = instant_utility
        self.terminal_penalty: PenalityFunction = terminal_penalty

        self.dt = dt
        self.beta: float = 1.0 / (1.0 + self.monthly_return)
        self.w_max: float = w_max
        self.w_step: float = w_step
        self.c_step: float = c_step
        self.save: bool = save

        # Derived / prepared attributes
        self.months: List[dt.date] = []
        self.n_months: int = 0
        self.wealth_grid: np.ndarray = np.arange(
            0.0, self.w_max, self.w_step, dtype=np.float32
        )

        self.months = [
            d.date()
            for d in pd.date_range(start=self.start_date, end=self.end_date, freq="MS")
        ]
        self.n_months = len(self.months)

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
