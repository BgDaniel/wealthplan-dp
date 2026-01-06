from abc import ABC, abstractmethod
import datetime as dt

class CashflowBase(ABC):
    """Abstract base class for any cashflow."""

    @abstractmethod
    def cashflow(self, delivery_date: dt.date) -> float:
        """
        Return the cashflow amount for a given delivery date.

        Args:
            delivery_date (date): The date for which the cashflow is calculated.

        Returns:
            float: Cashflow amount (positive for inflow, negative for outflow).
        """
        pass
