from abc import ABC, abstractmethod
import datetime as dt


class PensionPlan(ABC):
    """
    Abstract base class for all pension plans.
    """

    name: str
    monthly_payout_brutto: float

    @abstractmethod
    def monthly_contribution(self, date: dt.date) -> float:
        """
        Return the monthly contribution for a given date.

        Returns
        -------
        float
            Positive contribution amount (0.0 if none).
        """
        pass