import datetime as dt
from src.wealthplan.cashflows.base import Cashflow


class Pension(Cashflow):
    """
    Deterministic monthly pension cashflow starting at retirement.

    Attributes:
        monthly_amount (float): Monthly pension payment.
        retirement_date (dt.date): Date when pension payments start.
    """

    def __init__(self, monthly_amount: float, retirement_date: dt.date) -> None:
        """
        Initialize a Pension cashflow.

        Args:
            monthly_amount (float): Monthly payment amount.
            retirement_date (dt.date): Date when pension payments start.
        """
        self.monthly_amount: float = monthly_amount
        self.retirement_date: dt.date = retirement_date

    def cashflow(self, date: dt.date) -> float:
        """
        Return the cashflow for a given date.

        Args:
            date (dt.date): Date to evaluate cashflow.

        Returns:
            float: Monthly pension if date is at or after retirement, otherwise 0.
        """
        if date >= self.retirement_date:
            return self.monthly_amount
        return 0.0
