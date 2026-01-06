import datetime as dt
from src.wealthplan.cashflows.base import Cashflow


class Salary(Cashflow):
    """Represents a monthly salary until retirement."""

    def __init__(self, monthly_amount: float, retirement_date: dt.date) -> None:
        """
        Args:
            monthly_amount (float): Salary received every month.
            retirement_date (date): Date when salary stops.
        """
        self.monthly_amount = monthly_amount
        self.retirement_date = retirement_date

    def cashflow(self, delivery_date: dt.date) -> float:
        return self.monthly_amount if delivery_date < self.retirement_date else 0.0
