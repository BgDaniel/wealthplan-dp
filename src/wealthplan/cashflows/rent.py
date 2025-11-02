import datetime as dt
from src.wealthplan.cashflows.base import Cashflow

class Rent(Cashflow):
    """Represents monthly rent payments."""

    def __init__(self, monthly_rent: float) -> None:
        """
        Args:
            monthly_rent (float): Rent paid every month.
        """
        self.monthly_rent = monthly_rent

    def cashflow(self, delivery_date: dt.date) -> float:
        return -self.monthly_rent
