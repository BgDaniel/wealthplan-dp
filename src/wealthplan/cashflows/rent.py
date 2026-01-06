import datetime as dt
from src.wealthplan.cashflows.cashflow_base import CashflowBase

class Rent(CashflowBase):
    """Represents monthly rent payments."""

    def __init__(self, monthly_amount: float) -> None:
        """
        Args:
            monthly_amount (float): Rent paid every month.
        """
        self.monthly_amount = monthly_amount

    def cashflow(self, delivery_date: dt.date) -> float:
        return -self.monthly_amount
