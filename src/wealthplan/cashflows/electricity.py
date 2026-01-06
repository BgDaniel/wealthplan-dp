import datetime as dt
from src.wealthplan.cashflows.base import Cashflow


class Electricity(Cashflow):
    """Represents monthly electricity payments."""

    def __init__(self, monthly_amount: float) -> None:
        """
        Args:
            monthly_amount (float): Electricity bill paid every month.
        """
        self.monthly_amount = monthly_amount

    def cashflow(self, delivery_date: dt.date) -> float:
        # Always an outflow
        return -self.monthly_amount