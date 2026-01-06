import datetime as dt

from wealthplan.cashflows.cashflow_base import CashflowBase


class Telecommunication(CashflowBase):
    """Represents monthly telecommunication payments (cell phone, internet)."""

    def __init__(self, monthly_amount: float) -> None:
        """
        Args:
            monthly_amount (float): Telecommunication bill paid every month.
        """
        self.monthly_amount = monthly_amount

    def cashflow(self, delivery_date: dt.date) -> float:
        # Always an outflow
        return -self.monthly_amount
