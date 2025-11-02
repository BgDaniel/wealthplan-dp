import datetime as dt
from typing import Optional
from src.wealthplan.cashflows.base import Cashflow

class LifeInsurance(Cashflow):
    """Represents life insurance with monthly payments and a possible payout."""

    def __init__(self, monthly_payment: float, payout: float, end_date: Optional[dt.date] = None) -> None:
        """
        Args:
            monthly_payment (float): Payment every month.
            payout (float): Payout at the end of insurance.
            end_date (Optional[date]): End date of insurance coverage.
        """
        self.monthly_payment = monthly_payment
        self.payout = payout
        self.end_date = end_date

    def cashflow(self, delivery_date: dt.date) -> float:
        if self.end_date and delivery_date > self.end_date:
            return 0.0
        return -self.monthly_payment
