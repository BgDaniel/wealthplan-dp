import datetime as dt
from typing import Optional
from src.wealthplan.cashflows.base import Cashflow


class LifeInsurance(Cashflow):
    """
    Represents life insurance with monthly payments and a one-time payout.

    The monthly payment is negative (cash outflow), and the payout is positive (cash inflow)
    delivered once at the first delivery date on or after `payout_date`.
    """

    def __init__(self, monthly_payment: float, payout: float, payout_date: Optional[dt.date] = None) -> None:
        """
        Args:
            monthly_payment (float): Payment every month (negative cashflow).
            payout (float): One-time payout at the end of insurance.
            payout_date (Optional[date]): Date when payout is delivered.
        """
        self.monthly_payment = monthly_payment
        self.payout = payout
        self.payout_date = payout_date

    def cashflow(self, delivery_date: dt.date) -> float:
        """
        Compute cashflow for a given date.

        - Monthly payment is paid every month until payout.
        - Payout occurs once at the first delivery date >= payout_date.
        """
        if self.payout_date == delivery_date:
            return self.payout
        elif delivery_date < self.payout_date:
            return -self.monthly_payment
        else:
            return 0.0
