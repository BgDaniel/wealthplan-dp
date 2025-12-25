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
        self._payout_delivered = False  # internal flag to ensure payout occurs only once

    def cashflow(self, delivery_date: dt.date) -> float:
        """
        Compute cashflow for a given date.

        - Monthly payment is paid every month until payout.
        - Payout occurs once at the first delivery date >= payout_date.
        """
        cf = -self.monthly_payment  # outflow each month

        # Deliver payout if not delivered yet and delivery_date >= payout_date
        if self.payout_date and not self._payout_delivered and delivery_date >= self.payout_date:
            cf += self.payout
            self._payout_delivered = True

        return cf