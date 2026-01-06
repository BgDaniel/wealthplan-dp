# src/wealthplan/cashflows/insurances.py

import datetime as dt
from typing import Literal

from src.wealthplan.cashflows.cashflow_base import CashflowBase


class Insurance(CashflowBase):
    """
    Represents recurring insurance expenses.

    Insurance payments are modeled as fixed, non-discretionary cashflows,
    occurring either monthly or yearly. Payments are made on the first day
    of each month (for monthly frequency) or the first day of the year (for yearly frequency).

    Parameters
    ----------
    amount : float
        Insurance payment amount (positive value).
    frequency : {'M', 'Y'}
        Payment frequency: 'M' for monthly, 'Y' for yearly.
    """

    def __init__(self, amount: float, name: str, frequency: Literal["M", "Y"]) -> None:
        self.amount = amount
        self.name = name

        if frequency not in ("M", "Y"):
            raise ValueError("frequency must be 'M' (monthly) or 'Y' (yearly)")

        self.frequency = frequency

    def cashflow(self, date: dt.date) -> float:
        """
        Return the insurance expense for the given date.

        Payments occur only on the first day of the period according to the frequency.

        Parameters
        ----------
        date : datetime.date
            Date for which the cashflow is evaluated.

        Returns
        -------
        float
            Insurance expense as a negative cashflow on the payment date, else 0.
        """
        if self.frequency == "M" and date.day == 1:
            return -self.amount
        elif self.frequency == "Y" and date.day == 1 and date.month == 1:
            return -self.amount
        else:
            return 0.0
