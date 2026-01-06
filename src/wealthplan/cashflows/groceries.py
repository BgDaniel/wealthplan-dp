# src/wealthplan/cashflows/groceries.py

import datetime as dt

from src.wealthplan.cashflows.cashflow_base import CashflowBase


class Groceries(CashflowBase):
    """
    Represents recurring grocery expenses.

    Grocery expenses are modeled as fixed, non-discretionary monthly
    outflows and are expressed as negative cashflows.

    Parameters
    ----------
    monthly_amount : float
        Monthly grocery spending amount (positive value).
    """

    def __init__(self, monthly_amount: float) -> None:
        self.monthly_amount = monthly_amount

    def cashflow(self, date: dt.date) -> float:
        """
        Return the grocery expense for the given date.

        Parameters
        ----------
        date : datetime.date
            Date for which the cashflow is evaluated.

        Returns
        -------
        float
            Monthly grocery expense as a negative cashflow.
        """
        return -self.monthly_amount
