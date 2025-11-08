# src/wealthplan/cashflows/essential_expenses.py

import datetime as dt
from typing import Union
from src.wealthplan.cashflows.base import Cashflow

class EssentialExpenses(Cashflow):
    """
    Represents recurring, non-discretionary living expenses such as food,
    housing, utilities, and commuting costs.

    These are modeled as negative cashflows (outflows) that occur
    periodically (e.g., monthly) throughout the simulation horizon.

    Parameters
    ----------
    monthly_expenses : float
        Base monthly expense amount (positive value).
    start_date : datetime.date
        First date of expense.
    end_date : datetime.date
        Last date of expense.
    inflation_rate : float, optional
        Annual inflation rate to apply (default: 0.02).
    """

    def __init__(
        self,
        monthly_expenses: float,
    ) -> None:
        self.monthly_expenses = monthly_expenses

    def cashflow(self, date: dt.date) -> float:
        """Return the expense for the given date (negative value)."""
        return -self.monthly_expenses
