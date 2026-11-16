import datetime as dt
from typing import Optional

from wealthplan.cashflows.cashflow_base import CashflowBase


class ScheduledExpense(CashflowBase):
    """
    Monthly expense paid from ``start_date`` (inclusive) until ``end_date`` (exclusive), e.g. the
    voluntary health insurance between an early retirement and the public pension.
    """

    def __init__(self, name: str, monthly_amount: float, start_date: Optional[dt.date] = None,
                 end_date: Optional[dt.date] = None) -> None:
        if monthly_amount < 0:
            raise ValueError("monthly_amount must be >= 0 (expenses are positive amounts)")
        self.name = name
        self.monthly_amount = monthly_amount
        self.start_date = start_date
        self.end_date = end_date

    def cashflow(self, delivery_date: dt.date) -> float:
        if self.start_date is not None and delivery_date < self.start_date:
            return 0.0
        if self.end_date is not None and delivery_date >= self.end_date:
            return 0.0
        return -self.monthly_amount
