import datetime as dt
from dataclasses import dataclass
from typing import Optional

from wealthplan.cashflows.pension.pension_plan import PensionPlan


@dataclass
class PublicPensionPlan(PensionPlan):
    """
    Represents a public (state) pension plan.

    ``monthly_payout_brutto`` is in euros of ``base_date`` (Renteninformation); the payout grows
    with ``indexation`` per year (Rentenanpassung) from ``base_date`` on.
    """
    monthly_payout_brutto: float
    name: str = "Public"
    indexation: float = 0.0
    base_date: Optional[dt.date] = None

    def payout(self, date: dt.date) -> float:
        if self.base_date is None or self.indexation == 0.0:
            return self.monthly_payout_brutto
        years = (date.year - self.base_date.year) + (date.month - self.base_date.month) / 12.0
        return self.monthly_payout_brutto * (1.0 + self.indexation) ** int(max(years, 0))

    def monthly_contribution(self, date: dt.date) -> float:
        return 0.0
