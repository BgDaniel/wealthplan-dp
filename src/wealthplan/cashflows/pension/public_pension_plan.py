import datetime as dt
from dataclasses import dataclass

from wealthplan.cashflows.pension.pension_plan import PensionPlan


@dataclass
class PublicPensionPlan(PensionPlan):
    """
    Represents a public (state) pension plan.
    """
    monthly_payout_brutto: float

    def monthly_contribution(self, date: dt.date) -> float:
        return 0.0
