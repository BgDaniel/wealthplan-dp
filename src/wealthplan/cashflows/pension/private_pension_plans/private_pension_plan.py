import datetime as dt
from dataclasses import dataclass

from wealthplan.cashflows.pension.pension_plan import PensionPlan


@dataclass
class PrivatePensionPlan(PensionPlan):
    """
    Represents a private/company pension plan.

    Attributes
    ----------
    name : str
        Name of the pension plan.
    monthly_contribution : float
        Monthly contribution during working life (positive; cashflow negative).
    monthly_payout_brutto : float
        Monthly gross pension after retirement.
    taxable_earnings_share : float
        Fraction of the payout that is taxable (Ertragsanteil), e.g. 0.18 for 18%.
    """
    name: str
    monthly_payout_brutto: float
    _monthly_contribution: float
    taxable_earnings_share: float

    def monthly_contribution(self, date: dt.date) -> float:
        return self._monthly_contribution
