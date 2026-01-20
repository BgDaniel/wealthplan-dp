import datetime as dt
from typing import List

from wealthplan.cashflows.cashflow_base import CashflowBase
from wealthplan.cashflows.pension.pension_plan import PensionPlan
from wealthplan.cashflows.pension.private_pension_plans.private_pension_plan import PrivatePensionPlan
from wealthplan.cashflows.pension.public_pension_plan import PublicPensionPlan


class TotalPension(CashflowBase):
    """
    Aggregates multiple pension dataclasses and computes net cashflows.

    Parameters
    ----------
    pensions : List
        List of PrivatePensionPlan or PublicPensionPlan dataclasses.
    retirement_date : dt.date
        Reference retirement date for cashflow calculations.
    """

    def __init__(self, pensions: List[PensionPlan], retirement_date: dt.date):
        """
        Initialize TotalPension.

        Args:
            pensions (List): List of pension dataclasses
            retirement_date (dt.date): Reference retirement date
        """
        self.pensions = pensions
        self.retirement_date: dt.date = retirement_date

    @staticmethod
    def _progressive_tax(income: float) -> float:
        """Simplified German-like progressive tax rate."""
        if income <= 10000:
            return 0.0
        elif 10001 <= income <= 15000:
            return 0.17
        elif 15001 <= income <= 60000:
            return 0.31
        elif 60001 <= income <= 277000:
            return 0.42
        else:
            return 0.45

    def cashflow(self, delivery_date: dt.date) -> float:
        """
        Compute total net cashflow for a given date.

        - Before retirement: negative contributions (private pensions)
        - On/after retirement: positive gross payouts from all pensions, net of progressive tax
        """
        if delivery_date < self.retirement_date:
            # All pensions are still in contribution phase
            total = 0.0

            for p in self.pensions:
                total -= p.monthly_contribution(delivery_date)

            return total
        else:
            # After retirement → sum taxable portions and apply tax
            total_taxable_income = 0.0
            total_brutto = 0.0

            for p in self.pensions:
                total_brutto += p.monthly_payout_brutto

                if isinstance(p, PrivatePensionPlan):
                    # Only Ertragsanteil is taxable
                    total_taxable_income += (
                        p.monthly_payout_brutto * p.taxable_earnings_share
                    )
                elif isinstance(p, PublicPensionPlan):
                    # Public pension fully taxable
                    total_taxable_income += p.monthly_payout_brutto
                else:
                    raise TypeError(f"Unsupported pension type: {type(p).__name__}")

            tax_rate = self._progressive_tax(total_taxable_income)

            tax = total_taxable_income * tax_rate
            net_total = total_brutto - tax

            return net_total
