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

    # Simplified German-like brackets: (upper bound of yearly taxable income, rate).
    TAX_BRACKETS = ((10_000.0, 0.0), (15_000.0, 0.17), (60_000.0, 0.31), (277_000.0, 0.42),
                    (float("inf"), 0.45))

    @classmethod
    def _progressive_tax(cls, yearly_income: float) -> float:
        """Simplified German-like progressive tax rate for a *yearly* taxable income.

        The brackets are contiguous (the former version left gaps such as 10000 < x < 10001,
        which fell through to the 45 % rate).
        """
        for upper, rate in cls.TAX_BRACKETS:
            if yearly_income <= upper:
                return rate
        return cls.TAX_BRACKETS[-1][1]

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

            # brackets are yearly, the pension is monthly
            tax_rate = self._progressive_tax(12.0 * total_taxable_income)

            tax = total_taxable_income * tax_rate
            net_total = total_brutto - tax

            return net_total
