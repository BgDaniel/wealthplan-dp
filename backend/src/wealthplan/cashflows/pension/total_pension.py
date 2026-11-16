import datetime as dt
from typing import Callable, Dict, List, Optional

import numpy as np

from wealthplan.cashflows.cashflow_base import CashflowBase
from wealthplan.cashflows.pension.pension_plan import PensionPlan
from wealthplan.cashflows.pension.private_pension_plans.private_pension_plan import PrivatePensionPlan
from wealthplan.cashflows.pension.public_pension_plan import PublicPensionPlan
from wealthplan.cashflows.pension.ruerup_pension import RuerupPension
from wealthplan.tax.tax_model import TaxModel


class TotalPension(CashflowBase):
    """
    Aggregates all pension plans and computes their net cashflow.

    Before retirement: minus the contributions, plus the tax refund for Basisrente (Rürup)
    contributions. From retirement on: gross payouts minus the income tax on the taxable part.

    Taxable part of the payout:
    * public pension and Rürup: Besteuerungsanteil of the retirement year,
    * private / company pensions: their ``taxable_earnings_share`` (Ertragsanteil, 1.0 for bAV).

    Parameters
    ----------
    pensions : List[PensionPlan]
        Private, public and Rürup plans.
    retirement_date : dt.date
        First month of the payout phase.
    tax : TaxModel, optional
        Tax parameters (defaults: §32a tariff, 42 % marginal rate while working).
    """

    def __init__(self, pensions: List[PensionPlan], retirement_date: dt.date, tax: Optional[TaxModel] = None,
                 tax_scale: Optional[Callable[[dt.date], float]] = None,
                 refund: Optional[Callable[[dt.date, float], float]] = None,
                 contributions_end: Optional[dt.date] = None):
        self.pensions = pensions
        self.retirement_date: dt.date = retirement_date
        # contributions stop here (early retirement: before the pension start); default: retirement
        self.contributions_end: dt.date = min(contributions_end or retirement_date, retirement_date)
        self.tax = tax or TaxModel()
        # indexation of the tariff brackets / allowances (inflation), 1 if not indexed
        self.tax_scale: Callable[[dt.date], float] = tax_scale or (lambda d: 1.0)
        # exact Rürup refund (date, monthly contribution) -> monthly refund, e.g. from the payroll;
        # default: marginal tax rate of the TaxModel
        self.refund = refund

    def taxable_share(self, plan: PensionPlan) -> float:
        """Taxable share of a plan's payout."""
        if isinstance(plan, (PublicPensionPlan, RuerupPension)):
            return self.tax.taxable_share_basis_pension(self.retirement_date.year)
        if isinstance(plan, PrivatePensionPlan):
            return plan.taxable_earnings_share
        raise TypeError(f"Unsupported pension type: {type(plan).__name__}")

    def components(self, delivery_date: dt.date) -> Dict[str, float]:
        """
        Signed components of the month's pension cashflow (they sum to ``cashflow``).

        Keys: one "<plan name> contribution" per contributing plan and "Rürup tax refund"
        before retirement; "<plan name> pension (gross)" per plan and "Income tax on pensions"
        afterwards.
        """
        out: Dict[str, float] = {}
        if delivery_date < self.retirement_date:
            if delivery_date >= self.contributions_end:
                return out  # gap between the end of the contributions and the pension start
            ruerup = 0.0
            for p in self.pensions:
                c = p.monthly_contribution(delivery_date)
                if c:
                    out[f"{p.name} contribution"] = -c
                if isinstance(p, RuerupPension):
                    ruerup += c
            if ruerup:
                out["Rürup tax refund"] = (self.refund(delivery_date, ruerup) if self.refund is not None else
                                           self.tax.basis_pension_refund_monthly(ruerup, self.tax_scale(delivery_date)))
            return out

        taxable = 0.0
        for p in self.pensions:
            if self.is_stochastic(p):
                continue  # simulated per path by the optimizer (``stochastic_net_payout``)
            gross = p.payout(delivery_date)
            out[f"{p.name} pension (gross)"] = gross
            taxable += gross * self.taxable_share(p)
        out["Income tax on pensions"] = -self.tax.pension_income_tax_monthly(taxable, self.tax_scale(delivery_date))
        return out

    @staticmethod
    def is_stochastic(plan: PensionPlan) -> bool:
        """Fund-linked Rürup plans: their payout depends on the simulated equity path."""
        return isinstance(plan, RuerupPension) and plan.fund_linked

    @property
    def stochastic_plans(self) -> List[RuerupPension]:
        return [p for p in self.pensions if self.is_stochastic(p)]

    def stochastic_net_payout(self, delivery_date: dt.date, gross: np.ndarray) -> np.ndarray:
        """
        Net payout of the fund-linked Rürup pensions per path: gross minus the additional income
        tax they cause on top of the other (deterministic) pensions.

        The incremental tax is evaluated on a grid of gross amounts and interpolated (the tariff
        is piecewise smooth), which keeps thousands of paths cheap.
        """
        if delivery_date < self.retirement_date or not len(gross):
            return np.zeros_like(gross, dtype=float)
        scale = self.tax_scale(delivery_date)
        share = self.tax.taxable_share_basis_pension(self.retirement_date.year)
        others = sum(p.payout(delivery_date) * self.taxable_share(p) for p in self.pensions
                     if not self.is_stochastic(p))
        base = self.tax.pension_income_tax_monthly(others, scale)
        top = max(float(np.max(gross)), 1.0)
        nodes = np.linspace(0.0, top, 401)
        extra = np.array([self.tax.pension_income_tax_monthly(others + share * g, scale) - base for g in nodes])
        return gross - np.interp(gross, nodes, extra)

    def cashflow(self, delivery_date: dt.date) -> float:
        """Net pension cashflow of the month (see ``components``)."""
        return sum(self.components(delivery_date).values())
