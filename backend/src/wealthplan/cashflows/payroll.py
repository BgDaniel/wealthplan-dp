"""Salary with yearly bonus, gross-to-net payroll (German employee, values 2025).

Yearly view (Jahresabrechnung): social security contributions up to the contribution ceilings,
income tax (§32a tariff + Soli + optional church tax) on the gross minus Werbungskosten and
Vorsorgeaufwendungen. The fixed salary is paid in 12 equal monthly amounts; the bonus month
receives the rest of the yearly net, so the bonus carries the marginal tax and only the
contributions it adds above the fixed salary (usually little, as the ceilings are reached).

All thresholds (ceilings, Höchstbetrag, allowances) grow with the tax scale (inflation) if
``tax_brackets_indexed``; the gross amounts grow with inflation plus real salary growth.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from wealthplan.cashflows.cashflow_base import CashflowBase
from wealthplan.tax.tax_model import TaxModel


@dataclass
class PayrollModel:
    """
    Employee contributions and deductions.

    Attributes
    ----------
    pension_rate, unemployment_rate : float
        Employee shares of the public pension (9.3 %) and unemployment insurance (1.3 %).
    pension_ceiling : float
        Beitragsbemessungsgrenze pension / unemployment insurance per year (2025: 96 600 €).
    health_insurance : str
        "statutory" or "private".
    health_rate : float
        Employee share of the statutory health insurance incl. half the Zusatzbeitrag
        (7.3 % + 1.25 %).
    care_rate : float
        Employee share of the long-term care insurance (1.8 %, childless 2.4 %).
    health_ceiling : float
        Beitragsbemessungsgrenze health / care insurance per year (2025: 66 150 €).
    private_premium_monthly : float
        Private health + care premium per month (health_insurance = "private").
    private_employer_subsidy_monthly : float
        Employer subsidy to the private premium per month.
    church_tax_rate : float
        Kirchensteuer as share of the income tax (0, 0.08 or 0.09).
    werbungskosten : float
        Werbungskosten per year (Pauschbetrag 1 230 €).
    basis_pension_max : float
        Höchstbetrag Basisvorsorge per year (2025: 29 344 €); the deduction room for Rürup is
        this minus employee + employer pension contributions.
    """

    pension_rate: float = 0.093
    unemployment_rate: float = 0.013
    pension_ceiling: float = 96_600.0
    health_insurance: str = "statutory"
    health_rate: float = 0.0855
    care_rate: float = 0.018
    health_ceiling: float = 66_150.0
    private_premium_monthly: float = 0.0
    private_employer_subsidy_monthly: float = 0.0
    church_tax_rate: float = 0.0
    werbungskosten: float = 1_230.0
    basis_pension_max: float = 29_344.0

    def __post_init__(self) -> None:
        if self.health_insurance not in ("statutory", "private"):
            raise ValueError("health_insurance must be 'statutory' or 'private'")

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "PayrollModel":
        return cls(**(data or {}))

    def yearly(self, gross: float, scale: float = 1.0, basis_deduction: float = 0.0) -> Dict[str, float]:
        """
        Yearly payroll of a gross income.

        Parameters
        ----------
        gross : float
            Yearly gross salary incl. bonus (nominal).
        scale : float
            Indexation of ceilings, allowances and tariff (1 in the start year).
        basis_deduction : float
            Additional Basisvorsorge contributions (Rürup) to deduct, before the room limit.

        Returns
        -------
        dict
            social, income_tax (incl. Soli and church tax), net, taxable, basis_room.
        """
        pension_base = min(gross, scale * self.pension_ceiling)
        pension = self.pension_rate * pension_base
        unemployment = self.unemployment_rate * pension_base
        if self.health_insurance == "statutory":
            health_base = min(gross, scale * self.health_ceiling)
            health = self.health_rate * health_base
            care = self.care_rate * health_base
            deductible_health = 0.96 * health + care  # without the Krankengeld share
        else:
            premium = 12.0 * max(self.private_premium_monthly - self.private_employer_subsidy_monthly, 0.0)
            health, care = premium, 0.0
            deductible_health = premium
        social = pension + unemployment + health + care
        room = max(scale * self.basis_pension_max - 2.0 * pension, 0.0)
        taxable = max(gross - scale * self.werbungskosten - pension - deductible_health - scale * 36.0
                      - min(basis_deduction, room), 0.0)
        est = scale * TaxModel.income_tax_2025(taxable / scale)
        income_tax = est + scale * TaxModel.soli(est / scale) + self.church_tax_rate * est
        return {"social": social, "income_tax": income_tax, "net": gross - social - income_tax,
                "taxable": taxable, "basis_room": room}

    def basis_pension_refund_yearly(self, gross: float, contribution_yearly: float, scale: float = 1.0) -> float:
        """Exact tax saving of Basisrente contributions: tax without minus tax with deduction."""
        without = self.yearly(gross, scale)
        with_ = self.yearly(gross, scale, basis_deduction=contribution_yearly)
        return without["income_tax"] - with_["income_tax"]


class GrossSalary(CashflowBase):
    """
    Fixed yearly gross salary in 12 monthly payments plus a yearly bonus in ``bonus_month``,
    paid until ``retirement_date``; converted to net by the ``PayrollModel``.

    Amounts are in euros of ``base_date`` and grow yearly with ``growth`` (inflation + real
    growth); thresholds grow with ``tax_scale(date)``.
    """

    name = "Salary"
    kind = "GrossSalary"

    def __init__(self, yearly_fixed: float, yearly_bonus: float, bonus_month: int, retirement_date: dt.date,
                 payroll: PayrollModel, base_date: dt.date, growth: float = 0.0,
                 tax_scale: Optional[Callable[[dt.date], float]] = None) -> None:
        if not 1 <= bonus_month <= 12:
            raise ValueError("bonus_month must be 1..12")
        self.yearly_fixed = yearly_fixed
        self.yearly_bonus = yearly_bonus
        self.bonus_month = bonus_month
        self.retirement_date = retirement_date
        self.payroll = payroll
        self.base_date = base_date
        self.growth = growth
        self.tax_scale = tax_scale or (lambda d: 1.0)
        self._cache: Dict[int, Dict[str, float]] = {}

    def _factor(self, date: dt.date) -> float:
        years = (date.year - self.base_date.year) + (date.month - self.base_date.month) / 12.0
        return (1.0 + self.growth) ** int(max(years, 0.0))

    def year_payroll(self, date: dt.date) -> Dict[str, float]:
        """Yearly payroll split into the fixed part and the bonus (gross amounts of ``date``)."""
        # one payroll per plan year (from the base month): ceilings and tariff fixed within the year
        year = date.year if date.month >= self.base_date.month else date.year - 1
        anniversary = dt.date(year, self.base_date.month, 1)
        key = year
        if key not in self._cache:
            f, scale = self._factor(anniversary), self.tax_scale(anniversary)
            fixed, bonus = f * self.yearly_fixed, f * self.yearly_bonus
            only_fixed = self.payroll.yearly(fixed, scale)
            total = self.payroll.yearly(fixed + bonus, scale)
            self._cache[key] = {
                "fixed_gross": fixed, "bonus_gross": bonus,
                "fixed_social": only_fixed["social"], "fixed_tax": only_fixed["income_tax"],
                "bonus_social": total["social"] - only_fixed["social"],
                "bonus_tax": total["income_tax"] - only_fixed["income_tax"],
                "total_gross": fixed + bonus,
            }
        return self._cache[key]

    def components(self, delivery_date: dt.date) -> Dict[str, float]:
        if delivery_date >= self.retirement_date:
            return {}
        y = self.year_payroll(delivery_date)
        out = {"Salary (gross)": y["fixed_gross"] / 12.0,
               "Social security (salary)": -y["fixed_social"] / 12.0,
               "Income tax (salary)": -y["fixed_tax"] / 12.0}
        if delivery_date.month == self.bonus_month and y["bonus_gross"] > 0:
            out.update({"Bonus (gross)": y["bonus_gross"],
                        "Social security (bonus)": -y["bonus_social"],
                        "Income tax (bonus)": -y["bonus_tax"]})
        return out

    def categories(self) -> Dict[str, str]:
        return {"Salary (gross)": "Salary", "Bonus (gross)": "Salary",
                "Social security (salary)": "Social security", "Social security (bonus)": "Social security",
                "Income tax (salary)": "Income tax", "Income tax (bonus)": "Income tax"}

    def cashflow(self, delivery_date: dt.date) -> float:
        return sum(self.components(delivery_date).values())


class NetSalary(CashflowBase):
    """Net monthly salary plus a net yearly bonus in ``bonus_month`` (amounts given net)."""

    name = "Salary"
    kind = "NetSalary"

    def __init__(self, monthly_net: float, bonus_net: float, bonus_month: int, retirement_date: dt.date) -> None:
        self.monthly_net = monthly_net
        self.bonus_net = bonus_net
        self.bonus_month = bonus_month
        self.retirement_date = retirement_date

    def components(self, delivery_date: dt.date) -> Dict[str, float]:
        if delivery_date >= self.retirement_date:
            return {}
        out = {"Salary (net)": self.monthly_net}
        if delivery_date.month == self.bonus_month and self.bonus_net:
            out["Bonus (net)"] = self.bonus_net
        return out

    def categories(self) -> Dict[str, str]:
        return {"Salary (net)": "Salary", "Bonus (net)": "Salary"}

    def cashflow(self, delivery_date: dt.date) -> float:
        return sum(self.components(delivery_date).values())
