"""Inflation: price index and indexation of cashflows.

The optimizer works in real euros of the start date ("today's euros"): nominal cashflows are
divided by the price index P_t = (1 + rate)^(years since start), nominal returns by the
monthly inflation factor. Taxes are computed on nominal amounts (capital gains tax on nominal
returns, income tax on nominal pensions with optionally indexed tariff brackets).
"""

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional

from wealthplan.cashflows.cashflow_base import CashflowBase


def years_between(start: dt.date, date: dt.date) -> float:
    return (date.year - start.year) + (date.month - start.month) / 12.0


@dataclass
class InflationModel:
    """
    Inflation and indexation parameters.

    Attributes
    ----------
    rate : float
        Yearly inflation rate (0 disables inflation: everything nominal = real).
    salary_real_growth : float
        Yearly salary growth on top of inflation (career / productivity).
    expenses_indexed : bool
        Fixed expenses (rent, living costs, insurances) grow with inflation.
    public_pension_indexation : float
        Yearly nominal increase of the public pension (Rentenanpassung), from the start date;
        the configured pension is the amount in today's euros (Renteninformation).
    tax_brackets_indexed : bool
        Tariff brackets, allowances and the Basisvorsorge room grow with inflation.
    """

    rate: float = 0.0
    salary_real_growth: float = 0.0
    expenses_indexed: bool = True
    public_pension_indexation: float = 0.0
    tax_brackets_indexed: bool = True
    start_date: Optional[dt.date] = None

    def __post_init__(self) -> None:
        if not -0.05 < self.rate < 0.5:
            raise ValueError("inflation rate must be in (-5 %, 50 %)")

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None, start_date: dt.date) -> "InflationModel":
        return cls(**(data or {}), start_date=start_date)

    def price_index(self, date: dt.date) -> float:
        """P_t: price level of ``date`` relative to the start date."""
        if self.rate == 0.0 or self.start_date is None:
            return 1.0
        return (1.0 + self.rate) ** years_between(self.start_date, date)

    @property
    def monthly_factor(self) -> float:
        return (1.0 + self.rate) ** (1.0 / 12.0)

    def tax_scale(self, date: dt.date) -> float:
        """Factor applied to tariff brackets and allowances (1 if not indexed)."""
        return self.price_index(date) if self.tax_brackets_indexed else 1.0


class IndexedCashflow(CashflowBase):
    """
    A cashflow growing with a yearly rate from a base date (stepwise once a year, like rent or
    salary increases). The wrapped amount is in euros of the base date.
    """

    def __init__(self, inner: CashflowBase, yearly_rate: float, base_date: dt.date) -> None:
        self.inner = inner
        self.yearly_rate = yearly_rate
        self.base_date = base_date
        self.name = getattr(inner, "name", type(inner).__name__)
        self.kind = type(inner).__name__

    def cashflow(self, delivery_date: dt.date) -> float:
        years = int(years_between(self.base_date, delivery_date))
        return self.inner.cashflow(delivery_date) * (1.0 + self.yearly_rate) ** max(years, 0)
