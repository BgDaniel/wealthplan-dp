"""German taxes of the plan: income tax on pensions, capital gains tax, Basisrente deduction.

Simplifications (documented in the README):

* Capital gains are taxed when they accrue (monthly, losses credited immediately), not on
  realisation. This ignores the deferral advantage of buy-and-hold equity (slightly
  pessimistic) and the Sparer-Pauschbetrag.
* Pension income tax: either the §32a EStG 2025 tariff on the yearly taxable pension income
  (incl. Solidaritätszuschlag) or a flat, configurable rate.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TaxModel:
    """
    Tax parameters.

    Attributes
    ----------
    capital_gains_tax_rate : float
        Abgeltungsteuer incl. Solidaritätszuschlag (and church tax if any), e.g. 0.26375.
    equity_partial_exemption : float
        Teilfreistellung of equity funds / ETFs (0.30): only 70 % of the gain is taxed.
    tax_savings_interest : bool, tax_equity_returns : bool
        Switch the capital gains tax on interest / equity returns on or off.
    income_tax_mode : str
        "tariff" (§32a EStG 2025 + Soli on the yearly taxable pension income) or "flat".
    income_tax_rate_retirement : float
        Flat income tax rate on the taxable pension income (mode "flat").
    marginal_tax_rate_working : float
        Personal marginal income tax rate while working: Basisrente (Rürup) contributions
        are deducted at this rate (tax refund).
    basis_pension_deduction_room : float
        Yearly amount still deductible as Basisvorsorge (Höchstbetrag 2025: 29 344 € minus
        the employee + employer contributions to the public pension).
    pension_allowance_yearly : float
        Werbungskosten-Pauschbetrag for pensions (102 €/year).
    """

    capital_gains_tax_rate: float = 0.26375
    equity_partial_exemption: float = 0.30
    tax_savings_interest: bool = True
    tax_equity_returns: bool = True
    income_tax_mode: str = "tariff"
    income_tax_rate_retirement: float = 0.20
    marginal_tax_rate_working: float = 0.42
    basis_pension_deduction_room: float = 11_000.0
    pension_allowance_yearly: float = 102.0

    def __post_init__(self) -> None:
        if self.income_tax_mode not in ("tariff", "flat"):
            raise ValueError("income_tax_mode must be 'tariff' or 'flat'")
        for name in ("capital_gains_tax_rate", "equity_partial_exemption", "income_tax_rate_retirement",
                     "marginal_tax_rate_working"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "TaxModel":
        return cls(**(data or {}))

    # ------------------------------------------------------------------ capital gains

    @property
    def savings_tax_rate(self) -> float:
        """Tax rate on interest of the savings account."""
        return self.capital_gains_tax_rate if self.tax_savings_interest else 0.0

    @property
    def equity_tax_rate(self) -> float:
        """Effective tax rate on equity fund returns after the partial exemption."""
        if not self.tax_equity_returns:
            return 0.0
        return self.capital_gains_tax_rate * (1.0 - self.equity_partial_exemption)

    # ------------------------------------------------------------------ income tax

    @staticmethod
    def tariff_2025(taxable_yearly: float) -> float:
        """Income tax §32a EStG 2025 (single assessment) plus Solidaritätszuschlag."""
        x = float(int(max(taxable_yearly, 0.0)))
        if x <= 12_096:
            tax = 0.0
        elif x <= 17_443:
            y = (x - 12_096) / 10_000
            tax = (932.30 * y + 1_400) * y
        elif x <= 68_480:
            z = (x - 17_443) / 10_000
            tax = (176.64 * z + 2_397) * z + 1_015.13
        elif x <= 277_825:
            tax = 0.42 * x - 10_911.92
        else:
            tax = 0.45 * x - 19_246.67
        tax = float(int(tax))
        soli = 0.0 if tax <= 19_950 else min(0.055 * tax, 0.119 * (tax - 19_950))
        return tax + soli

    def pension_income_tax_monthly(self, taxable_monthly: float, scale: float = 1.0) -> float:
        """
        Income tax per month on the (nominal) monthly taxable pension income.

        ``scale`` > 1 indexes the tariff brackets and the allowance with inflation:
        tax = scale * tariff(income / scale).
        """
        taxable_yearly = max(12.0 * taxable_monthly - scale * self.pension_allowance_yearly, 0.0)
        if self.income_tax_mode == "flat":
            return self.income_tax_rate_retirement * taxable_yearly / 12.0
        return scale * self.tariff_2025(taxable_yearly / scale) / 12.0

    @staticmethod
    def taxable_share_basis_pension(start_year: int) -> float:
        """
        Besteuerungsanteil of public pensions and Basisrenten by year of the pension start
        (§22 EStG after the Wachstumschancengesetz: 2023 82.5 %, +0.5 %-points per year,
        100 % from 2058).
        """
        if start_year <= 2005:
            return 0.50
        if start_year <= 2020:
            return 0.50 + 0.02 * (start_year - 2005)
        if start_year <= 2022:
            return 0.80 + 0.01 * (start_year - 2020)
        return min(1.0, 0.825 + 0.005 * (start_year - 2023))

    def basis_pension_refund_monthly(self, contribution_monthly: float, scale: float = 1.0) -> float:
        """Tax refund per month for Basisrente contributions (100 % deductible since 2023)."""
        deductible_yearly = min(12.0 * contribution_monthly, scale * self.basis_pension_deduction_room)
        return self.marginal_tax_rate_working * deductible_yearly / 12.0
