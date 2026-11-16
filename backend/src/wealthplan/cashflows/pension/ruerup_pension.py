import datetime as dt
from dataclasses import dataclass, field

from wealthplan.cashflows.pension.pension_plan import PensionPlan


@dataclass
class RuerupPension(PensionPlan):
    """
    Basisrente (Rürup), fund-linked, with deterministic expected return.

    Savings phase: monthly contribution from ``start_date`` until the pension start; each
    contribution is reduced by the acquisition / distribution costs, the capital grows with
    the expected fund return minus the running costs. Contributions are tax deductible
    (refund in ``TotalPension`` via ``TaxModel.basis_pension_refund_monthly``).

    Payout phase: lifelong monthly pension = capital at pension start / 10 000 * pension_factor
    (Rentenfaktor). Taxed with the Besteuerungsanteil of the start year
    (``TaxModel.taxable_share_basis_pension``). No death benefit (no Hinterbliebenenschutz).

    Attributes
    ----------
    name : str
    monthly_contribution_amount : float
        Contribution per month (e.g. 600 €).
    start_date : dt.date
        First contribution month.
    pension_start : dt.date
        First payout month (= end of contributions).
    contribution_growth : float
        Yearly increase of the contribution (Dynamik), 0 for a constant contribution.
    expected_return : float
        Expected yearly fund return before costs.
    running_costs : float
        Yearly costs on the capital (fund TER + contract costs).
    acquisition_costs : float
        Share of every contribution lost to acquisition / distribution costs.
    pension_factor : float
        Rentenfaktor: monthly pension per 10 000 € capital at the pension start.
    initial_capital : float
        Capital already saved at ``start_date``.
    """

    name: str
    monthly_contribution_amount: float
    start_date: dt.date
    pension_start: dt.date
    contribution_growth: float = 0.0
    expected_return: float = 0.06
    running_costs: float = 0.012
    acquisition_costs: float = 0.04
    pension_factor: float = 28.0
    initial_capital: float = 0.0
    monthly_payout_brutto: float = field(init=False)

    def __post_init__(self) -> None:
        if self.pension_factor <= 0:
            raise ValueError("pension_factor must be > 0")
        if not 0.0 <= self.acquisition_costs < 1.0:
            raise ValueError("acquisition_costs must be in [0, 1)")
        self.monthly_payout_brutto = self.capital_at_pension_start() / 10_000.0 * self.pension_factor

    def monthly_contribution(self, date: dt.date) -> float:
        if date < self.start_date or date >= self.pension_start:
            return 0.0
        years = date.year - self.start_date.year
        return self.monthly_contribution_amount * (1.0 + self.contribution_growth) ** years

    def capital_at_pension_start(self) -> float:
        """Deterministic capital at the pension start (expected return net of costs)."""
        monthly_growth = ((1.0 + self.expected_return) * (1.0 - self.running_costs)) ** (1.0 / 12.0)
        capital = self.initial_capital
        y, m = self.start_date.year, self.start_date.month
        while (y, m) < (self.pension_start.year, self.pension_start.month):
            # contribution at the start of the month, then one month of growth
            capital = (capital + (1.0 - self.acquisition_costs) * self.monthly_contribution(
                dt.date(y, m, 1))) * monthly_growth
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return capital
