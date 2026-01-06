import datetime as dt
from src.wealthplan.cashflows.base import Cashflow


class PensionPlan(Cashflow):
    """
    Represents a company or private pension plan.

    During working life, the employee contributes a fixed monthly amount (negative cashflow).
    After retirement, the plan pays a fixed monthly pension (positive cashflow).

    Parameters
    ----------
    name : str
        Name of the pension plan (e.g., company, private_huk).
    monthly_contribution : float
        Monthly contribution amount during working life (positive value; cashflow will be negative).
    monthly_payout : float
        Monthly pension received after retirement (positive value; cashflow will be positive).
    retirement_date : datetime.date
        Date on which the pension payments start.
    """

    def __init__(
        self,
        name: str,
        monthly_contribution: float,
        monthly_payout: float,
        retirement_date: dt.date,
    ) -> None:
        self.name = name
        self.monthly_contribution = monthly_contribution
        self.monthly_payout = monthly_payout
        self.retirement_date = retirement_date

    def cashflow(self, date: dt.date) -> float:
        """
        Return the cashflow for a given date.

        - Negative during working life (contribution)
        - Positive after retirement (payout)
        - Only applies on the first day of each month

        Parameters
        ----------
        date : datetime.date
            Date for which the cashflow is evaluated.

        Returns
        -------
        float
            Cashflow amount for the given date.
        """
        if date.day != 1:
            return 0.0  # Only first day of month triggers cashflow

        if date < self.retirement_date:
            return -self.monthly_contribution  # contribution (outflow)
        else:
            return self.monthly_payout  # pension (inflow)
