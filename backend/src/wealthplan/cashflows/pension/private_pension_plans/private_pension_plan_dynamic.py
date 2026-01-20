from dataclasses import dataclass
import datetime as dt

from wealthplan.cashflows.pension.private_pension_plans.private_pension_plan import PrivatePensionPlan


@dataclass
class PrivatePensionPlanDynamic(PrivatePensionPlan):
    initial_monthly_contribution: float
    contribution_growth_rate: float
    start_date: dt.date

    def monthly_contribution(self, date: dt.date) -> float:
        years = date.year - self.start_date.year
        return self.initial_monthly_contribution * (
            (1 + self.contribution_growth_rate) ** years
        )
