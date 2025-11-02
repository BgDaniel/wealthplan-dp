import datetime as dt
from src.wealthplan.cashflows.base import Cashflow

class Salary(Cashflow):
    """Represents a monthly salary until retirement."""

    def __init__(self, monthly_salary: float, retirement_date: dt.date) -> None:
        """
        Args:
            monthly_salary (float): Salary received every month.
            retirement_date (date): Date when salary stops.
        """
        self.monthly_salary = monthly_salary
        self.retirement_date = retirement_date

    def cashflow(self, delivery_date: dt.date) -> float:
        return self.monthly_salary if delivery_date < self.retirement_date else 0.0
