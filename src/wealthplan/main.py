import datetime as dt

from src.wealthplan.cashflows.essential_expenses import EssentialExpenses
from src.wealthplan.wealth import Wealth
from src.wealthplan.cashflows.salary import Salary
from src.wealthplan.cashflows.rent import Rent
from src.wealthplan.cashflows.pension import Pension
from src.wealthplan.cashflows.life_insurance import LifeInsurance
from src.wealthplan.optimizer.deterministic.bellman.bellman_optimizer import (
    BellmanOptimizer,
)


def main():
    # Simulation dates
    start_date = dt.date(2025, 10, 1)
    end_date = dt.date(2075, 10, 1)
    retirement_date = dt.date(2053, 10, 1)

    # Cashflows
    salary = Salary(monthly_salary=6000, retirement_date=retirement_date)
    rent = Rent(monthly_rent=1300)
    insurance = LifeInsurance(
        monthly_payment=100, payout=100000, end_date=dt.date(2053, 10, 1)
    )
    pension = Pension(monthly_amount=3100, retirement_date=retirement_date)
    essential_expenses = EssentialExpenses(monthly_expenses=1500)
    cashflows = [salary, rent, insurance, pension, essential_expenses]

    # Wealth
    wealth = Wealth(initial_wealth=100000, yearly_return=0.05)

    # Bellman solver
    bell = BellmanOptimizer(
        start_date=start_date,
        end_date=end_date,
        wealth=wealth,
        cashflows=cashflows,
        beta=1.0,
    )

    bell.solve()
    bell.plot()


if __name__ == "__main__":
    main()
