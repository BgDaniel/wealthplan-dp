import datetime as dt

from src.wealthplan.cashflows.salary import Salary
from src.wealthplan.cashflows.rent import Rent
from src.wealthplan.cashflows.pension import Pension
from src.wealthplan.cashflows.life_insurance import LifeInsurance
from wealthplan.optimizer.stochastic.bellmann.binomial_tree.bintree_bellman_optimizer import (
    BinTreeBellmanOptimizer,
)


from wealthplan.optimizer.utility_functions import crra_utility


def main():
    # Simulation dates
    start_date = dt.date(2026, 1, 1)
    end_date = dt.date(2076, 1, 1)
    retirement_date = dt.date(2053, 10, 1)

    # Cashflows
    salary = Salary(monthly_salary=6800, retirement_date=retirement_date)
    rent = Rent(monthly_rent=1300)
    insurance = LifeInsurance(
        monthly_payment=130, payout=100000, payout_date=retirement_date
    )
    pension = Pension(monthly_amount=3200, retirement_date=retirement_date)
    # essential_expenses = EssentialExpenses(monthly_expenses=1500)
    cashflows = [salary, rent, insurance, pension]

    # Wealth
    wealth_0 = 140_000

    # utilities
    instant_utility = lambda c: crra_utility(c)
    terminal_penalty = lambda w: -(w**2)

    # Bellman solver
    bell = BinTreeBellmanOptimizer(
        run_id="test_bin_tree_optimizer",
        start_date=start_date,
        end_date=end_date,
        current_age=37,
        wealth_0=wealth_0,
        cashflows=cashflows,
        sigma=0.1,
        r=0.06,
        b=9.5e-5,
        c=0.085,
        instant_utility=instant_utility,
        terminal_penalty=terminal_penalty,
    )

    bell.solve()
    bell.plot()


if __name__ == "__main__":
    main()
