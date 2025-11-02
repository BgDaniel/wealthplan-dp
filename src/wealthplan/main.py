import datetime as dt

from src.wealthplan.cashflows.pension import Pension
from src.wealthplan.wealth import Wealth
from src.wealthplan.cashflows.salary import Salary
from src.wealthplan.cashflows.rent import Rent
from src.wealthplan.cashflows.pension import Pension
from src.wealthplan.cashflows.life_insurance import LifeInsurance
from src.wealthplan.bellman import BellmanOptimizer

def main():
    # Simulation dates
    start_date = dt.date(2025, 10, 1)
    end_date = dt.date(2075, 10, 1)
    retirement_date = dt.date(2053, 10, 1)

    # Cashflows
    salary = Salary(monthly_salary=6000, retirement_date=retirement_date)
    rent = Rent(monthly_rent=1300)
    insurance = LifeInsurance(monthly_payment=100, payout=100000, end_date=dt.date(2053, 10, 1))
    pension = Pension(monthly_amount=3100, retirement_date=retirement_date)
    cashflows = [salary, rent, insurance, pension]

    # Wealth
    wealth = Wealth(initial_wealth=100000, yearly_return=0.06)

    # Bellman solver
    bellman = BellmanOptimizer(
        start_date=start_date,
        end_date=end_date,
        wealth=wealth,
        cashflows=cashflows,
        beta=1.0
    )

    # Solve Bellman equation
    V, policy = bellman.solve()

    # Print first 12 months of policy
    print("Month | Optimal Consumption | Wealth Start")
    for date_t in list(policy.keys())[:12]:
        W0 = wealth.initial_wealth
        C_opt = policy[date_t][W0]
        print(f"{date_t} | {C_opt:.2f} | {W0:.2f}")

if __name__ == "__main__":
    main()
