import datetime as dt

from wealthplan.cashflows.salary import Salary
from wealthplan.cashflows.rent import Rent
from wealthplan.cashflows.pension import Pension
from wealthplan import LifeInsurance
from wealthplan.optimizer.stochastic.lsmc.lsmc_bellman_optimizer import (
    LSMCBellmanOptimizer,
)
from wealthplan.optimizer.stochastic.market_model.gbm_returns import GBM
from wealthplan.optimizer.stochastic.survival_process.survival_process import SurvivalProcess
from wealthplan.optimizer.utility_functions import crra_utility


def main():
    # Simulation dates
    start_date = dt.date(2026, 1, 1)
    end_date = dt.date(2076, 1, 1)
    retirement_date = dt.date(2053, 10, 1)

    # Cashflows
    salary = Salary(monthly_amount=6800, retirement_date=retirement_date)
    rent = Rent(monthly_amount=1300)
    insurance = LifeInsurance(
        monthly_payment=130, payout=100000, payout_date=retirement_date
    )
    pension = Pension(monthly_amount=3200, retirement_date=retirement_date)
    #essential_expenses = EssentialExpenses(monthly_expenses=1500)
    cashflows = [salary, rent, insurance, pension]

    # Wealth
    wealth_0 = 140_000

    gbm_returns = GBM(
        mu=0.05,
        sigma=0.0,
        seed=42
    )

    survival_process = SurvivalProcess(
        b=9.5e-5,  # calibrated baseline hazard
        c=0.085,   # aging rate
        age=37,    # current age
        seed=123
    )

    n_sims = 1000

    # utilities
    instant_utility = lambda c: crra_utility(c)
    terminal_penalty = lambda w: -(w ** 2)

    # Bellman solver
    bell = LSMCBellmanOptimizer(
        run_id="test_stochastic",
        start_date=start_date,
        end_date=end_date,
        wealth_0=wealth_0,
        cashflows=cashflows,
        gbm_returns=gbm_returns,
        survival_process=survival_process,
        n_sims=n_sims,
        instant_utility=instant_utility,
        terminal_penalty=terminal_penalty
    )

    bell.solve()
    bell.plot()


if __name__ == "__main__":
    main()
