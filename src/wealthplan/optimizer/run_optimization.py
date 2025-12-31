import datetime as dt
from typing import List

from src.wealthplan.cashflows.base import Cashflow
from src.wealthplan.cashflows.salary import Salary
from src.wealthplan.cashflows.rent import Rent
from src.wealthplan.cashflows.pension import Pension
from src.wealthplan.cashflows.life_insurance import LifeInsurance

from wealthplan.optimizer.deterministic.deterministic_bellman_optimizer import (
    DeterministicBellmanOptimizer,
)
from wealthplan.optimizer.stochastic.binomial_tree.bin_tree_bellman_optimizer import (
    BinTreeBellmanOptimizer,
)
from wealthplan.optimizer.stochastic.survival_process.survival_model import (
    SurvivalModel,
)


def main() -> None:
    """
    Run deterministic and stochastic Bellman optimizers for a lifecycle
    consumption–wealth planning problem.
    """

    # ------------------
    # Simulation horizon
    # ------------------
    start_date: dt.date = dt.date(2026, 1, 1)
    end_date: dt.date = dt.date(2031, 1, 1)
    retirement_date: dt.date = dt.date(2053, 10, 1)

    # ----------
    # Cashflows
    # ----------
    salary: Salary = Salary(
        monthly_salary=3_500,
        retirement_date=retirement_date,
    )

    rent: Rent = Rent(monthly_rent=1_100.0)

    insurance: LifeInsurance = LifeInsurance(
        monthly_payment=130.0,
        payout=100_000.0,
        payout_date=retirement_date,
    )

    pension: Pension = Pension(
        monthly_amount=1_300.0,
        retirement_date=retirement_date,
    )

    cashflows: List[Cashflow] = [
        salary,
        rent,
        insurance,
        pension
    ]

    # -------
    # Wealth
    # -------
    initial_wealth: float = 40_000.0

    yearly_return: float = 0.06

    save = True

    # ----------------------------
    # Deterministic Bellman solver
    # ----------------------------
    deterministic_optimizer: DeterministicBellmanOptimizer = (
        DeterministicBellmanOptimizer(
            run_id="test_deterministic",
            start_date=start_date,
            end_date=end_date,
            retirement_date=retirement_date,
            initial_wealth=initial_wealth,
            yearly_return=yearly_return,
            cashflows=cashflows,
            w_max=100_000.0,
            w_step=50.0,
            c_step=50.0,
            save=save
        )
    )

    deterministic_optimizer.solve()
    deterministic_optimizer.plot()

    # --------------------------------
    # Stochastic Binomial Tree solver
    # --------------------------------
    sigma: float = 0.10

    survival_model: SurvivalModel = SurvivalModel(
        b=9.5e-5,
        c=0.085,
    )

    current_age: int = 37

    bin_tree_optimizer: BinTreeBellmanOptimizer = BinTreeBellmanOptimizer(
        run_id="test_bin_tree_optimizer",
        start_date=start_date,
        end_date=end_date,
        retirement_date=retirement_date,
        initial_wealth=initial_wealth,
        yearly_return=yearly_return,
        cashflows=cashflows,
        sigma=sigma,
        survival_model=survival_model,
        current_age=current_age,
        w_max=50_000.0,
        w_step=50.0,
        c_step=50.0,
        stochastic=True,
        save=save
    )

    bin_tree_optimizer.solve()
    bin_tree_optimizer.plot()


if __name__ == "__main__":
    main()
