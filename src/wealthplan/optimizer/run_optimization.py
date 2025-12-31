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
    end_date: dt.date = dt.date(2041, 1, 1)
    retirement_date: dt.date = dt.date(2053, 10, 1)

    # ----------
    # Cashflows
    # ----------
    salary: Salary = Salary(
        monthly_salary=3_800.0,
        retirement_date=retirement_date,
    )

    rent: Rent = Rent(monthly_rent=1_000.0)

    insurance: LifeInsurance = LifeInsurance(
        monthly_payment=130.0,
        payout=100_000.0,
        payout_date=retirement_date,
    )

    pension: Pension = Pension(
        monthly_amount=3_200.0,
        retirement_date=retirement_date,
    )

    cashflows: List[Cashflow] = [
        salary,
        rent,
        insurance,
        pension,
    ]

    # -------
    # Wealth
    # -------
    initial_wealth: float = 40_000.0

    yearly_return: float = 0.05

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
            w_max=300_000.0,
            w_step=250.0,
            c_step=250.0,
        )
    )

    deterministic_optimizer.solve()
    deterministic_optimizer.plot()

    # --------------------------------
    # Stochastic Binomial Tree solver
    # --------------------------------
    r: float = 0.05
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
        w_max=300_000.0,
        w_step=250.0,
        c_step=250.0,
    )

    bin_tree_optimizer.solve()


if __name__ == "__main__":
    main()
