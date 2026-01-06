import datetime as dt
from typing import List

from financial_parameter_loader import FinancialParametersLoader
from wealthplan.optimizer.deterministic.deterministic_bellman_optimizer import (
    DeterministicBellmanOptimizer,
)
from src.wealthplan.cashflows.base import Cashflow


def main() -> None:
    """
    Run a deterministic Bellman optimizer for a lifecycle consumption–wealth
    planning problem using parameters loaded from a YAML file.

    Steps:
    1. Set simulation horizon and retirement date.
    2. Load all cashflow, insurance, and pension plan parameters from YAML.
    3. Set initial wealth and expected yearly return.
    4. Instantiate and run the deterministic Bellman optimizer.
    5. Solve the optimization problem and plot results.
    """
    # ----------------------------
    # Simulation horizon
    # ----------------------------
    start_date: dt.date = dt.date(2026, 1, 1)
    end_date: dt.date = dt.date(2076, 1, 1)
    retirement_date: dt.date = dt.date(2055, 7, 1)

    # ----------------------------
    # Load cashflows
    # ----------------------------
    cashflows: List[Cashflow] = FinancialParametersLoader(
        filename="lifecycle_params.yaml"
    ).load()

    # ----------------------------
    # Initial wealth and return
    # ----------------------------
    initial_wealth: float = 140_000.0
    yearly_return: float = 0.06
    save: bool = True

    # ----------------------------
    # Deterministic Bellman solver
    # ----------------------------
    deterministic_optimizer: DeterministicBellmanOptimizer = DeterministicBellmanOptimizer(
        run_id="test_deterministic",
        start_date=start_date,
        end_date=end_date,
        retirement_date=retirement_date,
        initial_wealth=initial_wealth,
        yearly_return=yearly_return,
        cashflows=cashflows,
        w_max=800_000.0,
        w_step=100.0,
        c_step=100.0,
        save=save,
    )

    # Solve and plot results
    deterministic_optimizer.solve()
    deterministic_optimizer.plot()


if __name__ == "__main__":
    main()
