import datetime as dt
from typing import List

from params.parameter_loader import FinancialParametersLoader
from wealthplan.cashflows.cashflow_base import CashflowBase

from wealthplan.optimizer.stochastic.binomial_tree.bin_tree_bellman_optimizer import (
    BinTreeBellmanOptimizer,
)
from wealthplan.optimizer.stochastic.survival_process.survival_model import (
    SurvivalModel,
)


def main() -> None:
    """
    Run a stochastic lifecycle consumption–wealth optimization using a
    Binomial Tree Bellman optimizer.

    Steps:
    1. Set simulation horizon and retirement date.
    2. Load cashflows (salary, rent, utilities, insurances, pensions) from YAML.
    3. Set initial wealth, expected return, and risk parameters.
    4. Initialize a survival model for stochastic mortality.
    5. Instantiate and run the stochastic Binomial Tree Bellman optimizer.
    6. Solve the optimization and plot results.
    """
    # ----------------------------
    # Simulation horizon
    # ----------------------------
    start_date: dt.date = dt.date(2026, 1, 1)
    end_date: dt.date = dt.date(2076, 1, 1)
    retirement_date: dt.date = dt.date(2053, 10, 1)

    # ----------------------------
    # Load cashflows
    # ----------------------------
    cashflows: List[CashflowBase] = FinancialParametersLoader(
        filename="lifecycle_params.yaml"
    ).load()

    # ----------------------------
    # Initial wealth and return
    # ----------------------------
    initial_wealth: float = 140_000.0
    yearly_return: float = 0.06
    save: bool = True

    # ----------------------------
    # Risk parameter for stochastic process
    # ----------------------------
    sigma: float = 0.15  # volatility of returns

    # ----------------------------
    # Survival model for mortality
    # ----------------------------
    survival_model: SurvivalModel = SurvivalModel(
        b=9.5e-5,
        c=0.085,
    )

    current_age: int = 37

    # ----------------------------
    # Instantiate Binomial Tree Bellman optimizer
    # ----------------------------
    bin_tree_optimizer_dynamic: BinTreeBellmanOptimizer = BinTreeBellmanOptimizer(
        run_config_id="test_bin_tree_optimizer_dynamic",
        start_date=start_date,
        end_date=end_date,
        retirement_date=retirement_date,
        initial_wealth=initial_wealth,
        yearly_return=yearly_return,
        cashflows=cashflows,
        sigma=sigma,
        survival_model=survival_model,
        current_age=current_age,
        w_max=800_000.0,
        w_step=400.0,
        c_step=400.0,
        use_dynamic_wealth_grid=False,
        use_cache=save,
    )

    # ----------------------------
    # Solve optimization and plot results
    # ----------------------------
    bin_tree_optimizer_dynamic.solve()
    bin_tree_optimizer_dynamic.plot()


if __name__ == "__main__":
    main()
