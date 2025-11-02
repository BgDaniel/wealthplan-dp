"""
Example: Consumption-only wealth planning.

This example demonstrates a simple wealth planning problem where:
- The agent only makes consumption decisions (no portfolio choice)
- All wealth is invested at a fixed interest rate
- Income and expenses vary over time
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from wealthplan import WealthPlanModel, UtilityFunction, DPSolver
from wealthplan.utils import plot_solution, create_income_profile


def main():
    """Run consumption-only example."""
    print("=" * 60)
    print("Wealth Planning Example: Consumption Only")
    print("=" * 60)
    
    # Model parameters
    T = 30  # 30 year horizon
    discount_factor = 0.96
    risk_free_rate = 0.03
    
    # Utility function (CRRA with moderate risk aversion)
    utility = UtilityFunction(utility_type="CRRA", gamma=2.0)
    
    # Create model (no risky asset, only risk-free)
    model = WealthPlanModel(
        T=T,
        discount_factor=discount_factor,
        risk_free_rate=risk_free_rate,
        risky_return=risk_free_rate,  # Same as risk-free (no risky asset)
        utility_function=utility,
    )
    
    # Set up income schedule: working for 20 years, then retirement
    income = create_income_profile(
        T=T,
        working_years=20,
        starting_income=50.0,
        income_growth=0.02,
        retirement_income=25.0
    )
    model.set_income_schedule(income)
    
    # Set up expenses (rent)
    expenses = np.ones(T) * 15.0  # Fixed rent of 15 per period
    model.set_expense_schedule(expenses)
    
    print(f"\nModel Configuration:")
    print(f"  Time horizon: {T} periods")
    print(f"  Discount factor: {discount_factor}")
    print(f"  Risk-free rate: {risk_free_rate}")
    print(f"  Starting income: {income[0]:.2f}")
    print(f"  Retirement income: {income[-1]:.2f}")
    print(f"  Fixed expenses: {expenses[0]:.2f}")
    
    # Create and solve DP problem
    print("\nSolving with Dynamic Programming...")
    solver = DPSolver(
        model=model,
        wealth_grid_size=100,
        wealth_min=0.0,
        wealth_max=500.0,
        consumption_grid_size=50,
        portfolio_grid_size=1,  # Only one option (100% risk-free)
    )
    
    solution = solver.solve(verbose=True)
    
    # Simulate optimal path
    print("\nSimulating optimal path from initial wealth = 50...")
    initial_wealth = 50.0
    path = solver.simulate_path(initial_wealth, verbose=False)
    
    # Display summary statistics
    print("\nSolution Summary:")
    print(f"  Initial wealth: {path['wealth'][0]:.2f}")
    print(f"  Final wealth: {path['wealth'][-1]:.2f}")
    print(f"  Average consumption: {np.mean(path['consumption']):.2f}")
    print(f"  Min consumption: {np.min(path['consumption']):.2f}")
    print(f"  Max consumption: {np.max(path['consumption']):.2f}")
    
    # Total income and expenses
    total_income = np.sum(income)
    total_expenses = np.sum(expenses)
    total_consumption = np.sum(path['consumption'])
    print(f"\n  Total income: {total_income:.2f}")
    print(f"  Total expenses: {total_expenses:.2f}")
    print(f"  Total consumption: {total_consumption:.2f}")
    print(f"  Wealth change: {path['wealth'][-1] - path['wealth'][0]:.2f}")
    
    # Plot solution
    try:
        print("\nGenerating plots...")
        plot_solution(
            path,
            model_params=model.get_model_params(),
            title="Consumption-Only Wealth Planning",
            save_path="consumption_only_solution.png"
        )
        print("✓ Plot saved successfully")
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
