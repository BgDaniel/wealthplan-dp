"""
Example: Wealth planning with portfolio allocation.

This example demonstrates a more complete wealth planning problem where:
- The agent makes both consumption and portfolio allocation decisions
- Wealth can be invested in risk-free and risky assets
- Income and expenses vary over time
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from wealthplan import WealthPlanModel, UtilityFunction, DPSolver
from wealthplan.utils import plot_solution, plot_policy_functions, create_income_profile


def main():
    """Run portfolio allocation example."""
    print("=" * 60)
    print("Wealth Planning Example: With Portfolio Choice")
    print("=" * 60)
    
    # Model parameters
    T = 25  # 25 year horizon
    discount_factor = 0.96
    risk_free_rate = 0.02
    risky_return = 0.07  # Higher expected return on risky asset
    risky_volatility = 0.15
    
    # Utility function (CRRA with moderate risk aversion)
    utility = UtilityFunction(utility_type="CRRA", gamma=3.0)
    
    # Create model
    model = WealthPlanModel(
        T=T,
        discount_factor=discount_factor,
        risk_free_rate=risk_free_rate,
        risky_return=risky_return,
        risky_volatility=risky_volatility,
        utility_function=utility,
    )
    
    # Set up income schedule: working for 20 years, then retirement
    income = create_income_profile(
        T=T,
        working_years=20,
        starting_income=60.0,
        income_growth=0.03,
        retirement_income=30.0
    )
    model.set_income_schedule(income)
    
    # Set up expenses
    expenses = np.ones(T) * 20.0  # Fixed expenses
    model.set_expense_schedule(expenses)
    
    # Add pension in retirement years
    pension = np.zeros(T)
    for t in range(20, T):
        pension[t] = 15.0
    model.set_pension_schedule(pension)
    
    print(f"\nModel Configuration:")
    print(f"  Time horizon: {T} periods")
    print(f"  Discount factor: {discount_factor}")
    print(f"  Risk-free rate: {risk_free_rate}")
    print(f"  Risky asset return: {risky_return}")
    print(f"  Risky asset volatility: {risky_volatility}")
    print(f"  Risk aversion (gamma): {utility.gamma}")
    print(f"  Starting income: {income[0]:.2f}")
    print(f"  Retirement income: {income[-1]:.2f}")
    print(f"  Retirement pension: {pension[-1]:.2f}")
    
    # Create and solve DP problem
    print("\nSolving with Dynamic Programming...")
    solver = DPSolver(
        model=model,
        wealth_grid_size=80,
        wealth_min=0.0,
        wealth_max=800.0,
        consumption_grid_size=40,
        portfolio_grid_size=11,  # 0%, 10%, ..., 100% in risky asset
    )
    
    solution = solver.solve(verbose=True)
    
    # Simulate optimal paths for different initial wealth levels
    print("\nSimulating optimal paths...")
    initial_wealths = [50.0, 100.0, 200.0]
    
    for w0 in initial_wealths:
        print(f"\n  Initial wealth = {w0:.2f}:")
        path = solver.simulate_path(w0, verbose=False)
        
        avg_consumption = np.mean(path['consumption'])
        avg_portfolio = np.mean(path['portfolio'])
        final_wealth = path['wealth'][-1]
        
        print(f"    Average consumption: {avg_consumption:.2f}")
        print(f"    Average risky asset weight: {avg_portfolio:.2%}")
        print(f"    Final wealth: {final_wealth:.2f}")
    
    # Use middle initial wealth for detailed analysis
    initial_wealth = 100.0
    print(f"\n\nDetailed Analysis for Initial Wealth = {initial_wealth:.2f}")
    print("-" * 60)
    path = solver.simulate_path(initial_wealth, verbose=False)
    
    # Display statistics by phase
    working_phase = slice(0, 20)
    retirement_phase = slice(20, T)
    
    print("\nWorking Phase (years 0-19):")
    print(f"  Avg consumption: {np.mean(path['consumption'][working_phase]):.2f}")
    print(f"  Avg risky weight: {np.mean(path['portfolio'][working_phase]):.2%}")
    print(f"  Wealth at end: {path['wealth'][20]:.2f}")
    
    print("\nRetirement Phase (years 20+):")
    print(f"  Avg consumption: {np.mean(path['consumption'][retirement_phase]):.2f}")
    print(f"  Avg risky weight: {np.mean(path['portfolio'][retirement_phase]):.2%}")
    print(f"  Final wealth: {path['wealth'][-1]:.2f}")
    
    # Plot solution
    try:
        print("\nGenerating plots...")
        plot_solution(
            path,
            model_params=model.get_model_params(),
            title="Wealth Planning with Portfolio Choice",
            save_path="portfolio_choice_solution.png"
        )
        
        # Plot policy functions at t=0
        plot_policy_functions(
            solver.wealth_grid,
            solver.consumption_policies[0],
            solver.portfolio_policies[0],
            period=0,
            save_path="policy_functions_t0.png"
        )
        
        print("✓ Plots saved successfully")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
