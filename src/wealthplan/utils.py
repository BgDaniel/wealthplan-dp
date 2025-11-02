"""
Utility functions and I/O helpers for wealth planning.

This module provides various helper functions including:
- Plotting and visualization
- Data import/export
- Common calculations
"""

import numpy as np
from typing import Dict, Any, Optional
import json


def plot_solution(
    solution: Dict[str, np.ndarray],
    model_params: Optional[Dict[str, Any]] = None,
    title: str = "Wealth Planning Solution",
    save_path: Optional[str] = None
):
    """
    Plot the solution trajectory.
    
    Args:
        solution: Dictionary with 'wealth', 'consumption', 'portfolio' arrays
        model_params: Optional model parameters for display
        title: Plot title
        save_path: If provided, save figure to this path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, cannot plot")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Wealth trajectory
    axes[0].plot(solution['wealth'], 'b-', linewidth=2)
    axes[0].set_ylabel('Wealth', fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Consumption trajectory
    T = len(solution['consumption'])
    axes[1].plot(range(T), solution['consumption'], 'g-', linewidth=2)
    axes[1].set_ylabel('Consumption', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Portfolio weight trajectory
    axes[2].plot(range(T), solution['portfolio'], 'r-', linewidth=2)
    axes[2].set_ylabel('Portfolio Weight (Risky)', fontsize=12)
    axes[2].set_xlabel('Time Period', fontsize=12)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_value_function(
    wealth_grid: np.ndarray,
    value_function: np.ndarray,
    period: int,
    save_path: Optional[str] = None
):
    """
    Plot the value function for a specific time period.
    
    Args:
        wealth_grid: Wealth grid points
        value_function: Value function values
        period: Time period
        save_path: If provided, save figure to this path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, cannot plot")
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(wealth_grid, value_function, 'b-', linewidth=2)
    plt.xlabel('Wealth', fontsize=12)
    plt.ylabel('Value Function', fontsize=12)
    plt.title(f'Value Function at Period {period}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_policy_functions(
    wealth_grid: np.ndarray,
    consumption_policy: np.ndarray,
    portfolio_policy: np.ndarray,
    period: int,
    save_path: Optional[str] = None
):
    """
    Plot policy functions for a specific time period.
    
    Args:
        wealth_grid: Wealth grid points
        consumption_policy: Consumption policy values
        portfolio_policy: Portfolio policy values
        period: Time period
        save_path: If provided, save figure to this path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, cannot plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    
    # Consumption policy
    axes[0].plot(wealth_grid, consumption_policy, 'g-', linewidth=2)
    axes[0].set_ylabel('Optimal Consumption', fontsize=12)
    axes[0].set_title(f'Policy Functions at Period {period}', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Portfolio policy
    axes[1].plot(wealth_grid, portfolio_policy, 'r-', linewidth=2)
    axes[1].set_ylabel('Optimal Portfolio Weight', fontsize=12)
    axes[1].set_xlabel('Wealth', fontsize=12)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_solution(solution: Dict[str, Any], filepath: str):
    """
    Save solution to file.
    
    Args:
        solution: Solution dictionary
        filepath: Path to save file (JSON format)
    """
    # Convert numpy arrays to lists for JSON serialization
    json_solution = {}
    for key, value in solution.items():
        if isinstance(value, np.ndarray):
            json_solution[key] = value.tolist()
        else:
            json_solution[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_solution, f, indent=2)
    
    print(f"Solution saved to {filepath}")


def load_solution(filepath: str) -> Dict[str, Any]:
    """
    Load solution from file.
    
    Args:
        filepath: Path to solution file (JSON format)
        
    Returns:
        Solution dictionary with numpy arrays
    """
    with open(filepath, 'r') as f:
        json_solution = json.load(f)
    
    # Convert lists back to numpy arrays
    solution = {}
    for key, value in json_solution.items():
        if isinstance(value, list):
            solution[key] = np.array(value)
        else:
            solution[key] = value
    
    print(f"Solution loaded from {filepath}")
    return solution


def compute_present_value(cash_flows: np.ndarray, discount_rate: float) -> float:
    """
    Compute present value of cash flow stream.
    
    Args:
        cash_flows: Array of cash flows over time
        discount_rate: Discount rate per period
        
    Returns:
        Present value
    """
    T = len(cash_flows)
    discount_factors = np.array([(1 / (1 + discount_rate)) ** t for t in range(T)])
    return np.sum(cash_flows * discount_factors)


def compute_annuity_payment(
    principal: float,
    rate: float,
    periods: int
) -> float:
    """
    Compute periodic payment for an annuity.
    
    Args:
        principal: Loan principal
        rate: Interest rate per period
        periods: Number of periods
        
    Returns:
        Payment per period
    """
    if rate == 0:
        return principal / periods
    
    return principal * (rate * (1 + rate) ** periods) / ((1 + rate) ** periods - 1)


def create_income_profile(
    T: int,
    working_years: int,
    starting_income: float,
    income_growth: float = 0.02,
    retirement_income: float = 0.0
) -> np.ndarray:
    """
    Create a typical income profile with working and retirement periods.
    
    Args:
        T: Time horizon
        working_years: Number of working years
        starting_income: Initial income
        income_growth: Annual income growth rate during working years
        retirement_income: Income during retirement (pension, etc.)
        
    Returns:
        Income schedule array
    """
    income = np.zeros(T)
    
    for t in range(min(working_years, T)):
        income[t] = starting_income * (1 + income_growth) ** t
    
    for t in range(working_years, T):
        income[t] = retirement_income
    
    return income
