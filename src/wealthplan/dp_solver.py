"""
Dynamic Programming solver using Bellman equation with interpolation.

This module implements backward induction for solving the wealth planning
problem using value function iteration with interpolation for continuous states.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Optional, List, Dict, Any
from .models import WealthPlanModel


class DPSolver:
    """
    Dynamic Programming solver for wealth planning problem.
    
    Uses backward induction with value function interpolation to solve
    the multi-period optimization problem.
    """
    
    def __init__(
        self,
        model: WealthPlanModel,
        wealth_grid_size: int = 100,
        wealth_min: float = 0.0,
        wealth_max: float = 1000.0,
        consumption_grid_size: int = 50,
        portfolio_grid_size: int = 11,
    ):
        """
        Initialize DP solver.
        
        Args:
            model: WealthPlanModel instance
            wealth_grid_size: Number of grid points for wealth state
            wealth_min: Minimum wealth in grid
            wealth_max: Maximum wealth in grid
            consumption_grid_size: Number of grid points for consumption control
            portfolio_grid_size: Number of grid points for portfolio weight (0 to 1)
        """
        self.model = model
        self.wealth_grid_size = wealth_grid_size
        self.wealth_min = wealth_min
        self.wealth_max = wealth_max
        self.consumption_grid_size = consumption_grid_size
        self.portfolio_grid_size = portfolio_grid_size
        
        # Create grids
        self.wealth_grid = np.linspace(wealth_min, wealth_max, wealth_grid_size)
        self.portfolio_grid = np.linspace(0.0, 1.0, portfolio_grid_size)
        
        # Storage for value functions and policies
        self.value_functions = []
        self.consumption_policies = []
        self.portfolio_policies = []
        
    def solve(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Solve the DP problem using backward induction.
        
        Args:
            verbose: If True, print progress
            
        Returns:
            Dictionary with solution information
        """
        T = self.model.T
        
        # Initialize storage
        self.value_functions = [None] * (T + 1)
        self.consumption_policies = [None] * T
        self.portfolio_policies = [None] * T
        
        # Terminal value function (bequest motive)
        self.value_functions[T] = np.array([
            self.model.terminal_utility(w) for w in self.wealth_grid
        ])
        
        if verbose:
            print(f"Solving DP problem with T={T}, wealth grid size={self.wealth_grid_size}")
        
        # Backward induction
        for t in range(T - 1, -1, -1):
            if verbose:
                print(f"  Period {t}/{T-1}")
                
            value_t = np.zeros(self.wealth_grid_size)
            consumption_policy_t = np.zeros(self.wealth_grid_size)
            portfolio_policy_t = np.zeros(self.wealth_grid_size)
            
            # Interpolate next period value function
            V_next_interp = interp1d(
                self.wealth_grid,
                self.value_functions[t + 1],
                kind='linear',
                bounds_error=False,
                fill_value=(self.value_functions[t + 1][0], self.value_functions[t + 1][-1])
            )
            
            # For each wealth level
            for i, wealth in enumerate(self.wealth_grid):
                best_value = -np.inf
                best_consumption = 0.0
                best_portfolio = 0.0
                
                # Create consumption grid (can't consume more than wealth + income - expenses)
                max_consumption = wealth + self.model.income_schedule[t] - self.model.expense_schedule[t]
                max_consumption = max(0.01, max_consumption)  # Ensure positive
                consumption_grid = np.linspace(0.01, max_consumption, self.consumption_grid_size)
                
                # Grid search over consumption and portfolio weight
                for c in consumption_grid:
                    for w_portfolio in self.portfolio_grid:
                        # Compute next wealth
                        wealth_next = self.model.wealth_dynamics(wealth, c, w_portfolio, t)
                        
                        # Ensure next wealth is within bounds for interpolation
                        wealth_next = np.clip(wealth_next, self.wealth_min, self.wealth_max)
                        
                        # Bellman equation
                        current_utility = self.model.utility(c)
                        continuation_value = V_next_interp(wealth_next)
                        total_value = current_utility + self.model.discount_factor * continuation_value
                        
                        # Update best
                        if total_value > best_value:
                            best_value = total_value
                            best_consumption = c
                            best_portfolio = w_portfolio
                
                value_t[i] = best_value
                consumption_policy_t[i] = best_consumption
                portfolio_policy_t[i] = best_portfolio
            
            self.value_functions[t] = value_t
            self.consumption_policies[t] = consumption_policy_t
            self.portfolio_policies[t] = portfolio_policy_t
        
        if verbose:
            print("DP solve complete!")
        
        return {
            "success": True,
            "value_functions": self.value_functions,
            "consumption_policies": self.consumption_policies,
            "portfolio_policies": self.portfolio_policies,
        }
    
    def get_policy(self, wealth: float, t: int) -> Tuple[float, float]:
        """
        Get optimal policy for given wealth and time.
        
        Args:
            wealth: Current wealth level
            t: Current time period
            
        Returns:
            Tuple of (optimal_consumption, optimal_portfolio_weight)
        """
        if t >= self.model.T:
            raise ValueError(f"Time period {t} exceeds horizon {self.model.T}")
            
        if self.consumption_policies[t] is None:
            raise ValueError("Must call solve() before getting policy")
        
        # Interpolate policy
        consumption_interp = interp1d(
            self.wealth_grid,
            self.consumption_policies[t],
            kind='linear',
            bounds_error=False,
            fill_value=(self.consumption_policies[t][0], self.consumption_policies[t][-1])
        )
        
        portfolio_interp = interp1d(
            self.wealth_grid,
            self.portfolio_policies[t],
            kind='linear',
            bounds_error=False,
            fill_value=(self.portfolio_policies[t][0], self.portfolio_policies[t][-1])
        )
        
        optimal_c = float(consumption_interp(wealth))
        optimal_w = float(portfolio_interp(wealth))
        
        return optimal_c, optimal_w
    
    def simulate_path(
        self,
        initial_wealth: float,
        verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Simulate optimal path from initial wealth.
        
        Args:
            initial_wealth: Initial wealth at t=0
            verbose: If True, print simulation progress
            
        Returns:
            Dictionary with arrays of wealth, consumption, and portfolio weights over time
        """
        if self.consumption_policies[0] is None:
            raise ValueError("Must call solve() before simulating path")
        
        T = self.model.T
        wealth_path = np.zeros(T + 1)
        consumption_path = np.zeros(T)
        portfolio_path = np.zeros(T)
        
        wealth_path[0] = initial_wealth
        
        for t in range(T):
            # Get optimal policy
            c_opt, w_opt = self.get_policy(wealth_path[t], t)
            
            consumption_path[t] = c_opt
            portfolio_path[t] = w_opt
            
            # Compute next wealth
            wealth_path[t + 1] = self.model.wealth_dynamics(
                wealth_path[t], c_opt, w_opt, t
            )
            
            if verbose:
                print(f"t={t}: W={wealth_path[t]:.2f}, c={c_opt:.2f}, w={w_opt:.2f}, W_next={wealth_path[t+1]:.2f}")
        
        return {
            "wealth": wealth_path,
            "consumption": consumption_path,
            "portfolio": portfolio_path,
        }
