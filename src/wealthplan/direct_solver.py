"""
Direct transcription / NLP solver for wealth planning problem.

This module provides an alternative solution approach using direct optimization
(collocation) instead of dynamic programming. This can be useful for:
- Handling stochastic elements
- Incorporating complex constraints
- Larger problems where DP becomes intractable
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Optional
from .models import WealthPlanModel


class DirectSolver:
    """
    Direct transcription solver using nonlinear programming.
    
    Converts the multi-period problem into a single large NLP by
    treating all time periods' decisions as optimization variables.
    """
    
    def __init__(
        self,
        model: WealthPlanModel,
        initial_wealth: float,
    ):
        """
        Initialize direct solver.
        
        Args:
            model: WealthPlanModel instance
            initial_wealth: Initial wealth at t=0
        """
        self.model = model
        self.initial_wealth = initial_wealth
        
        # Storage for solution
        self.solution = None
        
    def _pack_variables(
        self,
        consumption: np.ndarray,
        portfolio_weights: np.ndarray
    ) -> np.ndarray:
        """
        Pack decision variables into single vector.
        
        Args:
            consumption: Consumption decisions [T]
            portfolio_weights: Portfolio weight decisions [T]
            
        Returns:
            Packed variable vector [2*T]
        """
        return np.concatenate([consumption, portfolio_weights])
    
    def _unpack_variables(self, x: np.ndarray) -> tuple:
        """
        Unpack variable vector into consumption and portfolio decisions.
        
        Args:
            x: Packed variable vector [2*T]
            
        Returns:
            Tuple of (consumption[T], portfolio_weights[T])
        """
        T = self.model.T
        consumption = x[:T]
        portfolio_weights = x[T:]
        return consumption, portfolio_weights
    
    def _compute_wealth_trajectory(
        self,
        consumption: np.ndarray,
        portfolio_weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute wealth trajectory given decisions.
        
        Args:
            consumption: Consumption decisions [T]
            portfolio_weights: Portfolio weight decisions [T]
            
        Returns:
            Wealth trajectory [T+1]
        """
        T = self.model.T
        wealth = np.zeros(T + 1)
        wealth[0] = self.initial_wealth
        
        for t in range(T):
            wealth[t + 1] = self.model.wealth_dynamics(
                wealth[t],
                consumption[t],
                portfolio_weights[t],
                t
            )
        
        return wealth
    
    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function: negative of total discounted utility.
        
        Args:
            x: Decision variable vector
            
        Returns:
            Negative total utility (for minimization)
        """
        consumption, portfolio_weights = self._unpack_variables(x)
        
        # Compute total discounted utility
        total_utility = 0.0
        discount = 1.0
        
        for t in range(self.model.T):
            utility_t = self.model.utility(consumption[t])
            total_utility += discount * utility_t
            discount *= self.model.discount_factor
        
        # Add terminal utility
        wealth_trajectory = self._compute_wealth_trajectory(consumption, portfolio_weights)
        terminal_utility = self.model.terminal_utility(wealth_trajectory[-1])
        total_utility += discount * terminal_utility
        
        # Return negative (we minimize)
        return -total_utility
    
    def _constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Inequality constraints: wealth >= 0 at all times.
        
        Args:
            x: Decision variable vector
            
        Returns:
            Constraint values (should be >= 0)
        """
        consumption, portfolio_weights = self._unpack_variables(x)
        wealth_trajectory = self._compute_wealth_trajectory(consumption, portfolio_weights)
        
        # All wealth levels should be non-negative
        return wealth_trajectory
    
    def solve(
        self,
        initial_guess: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Solve the optimization problem using NLP.
        
        Args:
            initial_guess: Initial guess for decision variables (optional)
            verbose: If True, print solver progress
            
        Returns:
            Dictionary with solution information
        """
        T = self.model.T
        
        # Initial guess: equal consumption, 50% portfolio weight
        if initial_guess is None:
            avg_income = np.mean(self.model.income_schedule)
            consumption_guess = np.ones(T) * max(1.0, avg_income * 0.5)
            portfolio_guess = np.ones(T) * 0.5
            x0 = self._pack_variables(consumption_guess, portfolio_guess)
        else:
            x0 = initial_guess
        
        # Bounds: consumption > 0, portfolio weight in [0, 1]
        bounds = [(0.01, None)] * T + [(0.0, 1.0)] * T
        
        # Constraints: wealth >= 0
        constraints = {
            'type': 'ineq',
            'fun': self._constraints
        }
        
        # Solve
        result = minimize(
            self._objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': verbose, 'maxiter': 1000}
        )
        
        if result.success:
            consumption, portfolio_weights = self._unpack_variables(result.x)
            wealth_trajectory = self._compute_wealth_trajectory(consumption, portfolio_weights)
            
            self.solution = {
                "success": True,
                "consumption": consumption,
                "portfolio": portfolio_weights,
                "wealth": wealth_trajectory,
                "objective_value": -result.fun,  # Convert back to positive utility
                "message": result.message,
            }
        else:
            self.solution = {
                "success": False,
                "message": result.message,
            }
        
        return self.solution
    
    def get_solution(self) -> Optional[Dict[str, Any]]:
        """
        Get the solution if available.
        
        Returns:
            Solution dictionary or None
        """
        return self.solution
