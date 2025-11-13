"""
Model definitions, utility functions, and wealth dynamics.

This module defines the core components of the wealth planning model including:
- Utility functions for optimization
- Wealth dynamics (state transitions)
- Model parameter specifications
"""

import numpy as np
from typing import Optional, Callable, Dict, Any


class UtilityFunction:
    """
    Utility function for consumption optimization.
    
    Supports various utility function forms including:
    - CRRA (Constant Relative Risk Aversion)
    - Log utility
    - Quadratic utility
    """
    
    def __init__(self, utility_type: str = "CRRA", gamma: float = 2.0):
        """
        Initialize utility function.
        
        Args:
            utility_type: Type of utility function ("CRRA", "log", "quadratic")
            gamma: Risk aversion parameter (for CRRA)
        """
        self.utility_type = utility_type
        self.gamma = gamma
        
    def evaluate(self, consumption: np.ndarray) -> np.ndarray:
        """
        Evaluate utility of consumption.
        
        Args:
            consumption: Consumption amount(s)
            
        Returns:
            Utility value(s)
        """
        if self.utility_type == "CRRA":
            if self.gamma == 1.0:
                return np.log(consumption)
            else:
                return (consumption ** (1 - self.gamma)) / (1 - self.gamma)
        elif self.utility_type == "log":
            return np.log(consumption)
        elif self.utility_type == "quadratic":
            return consumption - 0.5 * self.gamma * consumption ** 2
        else:
            raise ValueError(f"Unknown utility type: {self.utility_type}")
    
    def marginal_utility(self, consumption: np.ndarray) -> np.ndarray:
        """
        Compute marginal utility of consumption.
        
        Args:
            consumption: Consumption amount(s)
            
        Returns:
            Marginal utility value(s)
        """
        if self.utility_type == "CRRA":
            return consumption ** (-self.gamma)
        elif self.utility_type == "log":
            return 1.0 / consumption
        elif self.utility_type == "quadratic":
            return 1.0 - self.gamma * consumption
        else:
            raise ValueError(f"Unknown utility type: {self.utility_type}")


class WealthPlanModel:
    """
    Multi-period wealth planning model with dynamics and constraints.
    
    This class encapsulates the dynamics of wealth evolution over time,
    including income, expenses, portfolio returns, and various life events.
    """
    
    def __init__(
        self,
        T: int,
        discount_factor: float = 0.95,
        risk_free_rate: float = 0.02,
        risky_return: float = 0.08,
        risky_volatility: float = 0.15,
        utility_function: Optional[UtilityFunction] = None,
    ):
        """
        Initialize the wealth planning model.
        
        Args:
            T: Time horizon (number of periods)
            discount_factor: Time discount factor (beta)
            risk_free_rate: Risk-free interest rate per period
            risky_return: Expected return on risky asset per period
            risky_volatility: Volatility of risky asset per period
            utility_function: Utility function for consumption
        """
        self.T = T
        self.discount_factor = discount_factor
        self.risk_free_rate = risk_free_rate
        self.risky_return = risky_return
        self.risky_volatility = risky_volatility
        
        if utility_function is None:
            self.utility_function = UtilityFunction()
        else:
            self.utility_function = utility_function
            
        # Income and expense schedules (can be customized)
        self.income_schedule = np.zeros(T)
        self.expense_schedule = np.zeros(T)
        self.pension_schedule = np.zeros(T)
        
    def set_income_schedule(self, income: np.ndarray):
        """Set income schedule over time horizon."""
        if len(income) != self.T:
            raise ValueError(f"Income schedule must have length {self.T}")
        self.income_schedule = income.copy()
        
    def set_expense_schedule(self, expenses: np.ndarray):
        """Set expense schedule (e.g., rent) over time horizon."""
        if len(expenses) != self.T:
            raise ValueError(f"Expense schedule must have length {self.T}")
        self.expense_schedule = expenses.copy()
        
    def set_pension_schedule(self, pension: np.ndarray):
        """Set pension/retirement income schedule."""
        if len(pension) != self.T:
            raise ValueError(f"Pension schedule must have length {self.T}")
        self.pension_schedule = pension.copy()
        
    def wealth_dynamics(
        self,
        wealth: float,
        consumption: float,
        portfolio_weight: float,
        t: int,
    ) -> float:
        """
        Compute next period wealth given current state and controls.
        
        In deterministic setting:
        W_{t+1} = (W_t - c_t + y_t - e_t) * [w * R_risky + (1-w) * R_f] + pension_t
        
        Args:
            wealth: Current wealth
            consumption: Current consumption
            portfolio_weight: Weight allocated to risky asset (0 to 1)
            t: Current time period
            
        Returns:
            Next period wealth
        """
        # Available for investment after consumption and expenses
        investable = wealth - consumption + self.income_schedule[t] - self.expense_schedule[t]
        
        # Enforce non-negative investable wealth
        investable = max(0.0, investable)
        
        # Portfolio return (deterministic expected return)
        portfolio_return = (
            portfolio_weight * (1 + self.risky_return)
            + (1 - portfolio_weight) * (1 + self.risk_free_rate)
        )
        
        # Next period wealth
        next_wealth = investable * portfolio_return
        
        # Add pension if applicable
        if t + 1 < self.T:
            next_wealth += self.pension_schedule[t + 1]
            
        return next_wealth
    
    def utility(self, consumption: float) -> float:
        """
        Evaluate utility of consumption.
        
        Args:
            consumption: Consumption amount
            
        Returns:
            Utility value
        """
        return self.utility_function.evaluate(np.array([consumption]))[0]
    
    def terminal_utility(self, wealth: float) -> float:
        """
        Terminal utility (bequest motive).
        
        Args:
            wealth: Terminal wealth
            
        Returns:
            Terminal utility value
        """
        # Simple bequest motive: utility of terminal wealth
        return self.utility_function.evaluate(np.array([max(0.0, wealth)]))[0]
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters as dictionary."""
        return {
            "T": self.T,
            "discount_factor": self.discount_factor,
            "risk_free_rate": self.risk_free_rate,
            "risky_return": self.risky_return,
            "risky_volatility": self.risky_volatility,
            "utility_type": self.utility_function.utility_type,
            "gamma": self.utility_function.gamma,
        }
