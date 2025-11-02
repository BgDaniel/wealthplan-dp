"""
wealthplan-dp: A deterministic, multi-period model for personal wealth planning.

This package provides tools for optimizing consumption and portfolio allocation
using dynamic programming (Bellman equation) while accounting for income, rent,
pensions, and insurance to maximize long-term financial well-being.
"""

__version__ = "0.1.0"

from .models import WealthPlanModel, UtilityFunction
from .dp_solver import DPSolver

__all__ = [
    "WealthPlanModel",
    "UtilityFunction",
    "DPSolver",
]
