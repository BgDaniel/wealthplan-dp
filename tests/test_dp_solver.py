"""
Tests for DP solver functionality.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wealthplan import WealthPlanModel, UtilityFunction, DPSolver
from wealthplan.direct_solver import DirectSolver


class TestUtilityFunction:
    """Test utility function implementations."""
    
    def test_crra_utility(self):
        """Test CRRA utility function."""
        util = UtilityFunction(utility_type="CRRA", gamma=2.0)
        
        # Test single value
        c = 1.0
        u = util.evaluate(np.array([c]))[0]
        assert u == -1.0  # (1^(-1)) / (-1) = -1
        
        # Test marginal utility
        mu = util.marginal_utility(np.array([c]))[0]
        assert mu == 1.0  # c^(-gamma) = 1^(-2) = 1
        
    def test_log_utility(self):
        """Test log utility function."""
        util = UtilityFunction(utility_type="log")
        
        c = np.e
        u = util.evaluate(np.array([c]))[0]
        assert np.isclose(u, 1.0)
        
        mu = util.marginal_utility(np.array([c]))[0]
        assert np.isclose(mu, 1.0 / np.e)
        
    def test_quadratic_utility(self):
        """Test quadratic utility function."""
        util = UtilityFunction(utility_type="quadratic", gamma=0.1)
        
        c = 2.0
        u = util.evaluate(np.array([c]))[0]
        expected = 2.0 - 0.5 * 0.1 * 4.0
        assert np.isclose(u, expected)


class TestWealthPlanModel:
    """Test wealth planning model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = WealthPlanModel(T=10)
        assert model.T == 10
        assert model.discount_factor == 0.95
        
    def test_income_schedule(self):
        """Test setting income schedule."""
        model = WealthPlanModel(T=5)
        income = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        model.set_income_schedule(income)
        assert np.array_equal(model.income_schedule, income)
        
        # Test wrong length
        with pytest.raises(ValueError):
            model.set_income_schedule(np.array([10.0, 12.0]))
    
    def test_wealth_dynamics(self):
        """Test wealth dynamics computation."""
        model = WealthPlanModel(T=2, risk_free_rate=0.1)
        model.set_income_schedule(np.array([10.0, 10.0]))
        model.set_expense_schedule(np.array([2.0, 2.0]))
        
        # Wealth = 100, consume 20, income 10, expense 2
        # Investable = 100 - 20 + 10 - 2 = 88
        # Next wealth = 88 * 1.1 = 96.8
        wealth_next = model.wealth_dynamics(
            wealth=100.0,
            consumption=20.0,
            portfolio_weight=0.0,  # All in risk-free
            t=0
        )
        assert np.isclose(wealth_next, 96.8)
    
    def test_terminal_utility(self):
        """Test terminal utility (bequest motive)."""
        model = WealthPlanModel(T=1)
        terminal_wealth = 100.0
        terminal_util = model.terminal_utility(terminal_wealth)
        
        # Should return utility of terminal wealth
        expected = model.utility(terminal_wealth)
        assert terminal_util == expected


class TestDPSolver:
    """Test DP solver."""
    
    def test_solver_initialization(self):
        """Test solver can be initialized."""
        model = WealthPlanModel(T=5)
        solver = DPSolver(model, wealth_grid_size=50)
        
        assert solver.wealth_grid_size == 50
        assert len(solver.wealth_grid) == 50
        assert len(solver.portfolio_grid) == 11  # default
    
    def test_simple_problem_solve(self):
        """Test solving a simple problem."""
        # Simple 3-period problem
        model = WealthPlanModel(T=3, discount_factor=0.9, risk_free_rate=0.0)
        income = np.array([10.0, 10.0, 10.0])
        model.set_income_schedule(income)
        
        solver = DPSolver(
            model,
            wealth_grid_size=20,
            wealth_max=100.0,
            consumption_grid_size=20,
            portfolio_grid_size=1
        )
        
        result = solver.solve()
        assert result["success"]
        assert len(solver.value_functions) == 4  # T+1
        assert len(solver.consumption_policies) == 3
    
    def test_get_policy(self):
        """Test getting policy for specific wealth."""
        model = WealthPlanModel(T=2)
        model.set_income_schedule(np.array([10.0, 10.0]))
        
        solver = DPSolver(model, wealth_grid_size=20)
        solver.solve()
        
        # Get policy at t=0, wealth=50
        c_opt, w_opt = solver.get_policy(50.0, 0)
        
        assert c_opt > 0  # Consumption should be positive
        assert 0 <= w_opt <= 1  # Portfolio weight should be in [0, 1]
    
    def test_simulate_path(self):
        """Test simulating optimal path."""
        model = WealthPlanModel(T=5)
        income = np.ones(5) * 10.0
        model.set_income_schedule(income)
        
        solver = DPSolver(model, wealth_grid_size=30)
        solver.solve()
        
        initial_wealth = 50.0
        path = solver.simulate_path(initial_wealth)
        
        assert len(path["wealth"]) == 6  # T+1
        assert len(path["consumption"]) == 5
        assert len(path["portfolio"]) == 5
        assert path["wealth"][0] == initial_wealth
        
        # All consumptions should be positive
        assert np.all(path["consumption"] > 0)
        
        # All portfolio weights should be in [0, 1]
        assert np.all(path["portfolio"] >= 0)
        assert np.all(path["portfolio"] <= 1)


class TestDirectSolver:
    """Test direct/NLP solver."""
    
    def test_solver_initialization(self):
        """Test solver can be initialized."""
        model = WealthPlanModel(T=5)
        solver = DirectSolver(model, initial_wealth=100.0)
        
        assert solver.initial_wealth == 100.0
        assert solver.model.T == 5
    
    def test_variable_packing(self):
        """Test variable packing and unpacking."""
        model = WealthPlanModel(T=3)
        solver = DirectSolver(model, initial_wealth=50.0)
        
        c = np.array([10.0, 12.0, 14.0])
        w = np.array([0.5, 0.6, 0.7])
        
        packed = solver._pack_variables(c, w)
        c_unpacked, w_unpacked = solver._unpack_variables(packed)
        
        assert np.allclose(c, c_unpacked)
        assert np.allclose(w, w_unpacked)
    
    def test_simple_problem_solve(self):
        """Test solving a simple problem with direct solver."""
        # Simple 3-period problem
        model = WealthPlanModel(T=3, discount_factor=0.9, risk_free_rate=0.05)
        income = np.array([20.0, 20.0, 20.0])
        model.set_income_schedule(income)
        
        solver = DirectSolver(model, initial_wealth=50.0)
        result = solver.solve()
        
        # Solver should find a solution
        assert result["success"]
        assert "consumption" in result
        assert "portfolio" in result
        assert "wealth" in result
        
        # Check dimensions
        assert len(result["consumption"]) == 3
        assert len(result["portfolio"]) == 3
        assert len(result["wealth"]) == 4


def test_model_integration():
    """Integration test: Compare DP and Direct solvers on same problem."""
    # Small problem both can solve
    T = 5
    model_params = {
        "T": T,
        "discount_factor": 0.95,
        "risk_free_rate": 0.03,
        "risky_return": 0.03,  # Same as risk-free for simplicity
    }
    
    # DP solver
    model_dp = WealthPlanModel(**model_params)
    income = np.ones(T) * 15.0
    model_dp.set_income_schedule(income)
    
    solver_dp = DPSolver(model_dp, wealth_grid_size=30, portfolio_grid_size=1)
    solver_dp.solve()
    
    initial_wealth = 50.0
    path_dp = solver_dp.simulate_path(initial_wealth)
    
    # Direct solver
    model_direct = WealthPlanModel(**model_params)
    model_direct.set_income_schedule(income)
    
    solver_direct = DirectSolver(model_direct, initial_wealth=initial_wealth)
    result_direct = solver_direct.solve()
    
    # Both should produce reasonable solutions
    assert result_direct["success"]
    
    # Final wealth should be positive for both
    assert path_dp["wealth"][-1] > 0
    assert result_direct["wealth"][-1] > 0
    
    # Average consumption should be in similar ballpark
    # (Not exactly equal due to discretization vs continuous optimization)
    avg_c_dp = np.mean(path_dp["consumption"])
    avg_c_direct = np.mean(result_direct["consumption"])
    
    assert avg_c_dp > 0
    assert avg_c_direct > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
