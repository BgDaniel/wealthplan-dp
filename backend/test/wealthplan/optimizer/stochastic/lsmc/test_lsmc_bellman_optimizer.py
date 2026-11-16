import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from config.stochastic.lsmc.lsmc_config_mapper import LSMCConfigMapper, solver_params_hash
from wealthplan.optimizer.stochastic.lsmc.lsmc_bellman_optimizer import (
    LSMCBellmanOptimizer,
    _optimal_investment,
    bellman_values,
    project_realised_values,
)
from wealthplan.optimizer.math_tools.utility_functions import crra_utility
from wealthplan.optimizer.stochastic.wealth_regression.wealth_regression import WealthRegressor

PARAMS = Path(__file__).resolve().parents[6] / "params" / "baseline_retire_67.yaml"


def small_params(**technical) -> dict:
    """The lifecycle example on a short horizon with few paths."""
    with open(PARAMS, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["simulation_parameters"]["end_date"] = data["simulation_parameters"]["start_date"].replace(year=2031)
    data["simulation_parameters"]["retirement_date"] = data["simulation_parameters"]["start_date"].replace(
        year=2029)
    data["lifecycle"]["salary"]["retirement_date"] = data["simulation_parameters"]["retirement_date"]
    data["total_pension"]["retirement_date"] = data["simulation_parameters"]["retirement_date"]
    data["stochastic"].update(n_sims=200, n_sims_oos=300)
    data["reserve"] = {"enabled": False}  # tests of the pure policy; the reserve has its own test
    data["technical"].update({"n_wealth": 60, "n_equity_shares": 6, "w_max": 1_000_000.0, "use_cache": False,
                              **technical})
    return data


class TestKernels(unittest.TestCase):
    """Numba kernels against plain numpy references."""

    def setUp(self) -> None:
        self.grid = 1e5 * np.linspace(0.0, 1.0, 41) ** 2
        self.rng = np.random.default_rng(0)

    def test_projection_matches_lstsq(self) -> None:
        n, n_a = 50, 3
        v_next = -1.0 / (self.grid[None, :] + 100.0) * self.rng.uniform(0.9, 1.1, (n, 1))
        gross = 1.0 + 0.05 * self.rng.standard_normal((n_a, n))
        reg = WealthRegressor(np.exp(0.1 * self.rng.standard_normal(n)), degree=2)
        proj, sum_y, sum_y2 = project_realised_values(self.grid, v_next, gross, reg.q, 0.99)
        # reference: explicit realised values and least squares
        y = np.empty((n, self.grid.size, n_a))
        for s in range(n):
            for k in range(n_a):
                x = self.grid * gross[k, s]
                y[s, :, k] = 0.99 * np.interp(x, self.grid, v_next[s])
                above = x > self.grid[-1]
                slope = (v_next[s, -1] - v_next[s, -2]) / (self.grid[-1] - self.grid[-2])
                y[s, above, k] = 0.99 * (v_next[s, -1] + slope * (x[above] - self.grid[-1]))
        coeffs, *_ = np.linalg.lstsq(reg.design, y.reshape(n, -1), rcond=None)
        reg.fit_projection(proj.reshape(proj.shape[0], -1), sum_y.ravel(), sum_y2.ravel())
        np.testing.assert_allclose(reg.coeffs, coeffs, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(sum_y.ravel(), y.reshape(n, -1).sum(axis=0), rtol=1e-10)

    def test_degree_zero_is_monte_carlo_mean(self) -> None:
        y = self.rng.standard_normal((100, 5))
        reg = WealthRegressor(np.exp(self.rng.standard_normal(100)), degree=0)
        fitted = reg.regress(y)
        np.testing.assert_allclose(fitted, np.broadcast_to(y.mean(axis=0), y.shape), atol=1e-12)

    def test_bellman_matches_brute_force(self) -> None:
        cont = crra_utility(self.grid * 1.01, 2.0, 1.0)[None, :]
        y, beta = 1500.0, 0.99
        values = bellman_values(self.grid, cont, np.array([y]), beta, 2.0, 1.0, 1.0)[0]
        for j in (0, 5, 20, 40):
            m = self.grid[j] + y
            x = np.linspace(0.0, m, 200_001)
            f = crra_utility(m - x, 2.0, 1.0) + beta * np.interp(x, self.grid, cont[0])
            self.assertAlmostEqual(values[j], f.max(), delta=1e-9 + 1e-7 * abs(f.max()))

    def test_closed_form_consumption(self) -> None:
        # linear continuation h(x) = a * x: u'(c) = beta * a -> c + eps = (beta a)^(-1/gamma)
        a, beta, gamma, eps = 1e-6, 0.99, 2.0, 1.0
        x, _, _ = _optimal_investment(50_000.0, self.grid, a * self.grid, 0, beta, gamma, eps)
        self.assertAlmostEqual(50_000.0 - x, (beta * a) ** (-1.0 / gamma) - eps, places=6)


class TestLSMCBellmanOptimizer(unittest.TestCase):
    """End-to-end runs on a short horizon."""

    def test_solve_and_summary(self) -> None:
        # a coarse wealth grid biases the in-sample value downwards (interpolation of the concave
        # value function), so this test uses the default grid density
        opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(small_params(n_wealth=200)))
        opt.solve()
        summary = opt.summary()
        self.assertEqual(opt.opt_wealth.shape, (opt.n_months, 300))  # noqa: E501
        self.assertTrue(0.0 <= summary["equity_share_start"] <= 1.0)
        self.assertGreater(summary["consumption_start"], 0.0)
        alive = opt.survival_paths.T > 0
        self.assertTrue(np.all(np.isfinite(opt.opt_wealth.values[alive])))
        self.assertTrue(np.all(opt.opt_wealth.values[alive] >= 0.0))
        # in-sample and out-of-sample value agree within a few standard errors
        self.assertLess(abs(summary["value_in_sample"] - summary["value_oos"]),
                        5 * summary["value_oos_stderr"] + 1e-3 * abs(summary["value_oos"]))

    def test_equity_share_follows_risk_premium(self) -> None:
        for mu, expected in ((0.10, 1.0), (0.0, 0.0)):
            data = small_params()
            data["stochastic"].update(yearly_return=mu, sigma=0.15)
            opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(data))
            opt.solve()
            self.assertAlmostEqual(float(np.nanmedian(opt.opt_equity_share.values[:12])), expected)

    def test_cache_hit_reproduces_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = small_params(use_cache=True)
            first = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(copy.deepcopy(data)), cache_root=tmp)
            first.solve()
            renamed = copy.deepcopy(data)
            renamed["simulation_parameters"]["run_config_id"] = "renamed"  # name is not part of the key
            renamed["stochastic"]["n_sims_oos"] = 300  # forward-only parameter
            second = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(renamed), cache_root=tmp)
            second.solve()
            self.assertFalse(first.from_cache)
            self.assertTrue(second.from_cache)
            np.testing.assert_allclose(second.opt_wealth.values, first.opt_wealth.values)

    def test_hash_ignores_forward_only_parameters(self) -> None:
        a = small_params()
        b = copy.deepcopy(a)
        b["stochastic"]["n_sims_oos"] = 12345
        b["simulation_parameters"]["run_config_id"] = "other"
        self.assertEqual(solver_params_hash(a), solver_params_hash(b))
        b["utility_function"]["gamma"] = 3.0
        self.assertNotEqual(solver_params_hash(a), solver_params_hash(b))

    def test_json_round_trip(self) -> None:
        """Job files store the parameters as JSON: dates become strings."""
        data = small_params()
        restored = json.loads(json.dumps(data, default=str))
        a = LSMCConfigMapper.map_yaml_to_params(data)
        b = LSMCConfigMapper.map_yaml_to_params(restored)
        self.assertEqual(a["start_date"], b["start_date"])
        self.assertEqual(a["params_hash"], b["params_hash"])
        opt = LSMCBellmanOptimizer(**b)
        np.testing.assert_allclose(opt.cf, LSMCBellmanOptimizer(**a).cf)

    def test_decide_matches_forward_simulation(self) -> None:
        """The steering decision of month 0 equals the first month of the simulation."""
        opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(small_params()))
        opt.solve()
        d = opt.decide(0, opt.initial_wealth)
        self.assertAlmostEqual(d["consumption"], float(opt.opt_consumption.values[0, 0]), places=6)
        self.assertAlmostEqual(d["equity_share"], float(opt.opt_equity_share.values[0, 0]), places=9)
        self.assertAlmostEqual(d["equity"] + d["savings"], d["invested"], places=9)

    def test_holdings_are_consistent(self) -> None:
        """Holdings before rebalancing add up to the wealth; capital gains tax lowers the wealth."""
        data = small_params()
        opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(data))
        opt.solve()
        total = opt.opt_equity_before.values + opt.opt_savings_before.values
        np.testing.assert_allclose(total, opt.opt_wealth.values, rtol=1e-12)
        data["tax"] = {"tax_savings_interest": False, "tax_equity_returns": False}
        untaxed = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(data))
        untaxed.solve()
        self.assertGreater(np.nanmedian(untaxed.opt_wealth.values[-12]), np.nanmedian(opt.opt_wealth.values[-12]))

    def test_cashflow_items_sum_to_cashflow(self) -> None:
        opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(small_params()))
        # items are nominal, cf is in real euros of the start date
        np.testing.assert_allclose(opt.cashflow_items().sum(axis=1).values / opt.price_index, opt.cf, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
