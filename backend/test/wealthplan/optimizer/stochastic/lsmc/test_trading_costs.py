import unittest

import numpy as np

from config.stochastic.lsmc.lsmc_config_mapper import LSMCConfigMapper
from wealthplan.optimizer.math_tools.utility_functions import crra_utility
from wealthplan.optimizer.stochastic.lsmc.lsmc_bellman_optimizer import (
    LSMCBellmanOptimizer,
    bellman_values,
    bellman_values_costs,
    exact_trade,
)

from test.wealthplan.optimizer.stochastic.lsmc.test_lsmc_bellman_optimizer import small_params


class TestTradingCosts(unittest.TestCase):
    """Proportional ETF trading costs."""

    def setUp(self) -> None:
        self.grid = 2e5 * np.linspace(0.0, 1.0, 81) ** 2
        self.cont = crra_utility(self.grid * 1.01 + 500.0, 2.0, 1.0)[None, :]

    def test_zero_costs_equal_plain_bellman(self) -> None:
        for y in (1_500.0, -800.0):
            ys = np.array([y])
            plain = bellman_values(self.grid, self.cont, ys, 0.99, 2.0, 1.0, 1.0)
            costs = bellman_values_costs(self.grid, self.cont, np.ones((1, self.grid.size)), ys, 0.99, 2.0, 1.0,
                                         1.0, 0.0, 0.0)
            np.testing.assert_allclose(costs, plain, rtol=1e-10, atol=1e-12)

    def test_costs_lower_the_value(self) -> None:
        ys = np.array([1_500.0])
        plain = bellman_values(self.grid, self.cont, ys, 0.99, 2.0, 1.0, 1.0)
        costs = bellman_values_costs(self.grid, self.cont, np.ones((1, self.grid.size)), ys, 0.99, 2.0, 1.0, 1.0,
                                     0.01, 0.01)
        self.assertTrue(np.all(costs <= plain + 1e-12))
        self.assertTrue(np.any(costs < plain - 1e-9))

    def test_exact_trade(self) -> None:
        self.assertAlmostEqual(exact_trade(1_000.0, 1.0, 0.0, 0.01, 0.02), 1_000.0 / 1.01)  # buy everything
        # sell 500 of 1000 equity to reach alpha = 0.5 of x = 1000: cost 0.02 * sold amount
        x_net = exact_trade(1_000.0, 0.5, 1_000.0, 0.01, 0.02)
        self.assertAlmostEqual(x_net, 1_000.0 - 0.02 * (1_000.0 - 0.5 * x_net))
        self.assertAlmostEqual(exact_trade(1_000.0, 0.5, 500.0, 0.01, 0.02), 1_000.0, places=9)  # at target: no trade
        self.assertEqual(exact_trade(1_000.0, 0.7, 100.0, 0.0, 0.0), 1_000.0)

    def test_run_with_costs(self) -> None:
        data = small_params()
        data["trading"] = {"buy_cost": 0.005, "sell_cost": 0.01}
        opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(data))
        opt.solve()
        summary = opt.summary()
        self.assertGreater(summary["trading_costs_mean"], 0.0)
        free = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(small_params()))
        free.solve()
        self.assertEqual(free.summary()["trading_costs_mean"], 0.0)
        self.assertLess(summary["value_oos"], free.summary()["value_oos"] + 1e-12)


if __name__ == "__main__":
    unittest.main()
