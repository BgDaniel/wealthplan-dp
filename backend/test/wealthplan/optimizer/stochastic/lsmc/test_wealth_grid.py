import unittest

import numpy as np

from wealthplan.optimizer.stochastic.lsmc.lsmc_bellman_optimizer import make_wealth_grid


class TestWealthGrid(unittest.TestCase):
    def test_log_grid(self):
        g = make_wealth_grid(3e6, 400, "log", 1e4)
        self.assertEqual(g[0], 0.0)
        self.assertEqual(g[-1], 3e6)
        self.assertTrue(np.all(np.diff(g) > 0))
        # constant relative spacing of (w + shift) above the shift
        ratio = (g[1:] + 1e4) / (g[:-1] + 1e4)
        self.assertTrue(np.allclose(ratio, ratio[0]))

    def test_quadratic_grid(self):
        g = make_wealth_grid(100.0, 11, "quadratic")
        self.assertTrue(np.allclose(g, 100.0 * np.linspace(0, 1, 11) ** 2))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            make_wealth_grid(1e6, 100, "cubic")
        with self.assertRaises(ValueError):
            make_wealth_grid(1e6, 100, "log", 0.0)


if __name__ == "__main__":
    unittest.main()
