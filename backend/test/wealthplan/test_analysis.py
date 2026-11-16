import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "optimizer" / "stochastic" / "lsmc"))
from test_lsmc_bellman_optimizer import small_params  # noqa: E402

from config.stochastic.lsmc.lsmc_config_mapper import LSMCConfigMapper  # noqa: E402
from wealthplan.analysis import depletion_risk, ruerup_vs_etf, yearly_taxes  # noqa: E402
from wealthplan.optimizer.stochastic.lsmc.lsmc_bellman_optimizer import LSMCBellmanOptimizer  # noqa: E402


class TestAnalysis(unittest.TestCase):
    """Depletion risk, taxes, Rürup vs. ETF, cashflow categories."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(small_params()))
        cls.opt.solve()

    def test_depletion_is_a_probability(self) -> None:
        risk = depletion_risk(self.opt)
        self.assertTrue(((risk["low_wealth"] >= 0) & (risk["low_wealth"] <= 1)).all())
        self.assertTrue(((risk["shortfall"] >= 0) & (risk["shortfall"] <= 1)).all())

    def test_yearly_capital_tax_adds_up(self) -> None:
        det, cap = yearly_taxes(self.opt)
        years = np.array([m.year for m in self.opt.months])
        full = np.isin(years, det.index.values)
        np.testing.assert_allclose(np.nansum(cap), np.nansum(self.opt.opt_capital_tax.values[full]), rtol=1e-10)
        self.assertTrue((det["rurup_refund"] >= 0).all())

    def test_capital_tax_matches_rates(self) -> None:
        """Tax / pre-tax return equals the configured rates for pure equity or pure savings months."""
        eq = self.opt.opt_equity_share.values[0, 0]
        x = self.opt.opt_invested.values[0, 0]
        after = self.opt.opt_equity_before.values[1, 0] + self.opt.opt_savings_before.values[1, 0]
        tax = self.opt.opt_capital_tax.values[0, 0]
        if eq in (0.0, 1.0):
            rate = self.opt.equity_tax_rate if eq == 1.0 else self.opt.savings_tax_rate
            self.assertAlmostEqual(tax, (after - x + tax) * rate, delta=1e-6 * max(1.0, abs(tax)))

    def test_ruerup_vs_etf_without_costs_and_taxes(self) -> None:
        """No costs, no taxes, same return: both capitals at retirement are equal."""
        data = small_params()
        data["total_pension"]["ruerup_plans"][0].update(running_costs=0.0, acquisition_costs=0.0,
                                                        expected_return=0.07)
        data["stochastic"]["yearly_return"] = 0.07
        data["tax"] = {"tax_savings_interest": False, "tax_equity_returns": False, "marginal_tax_rate_working": 0.0}
        opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(data))
        _, info = ruerup_vs_etf(opt, etf_ter=0.0)
        self.assertAlmostEqual(info["etf_capital_at_retirement"], info["ruerup_capital_at_retirement"],
                               delta=1e-6 * info["ruerup_capital_at_retirement"])

    def test_categories_cover_all_items(self) -> None:
        items = self.opt.cashflow_items()
        cats = self.opt.cashflow_categories()
        self.assertTrue(set(items.columns) <= set(cats))


if __name__ == "__main__":
    unittest.main()
