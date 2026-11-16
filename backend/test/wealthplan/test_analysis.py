import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "optimizer" / "stochastic" / "lsmc"))
from test_lsmc_bellman_optimizer import small_params  # noqa: E402

from config.stochastic.lsmc.lsmc_config_mapper import LSMCConfigMapper  # noqa: E402
from wealthplan.analysis import depletion_risk, yearly_taxes  # noqa: E402
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
            phi = self.opt.inflation_factor  # amounts are real: nominal = real * phi
            gain_nominal = after * phi - x + tax * phi
            self.assertAlmostEqual(tax * phi, gain_nominal * rate, delta=1e-6 * max(1.0, abs(tax)))

    def test_ruerup_contract_on_equity_paths(self) -> None:
        """Without costs the contract capital is the contributions compounded with the equity path."""
        data = small_params()
        plan = data["total_pension"]["ruerup_plans"][0]
        plan.update(running_costs=0.0, acquisition_costs=0.0, initial_capital=0.0, contribution_growth=0.0)
        data["inflation"] = {"rate": 0.0}
        opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(data))
        equity = opt._equity_paths(5, 7)
        sim = opt._ruerup_simulation(equity)
        r = opt.ruerup_index
        c = float(plan["monthly_contribution"])
        expected = sum(c * equity[:, r] / equity[:, t] for t in range(r))
        np.testing.assert_allclose(sim["capital"][r - 1] * equity[:, r] / equity[:, r - 1] + c * equity[:, r]
                                   / equity[:, r - 1], expected, rtol=1e-10)
        gross = expected / 10_000.0 * float(plan["pension_factor"])
        np.testing.assert_allclose(sim["extra"][r] + sim["tax"][r], gross, rtol=1e-10)

    def test_out_of_money(self) -> None:
        """No income, high fixed costs, little wealth: everybody runs out of money; shares consistent."""
        data = small_params()
        data["lifecycle"]["salary"] = {"mode": "net", "monthly_net": 0.0}
        data["simulation_parameters"]["initial_wealth"] = 20_000.0
        opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(data))
        opt.solve()
        oom = opt.out_of_money()
        self.assertTrue((oom["share_ever"] >= oom["share_now"] - 1e-12).all())
        summary = opt.summary()
        self.assertGreater(summary["prob_shortfall"], 0.9)
        self.assertGreater(summary["unpaid_fixed_costs_mean"], 0.0)
        self.assertEqual(oom["share_now"].values[-1], 0.0)  # last month: horizon end, not counted

    def test_categories_cover_all_items(self) -> None:
        items = self.opt.cashflow_items()
        cats = self.opt.cashflow_categories()
        self.assertTrue(set(items.columns) <= set(cats))


class TestInflationReserveMortality(unittest.TestCase):
    """Real-terms conversion, emergency reserve, mortality presets."""

    def test_real_returns_and_cashflows(self) -> None:
        data = small_params()
        data["inflation"] = {"rate": 0.03, "expenses_indexed": True, "public_pension_indexation": 0.03}
        data["tax"] = {"tax_savings_interest": False}
        data["simulation_parameters"]["yearly_return_savings"] = 0.03
        opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(data))
        self.assertAlmostEqual(opt.rf, 1.0, places=12)  # savings rate = inflation, no tax: real return 0
        # rent grows with inflation once a year -> constant in real terms up to the intra-year drift
        rent = opt.cashflow_items()["Rent"].values / opt.price_index
        self.assertAlmostEqual(rent[12], rent[0], delta=0.03 * abs(rent[0]))
        nominal = opt.cashflow_items()["Rent"].values
        self.assertAlmostEqual(nominal[24] / nominal[0], 1.03 ** 2, places=10)

    def test_reserve_stays_in_savings(self) -> None:
        data = small_params()
        data["reserve"] = {"enabled": True, "months_of_fixed_costs": 6}
        data["stochastic"]["yearly_return"] = 0.12  # equity very attractive: without reserve 100 %
        opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(data))
        opt.solve()
        savings = (1.0 - opt.opt_equity_share.values) * opt.opt_invested.values
        x = opt.opt_invested.values
        need = np.minimum(opt.reserve[:, None], x)
        ok = np.isfinite(x)
        ok[-1] = False  # last month: everything consumed
        self.assertTrue(np.all(savings[ok] >= need[ok] - 1e-6 * np.maximum(need[ok], 1.0)))
        self.assertGreater(float(np.nanmax(savings[:-1])), 0.0)

    def test_mortality_presets(self) -> None:
        from config.stochastic.stochastic_config_mapper import survival_parameters

        b, c = survival_parameters({"table": "destatis_male"})
        b3, c3 = survival_parameters({"table": "destatis_male", "age_shift": -3})
        self.assertEqual(c, c3)
        self.assertAlmostEqual(b3, b * np.exp(-3 * c))
        with self.assertRaises(ValueError):
            survival_parameters({"table": "unknown"})


if __name__ == "__main__":
    unittest.main()
