import datetime as dt
import unittest

from wealthplan.cashflows.pension.private_pension_plans.private_pension_plan import PrivatePensionPlan
from wealthplan.cashflows.pension.public_pension_plan import PublicPensionPlan
from wealthplan.cashflows.pension.ruerup_pension import RuerupPension
from wealthplan.cashflows.pension.total_pension import TotalPension
from wealthplan.tax.tax_model import TaxModel


class TestTaxModel(unittest.TestCase):
    """Income tax tariff, Besteuerungsanteil, capital gains rates."""

    def test_tariff_2025(self) -> None:
        self.assertEqual(TaxModel.tariff_2025(12_096), 0.0)
        self.assertAlmostEqual(TaxModel.tariff_2025(40_000), 7_320.0, delta=2.0)
        tax = 0.42 * 100_000 - 10_911.92  # Soli applies above 19 950 € tax (sliding zone)
        self.assertAlmostEqual(TaxModel.tariff_2025(100_000), tax + min(0.055 * tax, 0.119 * (tax - 19_950)), delta=1.0)
        # continuity at the zone limits
        for x in (17_443, 68_480, 277_825):
            self.assertAlmostEqual(TaxModel.tariff_2025(x), TaxModel.tariff_2025(x + 1), delta=5.0 + 0.06 * 1000)

    def test_besteuerungsanteil(self) -> None:
        self.assertAlmostEqual(TaxModel.taxable_share_basis_pension(2005), 0.50)
        self.assertAlmostEqual(TaxModel.taxable_share_basis_pension(2020), 0.80)
        self.assertAlmostEqual(TaxModel.taxable_share_basis_pension(2023), 0.825)
        self.assertAlmostEqual(TaxModel.taxable_share_basis_pension(2055), 0.985)
        self.assertAlmostEqual(TaxModel.taxable_share_basis_pension(2070), 1.0)

    def test_capital_gains_rates(self) -> None:
        tax = TaxModel()
        self.assertAlmostEqual(tax.savings_tax_rate, 0.26375)
        self.assertAlmostEqual(tax.equity_tax_rate, 0.26375 * 0.7)
        self.assertEqual(TaxModel(tax_equity_returns=False).equity_tax_rate, 0.0)

    def test_refund_respects_deduction_room(self) -> None:
        tax = TaxModel(marginal_tax_rate_working=0.42, basis_pension_deduction_room=6_000)
        self.assertAlmostEqual(tax.basis_pension_refund_monthly(600.0), 0.42 * 6_000 / 12)


class TestRuerupPension(unittest.TestCase):
    """Capital accumulation and payout of the Basisrente."""

    def test_no_costs_no_return(self) -> None:
        plan = RuerupPension(name="R", monthly_contribution_amount=600.0, start_date=dt.date(2026, 1, 1),
                             pension_start=dt.date(2036, 1, 1), expected_return=0.0, running_costs=0.0,
                             acquisition_costs=0.0, pension_factor=30.0)
        self.assertAlmostEqual(plan.capital_at_pension_start(), 120 * 600.0)
        self.assertAlmostEqual(plan.monthly_payout_brutto, 72_000 / 10_000 * 30.0)
        self.assertEqual(plan.monthly_contribution(dt.date(2036, 1, 1)), 0.0)

    def test_costs_reduce_capital(self) -> None:
        kwargs = dict(name="R", monthly_contribution_amount=600.0, start_date=dt.date(2026, 1, 1),
                      pension_start=dt.date(2056, 1, 1), expected_return=0.06)
        cheap = RuerupPension(**kwargs, running_costs=0.0, acquisition_costs=0.0)
        costly = RuerupPension(**kwargs, running_costs=0.012, acquisition_costs=0.04)
        self.assertGreater(cheap.capital_at_pension_start(), costly.capital_at_pension_start())


class TestTotalPension(unittest.TestCase):
    """Contributions, Rürup refund and taxed payouts."""

    def setUp(self) -> None:
        self.retirement = dt.date(2055, 7, 1)
        self.ruerup = RuerupPension(name="Rürup", monthly_contribution_amount=600.0, start_date=dt.date(2026, 1, 1),
                                    pension_start=self.retirement)
        self.pension = TotalPension(
            pensions=[
                PublicPensionPlan(monthly_payout_brutto=3100.0),
                PrivatePensionPlan(name="private", monthly_payout_brutto=200.0, _monthly_contribution=60.0,
                                   taxable_earnings_share=0.18),
                self.ruerup,
            ],
            retirement_date=self.retirement,
            tax=TaxModel(marginal_tax_rate_working=0.42),
        )

    def test_contributions_and_refund_before_retirement(self) -> None:
        parts = self.pension.components(dt.date(2030, 1, 1))
        self.assertAlmostEqual(parts["private contribution"], -60.0)
        self.assertAlmostEqual(parts["Rürup contribution"], -600.0)
        self.assertAlmostEqual(parts["Rürup tax refund"], 0.42 * 600.0)
        self.assertAlmostEqual(self.pension.cashflow(dt.date(2030, 1, 1)), -660.0 + 252.0)

    def test_tax_on_taxable_part_after_retirement(self) -> None:
        share = TaxModel.taxable_share_basis_pension(2055)
        taxable = (3100.0 + self.ruerup.monthly_payout_brutto) * share + 0.18 * 200.0
        expected_tax = TaxModel.tariff_2025(12 * taxable - 102.0) / 12
        parts = self.pension.components(dt.date(2060, 1, 1))
        self.assertAlmostEqual(-parts["Income tax on pensions"], expected_tax, places=6)
        gross = 3100.0 + 200.0 + self.ruerup.monthly_payout_brutto
        self.assertAlmostEqual(self.pension.cashflow(dt.date(2060, 1, 1)), gross - expected_tax, places=6)

    def test_flat_rate(self) -> None:
        self.pension.tax = TaxModel(income_tax_mode="flat", income_tax_rate_retirement=0.2)
        parts = self.pension.components(dt.date(2060, 1, 1))
        share = TaxModel.taxable_share_basis_pension(2055)
        taxable = (3100.0 + self.ruerup.monthly_payout_brutto) * share + 0.18 * 200.0
        self.assertAlmostEqual(-parts["Income tax on pensions"], 0.2 * (12 * taxable - 102.0) / 12, places=6)


if __name__ == "__main__":
    unittest.main()
