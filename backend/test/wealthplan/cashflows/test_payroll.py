import datetime as dt
import unittest

from wealthplan.cashflows.payroll import GrossSalary, NetSalary, PayrollModel
from wealthplan.tax.tax_model import TaxModel


class TestPayroll(unittest.TestCase):
    """Gross-to-net, bonus month, Rürup refund."""

    def test_contributions_respect_ceilings(self) -> None:
        p = PayrollModel()
        y = p.yearly(152_500)
        expected = (0.093 + 0.013) * 96_600 + (0.0855 + 0.018) * 66_150
        self.assertAlmostEqual(y["social"], expected, places=6)
        self.assertAlmostEqual(y["basis_room"], 29_344 - 2 * 0.093 * 96_600, places=6)

    def test_income_tax_on_taxable_income(self) -> None:
        p = PayrollModel()
        y = p.yearly(152_500)
        pension = 0.093 * 96_600
        health = 0.96 * 0.0855 * 66_150 + 0.018 * 66_150
        taxable = 152_500 - 1_230 - pension - health - 36
        self.assertAlmostEqual(y["taxable"], taxable, places=6)
        self.assertAlmostEqual(y["income_tax"], TaxModel.tariff_2025(taxable), places=6)

    def test_church_tax(self) -> None:
        a = PayrollModel().yearly(100_000)
        b = PayrollModel(church_tax_rate=0.09).yearly(100_000)
        est = TaxModel.income_tax_2025(a["taxable"])
        self.assertAlmostEqual(b["income_tax"] - a["income_tax"], 0.09 * est, places=6)

    def test_bonus_month_and_year_sum(self) -> None:
        s = GrossSalary(120_000, 32_500, 4, dt.date(2060, 1, 1), PayrollModel(), dt.date(2026, 1, 1))
        months = [dt.date(2026, m, 1) for m in range(1, 13)]
        total = sum(s.cashflow(m) for m in months)
        self.assertAlmostEqual(total, PayrollModel().yearly(152_500)["net"], places=6)
        self.assertIn("Bonus (gross)", s.components(dt.date(2026, 4, 1)))
        self.assertNotIn("Bonus (gross)", s.components(dt.date(2026, 5, 1)))
        self.assertEqual(s.cashflow(dt.date(2060, 1, 1)), 0.0)

    def test_growth_and_indexed_thresholds(self) -> None:
        """With everything indexed by the same rate, the net grows with that rate."""
        g = 0.02
        s = GrossSalary(120_000, 32_500, 4, dt.date(2060, 1, 1), PayrollModel(), dt.date(2026, 1, 1), growth=g,
                        tax_scale=lambda d: (1 + g) ** (d.year - 2026))
        y0 = sum(s.cashflow(dt.date(2026, m, 1)) for m in range(1, 13))
        y5 = sum(s.cashflow(dt.date(2031, m, 1)) for m in range(1, 13))
        self.assertAlmostEqual(y5 / y0, (1 + g) ** 5, delta=1e-3)

    def test_ruerup_refund_exact(self) -> None:
        p = PayrollModel()
        refund = p.basis_pension_refund_yearly(152_500, 7_200)
        a = p.yearly(152_500)
        self.assertAlmostEqual(refund, TaxModel.tariff_2025(a["taxable"]) - TaxModel.tariff_2025(a["taxable"] - 7_200),
                               places=6)

    def test_net_salary(self) -> None:
        s = NetSalary(5_000, 15_000, 4, dt.date(2060, 1, 1))
        self.assertEqual(s.cashflow(dt.date(2026, 4, 1)), 20_000)
        self.assertEqual(s.cashflow(dt.date(2026, 5, 1)), 5_000)


if __name__ == "__main__":
    unittest.main()
