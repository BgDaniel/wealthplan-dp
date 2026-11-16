import datetime as dt
import unittest

from wealthplan.cashflows.pension.private_pension_plans.private_pension_plan import PrivatePensionPlan
from wealthplan.cashflows.pension.public_pension_plan import PublicPensionPlan
from wealthplan.cashflows.pension.total_pension import TotalPension


class TestTotalPension(unittest.TestCase):
    """Pension contributions before and taxed payouts after retirement."""

    def setUp(self) -> None:
        self.pension = TotalPension(
            pensions=[
                PublicPensionPlan(monthly_payout_brutto=3100.0),
                PrivatePensionPlan(name="private", monthly_payout_brutto=200.0, _monthly_contribution=60.0,
                                   taxable_earnings_share=0.18),
            ],
            retirement_date=dt.date(2055, 7, 1),
        )

    def test_contributions_before_retirement(self) -> None:
        self.assertAlmostEqual(self.pension.cashflow(dt.date(2030, 1, 1)), -60.0)

    def test_tax_uses_yearly_income(self) -> None:
        # monthly taxable 3100 + 36 = 3136 -> 37 632 per year -> 31 % bracket
        net = self.pension.cashflow(dt.date(2060, 1, 1))
        self.assertAlmostEqual(net, 3300.0 - 0.31 * 3136.0)

    def test_brackets_are_contiguous(self) -> None:
        self.assertEqual(TotalPension._progressive_tax(10_000.0), 0.0)
        self.assertEqual(TotalPension._progressive_tax(10_000.5), 0.17)  # was 45 % (gap in the brackets)
        self.assertEqual(TotalPension._progressive_tax(60_000.5), 0.42)
        self.assertEqual(TotalPension._progressive_tax(1e6), 0.45)


if __name__ == "__main__":
    unittest.main()
