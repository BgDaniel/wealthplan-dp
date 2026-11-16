import copy
import datetime as dt
from typing import Any, Dict, List, Type

from wealthplan.cashflows.cashflow_base import CashflowBase
from wealthplan.cashflows.payroll import GrossSalary, NetSalary, PayrollModel
from wealthplan.cashflows.scheduled_expense import ScheduledExpense
from wealthplan.cashflows.rent import Rent
from wealthplan.cashflows.electricity import Electricity
from wealthplan.cashflows.telecommunication import Telecommunication
from wealthplan.cashflows.groceries import Groceries
from wealthplan.cashflows.insurances import Insurance
from wealthplan.cashflows.pension.total_pension import TotalPension
from wealthplan.cashflows.pension.private_pension_plans.private_pension_plan import (
    PrivatePensionPlan,
)
from wealthplan.cashflows.pension.private_pension_plans.private_pension_plan_dynamic import (
    PrivatePensionPlanDynamic,
)
from wealthplan.cashflows.pension.public_pension_plan import PublicPensionPlan
from wealthplan.cashflows.pension.ruerup_pension import RuerupPension
from wealthplan.tax.tax_model import TaxModel
from wealthplan.economy.inflation import IndexedCashflow, InflationModel


# ----------------------
# Lifecycle cashflow keys
# ----------------------
TYPE_SALARY: str = "salary"
TYPE_RENT: str = "rent"
TYPE_ELECTRICITY: str = "electricity"
TYPE_TELECOMMUNICATION: str = "telecommunication"
TYPE_GROCERIES: str = "groceries"

REGULAR_KEYS: List[str] = [
    TYPE_RENT,
    TYPE_ELECTRICITY,
    TYPE_TELECOMMUNICATION,
    TYPE_GROCERIES,
]

REGULAR_CLASS_MAP: Dict[str, Type[CashflowBase]] = {
    TYPE_RENT: Rent,
    TYPE_ELECTRICITY: Electricity,
    TYPE_TELECOMMUNICATION: Telecommunication,
    TYPE_GROCERIES: Groceries,
}

# ----------------------
# YAML keys
# ----------------------
KEY_SIMULATION: str = "simulation_parameters"
KEY_LIFECYCLE: str = "lifecycle"
KEY_INSURANCE: str = "insurances"
KEY_TOTAL_PENSION: str = "total_pension"

KEY_RUN_CONFIG_ID: str = "run_config_id"
KEY_START_DATE: str = "start_date"
KEY_END_DATE: str = "end_date"
KEY_RETIREMENT_DATE: str = "retirement_date"
KEY_INITIAL_WEALTH: str = "initial_wealth"
KEY_YEARLY_RETURN_SAVINGS: str = "yearly_return_savings"
KEY_CASHFLOWS: str = "cashflows"

KEY_NAME: str = "name"
KEY_MONTHLY_CONTRIBUTION: str = "monthly_contribution"
KEY_INITIAL_MONTHLY_CONTRIBUTION: str = "initial_monthly_contribution"
KEY_MONTHLY_PAYOUT_BRUTTO: str = "monthly_payout_brutto"
KEY_TAXABLE_EARNINGS_SHARE: str = "taxable_earnings_share"
KEY_CONTRIBUTION_GROWTH_RATE: str = "contribution_growth_rate"
KEY_START_DATE_PLAN: str = "start_date"

KEY_CURRENT_AGE: str = "current_age"
KEY_BETA: str = "beta"

KEY_INITIAL_EQUITY: str = "initial_equity"
KEY_PAYROLL: str = "payroll"
KEY_TAX: str = "tax"
KEY_INFLATION: str = "inflation"
KEY_INFLATION_RATE: str = "inflation_rate"
KEY_RESERVE: str = "reserve"
KEY_RESERVE_MONTHS: str = "reserve_months"
KEY_SAVINGS_TAX_RATE: str = "savings_tax_rate"
KEY_EQUITY_TAX_RATE: str = "equity_tax_rate"
KEY_RUERUP_PLANS: str = "ruerup_plans"

KEY_TECHNICAL: str = "technical"
KEY_USE_CACHE: str = "use_cache"
KEY_UTILITY_FUNCTION: str = "utility_function"
KEY_GAMMA: str = "gamma"
KEY_EPSILON: str = "epsilon"


def _payroll_refund(salary: GrossSalary):
    """Exact monthly Rürup refund from the payroll of the gross salary (date, monthly contribution)."""
    def refund(date, contribution_monthly: float) -> float:
        gross = salary.year_payroll(date)["total_gross"]
        return salary.payroll.basis_pension_refund_yearly(gross, 12.0 * contribution_monthly,
                                                          salary.tax_scale(date)) / 12.0
    return refund


class ConfigMapper:
    """
    Base mapper that converts YAML configuration dictionaries into
    domain-level optimizer parameters.

    Responsibilities:
    - Read simulation parameters
    - Create lifecycle cashflows
    - Create insurance cashflows
    - Aggregate private and public pension plans

    This class is solver-agnostic and contains no numerical or
    technical optimization parameters.
    """

    @staticmethod
    def normalize_dates(data: Any) -> Any:
        """
        Copy of the configuration with every ``*date`` value as ``datetime.date``.

        YAML parses unquoted ISO dates itself, JSON (job files) and quoted YAML values
        give strings.
        """
        if isinstance(data, dict):
            return {
                k: (dt.date.fromisoformat(str(v)[:10]) if k.endswith("date") and isinstance(v, (str, dt.datetime))
                    else ConfigMapper.normalize_dates(v))
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [ConfigMapper.normalize_dates(v) for v in data]
        return copy.copy(data)

    @classmethod
    def map_yaml_to_params(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        data = cls.normalize_dates(data)

        # Base simulation parameters
        sim: Dict[str, Any] = data[KEY_SIMULATION]

        tax = TaxModel.from_dict(data.get(KEY_TAX))
        inflation = InflationModel.from_dict(data.get(KEY_INFLATION), sim[KEY_START_DATE])

        salary = cls._load_salary(data, inflation)
        cashflows: List[CashflowBase] = [salary] if salary is not None else []
        cashflows.extend(cls._load_lifecycle_cashflows(data))
        cashflows.extend(cls._load_insurances(data))
        cashflows.extend(cls._load_scheduled_expenses(data))
        cashflows = cls._index_cashflows(cashflows, inflation, sim[KEY_START_DATE])
        cashflows.extend(cls._load_total_pension(data, tax, inflation, salary))

        reserve: Dict[str, Any] = data.get(KEY_RESERVE) or {}
        reserve_months = float(reserve.get("months_of_fixed_costs", 0.0)) if reserve.get("enabled", False) else 0.0

        params: Dict[str, Any] = {
            KEY_RUN_CONFIG_ID: sim[KEY_RUN_CONFIG_ID],
            KEY_START_DATE: sim[KEY_START_DATE],
            KEY_END_DATE: sim[KEY_END_DATE],
            KEY_RETIREMENT_DATE: sim[KEY_RETIREMENT_DATE],
            KEY_INITIAL_WEALTH: sim[KEY_INITIAL_WEALTH],
            KEY_YEARLY_RETURN_SAVINGS: sim[KEY_YEARLY_RETURN_SAVINGS],
            KEY_CURRENT_AGE: float(sim[KEY_CURRENT_AGE]),
            KEY_INITIAL_EQUITY: float(sim.get(KEY_INITIAL_EQUITY, 0.0)),
            KEY_BETA: float(sim[KEY_BETA]),
            KEY_CASHFLOWS: cashflows,
            KEY_SAVINGS_TAX_RATE: tax.savings_tax_rate,
            KEY_EQUITY_TAX_RATE: tax.equity_tax_rate,
            KEY_INFLATION_RATE: inflation.rate,
            KEY_RESERVE_MONTHS: reserve_months,
            "buy_cost": float((data.get("trading") or {}).get("buy_cost", 0.0)),
            "sell_cost": float((data.get("trading") or {}).get("sell_cost", 0.0)),
        }

        # ----------------------
        # Load utility function
        # ----------------------
        params.update(cls._load_crra_params(data))

        params[KEY_USE_CACHE] = data[KEY_TECHNICAL].get(KEY_USE_CACHE, False)

        return params

    # ----------------------
    # Lifecycle cashflows
    # ----------------------
    @staticmethod
    def _load_lifecycle_cashflows(data: Dict[str, Any]) -> List[CashflowBase]:
        """
        Instantiate regular lifecycle cashflows (salary, rent, etc.).

        Args:
            data: Parsed YAML configuration.

        Returns:
            List of instantiated CashflowBase objects.
        """
        lifecycle: Dict[str, Any] = data.get(KEY_LIFECYCLE, {})
        cashflows: List[CashflowBase] = []

        for key in REGULAR_KEYS:
            if key in lifecycle:
                cls: Type[CashflowBase] = REGULAR_CLASS_MAP[key]
                cashflows.append(cls(**lifecycle[key]))

        return cashflows

    @staticmethod
    def _load_salary(data: Dict[str, Any], inflation: InflationModel) -> CashflowBase | None:
        """
        Salary from ``lifecycle.salary``:

        * ``mode: gross`` (default if ``yearly_fixed`` is given): yearly fixed gross salary in 12
          payments plus ``yearly_bonus`` in ``bonus_month``, gross-to-net by the ``payroll``
          section; grows with inflation and ``salary_real_growth``.
        * ``mode: net``: ``monthly_net`` plus ``bonus_net`` in ``bonus_month`` (the former
          ``monthly_amount`` is read as ``monthly_net``).
        """
        cfg: Dict[str, Any] | None = data.get(KEY_LIFECYCLE, {}).get(TYPE_SALARY)
        if not cfg:
            return None
        retirement = cfg.get(KEY_RETIREMENT_DATE, data[KEY_SIMULATION][KEY_RETIREMENT_DATE])
        bonus_month = int(cfg.get("bonus_month", 4))
        mode = cfg.get("mode", "gross" if "yearly_fixed" in cfg else "net")
        if mode == "gross":
            growth = (1.0 + inflation.rate) * (1.0 + inflation.salary_real_growth) - 1.0
            return GrossSalary(yearly_fixed=float(cfg["yearly_fixed"]), yearly_bonus=float(cfg.get("yearly_bonus", 0.0)),
                               bonus_month=bonus_month, retirement_date=retirement,
                               payroll=PayrollModel.from_dict(data.get(KEY_PAYROLL)),
                               base_date=data[KEY_SIMULATION][KEY_START_DATE], growth=growth,
                               tax_scale=inflation.tax_scale)
        if mode == "net":
            monthly = float(cfg.get("monthly_net", cfg.get("monthly_amount", 0.0)))
            return NetSalary(monthly_net=monthly, bonus_net=float(cfg.get("bonus_net", 0.0)), bonus_month=bonus_month,
                             retirement_date=retirement)
        raise ValueError("salary mode must be 'gross' or 'net'")

    @staticmethod
    def _index_cashflows(cashflows: List[CashflowBase], inflation: InflationModel,
                         start: Any) -> List[CashflowBase]:
        """
        Let salary and fixed expenses grow (amounts in the YAML are in euros of the start date):
        salary with inflation and real growth, expenses with inflation if indexed.
        """
        out: List[CashflowBase] = []
        for cf in cashflows:
            if isinstance(cf, GrossSalary):
                rate = 0.0  # grows itself (thresholds of the payroll must follow the tax scale)
            elif isinstance(cf, NetSalary):
                rate = (1.0 + inflation.rate) * (1.0 + inflation.salary_real_growth) - 1.0
            else:
                rate = inflation.rate if inflation.expenses_indexed else 0.0
            out.append(IndexedCashflow(cf, rate, start) if rate != 0.0 else cf)
        return out

    @staticmethod
    def _load_scheduled_expenses(data: Dict[str, Any]) -> List[CashflowBase]:
        """``lifecycle.other_expenses``: monthly expenses with optional start / end date."""
        cfg = data.get(KEY_LIFECYCLE, {}).get("other_expenses", []) or []
        return [ScheduledExpense(name=e["name"], monthly_amount=float(e["monthly_amount"]),
                                 start_date=e.get("start_date"), end_date=e.get("end_date")) for e in cfg]

    @staticmethod
    def _load_insurances(data: Dict[str, Any]) -> List[CashflowBase]:
        """
        Instantiate insurance cashflows.

        Args:
            data: Parsed YAML configuration.

        Returns:
            List of Insurance cashflows.
        """
        lifecycle: Dict[str, Any] = data.get(KEY_LIFECYCLE, {})
        insurance_cfg: List[Dict[str, Any]] = lifecycle.get(KEY_INSURANCE, [])

        return [Insurance(**cfg) for cfg in insurance_cfg]

    # ----------------------
    # Pension handling
    # ----------------------
    @staticmethod
    def _load_total_pension(data: Dict[str, Any], tax: TaxModel, inflation: InflationModel | None = None,
                            salary: CashflowBase | None = None) -> List[CashflowBase]:
        """
        Create an aggregated TotalPension cashflow from all pension plans.

        Args:
            data: Parsed YAML configuration.
            tax: Tax parameters (pension income tax, Rürup refund).
            inflation: Indexation of the public pension and of the tariff brackets.
            salary: With a gross salary the Rürup refund is computed exactly from the payroll.

        Returns:
            List containing a single TotalPension instance,
            or an empty list if no pension configuration exists.
        """
        cfg: Dict[str, Any] | None = data.get(KEY_TOTAL_PENSION)
        if not cfg:
            return []

        retirement_date = cfg[KEY_RETIREMENT_DATE]
        # end of all contributions (early retirement); default: the pension start
        contributions_end = cfg.get("contributions_end_date", retirement_date)
        pension_plans: List[CashflowBase] = []

        # Private pension plans
        for p in cfg.get("private_pension_plans", []):
            if KEY_CONTRIBUTION_GROWTH_RATE in p:
                pension_plans.append(
                    PrivatePensionPlanDynamic(
                        name=p[KEY_NAME],
                        _monthly_contribution=float("nan"),
                        initial_monthly_contribution=p[KEY_INITIAL_MONTHLY_CONTRIBUTION],
                        monthly_payout_brutto=p[KEY_MONTHLY_PAYOUT_BRUTTO],
                        taxable_earnings_share=p[KEY_TAXABLE_EARNINGS_SHARE],
                        contribution_growth_rate=p[KEY_CONTRIBUTION_GROWTH_RATE],
                        start_date=p[KEY_START_DATE_PLAN],
                    )
                )
            else:
                pension_plans.append(
                    PrivatePensionPlan(
                        name=p[KEY_NAME],
                        _monthly_contribution=p[KEY_MONTHLY_CONTRIBUTION],
                        monthly_payout_brutto=p[KEY_MONTHLY_PAYOUT_BRUTTO],
                        taxable_earnings_share=p[KEY_TAXABLE_EARNINGS_SHARE],
                    )
                )

        # Public pension plans
        for p in cfg.get("public_pension_plan", []):
            pension_plans.append(
                PublicPensionPlan(
                    monthly_payout_brutto=p[KEY_MONTHLY_PAYOUT_BRUTTO],
                    indexation=inflation.public_pension_indexation if inflation else 0.0,
                    base_date=data[KEY_SIMULATION][KEY_START_DATE],
                )
            )

        # Basisrente (Rürup): contributions until, payout from the retirement date
        for p in cfg.get(KEY_RUERUP_PLANS, []):
            pension_plans.append(
                RuerupPension(
                    name=p.get(KEY_NAME, "Rürup"),
                    monthly_contribution_amount=float(p["monthly_contribution"]),
                    start_date=p.get(KEY_START_DATE_PLAN, data[KEY_SIMULATION][KEY_START_DATE]),
                    pension_start=retirement_date,
                    contribution_end=contributions_end,
                    contribution_growth=float(p.get("contribution_growth", 0.0)),
                    expected_return=float(p.get("expected_return", 0.06)),
                    running_costs=float(p.get("running_costs", 0.012)),
                    acquisition_costs=float(p.get("acquisition_costs", 0.04)),
                    pension_factor=float(p.get("pension_factor", 28.0)),
                    initial_capital=float(p.get("initial_capital", 0.0)),
                    fund_linked=bool(p.get("fund_linked", True)),
                )
            )

        return [TotalPension(pensions=pension_plans, retirement_date=retirement_date, tax=tax,
                             tax_scale=inflation.tax_scale if inflation else None,
                             refund=_payroll_refund(salary) if isinstance(salary, GrossSalary) else None,
                             contributions_end=contributions_end)]

    @staticmethod
    def _load_crra_params(data: Dict[str, Any]) -> Dict[str, float]:
        """
        Load the CRRA utility parameters from the configuration.

        Args:
            data: Parsed YAML configuration.

        Returns:
            Dictionary with gamma and epsilon.
        """
        utility_function: Dict[str, Any] = data.get(KEY_UTILITY_FUNCTION, {})

        return {
            KEY_GAMMA: float(utility_function.get(KEY_GAMMA, 0.5)),
            KEY_EPSILON: float(utility_function.get(KEY_EPSILON, 1e-8))
        }
