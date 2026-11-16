import copy
import datetime as dt
from typing import Any, Dict, List, Type

from wealthplan.cashflows.cashflow_base import CashflowBase
from wealthplan.cashflows.salary import Salary
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


# ----------------------
# Lifecycle cashflow keys
# ----------------------
TYPE_SALARY: str = "salary"
TYPE_RENT: str = "rent"
TYPE_ELECTRICITY: str = "electricity"
TYPE_TELECOMMUNICATION: str = "telecommunication"
TYPE_GROCERIES: str = "groceries"

REGULAR_KEYS: List[str] = [
    TYPE_SALARY,
    TYPE_RENT,
    TYPE_ELECTRICITY,
    TYPE_TELECOMMUNICATION,
    TYPE_GROCERIES,
]

REGULAR_CLASS_MAP: Dict[str, Type[CashflowBase]] = {
    TYPE_SALARY: Salary,
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

KEY_TECHNICAL: str = "technical"
KEY_USE_CACHE: str = "use_cache"
KEY_UTILITY_FUNCTION: str = "utility_function"
KEY_GAMMA: str = "gamma"
KEY_EPSILON: str = "epsilon"


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

        cashflows: List[CashflowBase] = []
        cashflows.extend(cls._load_lifecycle_cashflows(data))
        cashflows.extend(cls._load_insurances(data))
        cashflows.extend(cls._load_total_pension(data))

        params: Dict[str, Any] = {
            KEY_RUN_CONFIG_ID: sim[KEY_RUN_CONFIG_ID],
            KEY_START_DATE: sim[KEY_START_DATE],
            KEY_END_DATE: sim[KEY_END_DATE],
            KEY_RETIREMENT_DATE: sim[KEY_RETIREMENT_DATE],
            KEY_INITIAL_WEALTH: sim[KEY_INITIAL_WEALTH],
            KEY_YEARLY_RETURN_SAVINGS: sim[KEY_YEARLY_RETURN_SAVINGS],
            KEY_CURRENT_AGE: float(sim[KEY_CURRENT_AGE]),
            KEY_BETA: float(sim[KEY_BETA]),
            KEY_CASHFLOWS: cashflows,
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
    def _load_total_pension(data: Dict[str, Any]) -> List[CashflowBase]:
        """
        Create an aggregated TotalPension cashflow from all pension plans.

        Args:
            data: Parsed YAML configuration.

        Returns:
            List containing a single TotalPension instance,
            or an empty list if no pension configuration exists.
        """
        cfg: Dict[str, Any] | None = data.get(KEY_TOTAL_PENSION)
        if not cfg:
            return []

        retirement_date: str = cfg[KEY_RETIREMENT_DATE]
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
                    monthly_payout_brutto=p[KEY_MONTHLY_PAYOUT_BRUTTO]
                )
            )

        return [TotalPension(pensions=pension_plans, retirement_date=retirement_date)]

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
