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
# Constants for mapping
# ----------------------
TYPE_SALARY = "salary"
TYPE_RENT = "rent"
TYPE_ELECTRICITY = "electricity"
TYPE_TELECOMMUNICATION = "telecommunication"
TYPE_GROCERIES = "groceries"

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

KEY_SIMULATION = "simulation_parameters"
KEY_LIFECYCLE = "lifecycle"
KEY_INSURANCE = "insurances"
KEY_TECHNICAL = "technical"
KEY_TOTAL_PENSION = "total_pension"

KEY_RUN_CONFIG_ID = "run_config_id"
KEY_CASHFLOWS = "cashflows"
KEY_START_DATE = "start_date"
KEY_END_DATE = "end_date"
KEY_RETIREMENT_DATE = "retirement_date"
KEY_INITIAL_WEALTH = "initial_wealth"
KEY_YEARLY_RETURN = "yearly_return"
KEY_BETA = "beta"
KEY_W_MAX = "w_max"
KEY_W_STEP = "w_step"
KEY_C_STEP = "c_step"
KEY_USE_CACHE = "use_cache"
KEY_NAME = "name"
KEY_MONTHLY_CONTRIBUTION = "monthly_contribution"
KEY_INITIAL_MONTHLY_CONTRIBUTION = "initial_monthly_contribution"
KEY_MONTHLY_PAYOUT_BRUTTO = "monthly_payout_brutto"
KEY_TAXABLE_EARNINGS_SHARE = "taxable_earnings_share"
KEY_CONTRIBUTION_GROWTH_RATE = "contribution_growth_rate"
KEY_START_DATE_PLAN = "start_date"
KEY_RUN_TASK_ID = "run_task_id"


class ConfigMapper:
    """
    Maps already-loaded YAML params (as Python dict) into the parameter dictionary
    required by the deterministic Bellman optimizer.

    Responsibilities:
    - Transform lifecycle, insurance, and pension cashflows into proper CashflowBase objects.
    - Merge simulation and technical parameters into a single dict.

    This class does NOT fetch YAML itself; it works purely on dicts.
    """

    @staticmethod
    def map_yaml_to_params(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a loaded YAML dictionary into optimizer-ready parameters.

        Args:
            data: YAML params loaded as a Python dict.

        Returns:
            A dictionary with keys expected by DeterministicBellmanOptimizer, e.g.:
            - 'run_id', 'start_date', 'end_date', 'retirement_date'
            - 'initial_wealth', 'yearly_return', 'beta'
            - 'cashflows' (list of CashflowBase objects)
            - 'w_max', 'w_step', 'c_step', 'use_cache'
        """
        sim_params: Dict[str, Any] = data.get(KEY_SIMULATION, {})
        technical_params: Dict[str, Any] = data.get(KEY_TECHNICAL, {})

        cashflows: List[CashflowBase] = []
        cashflows.extend(ConfigMapper._load_lifecycle_cashflows(data))
        cashflows.extend(ConfigMapper._load_insurances(data))
        cashflows.extend(ConfigMapper._load_total_pension(data))

        return {
            KEY_RUN_CONFIG_ID: sim_params[KEY_RUN_CONFIG_ID],
            KEY_START_DATE: sim_params[KEY_START_DATE],
            KEY_END_DATE: sim_params[KEY_END_DATE],
            KEY_RETIREMENT_DATE: sim_params[KEY_RETIREMENT_DATE],
            KEY_INITIAL_WEALTH: sim_params[KEY_INITIAL_WEALTH],
            KEY_YEARLY_RETURN: sim_params[KEY_YEARLY_RETURN],
            KEY_BETA: sim_params[KEY_BETA],
            KEY_CASHFLOWS: cashflows,
            KEY_W_MAX: technical_params[KEY_W_MAX],
            KEY_W_STEP: technical_params[KEY_W_STEP],
            KEY_C_STEP: technical_params[KEY_C_STEP],
            KEY_USE_CACHE: technical_params[KEY_USE_CACHE]
        }

    # ----------------------
    # Lifecycle cashflows
    # ----------------------
    @staticmethod
    def _load_lifecycle_cashflows(data: Dict[str, Any]) -> List[CashflowBase]:
        """
        Load standard lifecycle cashflows from YAML params.

        Args:
            data: YAML params as dict.

        Returns:
            List of CashflowBase objects corresponding to lifecycle cashflows.
        """
        lifecycle: Dict[str, Any] = data.get(KEY_LIFECYCLE, {})
        cashflows: List[CashflowBase] = []

        for key in REGULAR_KEYS:
            if key not in lifecycle:
                continue
            params: Dict[str, Any] = lifecycle[key]
            cls: Type[CashflowBase] = REGULAR_CLASS_MAP[key]
            cashflows.append(cls(**params))

        return cashflows

    @staticmethod
    def _load_insurances(data: Dict[str, Any]) -> List[CashflowBase]:
        """
        Load insurance cashflows from YAML params.

        Args:
            data: YAML params as dict.

        Returns:
            List of Insurance objects.
        """
        lifecycle: Dict[str, Any] = data.get(KEY_LIFECYCLE, {})
        ins_list: List[Dict[str, Any]] = lifecycle.get(KEY_INSURANCE, [])

        return [Insurance(**i) for i in ins_list]

    @staticmethod
    def _load_total_pension(data: Dict[str, Any]) -> List[CashflowBase]:
        """
        Load total pension cashflows (private and public) from YAML params.

        Args:
            data: YAML params as dict.

        Returns:
            List with a single TotalPension object containing all pension plans.
        """
        cfg: Dict[str, Any] = data.get(KEY_TOTAL_PENSION)
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
                    monthly_payout_brutto=p[KEY_MONTHLY_PAYOUT_BRUTTO],
                )
            )

        return [TotalPension(pensions=pension_plans, retirement_date=retirement_date)]
