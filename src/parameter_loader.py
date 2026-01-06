import os
import datetime as dt
from typing import List, Dict, Type, Any
import yaml

# ----------------------
# Constants
# ----------------------
DATE_FORMAT: str = "%Y-%m-%d"
PARAMS_PATH_ENV: str = "FINANCIAL_PARAMS_PATH"

# Section keys
KEY_SIMULATION: str = "simulation_parameters"
KEY_LIFECYCLE: str = "lifecycle"
KEY_INSURANCE: str = "insurances"
KEY_TECHNICAL: str = "technical"
KEY_TOTAL_PENSION: str = "total_pension"

# Lifecycle cashflow types
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

# Cashflow fields
KEY_NAME: str = "name"
KEY_MONTHLY_AMOUNT: str = "monthly_amount"
KEY_AMOUNT: str = "amount"
KEY_FREQUENCY: str = "frequency"
KEY_MONTHLY_CONTRIBUTION: str = "monthly_contribution"
KEY_INITIAL_MONTHLY_CONTRIBUTION: str = "initial_monthly_contribution"
KEY_MONTHLY_PAYOUT: str = "monthly_payout"
KEY_MONTHLY_PAYOUT_BRUTTO: str = "monthly_payout_brutto"
KEY_TAXABLE_EARNINGS_SHARE: str = "taxable_earnings_share"
KEY_CONTRIBUTION_GROWTH_RATE: str = "contribution_growth_rate"
KEY_START_DATE_PLAN: str = "start_date"

# Technical parameter keys
KEY_W_MAX: str = "w_max"
KEY_W_STEP: str = "w_step"
KEY_C_STEP: str = "c_step"
KEY_SAVE: str = "save"

# Flat parameter dictionary keys
KEY_RUN_ID: str = "run_id"
KEY_CASHFLOWS: str = "cashflows"
KEY_START_DATE: str = "start_date"
KEY_END_DATE: str = "end_date"
KEY_RETIREMENT_DATE: str = "retirement_date"
KEY_INITIAL_WEALTH: str = "initial_wealth"
KEY_YEARLY_RETURN: str = "yearly_return"
KEY_BETA: str = 'beta'


# ----------------------
# Imports (cashflows)
# ----------------------
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

# Map lifecycle types to classes
REGULAR_CLASS_MAP: Dict[str, Type[CashflowBase]] = {
    TYPE_SALARY: Salary,
    TYPE_RENT: Rent,
    TYPE_ELECTRICITY: Electricity,
    TYPE_TELECOMMUNICATION: Telecommunication,
    TYPE_GROCERIES: Groceries,
}


class ParametersLoader:
    """Loads financial simulation parameters from YAML with full string constant usage."""

    def __init__(self, filename: str) -> None:
        base_path = os.getenv(PARAMS_PATH_ENV)
        if not base_path:
            raise ValueError(f"Environment variable '{PARAMS_PATH_ENV}' is not set")

        self.yaml_path = os.path.join(base_path, filename)

    def load(self) -> Dict[str, Any]:
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)

        sim_params = data.get(KEY_SIMULATION, {})
        technical_params = data.get(KEY_TECHNICAL, {})

        cashflows: List[CashflowBase] = []
        cashflows.extend(self._load_lifecycle_cashflows(data))
        cashflows.extend(self._load_insurances(data))
        cashflows.extend(self._load_total_pension(data))

        return {
            KEY_RUN_ID: sim_params.get(KEY_RUN_ID, "lifecycle_sim"),
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
            KEY_SAVE: technical_params[KEY_SAVE],
        }

    # ----------------------
    # Lifecycle cashflows
    # ----------------------
    @staticmethod
    def _load_lifecycle_cashflows(data: Dict) -> List[CashflowBase]:
        lifecycle = data.get(KEY_LIFECYCLE, {})
        cashflows: List[CashflowBase] = []

        for key in REGULAR_KEYS:
            if key not in lifecycle:
                continue

            params = lifecycle[key]
            cls = REGULAR_CLASS_MAP[key]

            cashflows.append(cls(**params))

        return cashflows

    @staticmethod
    def _load_insurances(data: Dict) -> List[CashflowBase]:
        lifecycle = data.get(KEY_LIFECYCLE, {})
        ins_list = lifecycle.get(KEY_INSURANCE, [])

        return [
            Insurance(
                **i,
            )
            for i in ins_list
        ]

    # ----------------------
    # Total Pension
    # ----------------------
    @staticmethod
    def _load_total_pension(data: Dict) -> List[CashflowBase]:
        cfg = data.get(KEY_TOTAL_PENSION)

        if not cfg:
            return []

        retirement_date = cfg[KEY_RETIREMENT_DATE]

        pension_plans: List = []

        # Private pension plans
        for p in cfg.get("private_pension_plans", []):
            if KEY_CONTRIBUTION_GROWTH_RATE in p:
                pension_plans.append(
                    PrivatePensionPlanDynamic(
                        name=p[KEY_NAME],
                        _monthly_contribution=float('nan'),
                        initial_monthly_contribution=p[
                            KEY_INITIAL_MONTHLY_CONTRIBUTION
                        ],
                        monthly_payout_brutto=p[KEY_MONTHLY_PAYOUT_BRUTTO],
                        taxable_earnings_share=p[KEY_TAXABLE_EARNINGS_SHARE],
                        contribution_growth_rate=p[KEY_CONTRIBUTION_GROWTH_RATE],
                        start_date=p[KEY_START_DATE],
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
