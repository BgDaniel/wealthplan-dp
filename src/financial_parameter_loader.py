import os
import datetime as dt
from typing import List, Dict, Type
import yaml

# ----------------------
# String constants
# ----------------------
DATE_FORMAT: str = "%Y-%m-%d"

KEY_CASHFLOWS: str = "cashflows"
KEY_REGULAR: str = "regular_cashflows"
KEY_INSURANCES: str = "insurances"
KEY_PENSION_PLANS: str = "pension_plans"
KEY_TYPE: str = "type"
KEY_NAME: str = "name"
KEY_RETIREMENT_DATE: str = "retirement_date"
KEY_MONTHLY_AMOUNT: str = "monthly_amount"
KEY_AMOUNT: str = "amount"
KEY_FREQUENCY: str = "frequency"
KEY_MONTHLY_CONTRIBUTION: str = "monthly_contribution"
KEY_MONTHLY_PAYOUT: str = "monthly_payout"

# Environment variable for base path
PARAMS_PATH_ENV: str = "FINANCIAL_PARAMS_PATH"

# Regular cashflow types
TYPE_SALARY: str = "salary"
TYPE_RENT: str = "rent"
TYPE_ELECTRICITY: str = "electricity"
TYPE_TELECOMMUNICATION: str = "telecommunication"
TYPE_PENSION: str = "pension"
TYPE_GROCERIES: str = "groceries"

REGULAR_KEYS: List[str] = [
    TYPE_SALARY,
    TYPE_RENT,
    TYPE_ELECTRICITY,
    TYPE_TELECOMMUNICATION,
    TYPE_PENSION,
    TYPE_GROCERIES,
]

# ----------------------
# Import Cashflow classes
# ----------------------
from src.wealthplan.cashflows.base import Cashflow
from src.wealthplan.cashflows.salary import Salary
from src.wealthplan.cashflows.rent import Rent
from src.wealthplan.cashflows.pension import Pension
from wealthplan.cashflows.electricity import Electricity
from wealthplan.cashflows.groceries import Groceries
from wealthplan.cashflows.insurances import Insurance
from wealthplan.cashflows.pension_plan import PensionPlan
from wealthplan.cashflows.telecommunication import Telecommunication


# Mapping from YAML type string to Cashflow class
REGULAR_CLASS_MAP: Dict[str, Type[Cashflow]] = {
    TYPE_SALARY: Salary,
    TYPE_RENT: Rent,
    TYPE_ELECTRICITY: Electricity,
    TYPE_TELECOMMUNICATION: Telecommunication,
    TYPE_PENSION: Pension,
    TYPE_GROCERIES: Groceries,
}


class FinancialParametersLoader:
    """
    Loads financial simulation parameters from a YAML file and returns
    a list of instantiated Cashflow objects.

    The YAML path is constructed by combining an environment variable base path
    with a filename passed to the constructor.
    """

    def __init__(self, filename: str) -> None:
        """
        Initialize loader with a YAML filename.

        Parameters
        ----------
        filename : str
            Name of the YAML file (e.g., 'simulation_parameters.yaml')
        """
        base_path: str = os.getenv(PARAMS_PATH_ENV)
        if not base_path:
            raise ValueError(f"Environment variable '{PARAMS_PATH_ENV}' is not set")

        self.yaml_path: str = os.path.join(base_path, filename)

    def load(self) -> List[Cashflow]:
        """
        Load parameters from YAML and instantiate Cashflow objects.

        Returns
        -------
        List[Cashflow]
            List of instantiated Cashflow objects.
        """
        with open(self.yaml_path, "r") as f:
            data: Dict = yaml.safe_load(f)

        cashflows: List[Cashflow] = []

        # --------------------
        # Regular cashflows
        # --------------------
        for key in REGULAR_KEYS:
            if key in data.get(KEY_CASHFLOWS, {}):
                params: Dict = data[KEY_CASHFLOWS][key].copy()
                cls = REGULAR_CLASS_MAP[key]

                # Handle optional retirement_date
                retirement_date: dt.date = None
                if KEY_RETIREMENT_DATE in params:
                    retirement_date = dt.datetime.strptime(
                        params[KEY_RETIREMENT_DATE], DATE_FORMAT
                    ).date()

                # Instantiate class with correct signature
                if cls in [Salary, Pension]:
                    obj = cls(
                        monthly_amount=params[KEY_MONTHLY_AMOUNT],
                        retirement_date=retirement_date,
                    )
                elif cls in [Rent, Electricity, Telecommunication, Groceries]:
                    obj = cls(monthly_amount=params[KEY_MONTHLY_AMOUNT])
                else:
                    raise ValueError(f"Unknown class for regular cashflow: {cls}")

                cashflows.append(obj)

        # --------------------
        # Insurances
        # --------------------
        for ins_params in data[KEY_CASHFLOWS].get(KEY_INSURANCES, []):
            if KEY_NAME not in ins_params:
                raise ValueError(f"Insurance entry missing '{KEY_NAME}': {ins_params}")
            obj = Insurance(
                amount=ins_params[KEY_AMOUNT],
                frequency=ins_params[KEY_FREQUENCY],
                name=ins_params[KEY_NAME],
            )
            cashflows.append(obj)

        # --------------------
        # Pension plans
        # --------------------
        for plan_params in data[KEY_CASHFLOWS].get(KEY_PENSION_PLANS, []):
            if KEY_NAME not in plan_params:
                raise ValueError(
                    f"Pension plan entry missing '{KEY_NAME}': {plan_params}"
                )
            retirement_date: dt.date = dt.datetime.strptime(
                plan_params[KEY_RETIREMENT_DATE], DATE_FORMAT
            ).date()
            obj = PensionPlan(
                name=plan_params[KEY_NAME],
                monthly_contribution=plan_params[KEY_MONTHLY_CONTRIBUTION],
                monthly_payout=plan_params[KEY_MONTHLY_PAYOUT],
                retirement_date=retirement_date,
            )
            cashflows.append(obj)

        return cashflows
