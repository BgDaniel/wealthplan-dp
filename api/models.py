from pydantic import BaseModel, Field
import datetime as dt

class OptimizationInput(BaseModel):
    """
    Input parameters for the wealth optimization model.
    """

    salary: float = Field(..., description="Monthly salary before retirement", example=6000)
    rent: float = Field(..., description="Monthly rent expense", example=1300)
    expenses: float = Field(..., description="Essential monthly living costs", example=1500)
    initial_wealth: float = Field(..., description="Initial wealth or savings", example=100000)
    investment_return: float = Field(..., description="Expected annual investment return rate", example=0.05)

    retirement_date: dt.date = Field(
        default_factory=lambda: dt.date.today().replace(year=dt.date.today().year + 20),
        description="Retirement date (default: 20 years from today)",
        example=str(dt.date.today().replace(year=dt.date.today().year + 20))
    )

    simulation_start_date: dt.date = Field(
        default_factory=dt.date.today,  # Default to today
        description="Start date of the simulation (YYYY-MM-DD)",
        example=str(dt.date.today())
    )

    simulation_end_date: dt.date = Field(
        default_factory=lambda: (dt.date.today().replace(year=dt.date.today().year + 20)
                                 .replace(year=dt.date.today().year + 20 + 15)),
        description="End date of the simulation (default: 15 years after retirement)",
        example=str(dt.date.today().replace(year=dt.date.today().year + 35))
    )
