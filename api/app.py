"""
WealthPlan Optimizer API
------------------------

This module exposes a REST API to run the Bellman-based wealth optimization model.
Clients can send input parameters as JSON, and the API returns optimized
wealth, consumption, and cashflow paths, including params suitable for plotting.

Endpoints:
- GET /            : Health check
- POST /optimize   : Run optimization with user-specified parameters
"""

import datetime as dt
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import logging

from .models import OptimizationInput
from .schemas import OptimizationResult

# Import your core wealth optimizer classes
from wealthplan.cashflows.groceries import EssentialExpenses
from wealthplan.cashflows import Salary
from wealthplan.cashflows.rent import Rent
from wealthplan.cashflows.pension import Pension
from wealthplan import LifeInsurance
from wealthplan import Wealth
from wealthplan.optimizer import (
    BellmanOptimizer,
)

logger = logging.getLogger("uvicorn.error")

# -----------------------
# App Initialization
# -----------------------

app: FastAPI = FastAPI(
    title="WealthPlan Optimizer API",
    version="0.1.0",
    description="API for optimizing monthly consumption, wealth, and cashflows using Bellman dynamic programming.",
)

# Enable CORS so frontend running on a different domain/port can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React/Next.js frontend default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger for exception reporting
logger = logging.getLogger("uvicorn.error")


# -----------------------
# Root endpoint (health check)
# -----------------------
@app.get("/", tags=["Health"])
def root() -> dict:
    """
    Health check endpoint to verify API is running.

    Returns:
        dict: Basic status information
    """
    return {"status": "ok", "service": "WealthPlan Optimizer API"}


# -----------------------
# Optimization endpoint
# -----------------------
@app.post("/optimize", response_model=OptimizationResult, tags=["Optimization"])
def optimize(params: OptimizationInput) -> OptimizationResult:
    """
    Run the wealth optimization using user-specified parameters.

    Args:
        params (OptimizationInput): Input parameters including salary, rent,
                                    expenses, initial wealth, expected return,
                                    retirement date, and simulation dates.

    Returns:
        OptimizationResult: Structured result including time series for wealth,
                            consumption, and cashflows, plus retirement date.
    """
    try:
        logger.info("Received optimization request")
        logger.info(f"Input parameters: {params.json()}")

        # -----------------------
        # Simulation Dates
        # -----------------------
        start_date: dt.date = params.simulation_start_date
        end_date: dt.date = params.simulation_end_date
        retirement_date: dt.date = params.retirement_date

        logger.info(
            f"Simulation start: {start_date}, end: {end_date}, retirement: {retirement_date}"
        )

        # -----------------------
        # Create Cashflow Objects
        # -----------------------
        salary = Salary(monthly_amount=params.salary, retirement_date=retirement_date)
        rent = Rent(monthly_amount=params.rent)
        insurance = LifeInsurance(
            monthly_payment=100, payout=100000, end_date=retirement_date
        )
        pension = Pension(monthly_amount=3100, retirement_date=retirement_date)
        essential_expenses = EssentialExpenses(monthly_expenses=params.expenses)

        cashflows: List = [salary, rent, insurance, pension, essential_expenses]
        logger.info(f"Cashflows created: {[type(cf).__name__ for cf in cashflows]}")

        # -----------------------
        # Create Wealth Object
        # -----------------------
        wealth = Wealth(
            initial_wealth=params.initial_wealth, yearly_return=params.investment_return
        )
        logger.info(
            f"Wealth object created with initial wealth: {wealth.initial_wealth}, yearly return: {wealth.yearly_return_savings}"
        )

        # -----------------------
        # Run Bellman Optimizer
        # -----------------------
        bell = BellmanOptimizer(
            start_date=start_date,
            end_date=end_date,
            wealth=wealth,
            cashflows=cashflows,
            beta=1.0,
        )
        logger.info("BellmanOptimizer initialized")

        bell.solve()  # Main optimization computation
        logger.info("Optimization completed successfully")

        # -----------------------
        # Prepare API Response
        # -----------------------
        result = OptimizationResult(
            success=True,
            message="Optimization completed successfully",
            summary={
                "start_date": start_date,
                "end_date": end_date,
                "retirement_date": retirement_date,
            },
            wealth_dates=bell.opt_wealth.index.tolist(),
            wealth_values=bell.opt_wealth.values.tolist(),
            consumption_dates=bell.opt_consumption.index.tolist(),
            consumption_values=bell.opt_consumption.values.tolist(),
            cashflow_dates=bell.monthly_cashflows.index.tolist(),
            cashflow_values=bell.monthly_cashflows.values.tolist(),
            retirement_date=retirement_date,
        )
        logger.info("Result object prepared, returning response")
        return result

    except Exception as e:
        logger.exception("Optimization failed")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")
