from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import datetime as dt

class OptimizationResult(BaseModel):
    """
    Output structure for optimization results, including summary statistics
    and time series data for frontend visualization.
    """

    success: bool = Field(..., description="Whether the optimization completed successfully")
    message: str = Field(..., description="Human-readable status message")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Key numerical results summary")
    raw: Optional[Any] = Field(None, description="Raw solver output (optional)")

    # --- Plot data ---
    wealth_dates: Optional[List[dt.date]] = Field(
        None, description="List of dates for optimized wealth timeline"
    )
    wealth_values: Optional[List[float]] = Field(
        None, description="Corresponding wealth values over time"
    )

    consumption_dates: Optional[List[dt.date]] = Field(
        None, description="List of dates for optimized consumption timeline"
    )
    consumption_values: Optional[List[float]] = Field(
        None, description="Corresponding consumption values over time"
    )

    cashflow_dates: Optional[List[dt.date]] = Field(
        None, description="List of dates for monthly cashflows"
    )
    cashflow_values: Optional[List[float]] = Field(
        None, description="Corresponding cashflow values over time"
    )
