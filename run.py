# run.py
"""
Launcher for WealthPlan Optimizer API.

This script starts the FastAPI application using Uvicorn.
It is intended for development/debugging in PyCharm.
"""

import uvicorn

if __name__ == "__main__":
    # Module path to FastAPI app: <folder>.<file>:<FastAPI instance>
    # api.app:app means "app" inside api/app.py
    uvicorn.run(
        "api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True  # Automatically reloads on code changes
    )
