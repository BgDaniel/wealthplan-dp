# ---------------------------------------------------------
# Dockerfile for WealthPlan Optimizer API
# ---------------------------------------------------------
# This Dockerfile builds a container for the FastAPI-based
# WealthPlan optimization service.
#
# It installs Python dependencies, copies source code, and
# runs the FastAPI app with Uvicorn.
#
# Usage:
#   Build: docker build -t wealthplan-api .
#   Run:   docker run -p 8000:8000 wealthplan-api
# ---------------------------------------------------------

# 1️⃣ Base image: Python 3.14 slim variant
FROM python:3.14-slim

# Debug message
RUN echo Base image python:3.14-slim selected

# 2️⃣ Set working directory inside the container
WORKDIR /app
RUN echo Working directory set to /app

# 3️⃣ Install Poetry
RUN pip install --no-cache-dir poetry && echo Poetry installed

# 4️⃣ Copy Poetry files first (to cache dependencies if they don't change)
COPY pyproject.toml poetry.lock /app/
RUN echo Poetry files copied

# 5️⃣ Configure Poetry to install packages system-wide
RUN poetry config virtualenvs.create false && echo Poetry configured to avoid virtualenvs

# 6️⃣ Install dependencies
RUN poetry install --only main --no-root --no-interaction --no-ansi && echo Dependencies installed

# 7️⃣ Copy source code into the container
COPY src/ /app/src/
COPY api/ /app/api/
RUN echo Source code copied

# 8️⃣ Expose port 8000 for the FastAPI app
EXPOSE 8000
RUN echo Port 8000 exposed

# 9️⃣ Start the FastAPI app using Uvicorn
#    CMD will be executed when the container runs
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
