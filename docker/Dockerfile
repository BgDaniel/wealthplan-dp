# ---------------------------------------------------------
# Dockerfile for WealthPlan Deterministic Bellman Optimizer
# ---------------------------------------------------------
# This Dockerfile builds a container for running the WealthPlan
# deterministic Bellman optimizer script.
#
# It installs Python dependencies via Poetry, copies source code,
# and runs the optimizer script which fetches parameters from S3.
#
# Usage:
#   Build: docker build -t wealthplan-optimizer .
#   Run:   docker run -e AWS_ACCESS_KEY_ID=... \
#                -e AWS_SECRET_ACCESS_KEY=... \
#                -e AWS_DEFAULT_REGION=eu-west-1 \
#                wealthplan-optimizer
# ---------------------------------------------------------

# 1️⃣ Base image: Python 3.14 slim variant
FROM python:3.14-slim

# Debug message
RUN echo "Base image python:3.14-slim selected"

# 2️⃣ Set working directory inside the container
WORKDIR /app
RUN echo "Working directory set to /app"

# 3️⃣ Install Poetry
RUN pip install --no-cache-dir poetry && echo "Poetry installed"

# 4️⃣ Copy Poetry files first (for caching dependencies if unchanged)
COPY ../pyproject.toml poetry.lock /app/
RUN echo "Poetry files copied"

# 5️⃣ Configure Poetry to install packages system-wide
RUN poetry config virtualenvs.create false && echo "Poetry configured to avoid virtualenvs"

# 6️⃣ Install dependencies
RUN poetry install --only main --no-root --no-interaction --no-ansi && echo "Dependencies installed"

# 7️⃣ Copy source code into the container
COPY ../backend/src /app/src/
RUN echo "Source code copied"

# 8️⃣ Set PYTHONPATH so imports from src/ work
ENV PYTHONPATH=/app/src

# 9️⃣ Set the default command to run the optimizer script
CMD ["python3", "src/scripts/run_optimizer.py"]
