# -----------------------------
# Dockerfile.dev
# DEV environment for WealthPlan deterministic optimizer
# -----------------------------

# 0️⃣ Base image: Python 3.14 slim variant
FROM python:3.14-slim
RUN echo "Base image python:3.14-slim selected"

# 1️⃣ Set environment variables for DEV S3 buckets and prefixes
ENV S3_BUCKET_ENV="wealthplan-dp-dev" \
    S3_PARAMS_PREFIX_ENV="params" \
    S3_OUTPUT_PREFIX_ENV="output" \
    TMP_FOLDER_ENV="/app/tmp"

# 2️⃣ Set working directory inside the container
WORKDIR /app
RUN echo "Working directory set to /app"

# 3️⃣ Install Poetry
RUN pip install --no-cache-dir poetry && echo "Poetry installed"

# 4️⃣ Copy Poetry files first (for caching dependencies if unchanged)
COPY ../../pyproject.toml ../../poetry.lock /app/
RUN echo "Poetry files copied"

# 5️⃣ Configure Poetry to install packages system-wide
RUN poetry config virtualenvs.create false && echo "Poetry configured to avoid virtualenvs"

# 6️⃣ Install dependencies
RUN poetry install --only main --no-root --no-interaction --no-ansi && echo "Dependencies installed"

# 7️⃣ Copy source code into the container
COPY ../../backend/src /app/src/
RUN echo "Source code copied"

# 8️⃣ Set PYTHONPATH so imports from src/ work
ENV PYTHONPATH=/app/src

# 🔟 Default command to run the optimizer script
# Allow passing params_file_name as an argument dynamically when running container
ENTRYPOINT ["python3", "src/scripts/run_optimizer.py"]
CMD ["--params-file", "lifecycle_params.yaml"]
