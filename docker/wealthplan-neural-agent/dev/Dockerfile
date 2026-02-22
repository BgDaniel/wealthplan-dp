# -----------------------------
# Dockerfile.dev
# DEV environment for training the neuronal net based agent for optimizing wealth
# -----------------------------

# 0️⃣ Valid CUDA base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 1️⃣ Set environment variables for S3 and PYTHONPATH
ENV S3_BUCKET="" \
    RUN_ID="" \
    PYTHONPATH=/app/src

# 2️⃣ Set working directory
WORKDIR /app
RUN echo "Working directory set to /app"

# 3️⃣ Install Python 3.11 and essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-distutils \
        python3-pip \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN echo "Python 3.11 and pip installed"

# 4️⃣ Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# 5️⃣ Install Poetry
RUN python3.11 -m pip install --no-cache-dir poetry

# 6️⃣ Copy Poetry project files for dependency caching
COPY ../../../pyproject.toml ../../poetry.lock /app/
RUN echo "Poetry project files copied"

# 7️⃣ Configure Poetry to install packages system-wide
RUN poetry config virtualenvs.create false && echo "Poetry configured to avoid virtualenvs"

# 8️⃣ Install Python dependencies
RUN poetry install --only main --no-root --no-interaction --no-ansi && echo "Dependencies installed"

# 9️⃣ Copy source code
COPY ../../../backend/src /app/src/
RUN echo "Source code copied"

# 🔟 Entrypoint for the training script
ENTRYPOINT ["python3.14", "src/scripts/local/stochastic/neural_agent/scripts/aws_ecs/run_training.py"]
CMD []