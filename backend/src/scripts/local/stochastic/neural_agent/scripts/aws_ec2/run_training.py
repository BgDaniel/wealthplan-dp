import os
import uuid
from typing import Dict, Any, List, Tuple

import torch

from scripts.local.stochastic.neural_agent.cache.s3_cache import S3Cache
from scripts.local.stochastic.neural_agent.core.hyper_params import HyperParameters
from scripts.local.stochastic.neural_agent.core.train_agent import train_agent
from scripts.local.stochastic.neural_agent.scripts.aws_ec2.s3_file_handler import S3FileHandler

# ==============================
# Environment variable constants
# ==============================

ENV_RUN_ID: str = "RUN_ID"
ENV_BUCKET: str = "S3_BUCKET"


def build_lifecycle_file_path(run_id: str) -> str:
    """
    Construct the default lifecycle file path on S3 for a given run.

    Parameters
    ----------
    run_id : str
        Unique identifier for the training run.

    Returns
    -------
    str
        Full S3 key for lifecycle parameters JSON.
    """
    return f"input/{run_id}/lifecycle_params.json"


def build_hyperparams_file_path(run_id: str) -> str:
    """
    Construct the default hyperparameters file path on S3 for a given run.

    Parameters
    ----------
    run_id : str
        Unique identifier for the training run.

    Returns
    -------
    str
        Full S3 key for hyperparameters JSON.
    """
    return f"input/{run_id}/hyperparams.json"


def run_training(
    run_id: str,
    bucket: str,
    life_cycle_file: str,
    hyperparams_file: str,
) -> Tuple[float, List[float]]:
    """
    Execute NeuralAgent training inside a containerized environment.

    Parameters
    ----------
    run_id : str
        Unique identifier for this training run.
    bucket : str
        S3 bucket used for cache, input parameters, and outputs.
    life_cycle_file : str
        S3 key of lifecycle/environment configuration file.
    hyperparams_file : str
        S3 key of hyperparameter configuration file.

    Returns
    -------
    Tuple[float, List[float]]
        Mean consumption objective and list of epoch rewards.
    """

    # Initialize S3 utilities
    s3_handler: S3FileHandler = S3FileHandler(bucket=bucket)
    cache: S3Cache = S3Cache(bucket=bucket)

    # Load configuration from S3
    life_cycle_params: Dict[str, Any] = s3_handler.download_dict(life_cycle_file)
    hyperparams_dict: Dict[str, Any] = s3_handler.download_dict(hyperparams_file)

    # Build hyperparameter object
    hyperparams: HyperParameters = HyperParameters(**hyperparams_dict)

    # Run training
    objective: float
    agent: Any
    rewards: List[float]

    objective, agent, rewards = train_agent(
        run_id=run_id,
        life_cycle_params=life_cycle_params,
        hyperparams=hyperparams,
        device="cuda",
        cache=cache,
    )

    print(f"[INFO] Training finished. Mean consumption: {objective:.2f}")
    return objective, rewards


if __name__ == "__main__":
    """
    Container entrypoint.

    Reads configuration from environment variables and launches training.
    """

    run_id: str | None = os.getenv(ENV_RUN_ID)

    if run_id is None:
        raise ValueError(f"Missing required environment variable: {ENV_RUN_ID}")


    bucket: str | None = os.getenv(ENV_BUCKET)

    if bucket is None:
        raise ValueError(f"Missing required environment variable: {ENV_BUCKET}")

    lifecycle_file: str = build_lifecycle_file_path(run_id)
    hyperparams_file: str = build_hyperparams_file_path(run_id)

    run_training(
        run_id=run_id,
        bucket=bucket,
        life_cycle_file=lifecycle_file,
        hyperparams_file=hyperparams_file,
    )