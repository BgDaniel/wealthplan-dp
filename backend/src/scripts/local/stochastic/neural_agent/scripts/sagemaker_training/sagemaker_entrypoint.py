import argparse
import uuid
from typing import Dict, Any, List, Tuple

import torch

from scripts.local.stochastic.neural_agent.cache.s3_cache import S3Cache
from scripts.local.stochastic.neural_agent.core.hyper_params import HyperParameters
from scripts.local.stochastic.neural_agent.core.train_agent import train_agent
from scripts.local.stochastic.neural_agent.scripts.sagemaker_training.s3_file_handler import S3FileHandler


def run_training(
    run_id: str,
    bucket: str,
    life_cycle_file: str,
    hyperparams_file: str,
) -> Tuple[float, List[float]]:
    """
    Execute NeuralAgent training inside a SageMaker container.

    Downloads JSON files from S3 based on given file names.

    Parameters
    ----------
    run_id : str
        Unique identifier for this training run.
    bucket : str
        S3 bucket to store cache and outputs.
    life_cycle_file : str
        S3 key / file name of lifecycle/environment parameters JSON.
    hyperparams_file : str
        S3 key / file name of hyperparameters JSON.

    Returns
    -------
    Tuple[float, List[float]]
        Mean consumption objective and list of epoch rewards.
    """
    # Initialize S3 file handler and cache
    s3_handler = S3FileHandler(bucket=bucket)
    cache = S3Cache(bucket=bucket)

    # Download JSONs from S3
    life_cycle_params: Dict[str, Any] = s3_handler.download_dict(life_cycle_file)
    hyperparams_dict: Dict[str, Any] = s3_handler.download_dict(hyperparams_file)
    hyperparams: HyperParameters = HyperParameters(**hyperparams_dict)

    # Choose device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Run training
    objective, agent, rewards = train_agent(
        run_id=run_id,
        life_cycle_params=life_cycle_params,
        hyperparams=hyperparams,
        device=device,
        cache=cache,
    )

    print(f"[INFO] Training finished. Mean consumption: {objective:.2f}")
    return objective, rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NeuralAgent training on SageMaker"
    )
    parser.add_argument("--run_id", type=str, default=str(uuid.uuid4().hex))
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket for cache")
    parser.add_argument(
        "--life_cycle_file",
        type=str,
        required=True,
        help="S3 JSON file for lifecycle parameters",
    )
    parser.add_argument(
        "--hyperparams_file",
        type=str,
        required=True,
        help="S3 JSON file for hyperparameters",
    )

    args = parser.parse_args()

    run_training(
        run_id=args.run_id,
        bucket=args.bucket,
        life_cycle_file=args.life_cycle_file,
        hyperparams_file=args.hyperparams_file,
    )
