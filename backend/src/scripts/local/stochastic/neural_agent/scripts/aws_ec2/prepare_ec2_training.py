import uuid
from dataclasses import asdict
from typing import Dict, Any


from io_handler.local_io_handler import LocalIOHandler
from scripts.local.stochastic.neural_agent.core.hyper_params import HyperParameters
from scripts.local.stochastic.neural_agent.scripts.aws_ec2.run_training import build_lifecycle_file_path, \
    build_hyperparams_file_path
from scripts.local.stochastic.neural_agent.scripts.aws_ec2.s3_file_handler import S3FileHandler


ROLE: str = "arn:aws:iam::<account-id>:role/SageMakerExecutionRole"
BUCKET: str = "wealthplan-neural-agent-dev"
ENTRY_POINT: str = "run_training.py"
SOURCE_DIR: str = "."
INSTANCE_TYPE: str = "ml.g4dn.xlarge"
INSTANCE_COUNT: int = 1
FRAMEWORK_VERSION: str = "2.1.0"
PY_VERSION: str = "py39"
WAIT: bool = True  # Whether to block until training job completes


def prepare_ec2_training() -> None:
    """
    Launch a SageMaker training job for the NeuralAgent.

    - Uploads lifecycle and hyperparameter JSON files to S3
    - Passes file names to the SageMaker container
    - Sets up a PyTorch SageMaker estimator
    - Submits the training job
    """
    # -------------------------
    # Run configuration
    # -------------------------
    run_id: str = uuid.uuid5(uuid.NAMESPACE_DNS, "sagemaker_test_run").hex[:8]

    print(f"[INFO] Using run ID: {run_id}")

    lifecycle_file: str = build_lifecycle_file_path(run_id)
    hyperparams_file: str = build_hyperparams_file_path(run_id)

    # -------------------------
    # Load local_training lifecycle params
    # -------------------------
    life_cycle_params: Dict[str, Any] = LocalIOHandler(
        params_file_name="stochastic/neural_agent/lifecycle_params_test.yaml"
    ).load_params()

    # -------------------------
    # Upload lifecycle params to S3
    # -------------------------
    s3_handler = S3FileHandler(bucket=BUCKET)
    s3_handler.upload_dict(life_cycle_params, lifecycle_file)

    # -------------------------
    # Define hyperparameters and upload to S3
    # -------------------------
    hyperparams = HyperParameters(
        hidden_layers=[64, 128, 64],
        activation="Softplus",
        dropout=0.1,
        lr=0.001,
        n_epochs=50,
        n_episodes=10000,
        lambda_penalty=1.0,
        terminal_penalty=0.05,
    )
    s3_handler.upload_dict(asdict(hyperparams), hyperparams_file)


if __name__ == "__main__":
    prepare_ec2_training()