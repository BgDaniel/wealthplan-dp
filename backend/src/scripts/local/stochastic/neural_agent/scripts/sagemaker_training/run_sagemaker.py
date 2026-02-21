import uuid
from dataclasses import asdict
from typing import Dict, Any
from sagemaker.pytorch import PyTorch


from io_handler.local_io_handler import LocalIOHandler
from scripts.local.stochastic.neural_agent.core.hyper_params import HyperParameters
from scripts.local.stochastic.neural_agent.scripts.sagemaker_training.s3_file_handler import S3FileHandler


ROLE: str = "arn:aws:iam::<account-id>:role/SageMakerExecutionRole"
BUCKET: str = "wealthplan-neural-agent-dev"
ENTRY_POINT: str = "sagemaker_entrypoint.py"
SOURCE_DIR: str = "."
INSTANCE_TYPE: str = "ml.g4dn.xlarge"
INSTANCE_COUNT: int = 1
FRAMEWORK_VERSION: str = "2.1.0"
PY_VERSION: str = "py39"
WAIT: bool = True  # Whether to block until training job completes


def main() -> None:
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
    run_id: str = uuid.uuid5(uuid.NAMESPACE_DNS, "sagemaker_test_run").hex

    lifecycle_file: str = f"input/{run_id}/lifecycle_params.json"
    hyperparams_file: str = f"input/{run_id}/hyperparams.json"

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

    # -------------------------
    # SageMaker Estimator
    # -------------------------
    estimator: PyTorch = PyTorch(
        entry_point=ENTRY_POINT,
        source_dir=SOURCE_DIR,
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        hyperparameters={
            "RUN_ID": run_id,
            "S3_BUCKET": BUCKET,
            "PARAMS_FILE": lifecycle_file,
            "HYPERPARAMS_FILE": hyperparams_file,
        },
    )

    # Submit the training job
    estimator.fit(wait=WAIT)
    print(f"[INFO] SageMaker training job submitted with run ID: {run_id}")


if __name__ == "__main__":
    main()