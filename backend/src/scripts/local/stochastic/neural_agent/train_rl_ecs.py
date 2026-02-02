import uuid
import io
import logging
import torch
from typing import Optional

from config.stochastic.neural_agent.neural_agent_config_mapper import NeuralAgentConfigMapper
from io_handler.s3_io_handler import S3IOHandler
from wealthplan.optimizer.stochastic.neural_agent.neural_agent_optimizer import NeuralAgentWealthOptimizer


def save_outputs_to_s3(
    s3_handler: S3IOHandler,
    optimizer: NeuralAgentWealthOptimizer,
    s3_output_prefix: str,
    run_task_id: str
) -> None:
    """
    Save trained model and output CSVs to S3.

    Parameters
    ----------
    s3_handler : S3IOHandler
        S3 handler object to interact with the bucket.
    optimizer : NeuralAgentWealthOptimizer
        Trained neural agent optimizer with policy_net and results.
    s3_output_prefix : str
        Prefix in S3 bucket to save files.
    run_task_id : str
        Unique identifier for this run (used in filenames).
    """
    logging.info("Saving trained model and results to S3...")

    model_key = f"{s3_output_prefix}/trained_policy_net_{run_task_id}.pth"
    savings_key = f"{s3_output_prefix}/opt_savings_{run_task_id}.csv"
    stocks_key = f"{s3_output_prefix}/opt_stocks_{run_task_id}.csv"
    consumption_key = f"{s3_output_prefix}/opt_consumption_{run_task_id}.csv"

    # Save PyTorch model
    buffer = io.BytesIO()
    torch.save(optimizer.policy_net.state_dict(), buffer)
    buffer.seek(0)
    s3_handler.upload_fileobj(buffer, key=model_key)

    # Save CSVs
    for df, key in zip(
        [optimizer.opt_savings, optimizer.opt_stocks, optimizer.opt_consumption],
        [savings_key, stocks_key, consumption_key]
    ):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        csv_buffer.seek(0)
        s3_handler.upload_fileobj(csv_buffer, key=key)

    logging.info(f"All outputs saved to S3 under prefix '{s3_output_prefix}'.")


def main(
    s3_bucket: str,
    params_file_name: str,
    s3_output_prefix: str,
    run_task_id: str = "",
    plot: bool = True,
    n_epochs: int = 500,
    batch_size: int = 1000,
    n_batches: int = 10,
) -> None:
    """
    Train a stochastic neural-agent optimizer and save outputs to S3.

    Parameters
    ----------
    s3_bucket : str
        Name of the S3 bucket containing the YAML config.
    params_file_name : str
        Key/path of the YAML config file in the S3 bucket.
    s3_output_prefix : str
        Prefix in S3 bucket where trained model and CSV outputs will be saved.
    run_task_id : str
        Unique identifier for this run (used in filenames).
    plot : bool
        If True, plot training progress after each epoch.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Number of simulation paths per batch.
    n_batches : int
        Number of batches per epoch.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info(f"Starting run {run_task_id}")

    # Load config from S3
    s3_handler = S3IOHandler(bucket_name=s3_bucket, params_file_name=params_file_name)
    yaml_dict = s3_handler.load_params()
    params = NeuralAgentConfigMapper.map_yaml_to_params(yaml_dict)

    # Initialize optimizer
    optimizer = NeuralAgentWealthOptimizer(**params)

    # Train agent
    logging.info("Starting training...")
    optimizer.train(
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_batches=n_batches,
        plot=plot
    )
    logging.info("Training completed.")

    # Save outputs to S3
    save_outputs_to_s3(s3_handler, optimizer, s3_output_prefix, run_task_id)


if __name__ == "__main__":
    # Example usage
    s3_bucket_name = "my-config-bucket"
    yaml_key = "stochastic/neural_agent/lifecycle_params.yaml"
    s3_output_prefix = "training_outputs/neural_agent"

    run_task_id = uuid.uuid4().hex

    main(
        s3_bucket=s3_bucket_name,
        params_file_name=yaml_key,
        s3_output_prefix=s3_output_prefix,
        run_task_id=run_task_id,
        plot=True,
        n_epochs=500,
        batch_size=1000,
        n_batches=10
    )
