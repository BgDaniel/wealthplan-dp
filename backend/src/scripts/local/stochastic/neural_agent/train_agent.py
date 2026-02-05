# train_agent.py
import os
import uuid
from typing import Dict

import torch
import torch.nn as nn

from config.stochastic.neural_agent.neural_agent_config_mapper import (
    NeuralAgentConfigMapper,
)
from io_handler.local_io_handler import LocalIOHandler
from wealthplan.optimizer.stochastic.neural_agent.neural_agent_optimizer import (
    NeuralAgentWealthOptimizer,
)
from wealthplan.optimizer.stochastic.neural_agent.simple_policy_network import (
    SimplePolicyNetwork,
)

# ----------------------------
# String constants
# ----------------------------
ACT_RELU = "ReLU"
ACT_LEAKY_RELU = "LeakyReLU"
ACT_TANH = "Tanh"

ENV_OUTPUT_FOLDER = "OUTPUT_FOLDER"
SUBDIR_NEURAL_AGENT = "neural_agent"

FILE_POLICY_NET = "trained_policy_net.pth"
FILE_SAVINGS = "opt_savings.csv"
FILE_STOCKS = "opt_stocks.csv"
FILE_CONSUMPTION = "opt_consumption.csv"

# ----------------------------
# Hyperparameter keys
# ----------------------------
HP_HIDDEN_LAYERS = "hidden_layers"
HP_ACTIVATION = "activation"
HP_DROPOUT = "dropout"
HP_LR = "lr"
HP_BATCH_SIZE = "batch_size"

# ----------------------------
# Activation mapping
# ----------------------------
ACTIVATIONS = {
    ACT_RELU: nn.ReLU,
    ACT_LEAKY_RELU: nn.LeakyReLU,
    ACT_TANH: nn.Tanh,
}


def _prepare_output_dir(run_task_id: str) -> str:
    """
    Prepare output directory:
        $OUTPUT_FOLDER/neural_agent/<run_task_id>/

    Returns:
        Absolute path to the run directory.
    """
    base_output = os.getenv(ENV_OUTPUT_FOLDER)
    if base_output is None:
        raise EnvironmentError(
            f"Environment variable '{ENV_OUTPUT_FOLDER}' is not set."
        )

    run_dir = os.path.join(base_output, SUBDIR_NEURAL_AGENT, run_task_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def train_agent(
    hyperparams: Dict[str, object],
    params_file_name: str,
    run_task_id: str,
    plot_training: bool = False,
    plot_results: bool = False,
    save: bool = False,
    device: str = "cpu",
) -> float:
    """
    Train NeuralAgentWealthOptimizer with given hyperparameters.

    Args:
        hyperparams: Dictionary with hyperparameters:
            - hidden_layers: List[int]
            - activation: str
            - dropout: float
            - lr: float
            - batch_size: int
        params_file_name: YAML configuration file
        run_task_id: Unique identifier for this run
        plot_training: Plot training diagnostics (loss curves, etc.)
        plot_results: Plot optimized policies / paths
        save: If True, save model and outputs
        device: Torch device ("cpu" or "cuda")

    Returns:
        Objective value (mean total consumption)
    """
    # ----------------------------
    # Prepare output directory
    # ----------------------------
    run_dir = _prepare_output_dir(run_task_id)

    # ----------------------------
    # Load configuration
    # ----------------------------
    io_handler = LocalIOHandler(params_file_name=params_file_name)
    yaml_dict = io_handler.load_params()
    params = NeuralAgentConfigMapper.map_yaml_to_params(yaml_dict)

    # ----------------------------
    # Initialize optimizer
    # ----------------------------
    agent = NeuralAgentWealthOptimizer(**params)

    # ----------------------------
    # Override policy network
    # ----------------------------
    activation_cls = ACTIVATIONS[hyperparams[HP_ACTIVATION]]

    agent.policy_net = SimplePolicyNetwork(
        hidden_dims=hyperparams[HP_HIDDEN_LAYERS],
        activation=activation_cls,
        output_activation=nn.Sigmoid,
        dropout=hyperparams[HP_DROPOUT],
    ).to(device)

    agent.optimizer = torch.optim.Adam(
        agent.policy_net.parameters(), lr=hyperparams[HP_LR]
    )

    # ----------------------------
    # Train agent
    # ----------------------------
    agent.train(
        n_epochs=50,  # shortened for HPO / testing
        batch_size=hyperparams[HP_BATCH_SIZE],
        n_batches=10,
        plot=plot_training,
    )

    # ----------------------------
    # Save outputs
    # ----------------------------
    if save:
        torch.save(
            agent.policy_net.state_dict(),
            os.path.join(run_dir, FILE_POLICY_NET),
        )
        agent.opt_savings.to_csv(os.path.join(run_dir, FILE_SAVINGS))
        agent.opt_stocks.to_csv(os.path.join(run_dir, FILE_STOCKS))
        agent.opt_consumption.to_csv(os.path.join(run_dir, FILE_CONSUMPTION))

    # ----------------------------
    # Plot optimization results
    # ----------------------------
    if plot_results:
        agent.plot()

    # ----------------------------
    # Objective
    # ----------------------------
    return float(agent.opt_consumption.sum(axis=0).mean())


# ----------------------------
# Main section
# ----------------------------
if __name__ == "__main__":
    PARAMS_FILE = "stochastic/neural_agent/lifecycle_params.yaml"
    RUN_TASK_ID = uuid.uuid4().hex
    DEVICE = "cpu"

    hyperparams = {
        HP_HIDDEN_LAYERS: [64, 64],
        HP_ACTIVATION: ACT_RELU,
        HP_DROPOUT: 0.1,
        HP_LR: 0.001,
        HP_BATCH_SIZE: 1000,
    }

    obj = train_agent(
        hyperparams=hyperparams,
        params_file_name=PARAMS_FILE,
        run_task_id=RUN_TASK_ID,
        plot_training=True,
        plot_results=True,
        save=False,
        device=DEVICE,
    )

    print(f"Training completed. Objective (mean total consumption): {obj:.2f}")
