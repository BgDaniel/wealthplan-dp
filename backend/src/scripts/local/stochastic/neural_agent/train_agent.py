# train_agent.py
import os
import uuid
from typing import Dict
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
ACT_SOFTPLUS = "Softplus"


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

# ----------------------------
# Activation mapping
# ----------------------------
ACTIVATIONS = {
    ACT_RELU: nn.ReLU,
    ACT_LEAKY_RELU: nn.LeakyReLU,
    ACT_TANH: nn.Tanh,
    ACT_SOFTPLUS: nn.Softplus,
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


def plot_policy_heatmaps(
    agent: NeuralAgentWealthOptimizer,
    inspection_date: dt.date,
    n_wealth: int = 500,
    n_savings_frac: int = 500,
):
    """
    Plot the trained neural network policy as 3D surfaces for consumption and savings actions.
    Two-row layout, cleaner axes, shades of red/blue, formatted ticks.
    """

    # Find time index
    t_idx = np.where(np.array(agent.months) == inspection_date)[0]
    if len(t_idx) == 0:
        raise ValueError(f"inspection_date {inspection_date} not in agent.months")
    t_idx = t_idx[0]

    # Create grid
    wealth_scaled_grid = np.linspace(0.0, 1.0, n_wealth)
    savings_frac_grid = np.linspace(0.0, 1.0, n_savings_frac)
    W_scaled, S_frac = np.meshgrid(wealth_scaled_grid, savings_frac_grid)

    # Flatten for batch
    W_flat = W_scaled.ravel()
    S_frac_flat = S_frac.ravel()

    # Normalized time feature
    t_norm = np.full_like(W_flat, fill_value=t_idx / agent.n_months, dtype=np.float32)

    # Build state tensor
    state_tensor = torch.from_numpy(
        np.stack([S_frac_flat, W_flat, t_norm], axis=1)
    ).float().to(next(agent.policy_net.parameters()).device)

    # Actions from policy network
    agent.policy_net.eval()
    actions = agent.policy_net(state_tensor).detach().cpu().numpy()
    consumption_rate = actions[:, 0].reshape(n_savings_frac, n_wealth)
    savings_rate = actions[:, 1].reshape(n_savings_frac, n_wealth)

    # -------------------------
    # Plot in two rows
    # -------------------------
    fig = plt.figure(figsize=(14, 12))

    # ---- Row 1: Consumption ----
    ax1 = fig.add_subplot(2, 1, 1, projection="3d")
    surf1 = ax1.plot_surface(
        W_scaled,
        S_frac,
        consumption_rate,
        cmap="Reds",
        linewidth=0,
        antialiased=True,
    )
    ax1.set_xlabel("Wealth (scaled)")
    ax1.set_ylabel("Savings Fraction")
    ax1.set_zlabel("Consumption Rate")
    ax1.set_title("Policy: Consumption Rate", pad=20)
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax1.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax1.zaxis.set_major_formatter(lambda v, _: f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.2e}")
    fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=15)

    # ---- Row 2: Savings Transfer ----
    ax2 = fig.add_subplot(2, 1, 2, projection="3d")
    surf2 = ax2.plot_surface(
        W_scaled,
        S_frac,
        savings_rate,
        cmap="Blues",
        linewidth=0,
        antialiased=True,
    )
    ax2.set_xlabel("Wealth (scaled)")
    ax2.set_ylabel("Savings Fraction")
    ax2.set_zlabel("Savings Transfer Rate")
    ax2.set_title("Policy: Savings Transfer Rate", pad=20)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax2.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax2.zaxis.set_major_formatter(lambda v, _: f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.2e}")
    fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=15)

    # Super title with space above
    fig.suptitle(f"Policy surfaces at inspection date: {inspection_date}", fontsize=16, y=0.95)
    plt.tight_layout(pad=3.0)
    plt.show()

def train_agent(
    hyperparams: Dict[str, object],
    params_file_name: str,
    run_task_id: str,
    n_epochs: int = 250,
    n_episodes: int = 5000,
    plot_training: bool = False,
    plot_results: bool = False,
    save: bool = False,
    device: str = "cpu",
    print_diagnostics: bool = False
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
        n_epochs=n_epochs,
        n_episodes=n_episodes,
        plot=plot_training
    )

    # ----------------------------
    # Save outputs
    # ----------------------------
    if save:
        cache_dir = agent.cache.cache_dir

        torch.save(
            agent.policy_net.state_dict(),
            os.path.join(cache_dir, FILE_POLICY_NET),
        )

        agent.opt_savings.to_csv(os.path.join(cache_dir, FILE_SAVINGS))
        agent.opt_stocks.to_csv(os.path.join(cache_dir, FILE_STOCKS))
        agent.opt_wealth.to_csv(os.path.join(cache_dir, FILE_STOCKS))
        agent.opt_consumption.to_csv(os.path.join(cache_dir, FILE_CONSUMPTION))

    # ----------------------------
    # Plot optimization results
    # ----------------------------
    if plot_results:
        agent.plot()

        first_month_date = agent.months[0]  # or choose any date
        plot_policy_heatmaps(agent, inspection_date=first_month_date)

        #agent.policy_net.plot_weight_distributions()

    if print_diagnostics:
        agent.policy_net.print_diagnostics(n_test=n_episodes)

    # ----------------------------
    # Objective
    # ----------------------------
    return float(agent.opt_consumption.sum(axis=0).mean())


# ----------------------------
# Main section
# ----------------------------
if __name__ == "__main__":
    PARAMS_FILE = "stochastic/neural_agent/lifecycle_params_test.yaml"
    RUN_TASK_ID = uuid.uuid4().hex
    DEVICE = "cpu"

    hyperparams = {
        HP_HIDDEN_LAYERS: [64, 128, 64],
        HP_ACTIVATION: ACT_SOFTPLUS,
        HP_DROPOUT: 0.1,
        HP_LR: 0.001
    }

    obj = train_agent(
        hyperparams=hyperparams,
        params_file_name=PARAMS_FILE,
        run_task_id=RUN_TASK_ID,
        n_epochs=250,
        n_episodes=10000,
        plot_training=True,
        plot_results=True,
        save=True,
        device=DEVICE,
    )

    print(f"Training completed. Objective (mean total consumption): {obj:.2f}")
