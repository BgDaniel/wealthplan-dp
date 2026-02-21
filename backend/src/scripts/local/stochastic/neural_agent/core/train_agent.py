import os
from dataclasses import asdict
from typing import Dict, Tuple, Optional, List, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import trange

from config.stochastic.neural_agent.neural_agent_config_mapper import NeuralAgentConfigMapper
from wealthplan.optimizer.stochastic.neural_agent.neural_agent_optimizer import NeuralAgentWealthOptimizer
from wealthplan.optimizer.stochastic.neural_agent.simple_policy_network import SimplePolicyNetwork
from scripts.local.stochastic.neural_agent.cache.base_cache import TrainingAgentCache


# Activation mapping
ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "Softplus": nn.Softplus,
}


def plot_training_progress(epoch_rewards: List[float], plot_config: Dict[str, Any] = None) -> None:
    """
    Plot training progress of the agent.

    Parameters
    ----------
    epoch_rewards : List[float]
        List of mean penalized rewards per epoch.
    plot_config : Dict[str, Any], optional
        Dictionary of plotting options. Supported keys:
        - 'figsize': Tuple[int, int], figure size (default: (10, 6))
        - 'xlabel': str, x-axis label (default: 'Epoch')
        - 'ylabel': str, y-axis label (default: 'Mean Penalized Reward')
        - 'title': str, plot title (default: 'Training Progress')
        - 'grid': bool, whether to show grid (default: True)
    """
    # Set defaults
    config = {
        "figsize": (10, 6),
        "xlabel": "Epoch",
        "ylabel": "Mean Penalized Reward",
        "title": "Training Progress",
        "grid": True,
    }

    # Override defaults with user-provided config
    if plot_config:
        config.update(plot_config)

    # Plot
    plt.figure(figsize=config["figsize"])
    plt.plot(epoch_rewards)
    plt.xlabel(config["xlabel"])
    plt.ylabel(config["ylabel"])
    plt.title(config["title"])
    if config["grid"]:
        plt.grid(True)
    plt.show()


def train_agent(
    run_id: str,
    life_cycle_dict: Dict[str, object],
    hyperparams: Dict[str, object],
    device: str = "cpu",
    cache: Optional[TrainingAgentCache] = None,
) -> Tuple[float, NeuralAgentWealthOptimizer]:
    """
    Train a NeuralAgentWealthOptimizer or load it from cache.

    This function is cache-backend agnostic. It relies only on the
    abstract `TrainingAgentCache` interface, so it works with:

    - Local filesystem cache
    - S3 cache
    - Any future storage backend

    Parameters
    ----------
    life_cycle_params : Dict[str, object]
        Environment/lifecycle parameters for the optimizer.
    hyperparams : Dict[str, object]
        Hyperparameters controlling model architecture and training.
    output_dir : str
        Directory where outputs are stored if no cache is used.
    device : str, default="cpu"
        Torch device.
    cache : Optional[TrainingAgentCache]
        Cache backend. If provided, it is used for loading/saving models.
    run_id : Optional[str]
        Unique run identifier required when cache is used.

    Returns
    -------
    Tuple[float, NeuralAgentWealthOptimizer]
        Objective value and trained agent.
    """

    # ------------------------------------------------------------
    # Initialize agent
    # ------------------------------------------------------------
    life_cycle_params = NeuralAgentConfigMapper.map_yaml_to_params(
        life_cycle_dict
    )

    agent = NeuralAgentWealthOptimizer(**life_cycle_params)

    activation_cls = ACTIVATIONS[hyperparams.activation]

    agent.policy_net = SimplePolicyNetwork(
        hidden_dims=hyperparams.hidden_layers,
        activation=activation_cls,
        output_activation=nn.Sigmoid,
        dropout=hyperparams.dropout,
    ).to(device)

    agent.optimizer = torch.optim.Adam(
        agent.policy_net.parameters(), lr=hyperparams.lr
    )

    # ------------------------------------------------------------
    # Cache loading
    # ------------------------------------------------------------
    if cache is not None and cache.model_exists(run_id):
        print(f"Cached model found for run_id={run_id}. Loading and skipping training.")

        epoch_rewards = cache.load(run_id, agent, torch.device(device))
        objective = float(agent.opt_consumption.sum(axis=0).mean())

        return objective, agent, epoch_rewards

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    n_epochs = hyperparams.n_epochs
    n_episodes = hyperparams.n_episodes
    lambda_penalty = hyperparams.lambda_penalty
    terminal_penalty_val = hyperparams.terminal_penalty

    epoch_rewards: List[float] = []

    for _ in trange(n_epochs, desc="Training Epochs"):
        savings_paths, stocks_paths, consumption_paths, wealth_paths = (
            agent._simulate_forward(batch_size=n_episodes)
        )

        rewards = agent.crra_utility_torch(consumption_paths)

        penalty = F.softplus(-wealth_paths)
        terminal_wealth = wealth_paths[:, -1]
        terminal_penalty_term = F.softplus(terminal_wealth)

        total_reward = (
            rewards.sum(dim=0).mean()
            - lambda_penalty * penalty.sum(dim=0).mean()
            - terminal_penalty_val * terminal_penalty_term.mean()
        )

        agent.optimizer.zero_grad()
        loss = -total_reward
        loss.backward()
        agent.optimizer.step()

        epoch_rewards.append(total_reward.item())

        # Convert to DataFrames immediately
        agent.opt_savings = pd.DataFrame(
            savings_paths.detach().cpu().numpy(), index=agent.months
        )

        agent.opt_stocks = pd.DataFrame(
            stocks_paths.detach().cpu().numpy(), index=agent.months
        )

        agent.opt_wealth = pd.DataFrame(
            wealth_paths.detach().cpu().numpy(), index=agent.months
        )

        # consumption is one month shorter
        agent.opt_consumption = pd.DataFrame(
            consumption_paths.detach().cpu().numpy(), index=agent.months
        )

    # ------------------------------------------------------------
    # Save to cache OR fallback to local_training dir
    # ------------------------------------------------------------
    if cache is not None and run_id is not None:
        cache.save(run_id, agent, epoch_rewards, life_cycle_dict, asdict(hyperparams))

    # ------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------
    objective = float(agent.opt_consumption.sum(axis=0).mean())

    return objective, agent, epoch_rewards