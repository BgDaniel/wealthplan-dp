# config/hyperparameters.py
from dataclasses import dataclass
from typing import List

@dataclass
class HyperParameters:
    """
    Hyperparameters for NeuralAgent training.

    Attributes:
        hidden_layers: List of integers specifying hidden layer sizes.
        activation: Activation function name for the model (e.g., 'Softplus').
        dropout: Dropout rate for the policy network.
        lr: Learning rate for optimizer.
        batch_size: Batch size for forward simulation.
        n_epochs: Number of training epochs.
        n_episodes: Number of episodes per epoch.
        lambda_penalty: Weight for negative wealth and terminal positive wealth penalty.
    """
    # Model parameters
    hidden_layers: List[int]
    activation: str
    dropout: float

    # Learning parameters
    lr: float
    n_epochs: int
    n_episodes: int

    # Penalty parameter
    lambda_penalty: float
    terminal_penalty: float