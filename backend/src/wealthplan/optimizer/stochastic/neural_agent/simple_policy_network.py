import torch
import torch.nn as nn
from typing import Callable, List


class SimplePolicyNetwork(nn.Module):
    """
    Flexible feedforward neural network for continuous policy.

    Fixed:
    - Input dimension: 3 (savings_fraction, wealth_scaled, normalized time)
    - Output dimension: 3 (consumption fraction, savings→stocks fraction, stocks→savings fraction)

    Flexible:
    - Hidden layer sizes
    - Hidden activation function
    - Output activation function
    - Dropout
    """

    INPUT_DIM = 3  # [savings_frac, stocks_frac, log_total_wealth, t_norm]
    OUTPUT_DIM = 2  # fixed

    def __init__(
        self,
        hidden_dims: List[int] = [64, 64],
        activation: Callable[[], nn.Module] = nn.ReLU,
        output_activation: Callable[[], nn.Module] = nn.Sigmoid,
        dropout: float = 0.0
    ):
        """
        Initialize the neural network.

        Parameters
        ----------
        hidden_dims : List[int]
            List of hidden layer sizes
        activation : Callable[[], nn.Module]
            Activation function for hidden layers (default ReLU)
        output_activation : Callable[[], nn.Module]
            Activation function for output layer (default Sigmoid)
        dropout : float
            Dropout probability for hidden layers (default 0.0)
        """
        super().__init__()

        layers = []
        in_dim = self.INPUT_DIM

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, self.OUTPUT_DIM))

        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input state tensor

        Returns
        -------
        torch.Tensor
            Action tensor with values in [0,1]
        """
        return self.net(x)
