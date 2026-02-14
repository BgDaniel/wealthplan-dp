import torch
import torch.nn as nn
from typing import Callable, List

from matplotlib import pyplot as plt


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

    def print_diagnostics(self, n_test: int = 2000) -> None:
        """
        Print diagnostic statistics of the trained network.

        This method probes the network using random inputs to detect
        potential causes of spiky or unstable policy functions.

        Diagnostics include:
        - weight statistics (mean/std/max per parameter tensor)
        - activation statistics across layers
        - local smoothness test via small input perturbations

        Parameters
        ----------
        n_test : int, optional
            Number of random input samples used for probing the network.
            Larger values give more stable statistics but increase runtime.
            Default is 2000.

        Returns
        -------
        None
        """
        device = next(self.parameters()).device

        print("\n========== POLICY NETWORK DIAGNOSTICS ==========\n")

        # -------------------------------
        # Weight stats
        # -------------------------------
        print("WEIGHT STATS\n")
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(
                    f"{name:30s} "
                    f"mean={p.data.mean():+.4f} "
                    f"std={p.data.std():.4f} "
                    f"max={p.data.abs().max():.4f}"
                )

        # -------------------------------
        # Activation test
        # -------------------------------
        print("\nACTIVATION TEST\n")

        x = torch.rand(n_test, self.INPUT_DIM, device=device)
        activations = []

        def hook_fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                activations.append(out.detach())

        hooks = []
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.ReLU, nn.Softplus, nn.Tanh)):
                hooks.append(m.register_forward_hook(hook_fn))

        with torch.no_grad():
            _ = self(x)

        for h in hooks:
            h.remove()

        for i, a in enumerate(activations):
            print(
                f"Layer {i:02d}: "
                f"mean={a.mean():+.3f} "
                f"std={a.std():.3f} "
                f"max={a.abs().max():.3f}"
            )

        # -------------------------------
        # Smoothness test
        # -------------------------------
        print("\nSMOOTHNESS TEST\n")

        x2 = x.clone()
        x2[:, 1] += 1e-3  # tiny perturbation in wealth dimension

        with torch.no_grad():
            y1 = self(x)
            y2 = self(x2)

        diff = (y2 - y1).abs()

        print(
            f"Mean output change: {diff.mean():.6f}\n"
            f"Max  output change: {diff.max():.6f}"
        )

        print("\n===============================================\n")

    def plot_weight_distributions(self) -> None:
        """
        Plot histograms of the weight values for each linear layer.

        This helps detect:
        - exploding weights
        - heavy-tailed distributions
        - badly scaled layers

        Returns
        -------
        None
        """
        weights = [
            p.detach().cpu().flatten().numpy()
            for name, p in self.named_parameters()
            if "weight" in name
        ]

        n = len(weights)
        plt.figure(figsize=(5 * n, 4))

        for i, w in enumerate(weights):
            plt.subplot(1, n, i + 1)
            plt.hist(w, bins=50)
            plt.title(f"Layer {i} weight distribution")
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()