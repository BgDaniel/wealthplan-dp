import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional


class AdamVisualizer:
    """
    Visualizes the Adam optimization process for a multi-variable function.

    Attributes:
        lr (float): Base learning rate.
        beta1 (float): Exponential decay rate for the first moment (momentum).
        beta2 (float): Exponential decay rate for the second moment (squared gradient).
        epsilon (float): Small number to prevent division by zero.
        positions (list[np.ndarray]): List of positions visited during optimization.
        gradients (list[np.ndarray]): List of gradients at each step.
        m_seq (list[np.ndarray]): List of Adam first moment vectors.
        v_seq (list[np.ndarray]): List of Adam second moment vectors.
    """

    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        """
        Initialize the Adam visualizer with hyperparameters.

        Args:
            learning_rate (float): Base learning rate for Adam.
            beta1 (float): Decay rate for first moment estimate.
            beta2 (float): Decay rate for second moment estimate.
            epsilon (float): Small number to avoid division by zero.
        """
        self.lr: float = learning_rate
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon

        self.positions: list[np.ndarray] = []
        self.gradients: list[np.ndarray] = []
        self.m_seq: list[np.ndarray] = []
        self.v_seq: list[np.ndarray] = []

    def compute(self, func: Callable[[np.ndarray], float], x0: np.ndarray, n_steps: int) -> None:
        """
        Perform Adam optimization on the given function.

        Args:
            func (Callable[[np.ndarray], float]): Function to minimize.
            x0 (np.ndarray): Initial position.
            n_steps (int): Number of optimization steps.
        """
        x = x0.astype(float)
        m = np.zeros_like(x)
        v = np.zeros_like(x)

        self.positions = [x.copy()]
        self.gradients = []
        self.m_seq = [m.copy()]
        self.v_seq = [v.copy()]

        for t in range(1, n_steps + 1):
            grad = self._numerical_grad(func, x)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad**2)

            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)

            x = x - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            self.positions.append(x.copy())
            self.gradients.append(grad.copy())
            self.m_seq.append(m.copy())
            self.v_seq.append(v.copy())

    def _numerical_grad(self, func: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Compute numerical gradient of a function at a given point.

        Args:
            func (Callable[[np.ndarray], float]): Function for which to compute gradient.
            x (np.ndarray): Point at which to compute gradient.
            h (float): Small perturbation for finite differences.

        Returns:
            np.ndarray: Gradient vector at x.
        """
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_forward = x.copy()
            x_backward = x.copy()
            x_forward[i] += h
            x_backward[i] -= h
            grad[i] = (func(x_forward) - func(x_backward)) / (2 * h)
        return grad

    def plot(
        self,
        func: Optional[Callable[[np.ndarray], float]] = None,
        xrange: tuple[float, float] = (-5, 5),
        yrange: tuple[float, float] = (-5, 5),
        resolution: int = 100
    ) -> None:
        """
        Plot the optimization path on the function's contour and Adam variables.

        Args:
            func (Optional[Callable[[np.ndarray], float]]): Function for contour plot (2D only).
            xrange (tuple[float, float]): X-axis range for contour plot.
            yrange (tuple[float, float]): Y-axis range for contour plot.
            resolution (int): Grid resolution for contour plot.
        """
        positions = np.array(self.positions)
        gradients = np.array(self.gradients)
        m_seq = np.array(self.m_seq)
        v_seq = np.array(self.v_seq)

        fig, axs = plt.subplots(2, 1, figsize=(8, 12))

        # Contour plot with path
        if func is not None and positions.shape[1] == 2:
            x = np.linspace(*xrange, resolution)
            y = np.linspace(*yrange, resolution)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[func(np.array([xi, yi])) for xi, yi in zip(X_row, Y_row)]
                          for X_row, Y_row in zip(X, Y)])
            axs[0].contourf(X, Y, Z, levels=50, cmap='viridis')
            axs[0].set_title("Adam Optimization Path")
            axs[0].set_xlabel("x1")
            axs[0].set_ylabel("x2")

            axs[0].plot(positions[:, 0], positions[:, 1], "ro-", label="Position")
            for i in range(len(gradients)):
                axs[0].arrow(
                    positions[i, 0], positions[i, 1],
                    -0.2 * gradients[i, 0], -0.2 * gradients[i, 1],
                    head_width=0.1, color='white', alpha=0.7
                )
            axs[0].legend()

        # Adam inner variables
        for i in range(m_seq.shape[1]):
            axs[1].plot(m_seq[:, i], '--', label=f'm_{i}')
            axs[1].plot(v_seq[:, i], '-', label=f'v_{i}')
        axs[1].set_title("Adam Momentum (m) and Squared Gradient (v)")
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("Value")
        axs[1].legend()
        plt.tight_layout()
        plt.show()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    def f(x: np.ndarray) -> float:
        return x[0] ** 2 + 2 * x[1] ** 2 + x[0] * x[1]

    viz = AdamVisualizer(learning_rate=0.1)
    viz.compute(f, x0=np.array([3.0, 2.0]), n_steps=20)
    viz.plot(func=f, xrange=(-1, 4), yrange=(-1, 4))
