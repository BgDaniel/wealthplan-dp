from typing import Callable
import numpy as np
from numba import njit

# Type alias
UtilityFunction = Callable[[np.ndarray], np.ndarray]


# -------------------------------------------------------------------
# Standard utility functions
# -------------------------------------------------------------------
def crra_utility(c: np.ndarray, gamma: float = 0.5, epsilon: float = 1e-8) -> np.ndarray:
    """
    Constant Relative Risk Aversion (CRRA) utility function.

    Args:
        c: Array of consumption values.
        gamma: Risk aversion parameter.
        epsilon: Small positive number to avoid zero consumption issues.

    Returns:
        Array of utility values corresponding to consumption c.
    """
    c_safe: np.ndarray = np.maximum(c, 0.0) + epsilon

    if gamma == 1.0:
        return np.log(c_safe)
    else:
        return (c_safe**(1 - gamma)) / (1 - gamma)


@njit()
def crra_utility_numba(c: np.ndarray, gamma: float = 0.5, epsilon: float = 1e-8) -> np.ndarray:
    """
    Numba-accelerated CRRA utility function for numerical efficiency.

    Args:
        c: Array of consumption values.
        gamma: Risk aversion parameter.
        epsilon: Small positive number to avoid zero consumption issues.

    Returns:
        Array of utility values.
    """
    n = c.shape[0]
    u = np.empty(n, dtype=np.float32)
    for i in range(n):
        ci = max(c[i], 0.0) + epsilon
        if gamma == 1.0:
            u[i] = np.log(ci)
        else:
            u[i] = (ci ** (1.0 - gamma)) / (1.0 - gamma)
    return u


def log_utility(c: np.ndarray) -> np.ndarray:
    """
    Logarithmic utility function: u(c) = log(c).

    Args:
        c: Array of consumption values.

    Returns:
        Array of utility values. Returns -inf for c <= 0, so small positive values are recommended.
    """
    return np.log(np.maximum(c, 1e-8))


@njit()
def log_utility_numba(c: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Numba-accelerated logarithmic utility function.

    Args:
        c: Array of consumption values.
        epsilon: Small positive number to avoid -inf for zero consumption.

    Returns:
        Array of utility values.
    """
    n = c.shape[0]
    u = np.empty(n, dtype=np.float32)
    for i in range(n):
        ci = max(c[i], epsilon)
        u[i] = np.log(ci)
    return u


# -------------------------------------------------------------------
# Demo / plotting
# -------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Consumption grid
    c_values = np.linspace(0.0, 10.0, 500)

    # Compute utilities
    u_log = log_utility(np.maximum(c_values, 1e-8))  # avoid -inf for plotting
    u_crra = crra_utility(c_values, gamma=0.5, epsilon=1e-3)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(c_values, u_log, label="log(c)")
    plt.plot(c_values, u_crra, label="CRRA(c, γ=0.5, ε=1e-3)")
    plt.xlabel("Consumption c")
    plt.ylabel("Utility u(c)")
    plt.title("Comparison of Instantaneous Utility Functions")
    plt.legend()
    plt.grid(True)
    plt.show()
