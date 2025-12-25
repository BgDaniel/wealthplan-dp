import numpy as np


# -------------------------------------------------------------------
# Standard utility functions
# -------------------------------------------------------------------
def crra_utility(c: np.ndarray, gamma: float = 0.5, epsilon: float = 1e-8) -> np.ndarray:
    """
    Constant relative risk aversion (CRRA) utility with numerical stability near zero.
    """
    c_safe = np.maximum(c, 0.0) + epsilon
    if gamma == 1.0:
        return np.log(c_safe)
    else:
        return (c_safe**(1 - gamma)) / (1 - gamma)


def log_utility(c: np.ndarray) -> np.ndarray:
    """
    Standard logarithmic utility function: u(c) = log(c)
    Warning: returns -inf for c <= 0.
    """
    return np.log(np.maximum(c, 10e-8))



# -------------------------------------------------------------------
# Demo / plotting
# -------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Consumption grid
    c_values = np.linspace(0.0, 10.0, 500)

    # Compute utilities
    u_log = log_utility(np.maximum(c_values, 1e-8))   # avoid -inf for plotting
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
