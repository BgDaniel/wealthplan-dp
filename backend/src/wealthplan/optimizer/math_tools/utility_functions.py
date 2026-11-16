from typing import Callable

import numpy as np
from numba import njit

# Type alias
UtilityFunction = Callable[[np.ndarray], np.ndarray]


def crra_utility(c: np.ndarray, gamma: float = 0.5, epsilon: float = 1e-4) -> np.ndarray:
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
        return (c_safe ** (1 - gamma)) / (1 - gamma)


@njit(cache=True)
def crra_utility_scalar(c: float, gamma: float, epsilon: float) -> float:
    """
    Numba CRRA utility of a single consumption value (used inside the optimizer kernels).

    Args:
        c: Consumption.
        gamma: Risk aversion parameter.
        epsilon: Small positive number to avoid zero consumption issues.

    Returns:
        Utility value.
    """
    ci = max(c, 0.0) + epsilon
    if gamma == 1.0:
        return np.log(ci)
    return ci ** (1.0 - gamma) / (1.0 - gamma)
