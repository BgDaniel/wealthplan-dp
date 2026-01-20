from typing import Callable

import numpy as np


PenalityFunction = Callable[[np.ndarray], np.ndarray]


# -------------------------------------------------------------------
# Standard penality functions
# -------------------------------------------------------------------
def square_penality(c: np.ndarray) -> np.ndarray:
    """
    Square penalty function: u(c) = -c^2
    """
    return -(c**2)