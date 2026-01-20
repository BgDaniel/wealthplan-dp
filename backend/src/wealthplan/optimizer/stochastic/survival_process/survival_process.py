import datetime as dt
from typing import List, Optional
import numpy as np


class SurvivalProcess:
    """
    Survival process with age-dependent hazard (Gompertz law),
    suitable for life insurance / human mortality modeling.

    Hazard function:
        λ(t) = b * exp(c * t)

    Survival indicator S_t:
        1 if alive at time t
        0 if dead
    """

    def __init__(self, b: float, c: float, age: float = 0.0, seed: Optional[int] = None):
        """
        Initialize Gompertz survival process.

        Parameters
        ----------
        b : float
            Baseline hazard (year^-1)
        c : float
            Aging rate (year^-1)
        age : float
            Current age at t=0 (years)
        seed : Optional[int]
            Random seed
        """
        if b < 0 or c < 0:
            raise ValueError("b and c must be positive for realistic mortality.")

        self.b = b
        self.c = c
        self.age = age
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def simulate(self, n_sims: int, dates: List[dt.date]) -> np.ndarray:
        """
        Simulate survival indicator paths.

        Parameters
        ----------
        n_sims : int
            Number of paths
        dates : List[datetime.date]
            Ordered time grid

        Returns
        -------
        np.ndarray
            Survival paths: shape (n_sims, len(dates)), values 0 or 1
        """
        if len(dates) < 2:
            raise ValueError("At least two dates are required for simulation.")

        # Time grid in years from start
        times = np.array([(d - dates[0]).days / 365.0 for d in dates])
        dt_grid = np.diff(times, prepend=0.0)

        paths = np.ones((n_sims, len(dates)), dtype=float)
        alive = np.ones(n_sims, dtype=bool)

        # Generate death times using inverse transform sampling
        # CDF: F(t) = 1 - exp(- (b/c) * (exp(c*(t+age)) - exp(c*age)))
        # Inverse CDF: t = (1/c) * log( - (c/b) * log(U) + exp(c*age)) - age
        U = np.random.uniform(size=n_sims)
        death_times = (1.0 / self.c) * np.log(- (self.c / self.b) * np.log(U) + np.exp(self.c * self.age)) - self.age

        for t_idx, t in enumerate(times):
            newly_dead = alive & (t >= death_times)
            alive[newly_dead] = False
            paths[~alive, t_idx] = 0.0
            paths[alive, t_idx] = 1.0

        return paths
