from __future__ import annotations

import datetime as dt
from typing import List, Optional

import numpy as np


class GBM:
    """
    Geometric Brownian Motion (GBM) simulator.

    The GBM is defined by the stochastic differential equation:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t

    where:
        - mu is the drift
        - sigma is the volatility
    """

    def __init__(self, mu: float, sigma: float) -> None:
        """
        Initialize the GBM model.

        Parameters
        ----------
        mu : float
            Drift parameter.
        sigma : float
            Volatility parameter.
        """
        self.mu = mu
        self.sigma = sigma

    def simulate(
        self,
        n_sims: int,
        dates: List[dt.date],
        seed: int,
        s0: float = 1.0,
    ) -> np.ndarray:
        """
        Simulate GBM paths.

        Parameters
        ----------
        n_sims : int
            Number of simulated paths.
        dates : List[datetime.date]
            Ordered list of dates defining the time grid.
        seed : int
            Random seed for reproducibility.
        s0 : float, default=1.0
            Initial value of the process.

        Returns
        -------
        np.ndarray
            Simulated GBM paths of shape (n_sims, len(dates)).
        """
        if len(dates) < 2:
            raise ValueError("At least two dates are required for simulation.")

        # Convert dates to year fractions
        times = np.array(
            [(dates[i] - dates[0]).days / 365.0 for i in range(len(dates))]
        )
        dt = np.diff(times)

        n_steps = len(dates)
        paths = np.zeros((n_sims, n_steps))
        paths[:, 0] = s0

        np.random.seed(seed)

        for t in range(1, n_steps):
            z = np.random.normal(size=n_sims)
            drift = (self.mu - 0.5 * self.sigma ** 2) * dt[t - 1]
            diffusion = self.sigma * np.sqrt(dt[t - 1]) * z
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion)

        return paths
