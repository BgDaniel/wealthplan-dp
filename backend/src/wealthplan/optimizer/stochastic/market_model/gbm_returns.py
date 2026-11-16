from __future__ import annotations

import datetime as dt
import math
from typing import List, Optional

import numpy as np


class GBM:
    """
    Geometric Brownian Motion (GBM) simulator for the equity index.

    The GBM is defined by the stochastic differential equation:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t

    with mu = log(1 + yearly_return), so that E[S_{t+1y} / S_t] = 1 + yearly_return.
    """

    def __init__(self, yearly_return: float, sigma: float, seed: Optional[int] = None) -> None:
        """
        Initialize the GBM model.

        Parameters
        ----------
        yearly_return : float
            Expected yearly (arithmetic) return, e.g. 0.07.
        sigma : float
            Yearly volatility.
        seed : int, optional
            Random seed for reproducibility.
        """
        if sigma < 0:
            raise ValueError("sigma must be >= 0")

        self.yearly_return = yearly_return
        self.mu = math.log(1.0 + yearly_return)
        self.sigma = sigma
        self.seed = seed

    def simulate(
        self,
        n_sims: int,
        dates: List[dt.date],
        s0: float = 1.0,
        seed: Optional[int] = None,
        moment_matching: bool = False,
    ) -> np.ndarray:
        """
        Simulate GBM paths on an arbitrary date grid (exact log-normal steps).

        Parameters
        ----------
        n_sims : int
            Number of simulated paths.
        dates : List[datetime.date]
            Ordered list of dates defining the time grid.
        s0 : float, default=1.0
            Initial value of the process.
        seed : int, optional
            Overrides the seed given in the constructor.
        moment_matching : bool, default=False
            Standardise the normal draws of every time step to mean 0 and variance 1 across
            the paths (variance reduction: the first two moments of every step are exact).

        Returns
        -------
        np.ndarray
            Simulated GBM paths of shape (n_sims, len(dates)).
        """
        if len(dates) < 2:
            raise ValueError("At least two dates are required for simulation.")

        times = np.array([(d - dates[0]).days / 365.0 for d in dates])
        dt_grid = np.diff(times)

        rng = np.random.default_rng(self.seed if seed is None else seed)
        z = rng.standard_normal((n_sims, dt_grid.size))
        if moment_matching and n_sims > 1:
            z = (z - z.mean(axis=0)) / z.std(axis=0)
        log_steps = (self.mu - 0.5 * self.sigma**2) * dt_grid + self.sigma * np.sqrt(dt_grid) * z

        paths = np.empty((n_sims, len(dates)))
        paths[:, 0] = s0
        paths[:, 1:] = s0 * np.exp(np.cumsum(log_steps, axis=1))
        return paths
