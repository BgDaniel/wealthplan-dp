import logging
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


class WealthRegressor:
    """
    Least-squares regression of continuation values on the market state (LSMC step).

    For every target column k (one column per post-decision wealth grid point and equity
    share) the conditional expectation of the realised next-period values is approximated by

        E[Y_k | S_t] ≈ Σ_i β_{ik} φ_i(z_s),    z_s = (log S_{t,s} - m) / sd,

    with the polynomial basis φ_i(z) = z^i, i = 0..degree, of the standardised log equity
    level S_t of path s. All targets share the design matrix, so one least-squares solve
    covers the whole wealth grid.

    The wealth dimension is no longer part of the basis (the former tensor-product
    polynomial in wealth fitted the steep CRRA value function poorly near zero wealth);
    wealth is handled exactly by the grid.
    """

    def __init__(self, state: np.ndarray, degree: int = 3) -> None:
        """
        Initialize the regressor.

        Parameters
        ----------
        state : np.ndarray, shape (n_sims,)
            Equity level S_t of each regression path (> 0).
        degree : int, default=3
            Polynomial degree in the standardised log level.
        """
        if state.ndim != 1:
            raise ValueError("state must be 1D.")
        if degree < 0:
            raise ValueError("degree must be >= 0.")

        log_state = np.log(state)
        self.mean: float = float(log_state.mean())
        std = float(log_state.std())
        # Degenerate state (e.g. t = 0, all paths start at S_0): constant basis only.
        self.std: float = std if std > 1e-12 else 0.0
        self.degree: int = degree if self.std > 0.0 else 0

        self.design: np.ndarray = self.basis(state)
        # Orthonormal basis of the design: design = q @ r. Projections on q give the
        # least-squares fit of any number of targets without forming the normal equations.
        self.q, self.r = np.linalg.qr(self.design)
        self.coeffs: Optional[np.ndarray] = None
        self.r_squared: float = float("nan")

    def basis(self, state: np.ndarray) -> np.ndarray:
        """
        Polynomial basis [1, z, ..., z^degree] of the standardised log level.

        Parameters
        ----------
        state : np.ndarray, shape (n,)

        Returns
        -------
        np.ndarray, shape (n, degree + 1)
        """
        z = (np.log(state) - self.mean) / self.std if self.std > 0.0 else np.zeros(state.shape)
        return np.vander(z, self.degree + 1, increasing=True)

    def fit_projection(self, proj: np.ndarray, sum_y: np.ndarray, sum_y2: np.ndarray) -> None:
        """
        Set the coefficients from precomputed projections (used by the Numba kernels).

        Parameters
        ----------
        proj : np.ndarray, shape (degree + 1, n_targets)
            q.T @ y, the projections of the targets on the orthonormal basis.
        sum_y, sum_y2 : np.ndarray, shape (n_targets,)
            Column sums of y and y**2 (for R²).
        """
        n = self.q.shape[0]
        self.coeffs = np.linalg.solve(self.r, proj)

        # Diagnostic: median R² over the targets that vary across paths. With i.i.d. returns
        # (GBM) the market state carries little information, so R² close to 0 is expected.
        ss_total = sum_y2 - sum_y**2 / n
        ss_res = ss_total - ((proj[1:] ** 2).sum(axis=0) if proj.shape[0] > 1 else 0.0)
        scale = np.maximum(1.0, sum_y2 / n)
        varying = ss_total > 1e-10 * scale * n
        self.r_squared = float(np.median(1.0 - ss_res[varying] / ss_total[varying])) if varying.any() else 1.0

    def regress(self, y: np.ndarray) -> np.ndarray:
        """
        Fit all target columns and return the fitted continuation values.

        Parameters
        ----------
        y : np.ndarray, shape (n_sims, n_targets)
            Realised continuation values per path and target.

        Returns
        -------
        np.ndarray, shape (n_sims, n_targets)
            Fitted conditional expectations.
        """
        proj = self.q.T @ y
        self.fit_projection(proj, y.sum(axis=0), (y**2).sum(axis=0))
        return self.q @ proj
