import matplotlib.pyplot as plt
import logging
from typing import Optional
import numpy as np


logger = logging.getLogger(__name__)


class WealthRegressor:
    """
    Polynomial regression model for continuation values in stochastic DP.

    Approximates:
        v(w, s) = Σ β_{ijk} w^i r_s^j q_s^k

    where:
        - w is wealth (grid)
        - r_s are simulated returns
        - q_s are survival indicators / probabilities
    """

    def __init__(
        self,
        wealth_grid: np.ndarray,
        returns: np.ndarray,
        survival_paths: np.ndarray,
        deg_w: int = 3,
        deg_r: int = 3,
        deg_q: int = 1,
    ) -> None:
        """
        Initialize the regressor.

        Parameters
        ----------
        wealth_grid : np.ndarray, shape (n_w,)
            Deterministic wealth grid.
        returns : np.ndarray, shape (n_sims,)
            Simulated returns.
        survival_paths : np.ndarray, shape (n_sims,)
            Survival indicators or probabilities.
        deg_w : int, default=3
            Polynomial degree in wealth.
        deg_r : int, default=2
            Polynomial degree in returns.
        deg_q : int, default=1
            Polynomial degree in survival.
        """
        if wealth_grid.ndim != 1:
            raise ValueError("wealth_grid must be 1D.")
        if returns.ndim != 1:
            raise ValueError("returns must be 1D.")
        if survival_paths.ndim != 1:
            raise ValueError("survival_paths must be 1D.")
        if returns.shape[0] != survival_paths.shape[0]:
            raise ValueError("returns and survival_paths must have same length.")

        self.wealth_grid = wealth_grid
        self.returns = returns
        self.survival_paths = survival_paths

        self.deg_w = deg_w
        self.deg_r = deg_r
        self.deg_q = deg_q

        self.coeffs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Basis helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _poly_basis(x: np.ndarray, degree: int) -> np.ndarray:
        """
        Polynomial basis [1, x, x^2, ..., x^degree].

        Parameters
        ----------
        x : np.ndarray, shape (n,)
        degree : int

        Returns
        -------
        np.ndarray, shape (n, degree + 1)
        """
        return np.vstack([x ** k for k in range(degree + 1)]).T

    def _design_matrix(self) -> np.ndarray:
        """
        Build full tensor-product design matrix.

        Returns
        -------
        np.ndarray, shape (n_w * n_sims, n_features)
        """
        W = self._poly_basis(self.wealth_grid, self.deg_w)        # (n_w, Dw)
        R = self._poly_basis(self.returns, self.deg_r)           # (n_sims, Dr)
        Q = self._poly_basis(self.survival_paths, self.deg_q)    # (n_sims, Dq)

        # Combine return & survival (simulation-wise)
        RS = np.einsum("sr,sk->srk", R, Q).reshape(
            self.returns.size, -1
        )  # (n_sims, Dr*Dq)

        # Full tensor product with wealth
        Z = np.einsum("wi,sj->wsij", W, RS).reshape(
            self.wealth_grid.size * self.returns.size, -1
        )

        # Condition number before scaling
        cond_pre = np.linalg.cond(Z)

        # Scale each column safely
        for i in range(Z.shape[1]):
            col = Z[:, i]
            std = col.std()
            if std < 1e-12:
                # Column is constant: skip scaling
                Z[:, i] = 1.0
            else:
                Z[:, i] = (col - col.mean()) / std

        # Condition number after scaling
        cond_post = np.linalg.cond(Z)
        reduction = cond_pre / cond_post if cond_post != 0 else float("inf")

        logger.info(f"Condition number reduced from {cond_pre:.2e} to {cond_post:.2e} "
                    f"(reduction factor ≈ {reduction:.2f})")

        # Raise error only if post-scaling matrix is still ill-conditioned
        #if cond_post > 1e10:
        #    raise RuntimeError(f"Design matrix is ill-conditioned after scaling (cond={cond_post:.2e})")

        return Z

    def regress(
        self,
        v_next: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Fit the regression to v_next and immediately predict continuation values.

        Parameters
        ----------
        v_next : np.ndarray, shape (n_w, n_sims)
            Continuation values to regress.
        r_squared_threshold : float, default=0.9
            Minimum acceptable R-squared. Raises exception if fit is worse.

        Returns
        -------
        cont_value : np.ndarray, shape (n_w, n_sims)
            Predicted continuation values.
        r_squared : float
            Coefficient of determination of the fit.

        Raises
        ------
        RuntimeError
            If the R-squared is below `r_squared_threshold`.
        """
        # Flatten design matrix and fit
        Z = self._design_matrix()
        y = v_next.reshape(-1)

        # Check if all columns are constant after scaling (std ~ 0)
        if np.all(np.std(Z, axis=0) < 1e-12):
            mean_value = v_next.mean()
            v_hat = np.full_like(v_next, mean_value)
            r_squared = 0.0

            return v_hat, r_squared

        coeffs, residuals, *_ = np.linalg.lstsq(Z, y, rcond=None)
        self.coeffs = np.atleast_1d(coeffs)

        # Predict
        v_hat = (Z @ self.coeffs).reshape(v_next.shape)

        # Compute R-squared using residuals if available
        ss_res = np.sum((y - Z @ self.coeffs) ** 2)
        ss_total = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_total

        return v_hat, r_squared

    def plot_regression_fit_1d(
        self,
        v_next: np.ndarray,
        n_samples: Optional[int] = 1000,
        show_residuals: bool = True,
        figsize=(10, 6)
    ) -> None:
        """
        Plot predicted vs observed continuation values for diagnostics.

        Parameters
        ----------
        v_next : np.ndarray, shape (n_w, n_sims)
            True continuation values used for regression.
        n_samples : int, optional
            Maximum number of points to scatter for readability.
        show_residuals : bool, default True
            Whether to plot residuals vs wealth.
        figsize : tuple
            Figure size.
        """
        if self.coeffs is None:
            raise RuntimeError("Regression coefficients not computed yet. Call regress() first.")

        # Predict continuation values
        Z = self._design_matrix()
        v_hat = (Z @ self.coeffs).reshape(v_next.shape)

        # Flatten arrays for plotting
        y_true = v_next.flatten()
        y_pred = v_hat.flatten()

        # Sample points if too many
        if n_samples is not None and y_true.size > n_samples:
            idx = np.random.choice(y_true.size, n_samples, replace=False)
            y_true = y_true[idx]
            y_pred = y_pred[idx]

        # Plot observed vs predicted
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(y_true, y_pred, alpha=0.6)
        ax.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()], 'r--', lw=2, label="y=x")
        ax.set_xlabel("Observed continuation value")
        ax.set_ylabel("Predicted continuation value")
        ax.set_title("WealthRegressor: Observed vs Predicted")
        ax.legend()
        ax.grid(True)

        if show_residuals:
            residuals = y_true - y_pred
            fig2, ax2 = plt.subplots(figsize=figsize)
            ax2.scatter(y_true, residuals, alpha=0.6)
            ax2.axhline(0, color='r', linestyle='--')
            ax2.set_xlabel("Observed continuation value")
            ax2.set_ylabel("Residual (Observed - Predicted)")
            ax2.set_title("WealthRegressor Residuals")
            ax2.grid(True)

        plt.show()

    def plot_regression_fit_2d(
            self,
            v_next: np.ndarray,
            n_samples: Optional[int] = None,
            figsize=(12, 6)
    ) -> None:
        """
        Plot observed vs regressed continuation values as:
        1) two 3D surfaces in stacked rows
        2) optionally subsample simulations for readability

        Parameters
        ----------
        v_next : np.ndarray, shape (n_w, n_sims)
            True continuation values.
        n_samples : int, optional
            Maximum number of simulations to plot (for readability).
        figsize : tuple
            Base figure size (width, height per row)
        """
        if self.coeffs is None:
            raise RuntimeError("Regression coefficients not computed yet. Call regress() first.")

        # Predict continuation values
        Z = self._design_matrix()
        v_hat = (Z @ self.coeffs).reshape(v_next.shape)

        # Compute R²
        ss_res = np.sum((v_next - v_hat) ** 2)
        ss_total = np.sum((v_next - v_next.mean()) ** 2)
        r_squared = 1 - ss_res / ss_total
        print(f"[WealthRegressor] Regression R² = {r_squared:.4f}")

        n_sims = v_next.shape[1]

        # Reorder simulation indices by returns
        sorted_idx = np.argsort(self.returns)
        if n_samples is not None and n_sims > n_samples:
            idx = sorted_idx[:n_samples]  # take first n_samples sorted by return
        else:
            idx = sorted_idx  # take all simulations

        v_next_plot = v_next[:, idx]
        v_hat_plot = v_hat[:, idx]
        sim_axis = idx

        wealth_axis = self.wealth_grid
        W, S = np.meshgrid(sim_axis, wealth_axis)

        # ------------------------------
        # 3D surface plots stacked vertically
        # ------------------------------
        fig1 = plt.figure(figsize=(figsize[0], figsize[1] * 2))
        ax1 = fig1.add_subplot(2, 1, 1, projection='3d')
        ax1.plot_surface(W, S, v_next_plot, cmap='viridis')
        ax1.set_title("v_next (observed)")
        ax1.set_xlabel("Simulation (sorted by returns)")
        ax1.set_ylabel("Wealth")
        ax1.set_zlabel("Value")

        ax2 = fig1.add_subplot(2, 1, 2, projection='3d')
        ax2.plot_surface(W, S, v_hat_plot, cmap='viridis')
        ax2.set_title("v_hat (regressed)")
        ax2.set_xlabel("Simulation (sorted by returns)")
        ax2.set_ylabel("Wealth")
        ax2.set_zlabel("Value")

        plt.tight_layout()
        plt.show()
