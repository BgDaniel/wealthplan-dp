import datetime as dt
import pandas as pd
from typing import List, Tuple
import unittest
import numpy as np
import matplotlib.pyplot as plt

from wealthplan.optimizer.stochastic import GBM


class TestGBM(unittest.TestCase):
    """
    Unittest tests for Geometric Brownian Motion (GBM) simulation.

    This class verifies that the empirical mean and variance from GBM
    simulations match the theoretical expectations within a specified tolerance.
    Optional plotting shows the ± corridor around theoretical values.
    """

    PLOT: bool = True  # Set True to visualize results

    @staticmethod
    def _year_fractions(dates: List[dt.date]) -> np.ndarray:
        return np.array([(d - dates[0]).days / 365.0 for d in dates])

    @staticmethod
    def _generate_monthly_dates(
            start_year: int = 2025,
            end_year: int = 2028,
    ) -> List[dt.date]:
        """
        Generate a list of monthly dates (month starts) between two years inclusive.

        Parameters
        ----------
        start_year : int, default=2025
            First calendar year (January included).
        end_year : int, default=2028
            Last calendar year (December included).

        Returns
        -------
        List[datetime.date]
            List of month-start dates.
        """
        date_index = pd.date_range(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-01",
            freq="MS",  # Month Start
        )

        return [d.date() for d in date_index]

    @staticmethod
    def _simulate(mu: float = 0.05, n_sims: int = 5_000) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        sigma: float = 0.2
        s0: float = 100.0
        seed: int = 123

        dates: List[dt.date] = TestGBM._generate_monthly_dates()
        gbm: GBM = GBM(mu=mu, sigma=sigma, seed=seed)
        paths: np.ndarray = gbm.simulate(n_sims=n_sims, dates=dates, s0=s0)
        times: np.ndarray = TestGBM._year_fractions(dates)

        return paths, times, mu, sigma, s0

    def test_empirical_mean_nonzero_mu(self, tol: float = 0.01) -> None:
        """
        Test that the empirical GBM mean matches the theoretical mean
        with a non-zero drift (default mu=0.05) within a relative tolerance.
        Always plots the results, marking points outside the ± corridor with red crosses.

        Parameters
        ----------
        tol : float
            Relative tolerance for mean assertion and ± corridor plotting.
        """
        paths, times, mu, _, s0 = self._simulate(mu=0.05)
        empirical_mean: np.ndarray = paths.mean(axis=0)
        theoretical_mean: np.ndarray = s0 * np.exp(mu * times)

        # Compute points outside relative tolerance
        tol_array: np.ndarray = tol * theoretical_mean
        outside_mask: np.ndarray = np.abs(empirical_mean - theoretical_mean) > tol_array

        # Plot always
        plt.figure()
        plt.plot(times, empirical_mean, label="Empirical Mean")
        plt.plot(times, theoretical_mean, "--", label="Theoretical Mean")
        plt.fill_between(times, theoretical_mean - tol_array, theoretical_mean + tol_array,
                         alpha=0.2, label=f"±{int(tol * 100)}% Corridor")

        # Mark outliers outside corridor
        if np.any(outside_mask):
            plt.scatter(times[outside_mask], empirical_mean[outside_mask], color='red', marker='x', s=80,
                        label="Outside ± Corridor")

        plt.title("GBM Mean (Non-zero Drift)")
        plt.xlabel("Time (years)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Assertion after plotting
        np.testing.assert_allclose(empirical_mean, theoretical_mean, rtol=tol)

    def test_empirical_mean_zero_mu(self, tol: float = 0.01) -> None:
        """
        Test that the empirical GBM mean matches the theoretical mean
        with zero drift (mu=0) within a relative tolerance.
        Always plots the results, marking points outside the ± corridor with red crosses.

        Parameters
        ----------
        tol : float
            Relative tolerance for mean assertion and ± corridor plotting.
        """
        paths, times, mu, _, s0 = self._simulate(mu=0.0)
        empirical_mean: np.ndarray = paths.mean(axis=0)
        theoretical_mean: np.ndarray = np.full_like(times, s0)

        # Compute points outside relative tolerance
        tol_array: np.ndarray = tol * theoretical_mean
        outside_mask: np.ndarray = np.abs(empirical_mean - theoretical_mean) > tol_array

        # Plot always
        plt.figure()
        plt.plot(times, empirical_mean, label="Empirical Mean")
        plt.plot(times, theoretical_mean, "--", label="Theoretical Mean")
        plt.fill_between(times, theoretical_mean - tol_array, theoretical_mean + tol_array,
                         alpha=0.2, label=f"±{int(tol * 100)}% Corridor")

        # Mark outliers outside corridor
        if np.any(outside_mask):
            plt.scatter(times[outside_mask], empirical_mean[outside_mask], color='red', marker='x', s=80,
                        label="Outside ± Corridor")

        plt.title("GBM Mean (Zero Drift)")
        plt.xlabel("Time (years)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Assertion after plotting
        np.testing.assert_allclose(empirical_mean, theoretical_mean, rtol=tol)

    def test_empirical_variance(self, rtol: float = 0.075) -> None:
        """
        Test that the empirical GBM variance matches the theoretical variance
        within a relative tolerance (default 10%). Always plots the results,
        marking points outside the ± corridor with red crosses.

        Parameters
        ----------
        rtol : float, default=0.1
            Relative tolerance for variance assertion and ± corridor plotting.
        """
        paths, times, mu, sigma, s0 = self._simulate()
        empirical_var: np.ndarray = paths.var(axis=0)
        theoretical_var: np.ndarray = s0 ** 2 * np.exp(2 * mu * times) * (np.exp(sigma ** 2 * times) - 1.0)

        # Compute points outside relative tolerance
        tol_array: np.ndarray = rtol * theoretical_var
        outside_mask: np.ndarray = np.abs(empirical_var - theoretical_var) > tol_array

        # Plot always
        plt.figure()
        plt.plot(times, empirical_var, label="Empirical Variance")
        plt.plot(times, theoretical_var, "--", label="Theoretical Variance")
        plt.fill_between(times, theoretical_var - tol_array, theoretical_var + tol_array,
                         alpha=0.2, label=f"±{int(rtol * 100)}% Corridor")

        # Mark outliers outside corridor
        if np.any(outside_mask):
            plt.scatter(times[outside_mask], empirical_var[outside_mask], color='red', marker='x', s=80,
                        label="Outside ± Corridor")

        plt.title("GBM Variance")
        plt.xlabel("Time (years)")
        plt.ylabel("Variance")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Assertion after plotting
        np.testing.assert_allclose(empirical_var, theoretical_var, rtol=rtol)

if __name__ == "__main__":
    unittest.main()
