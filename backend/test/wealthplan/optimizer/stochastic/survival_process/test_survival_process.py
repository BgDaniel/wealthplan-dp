import datetime as dt
from typing import List
import pandas as pd

import unittest
import numpy as np
import matplotlib.pyplot as plt

from wealthplan.optimizer.stochastic import SurvivalProcess


class TestSurvivalProcess(unittest.TestCase):
    """
    Unit tests for GompertzSurvivalProcess (age-dependent hazard),
    suitable for life insurance mortality modeling.
    """

    PLOT: bool = True

    @staticmethod
    def _monthly_dates(start: dt.date, end: dt.date) -> List[dt.date]:
        """Generate a list of month-start dates."""
        date_index = pd.date_range(start=start, end=end, freq="MS")
        return [d.date() for d in date_index]

    @staticmethod
    def _year_fractions(dates: List[dt.date]) -> np.ndarray:
        """Convert dates to year fractions from the first date."""
        return np.array([(d - dates[0]).days / 365.0 for d in dates])

    def test_shape_and_values(self) -> None:
        """Check that simulated paths have correct shape and only 0/1 values."""
        n_sims = 1000
        b, c, age = 0.0005, 0.085, 37  # Gompertz parameters

        dates = self._monthly_dates(dt.date(2025, 1, 1), dt.date(2125, 1, 1))
        model = SurvivalProcess(b=b, c=c, age=age, seed=42)
        paths = model.simulate(n_sims=n_sims, dates=dates)

        self.assertEqual(paths.shape, (n_sims, len(dates)))
        self.assertTrue(np.all(np.isin(paths, [0.0, 1.0])))

    def test_absorbing_state(self) -> None:
        """Check that once a path hits zero, it remains zero."""
        n_sims = 2000
        b, c, age = 0.0005, 0.085, 37

        dates = self._monthly_dates(dt.date(2025, 1, 1), dt.date(2125, 1, 1))
        model = SurvivalProcess(b=b, c=c, age=age, seed=123)
        paths = model.simulate(n_sims=n_sims, dates=dates)

        for path in paths:
            zeros = np.where(path == 0.0)[0]
            if zeros.size > 0:
                first_zero = zeros[0]
                self.assertTrue(np.all(path[first_zero:] == 0.0))

    def test_death_time_density(self) -> None:
        """Plot empirical PDF of death times and theoretical Gompertz PDF."""
        n_sims = 10_000
        b, c, age = 9.5e-5, 0.085, 37

        dates = self._monthly_dates(dt.date(2025, 1, 1), dt.date(2125, 1, 1))
        model = SurvivalProcess(b=b, c=c, age=age, seed=123)
        paths = model.simulate(n_sims=n_sims, dates=dates)
        times = self._year_fractions(dates)

        # Extract death times
        death_times = []
        for path in paths:
            zeros = np.where(path == 0.0)[0]
            if zeros.size > 0:
                death_times.append(times[zeros[0]])
        death_times = np.array(death_times)

        # Theoretical PDF (Gompertz)
        t_grid = np.linspace(0, death_times.max(), 500)
        hazard = b * np.exp(c * (age + t_grid))
        S = np.exp(- (b / c) * (np.exp(c * (age + t_grid)) - np.exp(c * age)))
        pdf = hazard * S

        if self.PLOT:
            plt.figure(figsize=(8, 5))
            plt.hist(death_times, bins=100, density=True, alpha=0.5, label="Empirical PDF")
            plt.plot(t_grid, pdf, "r--", linewidth=2, label="Theoretical Gompertz PDF")
            # Expected age (approx)
            expected_remaining = (1 / c) * np.log(1 + (c / b) * death_times.mean())
            plt.axvline(death_times.mean(), color="black", linestyle=":", linewidth=2,
                        label=f"Empirical mean = {death_times.mean():.1f} yrs")
            plt.title("Distribution of Death Times (Gompertz)")
            plt.xlabel("Years from start")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Sanity check: empirical mean > 0
        self.assertGreater(death_times.mean(), 0.0)

if __name__ == "__main__":
    unittest.main()
