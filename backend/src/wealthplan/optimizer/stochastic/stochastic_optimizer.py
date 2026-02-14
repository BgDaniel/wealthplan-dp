import logging
import datetime as dt
from typing import List, Optional, OrderedDict
import numpy as np
import pandas as pd
from abc import abstractmethod

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from wealthplan.cashflows.cashflow_base import CashflowBase

from wealthplan.optimizer.optimizer_base import OptimizerBase
from wealthplan.optimizer.stochastic.survival_process.survival_model import (
    SurvivalModel,
)

logger = logging.getLogger(__name__)


# ---------------------------
# Stochastic optimizer
# ---------------------------
class StochasticOptimizerBase(OptimizerBase):
    """
    Base class for stochastic lifecycle optimization.

    Adds support for:
    - Stochastic returns (GBM)
    - Mortality / survival model
    - Current age tracking
    """

    def __init__(
        self,
        run_config_id: str,
        start_date: dt.date,
        end_date: dt.date,
        retirement_date: dt.date,
        initial_wealth: float,
        yearly_return_savings: float,
        cashflows: List[CashflowBase],
        survival_model: SurvivalModel,
        current_age: int,
        stochastic: bool,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            run_config_id=run_config_id,
            start_date=start_date,
            end_date=end_date,
            retirement_date=retirement_date,
            initial_wealth=initial_wealth,
            yearly_return_savings=yearly_return_savings,
            cashflows=cashflows,
        )
        self.survival_model: SurvivalModel = survival_model
        self.current_age: int = current_age

        self.age_grid = self.time_grid + self.current_age

        self.survival_probabilities: np.ndarray = (
            self.survival_model.cumulative_survival_probabilities(
                self.age_grid, self.dt
            )
        )

        self.stochastic = stochastic

        # Storage for simulation results
        self.optimal_wealth: Optional[pd.DataFrame] = None
        self.optimal_consumption: Optional[pd.DataFrame] = None

        self.seed = seed

    @staticmethod
    def compute_mean_and_bands(
        df: pd.DataFrame, percentiles: tuple[float, ...] = (5, 10)
    ):
        """
        Compute mean and percentile bands for a DataFrame along axis=1.

        Parameters
        ----------
        df : pd.DataFrame
            Data with shape (time, n_sims)
        percentiles : tuple[float, ...]
            Percentiles to compute (e.g., (5, 10) → 5–95% and 10–90%)

        Returns
        -------
        mean : pd.Series
            Mean across simulations at each time step
        bands : dict
            Dictionary mapping percentile → (lower, upper) Series
        """
        mean = df.mean(axis=1)

        bands = {
            p: (df.quantile(p / 100, axis=1), df.quantile(1 - p / 100, axis=1))
            for p in percentiles
        }

        return mean, bands

    def _create_plot(
            self,
            percentiles=(5, 10),
            sample_sim=None,
            title_size=22,
            tick_size=14,
    ):
        """
        Base figure and axes with:
        1. Consumption + Survival probability
        2. Wealth
        3. Investment / Withdrawal
        """
        if self.opt_wealth is None or self.opt_consumption is None:
            raise RuntimeError("No solution available — call solve() or train() first.")

        months = self.months
        n_sims = len(self.opt_wealth.columns)
        if sample_sim is None:
            sample_sim = np.random.randint(n_sims)

        # Compute mean + bands
        cons_mean, cons_bands = self.compute_mean_and_bands(self.opt_consumption, percentiles)
        wealth_mean, wealth_bands = self.compute_mean_and_bands(self.opt_wealth, percentiles)
        inv_df = self.cf[:, np.newaxis] - self.opt_consumption.values
        inv_df = pd.DataFrame(inv_df, index=self.opt_consumption.index)
        inv_mean, inv_bands = self.compute_mean_and_bands(inv_df, percentiles)

        cons_sample = self.opt_consumption.iloc[:, sample_sim]
        wealth_sample = self.opt_wealth.iloc[:, sample_sim]
        inv_sample = inv_df.iloc[:, sample_sim]

        # --- figure and axes ---
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=False)

        # 1️⃣ Consumption + survival probability
        ax = axes[0]
        ax.plot(months, cons_mean, color="tab:green", lw=2, linestyle="--", label="Mean Consumption")

        for i, (p, (lo, hi)) in enumerate(cons_bands.items()):
            ax.fill_between(months, lo, hi, color="tab:green", alpha=0.15 + 0.15 * i,
                            label=f"{p}-{100 - p}% Consumption Band")

        ax2 = ax.twinx()  # create a secondary y-axis
        ax2.plot(months, self.survival_probabilities, color="tab:orange", lw=2)
        ax2.set_ylabel("Survival Probability")
        ax2.set_ylim(0, 1)

        ax.plot(months, self.survival_probabilities * cons_mean.max(),
                color="tab:orange", lw=2, label="Survival Probability (scaled)")

        ax.set_title("Consumption & Survival Probability", fontsize=title_size)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_size)

        # 2️⃣ Wealth
        ax = axes[1]
        ax.plot(months, wealth_mean, color="tab:blue", lw=2, linestyle="--", label="Mean Wealth")

        for i, (p, (lo, hi)) in enumerate(wealth_bands.items()):
            ax.fill_between(months, lo, hi, color="tab:blue", alpha=0.15 + 0.15 * i,
                            label=f"{p}-{100 - p}% Wealth Band")

        ax.plot(months, wealth_sample, color="red", lw=1, alpha=0.7, label="Sample Wealth")
        ax.set_title("Wealth Over Time", fontsize=title_size)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_size)

        # 3️⃣ Investment / Withdrawal
        ax = axes[2]
        ax.plot(months, inv_mean, color="tab:purple", lw=2, linestyle="--", label="Mean Investment")

        for i, (p, (lo, hi)) in enumerate(inv_bands.items()):
            ax.fill_between(months, lo, hi, color="tab:purple", alpha=0.15 + 0.15 * i,
                            label=f"{p}-{100 - p}% Investment Band")

        ax.plot(months, inv_sample, color="red", lw=1, alpha=0.8, label="Sample Investment")
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.7)
        ax.set_title("Monthly Investment / Withdrawal", fontsize=title_size)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_size)

        return fig, axes

    def plot(
            self,
            percentiles=(5, 10),
            sample_sim=None,
            title_size=22,
            legend_size=16,
            tick_size=16,
    ):
        """
        Base class plot with 3 axes.
        Legends now include full descriptive extension.
        """
        fig, axes = self._create_plot(percentiles, sample_sim, title_size, tick_size)

        # add legends for all subplots
        for ax in axes:
            ax.legend(fontsize=legend_size)

        plt.show()