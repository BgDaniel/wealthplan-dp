import logging
import datetime as dt
from typing import List, Optional
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
        yearly_return: float,
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
            yearly_return=yearly_return,
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

    @staticmethod
    def plot_with_bands(
        ax,
        x,
        mean,
        bands,
        color,
        title,
        title_size=22,
        legend_size=16,
        tick_size=16,
        yfmt=False,
    ):
        """
        Plot mean + percentile bands on a given axis.
        """
        all_values = [mean]

        for i, (p, (lo, hi)) in enumerate(bands.items()):
            ax.fill_between(
                x, lo, hi, color=color, alpha=0.15 + 0.15 * i, label=f"{p}-{100-p}%"
            )
            all_values.extend([lo, hi])

        ax.plot(x, mean, color=color, lw=2, linestyle="--", label="Mean")
        all_values = np.concatenate([np.ravel(v) for v in all_values])

        ymin, ymax = np.min(all_values), np.max(all_values)
        ymin *= 0.9 if ymin >= 0 else 1.1
        ymax *= 1.1 if ymax >= 0 else 0.9

        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=title_size)
        ax.legend(fontsize=legend_size)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_size)

        if yfmt:
            from matplotlib.ticker import FuncFormatter

            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    def _create_plot(self, percentiles=(5,10), sample_sim=None, title_size=22, legend_size=16, tick_size=16):
        """
        Creates figure and axes with base plots: consumption, wealth, investment, survival.
        Returns fig, axes (does NOT call plt.show())
        """
        if self.opt_wealth is None or self.opt_consumption is None:
            raise RuntimeError("No solution available — call solve() or train() first.")

        months = self.months
        n_sims = len(self.opt_wealth.columns)
        if sample_sim is None:
            np.random.seed()
            sample_sim = np.random.randint(n_sims)

        # Compute mean and bands
        cons_mean, cons_bands = self.compute_mean_and_bands(self.opt_consumption, percentiles)
        wealth_mean, wealth_bands = self.compute_mean_and_bands(self.opt_wealth, percentiles)
        inv_df = self.cf[:, np.newaxis] - self.opt_consumption.values
        inv_df = pd.DataFrame(inv_df, index=self.opt_consumption.index)
        inv_mean, inv_bands = self.compute_mean_and_bands(inv_df, percentiles)

        cons_sample = self.opt_consumption.iloc[:, sample_sim]
        wealth_sample = self.opt_wealth.iloc[:, sample_sim]
        inv_sample = inv_df.iloc[:, sample_sim]

        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)

        # Plot consumption
        self.plot_with_bands(axes[0], months, cons_mean, cons_bands, color="tab:green",
                              title="Optimal Consumption Over Time", title_size=title_size,
                              legend_size=legend_size, tick_size=tick_size, yfmt=True)
        axes[0].plot(months, cons_sample.values, color="red", lw=1.0, alpha=0.8, label="Sample Path")
        axes[0].plot(months, self.cf, color="blue", linestyle="--", lw=1.0, label="Deterministic Cashflows")
        axes[0].legend(fontsize=legend_size)

        # Plot wealth
        self.plot_with_bands(axes[1], months, wealth_mean, wealth_bands, color="tab:blue",
                              title="Wealth Over Time", title_size=title_size,
                              legend_size=legend_size, tick_size=tick_size)
        axes[1].plot(months, wealth_sample.values, color="red", lw=1.0, alpha=0.7, label="Sample Path")
        axes[1].legend(fontsize=legend_size)

        # Plot investment / withdrawal
        self.plot_with_bands(axes[2], months, inv_mean, inv_bands, color="tab:purple",
                              title="Monthly Investment / Withdrawal", title_size=title_size,
                              legend_size=legend_size, tick_size=tick_size)
        axes[2].plot(months, inv_sample.values, color="red", lw=1.0, alpha=0.8, label="Sample Path")
        axes[2].axhline(0.0, color="black", lw=1.0, alpha=0.7)
        axes[2].legend(fontsize=legend_size)

        # Survival probability
        axes[3].plot(self.age_grid, self.survival_probabilities, color="tab:orange", lw=2)
        axes[3].set_title("Survival Probability", fontsize=title_size)
        axes[3].set_ylim(0.0, 1.05)
        axes[3].grid(True)
        axes[3].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))

        return fig, axes

    def plot(self, percentiles=(5,10), sample_sim=None, title_size=22, legend_size=16, tick_size=16):
        _, _ = self._create_plot(percentiles, sample_sim, title_size, legend_size, tick_size)
        plt.show()