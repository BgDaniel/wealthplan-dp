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
from wealthplan.optimizer.stochastic.survival_process.survival_model import SurvivalModel

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
            stochastic: bool
    ) -> None:
        super().__init__(
            run_config_id=run_config_id,
            start_date=start_date,
            end_date=end_date,
            retirement_date=retirement_date,
            initial_wealth=initial_wealth,
            yearly_return=yearly_return,
            cashflows=cashflows
        )
        self.survival_model: SurvivalModel = survival_model
        self.current_age: int = current_age

        self.age_grid = self.time_grid + self.current_age

        self.survival_probabilities: np.ndarray = self.survival_model.cumulative_survival_probabilities(self.age_grid, self.dt)

        self.stochastic = stochastic

        # Storage for simulation results
        self.optimal_wealth: Optional[pd.DataFrame] = None
        self.optimal_consumption: Optional[pd.DataFrame] = None
        self.value_function: Optional[np.ndarray] = None

    def plot(
        self,
        percentiles: tuple[float, ...] = (5, 10),
        sample_sim: Optional[int] = None,
        *,
        title_size: int = 22,
        legend_size: int = 16,
        tick_size: int = 16,
    ) -> None:
        """
        Plot stochastic results with mean paths, percentile bands, sample paths,
        and survival probabilities.

        The figure contains four subplots:
            1. Optimal consumption over time
            2. Wealth over time
            3. Monthly investment / withdrawal
            4. Cumulative survival probability

        Parameters
        ----------
        percentiles : tuple[float, ...], default=(5, 10)
            Percentile levels (e.g. 5 → 5–95%, 10 → 10–90%).

        sample_sim : int, optional
            Index of simulation to plot as a sample path.
            If None, a random simulation is selected.

        title_size : int, default=18
            Font size for subplot titles.

        legend_size : int, default=14
            Font size for legend text.

        tick_size : int, default=14
            Font size for axis tick labels.

        Returns
        -------
        None
        """
        if self.opt_wealth.empty:
            raise RuntimeError("No solution available — call solve() first.")

        n_sims = len(self.opt_wealth.columns)

        percentiles = sorted(percentiles)

        # --- Pick sample simulation ---
        if sample_sim is None:
            np.random.seed()
            sample_sim = np.random.randint(n_sims)

        months = self.months

        # ============================
        # Helper: mean + percentile bands
        # ============================
        def mean_and_bands(
            df: pd.DataFrame,
        ) -> tuple[pd.Series, dict[float, tuple[pd.Series, pd.Series]]]:
            mean = df.mean(axis=1)
            bands = {
                p: (
                    df.quantile(p / 100, axis=1),
                    df.quantile(1 - p / 100, axis=1),
                )
                for p in percentiles
            }
            return mean, bands

        # ============================
        # Data preparation
        # ============================
        cons_mean, cons_bands = mean_and_bands(self.opt_consumption)
        wealth_mean, wealth_bands = mean_and_bands(self.opt_wealth)

        cons_sample = self.opt_consumption.iloc[:, sample_sim]
        wealth_sample = self.opt_wealth.iloc[:, sample_sim]

        # ============================
        # Plotting
        # ============================
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)

        def plot_with_bands(
            ax: plt.Axes,
            x: pd.Index | np.ndarray,
            mean: pd.Series,
            bands: dict[float, tuple[pd.Series, pd.Series]],
            color: str,
            title: str,
            yfmt: bool = False,
        ) -> None:
            all_values = [mean]

            for i, (p, (lo, hi)) in enumerate(bands.items()):
                ax.fill_between(
                    x,
                    lo,
                    hi,
                    color=color,
                    alpha=0.15 + 0.15 * i,
                    label=f"{p}–{100 - p}%",
                )
                all_values.extend([lo, hi])

            ax.plot(x, mean, color=color, lw=2, label="Mean", linestyle="--")

            if x[0] <= self.retirement_date <= x[-1]:
                ax.axvline(
                    self.retirement_date,
                    color="red",
                    linestyle="--",
                    lw=2,
                    label="Retirement",
                )

            # Dynamic y-limits
            all_values = np.concatenate([np.ravel(v) for v in all_values])
            ymin, ymax = np.min(all_values), np.max(all_values)

            # Adjust lower limit
            if ymin >= 0:
                ymin *= 0.9
            else:
                ymin *= 1.1  # expand downward for negative min

            # Adjust upper limit
            if ymax >= 0:
                ymax *= 1.1
            else:
                ymax *= 0.9  # expand upward for negative max

            ax.set_ylim(ymin, ymax)

            ax.set_title(title, fontsize=title_size)
            ax.legend(fontsize=legend_size)
            ax.grid(alpha=0.3)
            ax.tick_params(axis="both", labelsize=tick_size)

            if yfmt:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

        # --- 1. Consumption ---
        plot_with_bands(
            axes[0],
            months,
            cons_mean,
            cons_bands,
            color="tab:green",
            title="Optimal Consumption Over Time",
            yfmt=True,
        )
        axes[0].plot(
            cons_sample.index,
            cons_sample.values,
            color="red",
            lw=1.0,
            alpha=0.8,
            label="Sample Path",
        )
        axes[0].plot(
            months,
            self.cf,
            color="blue",
            linestyle="--",
            lw=1.0,
            label="Deterministic Cashflows",
        )
        axes[0].legend(fontsize=legend_size)
        axes[0].tick_params(axis="both", labelsize=tick_size)

        # --- 2. Wealth ---
        plot_with_bands(
            axes[1],
            months,
            wealth_mean,
            wealth_bands,
            color="tab:blue",
            title="Wealth Over Time",
            yfmt=True,
        )
        axes[1].plot(
            wealth_sample.index,
            wealth_sample.values,
            color="red",
            lw=1.0,
            alpha=0.8,
            label="Sample Path",
        )

        axes[1].legend(fontsize=legend_size)
        axes[1].tick_params(axis="both", labelsize=tick_size)

        # --- 3. Investment / Withdrawal ---
        inv_df = self.cf[:, np.newaxis] - self.opt_consumption.values
        inv_df = pd.DataFrame(inv_df, index=self.opt_consumption.index)

        inv_mean, inv_bands = mean_and_bands(inv_df)
        inv_sample = inv_df.iloc[:, sample_sim]

        plot_with_bands(
            axes[2],
            months,
            inv_mean,
            inv_bands,
            color="tab:purple",
            title="Monthly Investment / Withdrawal in Portfolio",
            yfmt=True,
        )
        axes[2].plot(
            inv_sample.index,
            inv_sample.values,
            color="red",
            lw=1.0,
            alpha=0.8,
            label="Sample Path",
        )
        axes[2].axhline(0.0, color="black", lw=1.0, alpha=0.7)
        axes[2].legend(fontsize=legend_size)
        axes[2].tick_params(axis="both", labelsize=tick_size)

        # --- 4. Survival probability ---
        cumulative_survival = np.cumprod(self.survival_probabilities)

        axes[3].plot(self.age_grid, cumulative_survival, color="tab:orange", lw=2)
        axes[3].set_title("Survival Probability", fontsize=title_size)
        axes[3].set_ylim(0.0, 1.05)
        axes[3].grid(True)
        axes[3].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        axes[3].tick_params(axis="both", labelsize=tick_size)

        plt.tight_layout()
        plt.show()