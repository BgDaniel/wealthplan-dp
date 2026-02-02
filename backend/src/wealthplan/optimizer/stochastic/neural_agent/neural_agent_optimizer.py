import datetime as dt
from typing import List, Callable, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import trange
from matplotlib.ticker import FuncFormatter

from wealthplan.cashflows.cashflow_base import CashflowBase
from wealthplan.optimizer.stochastic.neural_agent.simple_policy_network import (
    SimplePolicyNetwork,
)
from wealthplan.optimizer.stochastic.market_model.gbm_returns import GBM
from wealthplan.optimizer.stochastic.survival_process.survival_process import (
    SurvivalProcess,
)


class NeuralAgentWealthOptimizer:
    """
    Neural-network agent-based wealth-consumption optimizer with two assets.

    Features:
    - Savings account: fixed interest rate + deterministic cashflows.
    - Stock asset: stochastic returns via GBM.
    - Actions: monthly consumption and transfers between savings and stocks.
    - Reward: instantaneous utility of consumption multiplied by survival probability.
    - State: [savings, stocks, normalized time, survival probability].
    - Trained via forward simulation and policy gradient (REINFORCE).
    """

    def __init__(
        self,
        run_config_id: str,
        start_date: dt.date,
        end_date: dt.date,
        retirement_date: dt.date,
        initial_wealth: float,
        yearly_return: float,
        beta: float,
        cashflows: List[CashflowBase],
        gbm_returns: GBM,
        survival_process: SurvivalProcess,
        lr: float,
        device: str,
        instant_utility: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """
        Initialize the neural agent optimizer.

        Parameters
        ----------
        run_config_id : str
            Identifier for the current simulation configuration.
        start_date : dt.date
            Simulation start date.
        end_date : dt.date
            Simulation end date.
        retirement_date : dt.date
            Retirement date (may influence cashflows/pensions).
        initial_wealth : float
            Initial wealth in the savings account.
        yearly_return : float
            Annual interest rate for savings account.
        beta : float
            Discount factor for future utility (optional for agent).
        cashflows : List[CashflowBase]
            Deterministic monthly cashflows (salary, rent, utilities, etc.).
        n_sims : int
            Number of Monte Carlo simulations (parallel agents).
        gbm_returns : GBM
            Geometric Brownian motion simulator for stock returns.
        survival_process : SurvivalProcess
            Mortality/survival simulator.
        lr : float
            Learning rate for policy network optimizer.
        device : str
            Torch device for computations ("cpu" or "cuda").
        instant_utility : Callable[[np.ndarray], np.ndarray]
            Vectorized instantaneous utility function.
        """
        self.run_config_id: str = run_config_id
        self.start_date: dt.date = start_date
        self.end_date: dt.date = end_date
        self.retirement_date: dt.date = retirement_date
        self.months: List[dt.date] = [
            d.date() for d in pd.date_range(start=start_date, end=end_date, freq="MS")
        ]
        self.n_months: int = len(self.months)
        self.initial_wealth: float = initial_wealth
        self.yearly_return: float = yearly_return
        self.monthly_return: float = (1 + self.yearly_return) ** (1 / 12) - 1
        self.beta: float = beta
        self.cashflows: List[CashflowBase] = cashflows
        self.gbm_returns: GBM = gbm_returns
        self.survival_process: SurvivalProcess = survival_process
        self.instant_utility: Callable[[np.ndarray], np.ndarray] = instant_utility
        self.device: str = device

        # Neural network agent
        self.policy_net: SimplePolicyNetwork = SimplePolicyNetwork().to(
            self.device
        )
        self.optimizer: optim.Adam = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Output paths (filled after training)
        self.opt_consumption: Optional[pd.DataFrame] = None
        self.opt_savings: Optional[pd.DataFrame] = None
        self.opt_stocks: Optional[pd.DataFrame] = None

    def _build_state_tensor(
            self,
            available_savings: np.ndarray,
            available_stocks: np.ndarray,
            t: int
    ) -> torch.Tensor:
        """
        Construct the batch state tensor for the policy network in a vectorized manner.

        Parameters
        ----------
        available_savings : np.ndarray
            Current savings for each agent (batch_size,)
        available_stocks : np.ndarray
            Current stock holdings for each agent (batch_size,)
        t : int
            Current timestep

        Returns
        -------
        torch.Tensor
            State tensor of shape (batch_size, 4), with:
            - savings fraction
            - stocks fraction
            - log(total wealth)
            - normalized time
        """
        total_wealth = available_savings + available_stocks + 1e-8
        savings_frac = available_savings / total_wealth
        stocks_frac = available_stocks / total_wealth
        log_total_wealth = torch.log(torch.from_numpy(total_wealth).float()).to(self.device)
        t_norm = torch.full((available_savings.shape[0],), t / self.n_months, dtype=torch.float32, device=self.device)

        # Stack features into a single tensor
        state_tensor = torch.stack([
            torch.from_numpy(savings_frac).float().to(self.device),
            torch.from_numpy(stocks_frac).float().to(self.device),
            log_total_wealth,
            t_norm
        ], dim=1)

        return state_tensor

    def _state(
            self,
            savings: float,
            stocks: float,
            t: int
    ) -> torch.Tensor:
        """
        Construct the state tensor for the policy network.

        The state includes both fractional allocations, absolute wealth (log-scaled),
        and normalized time. This allows the network to make decisions based on both
        relative and absolute levels of wealth.

        Parameters
        ----------
        savings : float
            Current savings account balance.
        stocks : float
            Current stock account balance.
        t : int
            Current time step (month index).

        Returns
        -------
        torch.Tensor
            Input tensor of shape (4,) for the policy network:
            [savings_frac, stocks_frac, log_total_wealth, normalized_time].
        """
        total_wealth: float = savings + stocks + 1e-8  # avoid division by zero
        savings_frac: float = savings / total_wealth
        stocks_frac: float = stocks / total_wealth
        log_total_wealth: float = np.log(total_wealth + 1e-8)
        t_norm: float = t / self.n_months

        state_tensor: torch.Tensor = torch.tensor(
            [savings_frac, stocks_frac, log_total_wealth, t_norm],
            dtype=torch.float32
        ).to(self.device)

        return state_tensor

    def _simulate_forward(self, batch_size: int = None, batch_seed: int = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate forward one episode batch using the current policy.

        Stochastic paths (GBM returns and survival) are generated per batch to reduce memory usage
        and optionally introduce different seeds for each batch.

        Parameters
        ----------
        batch_size : int, optional
            Number of simulation paths to generate for this forward pass. Defaults to self.n_sims.
        batch_seed : int, optional
            Seed to initialize stochastic path generation. If None, randomness is used.

        Returns
        -------
        savings_path : np.ndarray
            Array of shape (n_months, batch_size) with monthly savings.
        stocks_path : np.ndarray
            Array of shape (n_months, batch_size) with monthly stocks.
        consumption_path : np.ndarray
            Array of shape (n_months, batch_size) with monthly consumption.
        """
        returns_paths = self.gbm_returns.simulate(n_sims=batch_size, dates=self.months, seed=batch_seed)
        survival_paths = self.survival_process.simulate(n_sims=batch_size, dates=self.months, seed=batch_seed)

        savings_paths = np.zeros((self.n_months, batch_size), dtype=np.float32)
        savings_paths[0] = np.full(batch_size, self.initial_wealth)

        stocks_paths = np.zeros((self.n_months, batch_size), dtype=np.float32)
        consumption_paths = np.zeros((self.n_months, batch_size), dtype=np.float32)

        for t in range(1, self.n_months):
            # ----------------------------
            # Compute deterministic cashflows for this month
            # ----------------------------
            cf_t_before = np.array(
                [sum(cf.cashflow(self.months[t - 1]) for cf in self.cashflows)] * batch_size,
                dtype=np.float32
            )

            # Add cashflows to savings BEFORE taking any actions
            available_savings_before = savings_paths[t - 1] + cf_t_before

            returns_t = returns_paths[:, t]
            alive_mask = survival_paths[:, t - 1]

            # Vectorized state for the whole batch
            state_tensor = self._build_state_tensor(
                available_savings_before,
                stocks_paths[t - 1],
                t - 1
            )

            # Get batch actions
            actions = self.policy_net(state_tensor).detach().cpu().numpy()

            # ----------------------------
            # Compute consumption & transfers
            # ----------------------------
            consumption = actions[:, 0] * available_savings_before
            transfer_s2x = actions[:, 1] * (available_savings_before - consumption)

            available_stocks_before = stocks_paths[t - 1]
            transfer_x2s = actions[:, 2] * available_stocks_before

            # zero out dead agents
            consumption *= alive_mask
            transfer_s2x *= alive_mask
            transfer_x2s *= alive_mask

            # ----------------------------
            # Update asset balances
            # ----------------------------
            # savings after consumption & transfers
            savings_after_rebalancing = available_savings_before - consumption - transfer_s2x + transfer_x2s
            # stocks after transfers
            stocks_after_rebalancing = available_stocks_before + transfer_s2x - transfer_x2s

            # record
            savings_paths[t, :] = savings_after_rebalancing * (1 + self.monthly_return)
            stocks_paths[t, :] = stocks_after_rebalancing * returns_t
            consumption_paths[t - 1, :] = consumption

        return savings_paths, stocks_paths, consumption_paths

    def _compute_reward(self, consumption: np.ndarray, t: int) -> np.ndarray:
        """
        Compute reward at a given time step.

        Reward = instantaneous utility of consumption * survival probability.

        Parameters
        ----------
        consumption : np.ndarray
            Consumption vector for all agents at time t.
        t : int
            Time step (month index).

        Returns
        -------
        np.ndarray
            Reward vector for all agents.
        """
        return self.instant_utility(consumption)

    def train(
            self,
            n_epochs: int = 500,
            batch_size: int = 1000,
            n_batches: int = 10,
            plot: bool = True
    ) -> None:
        """
        Train the policy network using REINFORCE with mini-batch updates.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        batch_size : int
            Number of simulation paths per batch.
        n_batches : int
            Number of batches per epoch.
        plot : bool
            If True, will plot training progress after each epoch.
        """

        epoch_rewards: List[float] = []

        for epoch in trange(n_epochs, desc="Training Epochs"):
            batch_rewards: List[float] = []

            for batch_idx in range(n_batches):
                # simulate batch_size paths
                savings_paths, stocks_paths, consumption_paths = self._simulate_forward(batch_size=batch_size)

                # compute reward
                total_reward = 0.0

                for t in range(self.n_months):
                    total_reward += self._compute_reward(consumption_paths[t, :], t).mean()

                # accumulate rewards for plotting
                batch_rewards.append(total_reward.item())

                # gradient ascent step for this batch
                self.optimizer.zero_grad()
                loss = -torch.tensor(total_reward, requires_grad=True)
                loss.backward()
                self.optimizer.step()

            # store mean reward for this epoch
            epoch_rewards.append(sum(batch_rewards) / n_batches)

        # Save optimal paths
        self.opt_savings = pd.DataFrame(savings_paths, index=self.months)
        self.opt_stocks = pd.DataFrame(stocks_paths, index=self.months)
        self.opt_consumption = pd.DataFrame(consumption_paths, index=self.months)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(epoch_rewards)), epoch_rewards, label="Mean Reward per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Mean Reward")
            plt.title("Neural Agent Training Progress")
            plt.grid(True)
            plt.legend()
            plt.show()

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

        percentiles = sorted(percentiles)

        # --- Pick sample simulation ---
        if sample_sim is None:
            np.random.seed(self.seed)
            sample_sim = np.random.randint(self.n_sims)

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

        if self.dynamic_grid_builder:
            axes[1].plot(
                months,
                self.opt_wealth_det,
                color="tab:cyan",
                lw=1.5,
                label="Optimal Wealth (det.)",
            )

            axes[1].plot(
                months, self.dynamic_grid_builder.upper_bounds, color="tab:grey", lw=2.0
            )

            axes[1].plot(
                months, self.dynamic_grid_builder.lower_bounds, color="tab:grey", lw=2.0
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
        cumulative_survival = np.cumprod(self.survival_probs)

        axes[3].plot(self.age_grid, cumulative_survival, color="tab:orange", lw=2)
        axes[3].set_title("Survival Probability", fontsize=title_size)
        axes[3].set_ylim(0.0, 1.05)
        axes[3].grid(True)
        axes[3].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        axes[3].tick_params(axis="both", labelsize=tick_size)

        plt.tight_layout()
        plt.show()
