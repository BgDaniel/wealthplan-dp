import datetime as dt
from typing import List, Tuple, Optional, OrderedDict
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import trange

from result_cache.result_cache import ResultCache
from wealthplan.cashflows.cashflow_base import CashflowBase
from wealthplan.optimizer.math_tools.penality_functions import square_penalty
from wealthplan.optimizer.stochastic.neural_agent.simple_policy_network import (
    SimplePolicyNetwork,
)
from wealthplan.optimizer.stochastic.market_model.gbm_returns import GBM
from wealthplan.optimizer.stochastic.stochastic_optimizer import StochasticOptimizerBase
from wealthplan.optimizer.stochastic.survival_process.survival_model import (
    SurvivalModel,
)
from wealthplan.optimizer.stochastic.survival_process.survival_process import (
    SurvivalProcess,
)

EPS: float = 10e-1


class NeuralAgentWealthOptimizer(StochasticOptimizerBase):
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
        current_age: int,
        retirement_date: dt.date,
        initial_wealth: float,
        yearly_return_savings: float,
        cashflows: List[CashflowBase],
        gbm_returns: GBM,
        survival_model: SurvivalModel,
        gamma: float,
        epsilon: float,
        lr: float,
        device: str,
        saving_min: float,
        buy_pct: float,
        sell_pct: float,
        max_wealth_factor: float,
        initial_savings_fraction: float,
        use_cache: bool,
        seed: Optional[int] = None,
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

        super().__init__(
            run_config_id=run_config_id,
            start_date=start_date,
            end_date=end_date,
            retirement_date=retirement_date,
            initial_wealth=initial_wealth,
            yearly_return_savings=yearly_return_savings,
            cashflows=cashflows,
            survival_model=survival_model,
            current_age=current_age,
            stochastic=False,
            seed=seed,
        )

        self.gbm_returns: GBM = gbm_returns

        self.gamma = gamma
        self.epsilon = epsilon

        self.device: str = device

        # Neural network agent
        self.policy_net: SimplePolicyNetwork = SimplePolicyNetwork().to(self.device)
        self.optimizer: optim.Adam = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.saving_min: float = saving_min

        self.buy_pct = buy_pct
        self.sell_pct = sell_pct

        self.max_wealth_factor: float = max_wealth_factor

        self.max_allowed_wealth = (
            self.max_wealth_factor * (self.initial_wealth - self.saving_min)
            + self.saving_min
        )

        self.initial_savings_fraction = initial_savings_fraction

        self.terminal_penalty = square_penalty

        self.use_cache = use_cache
        self.cache = ResultCache(run_id=run_config_id, enabled=self.use_cache)

    def crra_utility_torch(
            self, consumption: torch.Tensor
    ) -> torch.Tensor:
        """
        CRRA utility in PyTorch, gradient-compatible.

        u(c) = (c + eps)^(1 - gamma) / (1 - gamma)   if gamma != 1
               log(c + eps)                           if gamma == 1
        """
        if self.gamma == 1.0:
            return torch.log(consumption + self.epsilon)
        else:
            return ((consumption + self.epsilon) ** (1 - self.gamma)) / (1 - self.gamma)

    def _get_wealth_scaled(self, wealth: torch.Tensor) -> torch.Tensor:
        """
        Compute min-max scaled wealth for tensor inputs.

        Parameters
        ----------
        wealth : torch.Tensor
            Total wealth tensor of shape (batch_size,)

        Returns
        -------
        torch.Tensor
            Scaled wealth tensor of shape (batch_size,).
        """
        return (wealth - self.saving_min) / (self.max_allowed_wealth - self.saving_min)

    def _get_savings_fraction(
        self, savings: torch.Tensor, wealth: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the fraction of total wealth allocated to savings.

        Parameters
        ----------
        savings : torch.Tensor
            Savings tensor of shape (batch_size,)
        wealth : torch.Tensor
            Total wealth tensor of shape (batch_size,)

        Returns
        -------
        torch.Tensor
            Fraction of wealth in savings, shape (batch_size,).
        """
        return (savings - self.saving_min) / (wealth - self.saving_min)

    def _get_stocks_fraction(
        self, stocks: torch.Tensor, wealth: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the fraction of total wealth allocated to stocks.

        Parameters
        ----------
        stocks : torch.Tensor
            Stocks tensor of shape (batch_size,)
        wealth : torch.Tensor
            Total wealth tensor of shape (batch_size,)

        Returns
        -------
        torch.Tensor
            Fraction of wealth in stocks, shape (batch_size,).
        """
        return stocks / (wealth - self.saving_min)

    def _build_state_tensor(
        self, available_savings: torch.Tensor, available_stocks: torch.Tensor, t: int
    ) -> torch.Tensor:
        """
        Construct the batch state tensor for the policy network in a vectorized manner.
        Fully tensorized, gradient-compatible.

        Parameters
        ----------
        available_savings : torch.Tensor
            Current savings for each agent (batch_size,)
        available_stocks : torch.Tensor
            Current stock holdings for each agent (batch_size,)
        t : int
            Current timestep

        Returns
        -------
        torch.Tensor
            State tensor of shape (batch_size, 3), with:
            - savings fraction
            - scaled total wealth
            - normalized time
        """
        total_wealth = available_savings + available_stocks

        # Compute fractions using tensor operations
        savings_frac = self._get_savings_fraction(available_savings, total_wealth)

        # Scaled total wealth
        wealth_scaled = self._get_wealth_scaled(total_wealth)

        # Normalized time
        batch_size = available_savings.shape[0]

        t_norm = torch.full(
            (batch_size,), t / self.n_months, dtype=torch.float32, device=self.device
        )

        # Stack features
        state_tensor = torch.stack([savings_frac, wealth_scaled, t_norm], dim=1)

        return state_tensor

    def _enforce_max_wealth(
        self, savings: torch.Tensor, stocks: torch.Tensor, consumption: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enforce maximum allowed wealth by proportionally increasing consumption
        and reducing savings & stocks if scaled total wealth exceeds max_wealth_factor.

        Args:
            savings (np.ndarray): Current savings (batch_size,)
            stocks (np.ndarray): Current stock holdings (batch_size,)
            consumption (np.ndarray): Current consumption (batch_size,)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Adjusted (savings, stocks, consumption)

        Raises:
            ValueError: If final values violate the wealth boundaries beyond eps tolerance.
        """
        # Compute scaled total wealth
        total_wealth = savings + stocks
        mask_exceed = total_wealth > self.max_allowed_wealth

        if mask_exceed.any():
            excess = total_wealth[mask_exceed] - self.max_allowed_wealth

            frac_savings = self._get_savings_fraction(
                savings[mask_exceed], total_wealth[mask_exceed]
            )
            frac_stocks = self._get_stocks_fraction(
                stocks[mask_exceed], total_wealth[mask_exceed]
            )

            consumption[mask_exceed] += excess
            savings[mask_exceed] -= excess * frac_savings
            stocks[mask_exceed] -= excess * frac_stocks

        # Enforce minimum constraints
        savings = torch.maximum(
            savings, torch.tensor(self.saving_min, device=savings.device)
        )
        stocks = torch.maximum(stocks, torch.tensor(0.0, device=stocks.device))

        return savings, stocks, consumption

    def _simulate_forward(
        self, batch_size: int = None, batch_seed: int = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # ---------- Storage lists (graph safe) ----------
        savings_list: list[torch.Tensor] = []
        stocks_list: list[torch.Tensor] = []
        wealth_list: list[torch.Tensor] = []
        consumption_list: list[torch.Tensor] = []

        # ---------- Initial allocation ----------
        savings_prev = torch.full(
            (batch_size,),
            self.initial_wealth * self.initial_savings_fraction,
            device=self.device,
            dtype=torch.float32,
        )

        stocks_prev = torch.full(
            (batch_size,),
            self.initial_wealth * (1.0 - self.initial_savings_fraction),
            device=self.device,
            dtype=torch.float32,
        )

        wealth_prev = savings_prev + stocks_prev

        savings_list.append(savings_prev)
        stocks_list.append(stocks_prev)
        wealth_list.append(wealth_prev)

        # ---------- Stochastic paths ----------
        returns_paths = torch.tensor(
            self.gbm_returns.simulate(
                n_sims=batch_size, dates=self.months, seed=batch_seed
            ),
            device=self.device,
            dtype=torch.float32,
        )

        survival_paths = torch.tensor(
            self.survival_model.simulate_survival(
                age_t=self.age_grid, dt=self.dt, n_sims=batch_size
            ),
            device=self.device,
            dtype=torch.float32,
        )

        # ---------- Forward simulation ----------
        for t in range(1, self.n_months):
            alive_mask = survival_paths[t] > 0.0

            # deterministic cashflows
            cf_t = torch.tensor(self.cf[t - 1], device=self.device, dtype=torch.float32)

            savings_before = savings_prev + cf_t
            stocks_before = stocks_prev

            # enforce max wealth BEFORE actions
            savings_before, stocks_before, _ = self._enforce_max_wealth(
                savings_before,
                stocks_before,
                torch.zeros_like(savings_before),
            )

            # ---------- State for policy ----------
            state_tensor = self._build_state_tensor(
                savings_before,
                stocks_before,
                t - 1,
            )

            actions = self.policy_net(state_tensor)
            consumption_rate = actions[:, 0]
            savings_rate = actions[:, 1]

            # ---------- Compute portfolio ----------
            total_wealth = savings_before + stocks_before
            available = total_wealth - self.saving_min

            consumption = consumption_rate * available
            savings_after = self.saving_min + savings_rate * (available - consumption)
            stocks_after = available - consumption - savings_after

            # ---------- Transaction costs (NO inplace!) ----------
            delta = stocks_after - stocks_before

            buy_cost = torch.where(delta > 0, delta * self.buy_pct / 100.0, 0.0)
            sell_cost = torch.where(delta < 0, (-delta) * self.sell_pct / 100.0, 0.0)

            stocks_after_transaction_costs = stocks_after - buy_cost - sell_cost

            # ---------- Death handling (NO inplace masking) ----------
            consumption = torch.where(
                alive_mask, consumption, torch.zeros_like(consumption)
            )
            savings_after = torch.where(alive_mask, savings_after, savings_prev)
            stocks_after = torch.where(alive_mask, stocks_after, stocks_prev)

            # ---------- Enforce max wealth AFTER actions ----------
            savings_after, stocks_after, consumption = self._enforce_max_wealth(
                savings_after, stocks_after, consumption
            )

            # ---------- Apply returns ----------
            savings_next = savings_after * (1.0 + self.monthly_return_savings)
            stocks_next = stocks_after * returns_paths[:, t]

            wealth_next = savings_next + stocks_next

            # ---------- Append to lists ----------
            savings_list.append(savings_next)
            stocks_list.append(stocks_next)
            wealth_list.append(wealth_next)
            consumption_list.append(consumption)

            # ---------- Move forward ----------
            savings_prev = savings_next
            stocks_prev = stocks_next

        # ---------- Stack results ----------
        savings_paths = torch.stack(savings_list)
        stocks_paths = torch.stack(stocks_list)
        wealth_paths = torch.stack(wealth_list)
        consumption_paths = torch.stack(consumption_list)

        # add final zero consumption month
        zero_last = torch.zeros_like(consumption_paths[0]).unsqueeze(0)
        consumption_paths = torch.cat([consumption_paths, zero_last], dim=0)

        return savings_paths, stocks_paths, consumption_paths, wealth_paths

    def plot_wealth_evolution(
        self,
        percentiles=(5, 10),
        sample_sim=None,
        title_size=22,
        legend_size=14,
        tick_size=14,
    ):
        """
        Child class plot with 2 independent figures:

        Figure 1: Consumption + survival + investment/withdrawal
        Figure 2: Wealth, Savings, Stocks (3 row subplots)

        Legends below each row; confidence bands preserved.
        """
        months = self.months
        n_sims = len(self.opt_wealth.columns)

        if sample_sim is None:
            sample_sim = np.random.randint(n_sims)

        # ---------------------
        # FIGURE 1: consumption + survival + investment/withdrawal
        # ---------------------
        fig1, ax_list1 = plt.subplots(2, 1, figsize=(14, 10))

        # ---- Row 1: Consumption + Survival ----
        ax = ax_list1[0]
        cons_mean, cons_bands = self.compute_mean_and_bands(
            self.opt_consumption, percentiles
        )
        cons_sample = self.opt_consumption.iloc[:, sample_sim]

        ax.plot(
            months,
            cons_mean,
            color="tab:green",
            lw=2,
            linestyle="--",
            label="Mean Consumption",
        )

        for i, (p, (lo, hi)) in enumerate(cons_bands.items()):
            ax.fill_between(
                months,
                lo,
                hi,
                color="tab:green",
                alpha=0.15 + 0.15 * i,
                label=f"{p}-{100 - p}% Consumption Band",
            )

        ax.plot(
            months,
            cons_sample,
            color="red",
            lw=1,
            alpha=0.8,
            label="Sample Consumption",
        )

        ax.plot(
            months,
            self.cf,
            color="blue",
            lw=1,
            linestyle="--",
            label="Deterministic Cashflows",
        )

        ax2 = ax.twinx()
        ax2.plot(
            months,
            self.survival_probabilities,
            color="tab:orange",
            lw=2,
            label="Survival Probability",
        )
        ax2.set_ylim(0, 1.1)
        ax.set_title("Consumption & Survival Probability", fontsize=title_size)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=legend_size, ncol=3
        )

        # ---- Row 2: Investment / Withdrawal ----
        ax = ax_list1[1]
        inv_df = self.cf[:, np.newaxis] - self.opt_consumption.values
        inv_mean, inv_bands = self.compute_mean_and_bands(
            pd.DataFrame(inv_df), percentiles
        )
        inv_sample = pd.DataFrame(inv_df).iloc[:, sample_sim]
        ax.plot(
            months,
            inv_mean,
            color="tab:purple",
            lw=2,
            linestyle="--",
            label="Mean Investment",
        )

        for i, (p, (lo, hi)) in enumerate(inv_bands.items()):
            ax.fill_between(
                months,
                lo,
                hi,
                color="tab:purple",
                alpha=0.15 + 0.15 * i,
                label=f"{p}-{100 - p}% Investment Band",
            )

        ax.plot(
            months,
            inv_sample,
            color="brown",
            lw=1,
            alpha=0.8,
            label="Sample Investment",
        )
        ax.axhline(0.0, color="black", lw=1, alpha=0.7)
        ax.set_title("Monthly Investment / Withdrawal", fontsize=title_size)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=legend_size, ncol=3
        )

        fig1.tight_layout()

        # ---------------------
        # FIGURE 2: Wealth, Savings, Stocks
        # ---------------------
        fig2, ax_list2 = plt.subplots(3, 1, figsize=(14, 12))

        # ---- Row 1: Wealth ----
        ax = ax_list2[0]
        wealth_mean, wealth_bands = self.compute_mean_and_bands(
            self.opt_wealth, percentiles
        )
        wealth_sample = self.opt_wealth.iloc[:, sample_sim]
        ax.plot(
            months,
            wealth_mean,
            color="tab:blue",
            lw=2,
            linestyle="--",
            label="Mean Wealth",
        )

        for i, (p, (lo, hi)) in enumerate(wealth_bands.items()):
            ax.fill_between(
                months,
                lo,
                hi,
                color="tab:blue",
                alpha=0.15 + 0.15 * i,
                label=f"{p}-{100 - p}% Wealth Band",
            )

        ax.plot(
            months, wealth_sample, color="red", lw=1, alpha=0.7, label="Sample Wealth"
        )
        ax.set_title("Wealth Over Time", fontsize=title_size)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=legend_size, ncol=3
        )

        # ---- Row 2: Savings ----
        ax = ax_list2[1]
        savings_mean, savings_bands = self.compute_mean_and_bands(
            self.opt_savings, percentiles
        )
        savings_sample = self.opt_savings.iloc[:, sample_sim]
        ax.plot(
            months,
            savings_mean,
            color="orange",
            lw=2,
            linestyle="--",
            label="Mean Savings",
        )

        for i, (p, (lo, hi)) in enumerate(savings_bands.items()):
            ax.fill_between(
                months,
                lo,
                hi,
                color="orange",
                alpha=0.15 + 0.15 * i,
                label=f"{p}-{100 - p}% Savings Band",
            )
        ax.plot(
            months, savings_sample, color="red", lw=1, alpha=0.8, label="Sample Savings"
        )
        ax.set_title("Savings Over Time", fontsize=title_size)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=legend_size, ncol=3
        )

        # ---- Row 3: Stocks ----
        ax = ax_list2[2]
        stocks_mean, stocks_bands = self.compute_mean_and_bands(
            self.opt_stocks, percentiles
        )
        stocks_sample = self.opt_stocks.iloc[:, sample_sim]

        ax.plot(
            months,
            stocks_mean,
            color="purple",
            lw=2,
            linestyle="--",
            label="Mean Stocks",
        )

        for i, (p, (lo, hi)) in enumerate(stocks_bands.items()):
            ax.fill_between(
                months,
                lo,
                hi,
                color="purple",
                alpha=0.15 + 0.15 * i,
                label=f"{p}-{100 - p}% Stocks Band",
            )
        ax.plot(
            months, stocks_sample, color="brown", lw=1, alpha=0.8, label="Sample Stocks"
        )
        ax.set_title("Stocks Over Time", fontsize=title_size)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=legend_size, ncol=3
        )

        fig2.tight_layout()
        plt.show()

    def plot_neuronal_net(
        self,
        inspection_date: dt.date,
        n_wealth: int = 500,
        n_savings_frac: int = 500,
    ):
        """
        Plot the trained neural network policy as 3D surfaces for consumption and savings actions.
        Two-row layout, cleaner axes, shades of red/blue, formatted ticks.
        """

        # Find time index
        t_idx = np.where(np.array(self.months) == inspection_date)[0]
        if len(t_idx) == 0:
            raise ValueError(f"inspection_date {inspection_date} not in months")
        t_idx = t_idx[0]

        # Create grid
        wealth_scaled_grid = np.linspace(0.0, 1.0, n_wealth)
        savings_frac_grid = np.linspace(0.0, 1.0, n_savings_frac)
        W_scaled, S_frac = np.meshgrid(wealth_scaled_grid, savings_frac_grid)

        # Flatten for batch
        W_flat = W_scaled.ravel()
        S_frac_flat = S_frac.ravel()

        # Normalized time feature
        t_norm = np.full_like(W_flat, fill_value=t_idx / self.n_months, dtype=np.float32)

        # Build state tensor
        state_tensor = torch.from_numpy(
            np.stack([S_frac_flat, W_flat, t_norm], axis=1)
        ).float().to(next(self.policy_net.parameters()).device)

        # Actions from policy network
        self.policy_net.eval()
        actions = self.policy_net(state_tensor).detach().cpu().numpy()
        consumption_rate = actions[:, 0].reshape(n_savings_frac, n_wealth)
        savings_rate = actions[:, 1].reshape(n_savings_frac, n_wealth)

        # -------------------------
        # Plot in two rows
        # -------------------------
        fig = plt.figure(figsize=(14, 12))

        # ---- Row 1: Consumption ----
        ax1 = fig.add_subplot(2, 1, 1, projection="3d")
        surf1 = ax1.plot_surface(
            W_scaled,
            S_frac,
            consumption_rate,
            cmap="Reds",
            linewidth=0,
            antialiased=True,
        )
        ax1.set_xlabel("Wealth (scaled)")
        ax1.set_ylabel("Savings Fraction")
        ax1.set_zlabel("Consumption Rate")
        ax1.set_title("Policy: Consumption Rate", pad=20)
        ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
        ax1.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
        ax1.zaxis.set_major_formatter(lambda v, _: f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.2e}")
        fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=15)

        # ---- Row 2: Savings Transfer ----
        ax2 = fig.add_subplot(2, 1, 2, projection="3d")
        surf2 = ax2.plot_surface(
            W_scaled,
            S_frac,
            savings_rate,
            cmap="Blues",
            linewidth=0,
            antialiased=True,
        )
        ax2.set_xlabel("Wealth (scaled)")
        ax2.set_ylabel("Savings Fraction")
        ax2.set_zlabel("Savings Transfer Rate")
        ax2.set_title("Policy: Savings Transfer Rate", pad=20)
        ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
        ax2.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
        ax2.zaxis.set_major_formatter(lambda v, _: f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.2e}")
        fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=15)

        # Super title with space above
        fig.suptitle(f"Policy surfaces at inspection date: {inspection_date}", fontsize=16, y=0.95)
        plt.tight_layout(pad=3.0)
        plt.show()

