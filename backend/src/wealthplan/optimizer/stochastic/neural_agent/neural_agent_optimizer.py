import datetime as dt
from typing import List, Tuple, Optional
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


def crra_utility_torch(
    consumption: torch.Tensor, gamma: float, eps: float = 1e-8
) -> torch.Tensor:
    """
    CRRA utility in PyTorch, gradient-compatible.

    u(c) = (c + eps)^(1 - gamma) / (1 - gamma)   if gamma != 1
           log(c + eps)                           if gamma == 1
    """
    if gamma == 1.0:
        return torch.log(consumption + eps)
    else:
        return ((consumption + eps) ** (1 - gamma)) / (1 - gamma)


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
        yearly_return: float,
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
        max_wealth: float,
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
            yearly_return=yearly_return,
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

        self.max_wealth: float = max_wealth

        self.max_allowed_wealth = (
            self.max_wealth_factor * (self.initial_wealth - self.saving_min)
            + self.saving_min
        )

        self.initial_savings_fraction = initial_savings_fraction

        self.terminal_penalty = square_penalty

        self.use_cache = use_cache
        self.cache = ResultCache(run_id=run_config_id, enabled=self.use_cache)

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
        return (wealth - self.saving_min) / (self.initial_wealth - self.saving_min)

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
        savings_frac = (available_savings - self.saving_min) / (
            total_wealth - self.saving_min + 1e-8
        )
        # Optional: clip to [0,1] to avoid numerical issues
        savings_frac = torch.clamp(savings_frac, 0.0, 1.0)

        # Min-max scaled total wealth
        wealth_scaled = (total_wealth - self.saving_min) / (
            self.initial_wealth - self.saving_min + 1e-8
        )
        wealth_scaled = torch.clamp(wealth_scaled, 0.0, self.max_wealth_factor)

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
            frac_savings = (savings[mask_exceed] - self.saving_min) / (
                total_wealth[mask_exceed] - self.saving_min
            )
            frac_stocks = stocks[mask_exceed] / (
                total_wealth[mask_exceed] - self.saving_min
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
            consumption = torch.where(alive_mask, consumption, torch.zeros_like(consumption))
            savings_after = torch.where(alive_mask, savings_after, savings_prev)
            stocks_after = torch.where(alive_mask, stocks_after, stocks_prev)

            # ---------- Enforce max wealth AFTER actions ----------
            savings_after, stocks_after, consumption = self._enforce_max_wealth(
                savings_after, stocks_after, consumption
            )

            # ---------- Apply returns ----------
            savings_next = savings_after * (1.0 + self.monthly_return)
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

    def train(
        self,
        n_epochs: int = 500,
        n_episodes: int = 5000,
        lambda_penalty: float = 1.0,
        plot: bool = True,
    ) -> None:
        """
        Train the policy network using fully differentiable
        Monte Carlo policy optimization.

        Each epoch:
            - Simulates `n_episodes` wealth paths
            - Computes CRRA utility
            - Applies softplus penalty to negative wealth
            - Performs one gradient ascent step

        Parameters
        ----------
        n_epochs : int
            Number of training epochs (optimizer steps).
        n_episodes : int
            Number of simulated episodes per epoch.
            This is your Monte Carlo batch size.
        lambda_penalty : float
            Strength of the soft constraint for negative wealth.
            Higher values enforce stronger non-negativity.
        plot : bool
            Whether to plot training progress.
        """

        epoch_rewards: List[float] = []

        for _ in trange(n_epochs, desc="Training Epochs"):
            # === Forward simulation (fully in torch) ===
            savings_paths, stocks_paths, consumption_paths, wealth_paths = (
                self._simulate_forward(batch_size=n_episodes)
            )

            # === Utility ===
            rewards = crra_utility_torch(consumption_paths, self.gamma, self.epsilon)

            # === Softplus penalty for negative wealth ===
            # penalizes only when wealth < 0
            penalty = F.softplus(-wealth_paths)

            # === Objective ===
            total_reward = rewards.mean() - lambda_penalty * penalty.mean()

            # === Gradient ascent ===
            self.optimizer.zero_grad()
            loss = -total_reward
            loss.backward()
            self.optimizer.step()

            epoch_rewards.append(total_reward.item())

        # Convert to DataFrames immediately
        self.opt_savings = pd.DataFrame(
            savings_paths.detach().cpu().numpy(),
            index=self.months
        )

        self.opt_stocks = pd.DataFrame(
            stocks_paths.detach().cpu().numpy(),
            index=self.months
        )

        self.opt_wealth = pd.DataFrame(
            wealth_paths.detach().cpu().numpy(),
            index=self.months
        )

        # consumption is one month shorter
        self.opt_consumption = pd.DataFrame(
            consumption_paths.detach().cpu().numpy(),
            index=self.months
        )

        # === Plot training curve ===
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(epoch_rewards)
            plt.xlabel("Epoch")
            plt.ylabel("Mean Penalized Reward")
            plt.title("Training Progress")
            plt.grid(True)
            plt.show()
