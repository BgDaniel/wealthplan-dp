import datetime as dt
from typing import List, Callable, Tuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import trange

from result_cache.result_cache import ResultCache
from wealthplan.cashflows.cashflow_base import CashflowBase
from wealthplan.optimizer.math_tools.penality_functions import square_penalty
from wealthplan.optimizer.math_tools.utility_functions import crra_utility_numba
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
        yearly_return: float,
        cashflows: List[CashflowBase],
        gbm_returns: GBM,
        survival_model: SurvivalModel,
        gamma: float,
        epsilon: float,
        stochastic: bool,
        lr: float,
        device: str,
        saving_min: float,
        buy_pct: float,
        sell_pct: float,
        max_wealth_factor: float,
        initial_savings_fraction: float,
        use_cache: bool
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
            stochastic=stochastic,
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

    def _get_wealth_scaled(self, wealth: np.ndarray) -> np.ndarray:
        return (wealth - self.saving_min) / (self.initial_wealth - self.saving_min)

    def _get_savings_fraction(
        self, savings: np.ndarray, wealth: np.ndarray
    ) -> np.ndarray:
        return (savings - self.saving_min) / (wealth - self.saving_min)

    def _get_stocks_fraction(
        self, stocks: np.ndarray, wealth: np.ndarray
    ) -> np.ndarray:
        return stocks / (wealth - self.saving_min)


    def _build_state_tensor(
        self, available_savings: np.ndarray, available_stocks: np.ndarray, t: int
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
        total_wealth = available_savings + available_stocks

        savings_frac = self._get_savings_fraction(available_savings, total_wealth)

        # Validate fractions are within [0, 1]
        if np.any(savings_frac < -EPS) or np.any(savings_frac > 1.0 + EPS):
            offending = savings_frac[(savings_frac < 0.0) | (savings_frac > 1.0)]

            raise ValueError(
                f"savings_frac out of bounds [0,1]! Found values: {offending}"
            )

        # Min-max scaled absolute wealth
        wealth_scaled = self._get_wealth_scaled(total_wealth)

        if np.any(wealth_scaled > self.max_wealth_factor + EPS):
            offending = wealth_scaled[wealth_scaled > self.max_wealth_factor + EPS]

            raise ValueError(
                f"Scaled wealth exceeds max_wealth_factor!\n"
                f"Max allowed: {self.max_wealth_factor}, "
                f"Found values: {offending}"
            )

        # Normalized time
        t_norm = torch.full(
            (available_savings.shape[0],),
            t / self.n_months,
            dtype=torch.float32,
            device=self.device,
        )

        # Stack features into a single tensor
        state_tensor = torch.stack(
            [
                torch.from_numpy(savings_frac).float().to(self.device),
                torch.from_numpy(wealth_scaled).float().to(self.device),
                t_norm,
            ],
            dim=1,
        )

        return state_tensor

    def _enforce_max_wealth(
        self, savings: np.ndarray, stocks: np.ndarray, consumption: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        # Identify paths exceeding max wealth factor
        mask_exceed = total_wealth > self.max_allowed_wealth

        if np.any(mask_exceed):
            excess = total_wealth[mask_exceed] - self.max_allowed_wealth

            # Compute fraction of wealth in savings/stocks
            frac_savings = self._get_savings_fraction(
                savings[mask_exceed], total_wealth[mask_exceed]
            )
            frac_stocks = self._get_stocks_fraction(
                stocks[mask_exceed], total_wealth[mask_exceed]
            )

            # Increase consumption
            consumption[mask_exceed] += excess

            # Reduce savings and stocks proportionally
            savings[mask_exceed] -= excess * frac_savings
            stocks[mask_exceed] -= excess * frac_stocks

        # Enforce minimum constraints
        savings = np.maximum(savings, self.saving_min)
        stocks = np.maximum(stocks, 0.0)
        total_wealth = savings + stocks

        if np.any(savings < self.saving_min - EPS):
            raise ValueError(
                f"Some savings fell below the minimum (eps={EPS}): {savings[savings < self.saving_min - EPS]}"
            )

        if np.any(stocks < -EPS):
            raise ValueError(
                f"Some stocks are negative (eps={EPS}): {stocks[stocks < -EPS]}"
            )

        if np.any(total_wealth > self.max_allowed_wealth + EPS):
            offending = total_wealth[total_wealth > self.max_allowed_wealth + EPS]

            raise ValueError(
                f"Total wealth exceeds the maximum allowed wealth.\n"
                f"Max allowed wealth: {self.max_allowed_wealth}\n"
                f"Offending total wealth values: {offending}"
            )

        return savings, stocks, consumption

    def _simulate_forward(
        self, batch_size: int = None, batch_seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        returns_paths = self.gbm_returns.simulate(
            n_sims=batch_size, dates=self.months, seed=batch_seed
        )
        survival_paths = self.survival_model.simulate_survival(
            age_t=self.age_grid, dt=self.dt, n_sims=batch_size
        )

        savings_paths = np.zeros((self.n_months, batch_size), dtype=np.float32)
        savings_paths[0] = np.full(
            batch_size,
            self.initial_wealth * self.initial_savings_fraction,
            dtype=np.float32,
        )

        stocks_paths = np.zeros((self.n_months, batch_size), dtype=np.float32)
        stocks_paths[0] = np.full(
            batch_size,
            self.initial_wealth * (1.0 - self.initial_savings_fraction),
            dtype=np.float32,
        )

        consumption_paths = np.zeros((self.n_months, batch_size), dtype=np.float32)

        wealth_paths = np.zeros((self.n_months, batch_size), dtype=np.float32)
        wealth_paths[0] = np.full(
            batch_size,
            self.initial_wealth,
            dtype=np.float32,
        )

        for t in range(1, self.n_months):
            alive_mask = survival_paths[t, :] == 1

            # ----------------------------
            # Compute deterministic cashflows for this month
            # ----------------------------
            cf_t_before = self.cf[t - 1]

            # Add cashflows to savings BEFORE taking any actions
            available_savings_before = savings_paths[t - 1] + cf_t_before

            returns_t = returns_paths[:, t]

            # ----------------------------
            # Ensure max wealth is not violated after cashflows
            # ----------------------------
            available_savings_before, stocks_paths[t - 1], _ = self._enforce_max_wealth(
                available_savings_before,
                stocks_paths[t - 1],
                consumption=np.zeros_like(
                    available_savings_before
                ),  # no extra consumption yet
            )

            # Vectorized state for the whole batch
            state_tensor = self._build_state_tensor(
                available_savings_before, stocks_paths[t - 1], t - 1
            )

            # Get batch actions
            actions = self.policy_net(state_tensor).detach().cpu().numpy()

            # ----------------------------
            # Compute consumption & transfers
            # ----------------------------
            consumption_rate = actions[:, 0]

            total_wealth = available_savings_before + stocks_paths[t - 1]
            total_available_wealth = total_wealth - self.saving_min

            consumption = consumption_rate * total_available_wealth

            total_available_wealth_after_consumption = (
                total_available_wealth - consumption
            )

            savings_rate = actions[:, 1]

            savings_after_rebalancing = self.saving_min + savings_rate * (
                total_available_wealth_after_consumption - self.saving_min
            )

            stocks_after_rebalancing = (
                total_available_wealth_after_consumption - savings_after_rebalancing
            )

            # ----------------------------
            # Apply transaction costs
            # ----------------------------
            delta_stocks = stocks_after_rebalancing - stocks_paths[t - 1]

            buy_mask = delta_stocks > 0
            sell_mask = delta_stocks < 0

            stocks_after_rebalancing[buy_mask] -= (
                delta_stocks[buy_mask] * self.buy_pct / 100.0
            )
            stocks_after_rebalancing[sell_mask] -= (
                (-delta_stocks[sell_mask]) * self.sell_pct / 100.0
            )

            # Zero consumption for dead agents
            consumption[~alive_mask] = 0.0

            # Keep previous savings/stocks for dead agents
            savings_after_rebalancing[~alive_mask] = savings_paths[t - 1, ~alive_mask]
            stocks_after_rebalancing[~alive_mask] = stocks_paths[t - 1, ~alive_mask]

            # check constraints
            if np.any(savings_after_rebalancing < self.saving_min):
                raise ValueError(
                    f"Some savings fell below the minimum after rebalancing at t={t}: "
                    f"{savings_after_rebalancing[savings_after_rebalancing < self.saving_min]}"
                )

            # record
            savings_next = savings_after_rebalancing * (1 + self.monthly_return)
            stocks_next = stocks_after_rebalancing * returns_t

            # ----------------------------
            # Enforce max wealth for current timestep
            # ----------------------------
            savings_next, stocks_next, consumption = self._enforce_max_wealth(
                savings_next, stocks_next, consumption
            )

            wealth_next = savings_next + stocks_next

            # ----------------------------
            # Record paths
            # ----------------------------
            savings_paths[t, :] = savings_next
            stocks_paths[t, :] = stocks_next
            consumption_paths[t - 1, :] = consumption
            wealth_paths[t, :] = wealth_next

        return savings_paths, stocks_paths, consumption_paths, wealth_paths

    def train(
        self,
        n_epochs: int = 500,
        batch_size: int = 1000,
        n_batches: int = 10,
        plot: bool = True,
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
        self.batch_size = batch_size

        epoch_rewards: List[float] = []

        for _ in trange(n_epochs, desc="Training Epochs"):
            batch_rewards: List[float] = []

            for batch_idx in range(n_batches):
                # simulate batch_size paths
                savings_paths, stocks_paths, consumption_paths, wealth_paths = (
                    self._simulate_forward(batch_size=batch_size)
                )

                # compute reward
                total_reward = 0.0

                for t in range(self.n_months):
                    total_reward += crra_utility_numba(
                        consumption_paths[t, :], self.gamma, self.epsilon
                    ).mean()

                # ----------------------------
                # Apply pathwise terminal penalty
                # ----------------------------
                # survival paths for this batch
                survival_paths = self.survival_model.simulate_survival(
                    age_t=self.age_grid, dt=self.dt, n_sims=batch_size
                )

                terminal_wealth = np.zeros(batch_size, dtype=np.float32)

                for i in range(batch_size):
                    # find first timestep agent dies
                    death_indices = np.where(survival_paths[:, i] == 0)[0]
                    if death_indices.size > 0:
                        death_t = death_indices[0]
                        terminal_wealth[i] = wealth_paths[death_t, i]
                    else:
                        # survived entire simulation
                        terminal_wealth[i] = wealth_paths[-1, i]

                # apply penalty to terminal wealth
                penalty = self.terminal_penalty(terminal_wealth)
                total_reward += penalty.mean()

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
        self.opt_wealth = pd.DataFrame(wealth_paths, index=self.months)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(len(epoch_rewards)), epoch_rewards, label="Mean Reward per Epoch"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Mean Reward")
            plt.title("Neural Agent Training Progress")
            plt.grid(True)
            plt.legend()
            plt.show()
