from typing import Any, Dict

from config.stochastic.stochastic_config_mapper import StochasticConfigMapper, KEY_STOCHASTIC
from wealthplan.optimizer.math_tools.utility_functions import (
    UtilityFunction,
    crra_utility,
    log_utility,
)
from wealthplan.optimizer.stochastic.market_model.gbm_returns import GBM

KEY_GBM_RETURNS: str = "gbm_returns"
KEY_GBM_MU: str = "mu"
KEY_GBM_SIGMA: str = "sigma"
KEY_GBM_SEED: str = "seed"

# ----------------------
# Neural-agent keys
# ----------------------
KEY_NEURAL_AGENT: str = "neural_agent"
KEY_LR: str = "lr"
KEY_DEVICE: str = "device"
KEY_SAVING_MIN: str = "saving_min"
KEY_MAX_WEALTH_FACTOR: str = "max_wealth_factor"

# ----------------------
# Instant utility keys
# ----------------------
KEY_INSTANT_UTILITY: str = "instant_utility"
KEY_UTILITY_TYPE: str = "type"
KEY_UTILITY_PARAMS: str = "params"
KEY_UTILITY_GAMMA: str = "gamma"
KEY_UTILITY_EPSILON: str = "epsilon"

# ----------------------
# Transaction cost keys
# ----------------------
KEY_TRANSACTION_COSTS: str = "transaction_costs"
KEY_TRANSACTION_BUY_PCT: str = "buy_pct"
KEY_TRANSACTION_SELL_PCT: str = "sell_pct"


class NeuralAgentConfigMapper(StochasticConfigMapper):
    """
    Configuration mapper for neural-agent-based stochastic optimization.

    Extends BinTreeStochasticConfigMapper by adding:
    - learning rate (`lr`)
    - computation device (`device`)
    - fixed interest rate (`fixed_interest`)
    - vectorized instant utility function (`instant_utility`)
    """

    @classmethod
    def map_yaml_to_params(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map YAML configuration into parameters required by
        NeuralAgentWealthOptimizer.

        Args:
            data (Dict[str, Any]): Parsed YAML configuration dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing domain parameters,
                            stochastic parameters, and neural-agent-specific parameters.
        """
        # ----------------------
        # Base + stochastic params
        # ----------------------
        params: Dict[str, Any] = super().map_yaml_to_params(data)

        stochastic: Dict[str, Any] = data[KEY_STOCHASTIC]

        # ----------------------
        # GBM returns
        # ----------------------
        gbm_returns: Dict[str, Any] = stochastic[KEY_GBM_RETURNS]

        mu = gbm_returns[KEY_GBM_MU]
        sigma = gbm_returns[KEY_GBM_SIGMA]

        gbm: GBM = GBM(
            mu=mu,
            sigma=sigma
        )

        params.update(
            {
                KEY_GBM_RETURNS: gbm
            }
        )

        # ----------------------
        # Neural-agent parameters
        # ----------------------
        neural_cfg: Dict[str, Any] = data[KEY_NEURAL_AGENT]
        lr: float = neural_cfg[KEY_LR]
        device: str = neural_cfg[KEY_DEVICE]

        saving_min: float = neural_cfg[KEY_SAVING_MIN]
        max_wealth_factor: float = neural_cfg[KEY_MAX_WEALTH_FACTOR]

        # ----------------------
        # Update final parameters
        # ----------------------
        params.update(
            {
                KEY_LR: lr,
                KEY_DEVICE: device,
                KEY_SAVING_MIN: saving_min,
                KEY_MAX_WEALTH_FACTOR: max_wealth_factor
            }
        )

        # ----------------------
        # Transaction costs
        # ----------------------
        transaction_cfg: Dict[str, Any] = data.get(KEY_TRANSACTION_COSTS, {})
        buy_pct: float = transaction_cfg.get(KEY_TRANSACTION_BUY_PCT, 0.0)  # default 0%
        sell_pct: float = transaction_cfg.get(KEY_TRANSACTION_SELL_PCT, 0.0)  # default 0%

        params.update(
            {
                KEY_TRANSACTION_BUY_PCT: buy_pct,
                KEY_TRANSACTION_SELL_PCT: sell_pct
            }
        )

        return params
