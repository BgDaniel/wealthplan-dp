from typing import Any, Dict

from config.stochastic.stochastic_config_mapper import StochasticConfigMapper
from wealthplan.optimizer.math_tools.utility_functions import (
    UtilityFunction,
    crra_utility,
    log_utility,
)

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

        # ----------------------
        # Neural-agent parameters
        # ----------------------
        neural_cfg: Dict[str, Any] = data[KEY_NEURAL_AGENT]
        lr: float = neural_cfg[KEY_LR]
        device: str = neural_cfg[KEY_DEVICE]

        saving_min: float = neural_cfg[KEY_SAVING_MIN]
        max_wealth_factor: float = neural_cfg[KEY_MAX_WEALTH_FACTOR]

        # ----------------------
        # Instant utility function
        # ----------------------
        utility_cfg: Dict[str, Any] = data[KEY_INSTANT_UTILITY]
        util_type: str = utility_cfg[KEY_UTILITY_TYPE]
        util_params: Dict[str, Any] = utility_cfg.get(KEY_UTILITY_PARAMS, {})

        instant_utility: UtilityFunction

        if util_type == "crra":
            gamma: float = float(util_params.get(KEY_UTILITY_GAMMA, 1.0))
            epsilon: float = float(util_params.get(KEY_UTILITY_EPSILON, 1e-8))

            # Vectorized NumPy CRRA utility
            instant_utility = lambda c, g=gamma, e=epsilon: crra_utility(c, g, e)
        elif util_type == "log":
            instant_utility = log_utility
        else:
            raise ValueError(f"Unsupported instant utility type: {util_type}")

        # ----------------------
        # Update final parameters
        # ----------------------
        params.update(
            {
                KEY_LR: lr,
                KEY_DEVICE: device,
                KEY_INSTANT_UTILITY: instant_utility,
                KEY_SAVING_MIN: saving_min,
                KEY_MAX_WEALTH_FACTOR: max_wealth_factor
            }
        )

        return params
