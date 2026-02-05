from typing import Any, Dict


from config.config_mapper import KEY_FUNCTIONS, KEY_UTILITY_FUNCTION, KEY_TERMINAL_PENALTY, KEY_SIMULATION, \
    KEY_USE_CACHE, KEY_TECHNICAL, KEY_W_MAX, KEY_W_STEP, KEY_C_STEP
from config.stochastic.neural_agent.neural_agent_config_mapper import KEY_GBM_SEED
from config.stochastic.stochastic_config_mapper import StochasticConfigMapper, KEY_STOCHASTIC

from wealthplan.optimizer.math_tools.utility_functions import crra_utility_numba, \
    log_utility_numba
from wealthplan.optimizer.math_tools.penality_functions import square_penalty


KEY_BETA: str = "beta"
KEY_SIGMA: str = "sigma"
KEY_SEED: str = "seed"
KEY_N_SIMS: str = "n_sims"


class BinTreeConfigMapper(StochasticConfigMapper):
    """
    Extends BaseConfigMapper with parameters required for
    stochastic optimization.

    Adds:
    - GBM return process
    - Survival process
    - Number of Monte Carlo simulations
    """

    @classmethod
    def map_yaml_to_params(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map YAML configuration into stochastic optimizer parameters.

        Args:
            data: Parsed YAML configuration.

        Returns:
            Dictionary containing domain parameters plus
            stochastic process objects and simulation count.
        """
        params: Dict[str, Any] = super().map_yaml_to_params(data)

        beta = data[KEY_SIMULATION][KEY_BETA]
        params[KEY_BETA] = beta

        technical: Dict[str, Any] = data[KEY_TECHNICAL]

        params.update({
            KEY_W_MAX: technical[KEY_W_MAX],
            KEY_W_STEP: technical[KEY_W_STEP],
            KEY_C_STEP: technical[KEY_C_STEP],
            KEY_USE_CACHE: technical[KEY_USE_CACHE],
        })

        stochastic: Dict[str, Any] = data[KEY_STOCHASTIC]

        params.update(
            {
                KEY_SIGMA:  stochastic[KEY_SIGMA],
                KEY_SEED:  stochastic[KEY_SEED],
                KEY_N_SIMS :  stochastic[KEY_N_SIMS]
            }
        )

        # --- Functions ---
        functions: Dict[str, Any] = data.get(KEY_FUNCTIONS, {})

        # Map penalty
        penalty_config = functions.get(KEY_TERMINAL_PENALTY, {})
        penalty_type = penalty_config.get("type", "square").lower()

        if penalty_type == "square":
            params[KEY_TERMINAL_PENALTY] = square_penalty
        else:
            raise ValueError(f"Unknown penalty type: {penalty_type}")

        return params
