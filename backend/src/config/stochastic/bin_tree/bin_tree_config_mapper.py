from typing import Any, Dict


from config.config_mapper import KEY_SIMULATION, \
    KEY_TECHNICAL, KEY_W_MAX, KEY_W_STEP, KEY_C_STEP
from config.stochastic.stochastic_config_mapper import StochasticConfigMapper, KEY_STOCHASTIC



KEY_BETA: str = "beta"
KEY_SIGMA: str = "sigma"
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
            KEY_C_STEP: technical[KEY_C_STEP]
        })

        stochastic: Dict[str, Any] = data[KEY_STOCHASTIC]

        params.update(
            {
                KEY_STOCHASTIC: stochastic[KEY_STOCHASTIC],
                KEY_SIGMA:  stochastic[KEY_SIGMA],
                KEY_N_SIMS :  stochastic[KEY_N_SIMS]
            }
        )

        return params
