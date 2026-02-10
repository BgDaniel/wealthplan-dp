from typing import Dict, Any

from config.config_mapper import ConfigMapper, \
    KEY_SIMULATION, KEY_BETA, KEY_W_MAX, KEY_W_STEP, KEY_C_STEP


KEY_TECHNICAL: str = "technical"


class BellmanConfigMapper(ConfigMapper):
    """
    Extends BaseConfigMapper with deterministic Bellman solver parameters.

    Adds:
    - wealth grid configuration
    - consumption grid configuration
    - caching behavior
    """

    @classmethod
    def map_yaml_to_params(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map YAML configuration into full Bellman optimizer parameters.

        Args:
            data: Parsed YAML configuration.

        Returns:
            Dictionary containing domain parameters plus
            Bellman solver technical parameters.
        """
        params: Dict[str, Any] = super().map_yaml_to_params(data)

        beta = data[KEY_SIMULATION][KEY_BETA]
        params[KEY_BETA] = beta

        technical: Dict[str, Any] = data[KEY_TECHNICAL]

        params.update({
            KEY_W_MAX: technical[KEY_W_MAX],
            KEY_W_STEP: technical[KEY_W_STEP],
            KEY_C_STEP: technical[KEY_C_STEP],
        })

        return params
