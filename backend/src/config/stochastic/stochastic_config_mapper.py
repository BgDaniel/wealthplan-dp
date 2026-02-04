from typing import Any, Dict


from config.config_mapper import ConfigMapper, KEY_SIMULATION
from wealthplan.optimizer.stochastic.market_model.gbm_returns import GBM
from wealthplan.optimizer.stochastic.survival_process.survival_process import (
    SurvivalProcess,
)

KEY_STOCHASTIC: str = "stochastic"

KEY_GBM_RETURNS: str = "gbm_returns"
KEY_GBM_MU: str = "mu"
KEY_GBM_SIGMA: str = "sigma"
KEY_GBM_SEED: str = "seed"

KEY_SURVIVAL_PROCESS: str = "survival_process"
KEY_SURVIVAL_B: str = "b"
KEY_SURVIVAL_C: str = "c"
KEY_SURVIVAL_AGE: str = "age"
KEY_SURVIVAL_SEED: str = "seed"

KEY_CURRENT_AGE: str = "current_age"


class StochasticConfigMapper(ConfigMapper):
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

        stochastic: Dict[str, Any] = data[KEY_STOCHASTIC]

        # ----------------------
        # GBM return process
        # ----------------------
        gbm_cfg: Dict[str, Any] = stochastic[KEY_GBM_RETURNS]

        gbm_returns: GBM = GBM(
            mu=gbm_cfg[KEY_GBM_MU],
            sigma=gbm_cfg[KEY_GBM_SIGMA],
        )

        # ----------------------
        # Survival process
        # ----------------------
        surv_cfg: Dict[str, Any] = stochastic[KEY_SURVIVAL_PROCESS]

        survival_process: SurvivalProcess = SurvivalProcess(
            b=surv_cfg[KEY_SURVIVAL_B],
            c=surv_cfg[KEY_SURVIVAL_C],
            age=surv_cfg[KEY_SURVIVAL_AGE]
        )

        params.update(
            {
                KEY_GBM_RETURNS: gbm_returns,
                KEY_SURVIVAL_PROCESS: survival_process,
            }
        )

        current_age = data[KEY_SIMULATION].get(KEY_CURRENT_AGE, 0)

        params[KEY_CURRENT_AGE] = current_age

        return params
