from typing import Any, Dict


from config.config_mapper import ConfigMapper, KEY_CURRENT_AGE
from wealthplan.optimizer.stochastic.market_model.gbm_returns import GBM
from wealthplan.optimizer.stochastic.survival_process.survival_model import SurvivalModel
from wealthplan.optimizer.stochastic.survival_process.survival_process import (
    SurvivalProcess,
)

KEY_STOCHASTIC: str = "stochastic"
KEY_SEED: str = "seed"
KEY_YEARLY_RETURN: str = "yearly_return"
KEY_SIGMA: str = "sigma"
KEY_N_SIMS: str = "n_sims"
KEY_N_SIMS_OOS: str = "n_sims_oos"

KEY_GBM_RETURNS: str = "gbm_returns"
KEY_SURVIVAL_MODEL: str = "survival_model"
KEY_SURVIVAL_PROCESS: str = "survival_process"
KEY_SURVIVAL_B: str = "b"
KEY_SURVIVAL_C: str = "c"
KEY_SURVIVAL_SEED: str = "seed"


class StochasticConfigMapper(ConfigMapper):
    """
    Extends ConfigMapper with the stochastic model parts.

    Adds:
    - GBM equity return process
    - Survival model (probabilities) and survival process (simulated death times)
    - Number of Monte Carlo simulations and seed
    """

    @classmethod
    def map_yaml_to_params(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map YAML configuration into stochastic optimizer parameters.

        Args:
            data: Parsed YAML configuration.

        Returns:
            Dictionary containing domain parameters plus
            stochastic process objects and simulation counts.
        """
        params: Dict[str, Any] = super().map_yaml_to_params(data)

        stochastic: Dict[str, Any] = data[KEY_STOCHASTIC]
        seed = stochastic.get(KEY_SEED)
        params[KEY_SEED] = seed
        params[KEY_N_SIMS] = int(stochastic[KEY_N_SIMS])
        params[KEY_N_SIMS_OOS] = int(stochastic.get(KEY_N_SIMS_OOS, stochastic[KEY_N_SIMS]))

        params[KEY_GBM_RETURNS] = GBM(
            yearly_return=float(stochastic[KEY_YEARLY_RETURN]),
            sigma=float(stochastic[KEY_SIGMA]),
            seed=seed,
        )

        surv_cfg: Dict[str, Any] = stochastic[KEY_SURVIVAL_MODEL]
        b, c = float(surv_cfg[KEY_SURVIVAL_B]), float(surv_cfg[KEY_SURVIVAL_C])
        params[KEY_SURVIVAL_MODEL] = SurvivalModel(b=b, c=c)
        params[KEY_SURVIVAL_PROCESS] = SurvivalProcess(
            b=b, c=c, age=params[KEY_CURRENT_AGE], seed=surv_cfg.get(KEY_SURVIVAL_SEED)
        )

        return params
