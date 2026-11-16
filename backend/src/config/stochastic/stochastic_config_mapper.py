import math
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
KEY_SURVIVAL_TABLE: str = "table"
KEY_SURVIVAL_AGE_SHIFT: str = "age_shift"

# Gompertz fits (hazard b * exp(c * age)) to the remaining life expectancy at 40 and 65 of the
# Destatis period life table 2021/23 (men 39.0 / 17.6 years, women 43.6 / 21.0 years; unisex:
# their mean).
SURVIVAL_TABLES: Dict[str, tuple] = {
    "destatis_unisex": (2.592e-05, 0.0945),
    "destatis_male": (3.752e-05, 0.0925),
    "destatis_female": (1.590e-05, 0.0980),
}


def survival_parameters(cfg: Dict[str, Any]) -> tuple:
    """
    Gompertz (b, c) from the configuration: a preset ``table`` (``SURVIVAL_TABLES``) or
    ``custom`` b / c. ``age_shift`` (years) shifts the hazard to younger / older ages: -3 makes
    you 3 years "younger" in mortality terms (rough cohort adjustment of a period table),
    i.e. b -> b * exp(c * age_shift).
    """
    table = cfg.get(KEY_SURVIVAL_TABLE, "custom")
    if table == "custom":
        b, c = float(cfg[KEY_SURVIVAL_B]), float(cfg[KEY_SURVIVAL_C])
    elif table in SURVIVAL_TABLES:
        b, c = SURVIVAL_TABLES[table]
    else:
        raise ValueError(f"unknown survival table {table!r}; use custom or one of {list(SURVIVAL_TABLES)}")
    shift = float(cfg.get(KEY_SURVIVAL_AGE_SHIFT, 0.0))
    return b * math.exp(c * shift), c


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
        b, c = survival_parameters(surv_cfg)
        params[KEY_SURVIVAL_MODEL] = SurvivalModel(b=b, c=c)
        params[KEY_SURVIVAL_PROCESS] = SurvivalProcess(
            b=b, c=c, age=params[KEY_CURRENT_AGE], seed=surv_cfg.get(KEY_SURVIVAL_SEED)
        )

        return params
