import copy
from typing import Any, Dict

from config.config_mapper import KEY_RUN_CONFIG_ID, KEY_SIMULATION, KEY_TECHNICAL, KEY_USE_CACHE
from config.stochastic.stochastic_config_mapper import (
    KEY_N_SIMS_OOS,
    KEY_STOCHASTIC,
    KEY_SURVIVAL_MODEL,
    KEY_SURVIVAL_SEED,
    StochasticConfigMapper,
)
from result_cache.result_cache import params_hash

KEY_W_MAX: str = "w_max"
KEY_N_WEALTH: str = "n_wealth"
KEY_N_EQUITY_SHARES: str = "n_equity_shares"
KEY_GRID_TYPE: str = "grid_type"
KEY_GRID_SHIFT: str = "grid_shift"
KEY_MAX_EQUITY_SHARE: str = "max_equity_share"
KEY_REGRESSION_DEGREE: str = "regression_degree"
KEY_SHORTFALL_PENALTY: str = "shortfall_penalty"
KEY_APPLY_TERMINAL_PENALTY: str = "apply_terminal_penalty"
KEY_PARAMS_HASH: str = "params_hash"

# Part of the cache key: bump whenever the backward induction changes its results.
SOLVER_VERSION: str = "lsmc-11"

TECHNICAL_DEFAULTS: Dict[str, Any] = {
    KEY_W_MAX: 3_000_000.0,
    KEY_N_WEALTH: 400,
    KEY_GRID_TYPE: "log",
    KEY_GRID_SHIFT: 10_000.0,
    KEY_N_EQUITY_SHARES: 11,
    KEY_MAX_EQUITY_SHARE: 1.0,
    KEY_REGRESSION_DEGREE: 0,
    KEY_SHORTFALL_PENALTY: 1.0,
    KEY_APPLY_TERMINAL_PENALTY: False,
}


def solver_params_hash(data: Dict[str, Any]) -> str:
    """
    Hash of everything the backward induction depends on.

    The run name, the current equity holding, the number of out-of-sample paths and the
    survival seed only affect the forward simulation, so changing them reuses the cached
    regression coefficients.
    """
    relevant = copy.deepcopy(data)
    relevant.get(KEY_SIMULATION, {}).pop(KEY_RUN_CONFIG_ID, None)
    relevant.get(KEY_SIMULATION, {}).pop("initial_equity", None)
    stochastic = relevant.get(KEY_STOCHASTIC, {})
    stochastic.pop(KEY_N_SIMS_OOS, None)
    stochastic.get(KEY_SURVIVAL_MODEL, {}).pop(KEY_SURVIVAL_SEED, None)
    # technical defaults included, so a changed default is a different key
    relevant[KEY_TECHNICAL] = {**TECHNICAL_DEFAULTS, **(relevant.get(KEY_TECHNICAL) or {})}
    relevant[KEY_TECHNICAL].pop(KEY_USE_CACHE, None)
    relevant["solver_version"] = SOLVER_VERSION
    return params_hash(relevant)


class LSMCConfigMapper(StochasticConfigMapper):
    """Maps a YAML configuration to the arguments of ``LSMCBellmanOptimizer``."""

    @classmethod
    def map_yaml_to_params(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map YAML configuration into LSMC optimizer parameters.

        Args:
            data: Parsed YAML configuration.

        Returns:
            Keyword arguments for LSMCBellmanOptimizer.
        """
        params: Dict[str, Any] = super().map_yaml_to_params(data)

        technical: Dict[str, Any] = data.get(KEY_TECHNICAL, {})
        for key, default in TECHNICAL_DEFAULTS.items():
            params[key] = type(default)(technical.get(key, default))

        params[KEY_PARAMS_HASH] = solver_params_hash(data)
        return params
