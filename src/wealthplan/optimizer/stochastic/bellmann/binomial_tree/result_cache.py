import os
import logging
import pickle
import datetime as dt
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Environment / file configuration
# ---------------------------------------------------------------------
ENV_CACHE_FOLDER = "WEALTHPLAN_CACHE_FOLDER"

# Cache entry keys
CACHE_KEY_VALUE_FUNCTION = "value_function"
CACHE_KEY_POLICY = "policy"
CACHE_KEY_R_SQUARED = "r_squared"


class ResultCache:
    """
    Append-only cache for stochastic Bellman backward induction.

    Cache layout:

        <CACHE_FOLDER>/
            ├── run_<run_id_1>/
            │     ├── 2035-01-01.pkl
            │     ├── 2034-12-01.pkl
            │     └── ...
            ├── run_<run_id_2>/
            │     └── ...
            └── ...

    Each date file contains:
        {
            "value_function": np.ndarray,
            "policy": np.ndarray,
            "r_squared": float
        }
    """

    def __init__(self, enabled: bool = True, run_id: Optional[str] = None):
        self.enabled = enabled
        self.run_id = run_id or "default"

    # -----------------------------------------------------------------
    # Path helpers
    # -----------------------------------------------------------------
    def _get_cache_folder(self) -> str:
        base_folder = os.environ.get(ENV_CACHE_FOLDER, "./cache")
        run_folder = f"run_{self.run_id}"

        path = os.path.join(base_folder, run_folder)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_date_file(self, date_t: dt.date) -> str:
        return os.path.join(
            self._get_cache_folder(),
            f"{date_t.isoformat()}.pkl",
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def has(self, date_t: dt.date) -> bool:
        if not self.enabled:
            return False
        return os.path.isfile(self._get_date_file(date_t))

    def load_date(
        self, date_t: dt.date
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Load cached results for a given date.
        """
        if not self.enabled:
            raise RuntimeError("Cache is disabled")

        path = self._get_date_file(date_t)
        logger.info(
            "Loading Bellman cache for %s (run_id=%s)",
            date_t,
            self.run_id,
        )

        with open(path, "rb") as f:
            entry = pickle.load(f)

        return (
            entry[CACHE_KEY_VALUE_FUNCTION],
            entry[CACHE_KEY_POLICY]
        )

    def store_date(
        self,
        date_t: dt.date,
        value_function: np.ndarray,
        policy: np.ndarray
    ) -> None:
        """
        Store results for a single date (O(1) write).
        """
        if not self.enabled:
            return

        path = self._get_date_file(date_t)
        logger.debug(
            "Saving Bellman cache for %s (run_id=%s)",
            date_t,
            self.run_id,
        )

        payload = {
            CACHE_KEY_VALUE_FUNCTION: value_function,
            CACHE_KEY_POLICY: policy
        }

        # Atomic write for safety
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        os.replace(tmp_path, path)
