import os
import pickle
import datetime as dt
from pathlib import Path
from typing import Dict, Any

# Public keys used by optimizers
VALUE_FUNCTION_KEY: str = "value_function"
POLICY_KEY: str = "policy"

CACHE_FOLDER_ENV: str = "CACHE_FOLDER"


class ResultCache:
    """
    Local filesystem cache for Bellman optimizer results.

    This cache stores intermediate results (value function and policy)
    per time step (date) as pickle files on the local_training filesystem.

    The cache can be enabled or disabled via the ``enabled`` flag.
    When disabled, all cache operations become no-ops.

    The base cache directory can be configured via the environment variable
    ``CACHE_FOLDER``. If not set, ``./.cache`` is used.
    """

    def __init__(self, run_id: str, *, enabled: bool = True) -> None:
        """
        Initialize a local_training result cache for a given optimizer run.

        Parameters
        ----------
        run_id : str
            Unique identifier for the optimizer run.
            Used to namespace cached files.

        enabled : bool, default=True
            Whether the cache is active.
            If False, the cache is fully disabled.
        """
        self.enabled: bool = enabled

        if not self.enabled:
            # No filesystem side effects if cache is disabled
            self.cache_dir: Path | None = None
            return

        base_path: Path = Path(
            os.getenv(CACHE_FOLDER_ENV, "./.cache")
        )

        self.cache_dir = base_path / run_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _date_to_filename(date: dt.date) -> str:
        """
        Convert a date to a cache filename.

        Parameters
        ----------
        date : datetime.date
            Date identifying the optimization step.

        Returns
        -------
        str
            Filename used for caching (ISO date with .pkl suffix).
        """
        return f"{date.isoformat()}.pkl"

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def has(self, date: dt.date) -> bool:
        """
        Check whether cached results exist for a given date.

        If caching is disabled, this always returns False.

        Parameters
        ----------
        date : datetime.date
            Date identifying the optimization step.

        Returns
        -------
        bool
            True if cached results exist and cache is enabled,
            False otherwise.
        """
        if not self.enabled or self.cache_dir is None:
            return False

        return (self.cache_dir / self._date_to_filename(date)).exists()

    def load_date(self, date: dt.date) -> Dict[str, Any]:
        """
        Load cached results for a given date.

        Parameters
        ----------
        date : datetime.date
            Date identifying the optimization step.

        Returns
        -------
        Dict[str, Any]
            Cached params dictionary.

        Raises
        ------
        RuntimeError
            If cache is disabled.
        FileNotFoundError
            If no cache file exists for the given date.
        """
        if not self.enabled or self.cache_dir is None:
            raise RuntimeError("ResultCache is disabled")

        path: Path = self.cache_dir / self._date_to_filename(date)

        with open(path, "rb") as f:
            return pickle.load(f)

    def store_date(self, date_t: dt.date, data: Dict[str, Any]) -> None:
        """
        Store optimization results for a given date in the cache.

        If caching is disabled, this method does nothing.

        Parameters
        ----------
        date_t : datetime.date
            Date identifying the optimization step.

        data : Dict[str, Any]
            Dictionary containing optimization results.
        """
        if not self.enabled or self.cache_dir is None:
            return

        path: Path = self.cache_dir / self._date_to_filename(date_t)

        with open(path, "wb") as f:
            pickle.dump(data, f)
