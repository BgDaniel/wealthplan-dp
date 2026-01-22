from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class AbstractIOHandler(ABC):
    """
    Abstract base class for loading configuration and saving results.

    Implementations can load YAML configuration and save Pandas Series results
    either locally or to cloud storage (e.g., S3).
    """
    def __init__(self, params_file_name: str) -> None:
        """
        Initialize the IO handler with the parameter file name.

        Parameters
        ----------
        params_file_name : str
            Name of the YAML configuration file to load.
        """
        self.params_file_name = params_file_name

    @abstractmethod
    def load_params(self) -> dict[str, Any]:
        """
        Load a YAML configuration and map it to optimizer parameters.

        Returns
        -------
        dict
            Dictionary of parameters suitable for initializing the optimizer.
        """
        pass

    @abstractmethod
    def save_results(self, results: pd.DataFrame, run_id: str) -> None:
        """
        Save a Pandas Series or DataFrame to the storage backend.

        Parameters
        ----------
        results : pd.DataFrame
            The Pandas object containing results (e.g., wealth path, consumption).
        run_id : str
            Run identifier; can be used to organize results by run.
        """
        pass
