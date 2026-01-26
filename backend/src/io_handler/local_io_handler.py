import os
from pathlib import Path
from typing import Any
import yaml
import pandas as pd
import logging

from io_handler.io_handler_base import AbstractIOHandler


PARAMETERIZATION_FOLDER_ENV: str = "PARAMETERIZATION_FOLDER"
OUTPUT_FOLDER_ENV: str = "OUTPUT_FOLDER"


# Configure logging
logger = logging.getLogger(__name__)


class LocalIOHandler(AbstractIOHandler):
    """
    Local IO Handler:
    - Loads YAML from local folder defined via environment variable
    - Saves results to a local output directory, optionally organized by run_id
    """

    def __init__(self, params_file_name: str) -> None:
        """
        Initialize the local IO handler.

        Parameters
        ----------
        params_file_name : str
            Name of the YAML configuration file to load.
        """
        super().__init__(params_file_name=params_file_name)

        # Read output directory from environment variable
        output_dir = os.getenv(OUTPUT_FOLDER_ENV)

        if not output_dir:
            raise ValueError(f"Environment variable '{OUTPUT_FOLDER_ENV}' not set")

        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def load_params(self) -> dict[str, Any]:
        """
        Load a YAML configuration file from the local base path (PARAMETERIZATION_FOLDER_ENV)
        and map it to optimizer parameters.

        Returns
        -------
        dict
            Optimizer parameters.
        """
        base_path = os.getenv(PARAMETERIZATION_FOLDER_ENV)

        if not base_path:
            raise ValueError(f"Environment variable '{PARAMETERIZATION_FOLDER_ENV}' not set")

        path = Path(base_path) / self.params_file_name

        if not path.exists():
            raise FileNotFoundError(path)

        logger.info(f"Loading YAML parameter file from: {path}")

        with path.open("r") as f:
            yaml_data = yaml.safe_load(f)

        logger.info(f"Successfully loaded parameters from {path}")

        return yaml_data

    def save_results(self, results: pd.DataFrame, run_config_id: str, run_task_id: str = "") -> None:
        """
        Save a Pandas DataFrame as CSV in the local output directory under a run-specific subfolder.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing results.
        run_config_id : str
            Run identifier; creates a subfolder for this run.
        run_task_id: str
            Optional run task ID for optimization run. (default empty).
        """
        output_dir = self.base_output_dir / run_config_id

        if run_task_id != "":
            output_dir /= run_task_id

        output_dir.mkdir(parents=True, exist_ok=True)

        # Use DataFrame name or fallback to "results.csv"
        path = output_dir / "optimization_results.csv"

        logger.info(f"Saving results DataFrame to: {path}")

        results.to_csv(path, header=True)

        logger.info(f"Successfully saved results → {path}")
