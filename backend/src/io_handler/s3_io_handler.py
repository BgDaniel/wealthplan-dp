from pathlib import Path
from typing import Any, Optional, Dict
import os
import yaml
import pandas as pd
import boto3
import logging

from io_handler.io_handler_base import AbstractIOHandler


# Environment variables
S3_BUCKET_ENV = "S3_BUCKET_ENV"             # Single bucket for both params and results
S3_PARAMS_PREFIX_ENV = "S3_PARAMS_PREFIX"   # e.g., "params"
S3_OUTPUT_PREFIX_ENV = "S3_OUTPUT_PREFIX"   # e.g., "output"
PARAMS_FOLDER_ENV = "PARAMS_FOLDER"         # Local folder for uploading params
TMP_FOLDER_ENV = "TMP_FOLDER"               # Temporary folder for downloads/uploads


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class S3IOHandler(AbstractIOHandler):
    """
    S3 IO Handler for loading optimizer parameters and saving results
    to a single S3 bucket with separate prefixes for params and output.

    Attributes
    ----------
    bucket : str
        Name of the S3 bucket used for both parameters and results.
    params_prefix : str
        Prefix (folder) in S3 bucket for parameter files.
    output_prefix : str
        Prefix (folder) in S3 bucket for output results.
    s3 : boto3.client
        Boto3 S3 client.
    params_file_name : str
        Name of the YAML configuration file in S3.
    """

    def __init__(self, params_file_name: str, region: Optional[str] = None) -> None:
        """
        Initialize S3IOHandler.

        Parameters
        ----------
        params_file_name : str
            Name of the YAML configuration file in S3.
        region : Optional[str], default=None
            AWS region (if not provided, boto3 default resolution is used).
        """
        super().__init__(params_file_name=params_file_name)

        self.bucket: str = os.getenv(S3_BUCKET_ENV)

        if not self.bucket:
            raise ValueError(f"Environment variable '{S3_BUCKET_ENV}' must be set")

        self.params_prefix: str = os.getenv(S3_PARAMS_PREFIX_ENV, "params")
        self.output_prefix: str = os.getenv(S3_OUTPUT_PREFIX_ENV, "output")

        logger.info(f"S3 bucket: {self.bucket}, params prefix: {self.params_prefix}, output prefix: {self.output_prefix}")

        self.s3 = boto3.client("s3", region_name=region)

        logger.info("Initialized S3 client")

    def _get_tmp_dir(self) -> Path:
        """
        Get temporary folder path from environment variable and create it if missing.

        Returns
        -------
        Path
            Path to temporary directory.

        Raises
        ------
        EnvironmentError
            If TMP_FOLDER_ENV is not set.
        """
        tmp_folder: Optional[str] = os.getenv(TMP_FOLDER_ENV)

        if not tmp_folder:
            raise EnvironmentError(f"Environment variable '{TMP_FOLDER_ENV}' must be set")

        tmp_dir: Path = Path(tmp_folder)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        return tmp_dir

    def _tmp_file_path(self, filename: str) -> Path:
        """
        Construct a temporary file path.

        Parameters
        ----------
        filename : str
            File name to append to temporary folder.

        Returns
        -------
        Path
            Full path to temporary file.
        """
        return self._get_tmp_dir() / filename

    def load_params(self) -> Dict[str, Any]:
        """
        Download YAML parameter file from S3 and return its contents as a dictionary.

        Returns
        -------
        dict
            Parsed YAML parameters.

        Raises
        ------
        EnvironmentError
            If TMP_FOLDER_ENV is not set.
        """
        s3_key: str = f"{self.params_prefix}/{self.params_file_name}"
        tmp_file: Path = self._tmp_file_path(self.params_file_name)

        logger.info(f"Downloading '{s3_key}' → '{tmp_file}'")

        try:
            self.s3.download_file(self.bucket, s3_key, str(tmp_file))

            with tmp_file.open("r") as f:
                yaml_data: Dict[str, Any] = yaml.safe_load(f)

            logger.info(f"Successfully loaded parameters from '{s3_key}'")

            return yaml_data
        finally:
            if tmp_file.exists():
                tmp_file.unlink()
                logger.debug(f"Deleted temp file: {tmp_file}")

    def upload_params(self, param_file_name: str) -> None:
        """
        Upload a local YAML file to S3 under the parameters prefix.

        Parameters
        ----------
        param_file_name : str
            Name of the local YAML file to upload.

        Raises
        ------
        EnvironmentError
            If PARAMS_FOLDER_ENV is not set.
        FileNotFoundError
            If the local file does not exist.
        """
        params_folder: Optional[str] = os.getenv(PARAMS_FOLDER_ENV)

        if not params_folder:
            raise EnvironmentError(f"Environment variable '{PARAMS_FOLDER_ENV}' must be set")

        local_file: Path = Path(params_folder) / param_file_name

        if not local_file.exists():
            raise FileNotFoundError(f"YAML file not found: {local_file}")

        s3_key: str = f"{self.params_prefix}/{param_file_name}"
        logger.info(f"Uploading '{local_file}' → s3://{self.bucket}/{s3_key}")

        self.s3.upload_file(str(local_file), self.bucket, s3_key)
        logger.info(f"Successfully uploaded parameters → s3://{self.bucket}/{s3_key}")

    def save_results(self, results: pd.DataFrame, run_id: str, run_task_id: str = "") -> None:
        """
        Save a Pandas DataFrame as CSV to S3 under output prefix using run_id.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame to upload.
        run_id : str
            Identifier for the run; used as folder name in S3.
        run_task_id: str
            Optional run task ID for optimization run. (default empty).
        """
        filename: str = "optimization_results.csv"

        if run_task_id != "":
            s3_key: str = f"{self.output_prefix}/{run_id}/{run_task_id}/{filename}"
        else:
            s3_key: str = f"{self.output_prefix}/{run_id}/{filename}"

        tmp_file: Path = self._tmp_file_path(filename)

        logger.info(f"Writing results to temporary file '{tmp_file}'")

        try:
            results.to_csv(tmp_file, header=True, index=False)
            logger.info(f"Uploading results → s3://{self.bucket}/{s3_key}")

            self.s3.upload_file(str(tmp_file), self.bucket, s3_key)

            logger.info(f"Successfully uploaded results → s3://{self.bucket}/{s3_key}")
        finally:
            if tmp_file.exists():
                tmp_file.unlink()
                logger.debug(f"Deleted temp file: {tmp_file}")


if __name__ == "__main__":
    os.environ[S3_BUCKET_ENV] = "wealthplan-dp-dev"
    os.environ[S3_PARAMS_PREFIX_ENV] = "params"
    os.environ[S3_OUTPUT_PREFIX_ENV] = "output"

    handler = S3IOHandler(params_file_name="lifecycle_params.yaml")

    param_file_name = "lifecycle_params.yaml"
    handler.upload_params(param_file_name)

    yaml_data = handler.load_params()
    print(yaml_data)




