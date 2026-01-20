import os
from pathlib import Path
from typing import Final

import boto3

from parameterization.parameter_loader import PARAMS_PATH_ENV


#: Name of the S3 bucket that stores lifecycle input files
BUCKET_NAME: Final[str] = "wealthplan-dp"

#: Object key (path) inside the S3 bucket
S3_KEY: Final[str] = "consumption-input/lifecycle_params.yaml"


def upload_lifecycle_params_to_s3(filename: str) -> None:
    """
    Upload a lifecycle parameter YAML file to S3.

    The base directory of the YAML file is read from the environment variable
    specified by ``PARAMS_PATH_ENV``. The file is validated as YAML before
    being uploaded.

    Args:
        filename: Name of the YAML file to upload (e.g. ``"lifecycle_params.yaml"``).

    Raises:
        ValueError: If the required environment variable is not set.
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        botocore.exceptions.ClientError: If the S3 upload fails.
    """
    base_path: str | None = os.getenv(PARAMS_PATH_ENV)

    if base_path is None:
        raise ValueError(
            f"Environment variable '{PARAMS_PATH_ENV}' is not set"
        )

    yaml_path: Path = Path(base_path) / filename

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    s3 = boto3.client("s3")

    s3.upload_file(
        Filename=str(yaml_path),
        Bucket=BUCKET_NAME,
        Key=S3_KEY,
    )

    print(f"Uploaded to s3://{BUCKET_NAME}/{S3_KEY}")


if __name__ == "__main__":
    upload_lifecycle_params_to_s3("lifecycle_params.yaml")
