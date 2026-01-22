import os
from pathlib import Path
from typing import Final
import boto3
import yaml
from params.parameter_loader import PARAMS_PATH_ENV


class S3FileManager:
    """
    Handles uploading and downloading YAML files to/from S3.
    """

    def __init__(self, bucket: str, key: str, region: str | None = None) -> None:
        self.bucket = bucket
        self.key = key
        self.s3 = boto3.client("s3", region_name=region)

    def upload_yaml(self, filename: str) -> None:
        """
        Upload a YAML file to S3, using the environment variable for base path.
        Validates the file as YAML before uploading.
        """
        base_path: str | None = os.getenv(PARAMS_PATH_ENV)
        if base_path is None:
            raise ValueError(f"Environment variable '{PARAMS_PATH_ENV}' is not set")

        yaml_path: Path = Path(base_path) / filename
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        # Validate YAML
        with open(yaml_path, "r") as f:
            yaml.safe_load(f)

        self.s3.upload_file(str(yaml_path), self.bucket, self.key)
        print(f"Uploaded {yaml_path} → s3://{self.bucket}/{self.key}")

    def download_folder(self, prefix: str, target_dir: Path) -> None:
        """
        Download all objects from S3 under a prefix to a local directory.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)

        for obj in response.get("Contents", []):
            key = obj["Key"]
            filename = target_dir / Path(key).name
            self.s3.download_file(self.bucket, key, str(filename))
            print(f"Downloaded s3://{self.bucket}/{key} → {filename}")
