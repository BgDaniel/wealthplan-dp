import boto3
import yaml
from typing import Any, Dict


def load_params_from_s3(bucket: str, key: str) -> Dict[str, Any]:
    """
    Fetch a YAML file from an S3 bucket and parse it into a Python dictionary.

    This function connects to AWS S3 using the default credentials
    (environment variables, IAM role, or AWS CLI config), downloads the
    object at the specified bucket/key, and parses its YAML content.

    Args:
        bucket (str): Name of the S3 bucket where the YAML file is stored.
        key (str): Key (path) of the YAML file inside the S3 bucket.

    Returns:
        Dict[str, Any]: The contents of the YAML file parsed into a dictionary.

    Raises:
        botocore.exceptions.ClientError: If the S3 get_object request fails.
        yaml.YAMLError: If the downloaded file is not valid YAML.
    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    yaml_content: bytes = obj['Body'].read()

    return yaml.safe_load(yaml_content)
