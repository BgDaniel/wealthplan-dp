import json
from typing import Dict, Any, Optional
import boto3
import io
import logging
import datetime as dt


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DateJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime.date objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, dt.date):
            return obj.isoformat()
        return super().default(obj)


class S3FileHandler:
    """
    Minimal S3 handler for uploading and downloading JSON dictionaries.

    Attributes
    ----------
    bucket : str
        Name of the S3 bucket.
    s3 : boto3.client
        Boto3 S3 client.
    """

    def __init__(self, bucket: str, region: Optional[str] = None) -> None:
        """
        Initialize S3FileHandler.

        Parameters
        ----------
        bucket : str
            Name of the S3 bucket.
        region : Optional[str]
            AWS region of the bucket. If None, uses boto3 default.
        """
        self.bucket: str = bucket
        self.s3: boto3.client = boto3.client("s3", region_name=region)

        logger.info(f"S3FileHandler initialized for bucket: {self.bucket}")

    def upload_dict(self, data: Dict[str, Any], s3_key: str) -> None:
        """
        Upload a dictionary to S3 as a JSON file.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary to upload.
        s3_key : str
            Full S3 key (path + filename) where the JSON will be stored.
        """
        buffer = io.BytesIO(json.dumps(data, indent=2, cls=DateJSONEncoder).encode("utf-8"))

        self.s3.upload_fileobj(buffer, self.bucket, s3_key)

        logger.info(f"Uploaded JSON to s3://{self.bucket}/{s3_key}")

    def download_dict(self, s3_key: str) -> Dict[str, Any]:
        """
        Download a JSON file from S3 and parse it as a dictionary.

        Parameters
        ----------
        s3_key : str
            Full S3 key (path + filename) of the JSON file.

        Returns
        -------
        Dict[str, Any]
            Parsed JSON contents as a dictionary.
        """
        buffer = io.BytesIO()
        self.s3.download_fileobj(self.bucket, s3_key, buffer)

        buffer.seek(0)

        def parse_dates(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: parse_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [parse_dates(v) for v in obj]
            elif isinstance(obj, str):
                try:
                    return dt.date.fromisoformat(obj)
                except ValueError:
                    return obj
            else:
                return obj

        data_raw: Dict[str, Any] = json.load(buffer)
        data: Dict[str, Any] = parse_dates(data_raw)

        logger.info(f"Downloaded JSON from s3://{self.bucket}/{s3_key}")

        return data