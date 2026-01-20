from pathlib import Path
import time
import uuid
import boto3


class OptimizerClient:
    """
    Client for running WealthPlan optimization jobs in the cloud.

    Responsibilities:
    - Upload input YAML to S3
    - Submit optimizer job
    - Poll job status
    - Download results from S3
    """

    def __init__(
        self,
        bucket: str,
        input_key: str,
        output_prefix: str,
        region: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.input_key = input_key
        self.output_prefix = output_prefix

        self.s3 = boto3.client("s3", region_name=region)

    # ----------------------------
    # Input
    # ----------------------------
    def upload_input(self, local_yaml: Path) -> None:
        """Upload local YAML parameter file to S3."""
        self.s3.upload_file(str(local_yaml), self.bucket, self.input_key)

    # ----------------------------
    # Job submission
    # ----------------------------
    def submit_job(self) -> str:
        """
        Submit the optimizer job.

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())

        # TODO: Replace with ECS / Batch / Lambda call
        print(f"Submitted job {job_id}")
        print(f"Input: s3://{self.bucket}/{self.input_key}")
        print(f"Output: s3://{self.bucket}/{self.output_prefix}")

        return job_id

    # ----------------------------
    # Polling
    # ----------------------------
    def wait_for_completion(self, job_id: str, poll_seconds: int = 10) -> None:
        """Wait until the job has finished."""
        print(f"Waiting for job {job_id}...")

        # Dummy polling logic
        for _ in range(5):
            time.sleep(poll_seconds)
            print("Job still running...")

        print("Job finished.")

    # ----------------------------
    # Output
    # ----------------------------
    def download_results(self, target_dir: Path) -> None:
        """Download all output files from S3."""
        target_dir.mkdir(parents=True, exist_ok=True)

        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.output_prefix,
        )

        for obj in response.get("Contents", []):
            key = obj["Key"]
            filename = target_dir / Path(key).name
            self.s3.download_file(self.bucket, key, str(filename))

    # ----------------------------
    # Full pipeline
    # ----------------------------
    def run(self, local_yaml: Path, target_dir: Path) -> None:
        """Run full optimization pipeline."""
        self.upload_input(local_yaml)

        job_id = self.submit_job()
        self.wait_for_completion(job_id)

        self.download_results(target_dir)
