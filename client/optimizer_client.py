import time
import uuid
from pathlib import Path
import boto3
from s3_file_manager import S3FileManager


class OptimizerClient:
    """
    Client for running WealthPlan optimization jobs in ECS Fargate.
    """

    def __init__(
        self,
        s3_manager: S3FileManager,
        output_prefix: str = "outputs/run-001/",
        region: str | None = None,
        cluster_name: str = "wealthplan-cluster",
        task_definition: str = "wealthplan-optimizer-task",
        container_name: str = "optimizer",
        subnets: list[str] | None = None,
        security_groups: list[str] | None = None,
    ) -> None:
        self.s3_manager = s3_manager
        self.output_prefix = output_prefix

        self.cluster_name = cluster_name
        self.task_definition = task_definition
        self.container_name = container_name
        self.subnets = subnets or ["subnet-xxxxxxxx"]  # replace with your subnets
        self.security_groups = security_groups or ["sg-xxxxxxxx"]  # replace with your SGs

        self.ecs = boto3.client("docker", region_name=region)

    # ----------------------------
    # Job submission
    # ----------------------------
    def submit_job(self) -> str:
        """
        Submit the optimizer job to ECS Fargate.

        Returns:
            task_arn: ECS task ARN as job ID
        """
        job_id = str(uuid.uuid4())

        response = self.ecs.run_task(
            cluster=self.cluster_name,
            launchType="FARGATE",
            taskDefinition=self.task_definition,
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": self.subnets,
                    "securityGroups": self.security_groups,
                    "assignPublicIp": "ENABLED",
                }
            },
            overrides={
                "containerOverrides": [
                    {
                        "name": self.container_name,
                        "environment": [
                            {"name": "JOB_ID", "value": job_id},
                            {"name": "S3_BUCKET", "value": self.s3_manager.bucket},
                            {"name": "S3_INPUT_KEY", "value": self.s3_manager.key},
                            {"name": "S3_OUTPUT_PREFIX", "value": self.output_prefix},
                        ],
                    }
                ]
            },
        )

        task_arn = response["tasks"][0]["taskArn"]
        print(f"Submitted ECS task: {task_arn}")
        return task_arn

    # ----------------------------
    # Polling
    # ----------------------------
    def wait_for_completion(self, task_arn: str, poll_seconds: int = 10) -> None:
        """Poll ECS task status until it stops."""
        print(f"Waiting for ECS task {task_arn} to complete...")
        while True:
            response = self.ecs.describe_tasks(
                cluster=self.cluster_name,
                tasks=[task_arn],
            )
            status = response["tasks"][0]["lastStatus"]
            print(f"Task status: {status}")

            if status == "STOPPED":
                print("Task completed.")
                break
            time.sleep(poll_seconds)

    # ----------------------------
    # Full pipeline
    # ----------------------------
    def run(self, filename: str, target_dir: Path) -> None:
        """
        Run the full pipeline: upload input, submit ECS job, wait, download results.
        """
        self.s3_manager.upload_yaml(filename)
        task_arn = self.submit_job()
        self.wait_for_completion(task_arn)
        self.s3_manager.download_folder(self.output_prefix, target_dir)
