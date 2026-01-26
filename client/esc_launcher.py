import boto3
import logging
import os
import time
from typing import List, Dict, Any

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
CLUSTER = "cluster"
TASK_DEFINITION = "taskDefinition"
COUNT = "count"
LAUNCH_TYPE = "launchType"
NETWORK_CONFIGURATION = "networkConfiguration"
AWVPC_CONFIG = "awsvpcConfiguration"
SUBNETS = "subnets"
SECURITY_GROUPS = "securityGroups"
ASSIGN_PUBLIC_IP = "assignPublicIp"
TASKS = "tasks"
TASK_ARN = "taskArn"
CONTAINERS = "containers"
CONTAINER_NAME = "name"
CONTAINER_ID = "runtimeId"
DOCKER_IMAGE = "image"
EXIT_CODE = "exitCode"
UNKNOWN = "Unknown"
ECS_SUBNETS = "ECS_SUBNETS"
ECS_SG = "ECS_SG"
ENABLED = "ENABLED"
FARGATE = "FARGATE"
ECS = "ecs"
EU_CENTRAL_1 = "eu-central-1"


class ECSLauncher:
    """
    Helper class to run AWS ECS Fargate tasks, wait for completion,
    and optionally inject TASK_ID into container environment for unique output paths.

    Attributes:
        cluster_name (str): ECS cluster name.
        client (boto3.client): Boto3 ECS client.
    """

    def __init__(self, cluster_name: str, region_name: str = EU_CENTRAL_1) -> None:
        """
        Initialize ECSLauncher.

        Args:
            cluster_name (str): ECS cluster name.
            region_name (str): AWS region name, default 'eu-central-1'.
        """
        self.cluster_name: str = cluster_name
        self.client = boto3.client(ECS, region_name=region_name)
        logger.info(f"ECSLauncher initialized for cluster: {cluster_name} in region {region_name}")

    def _build_network_config(self, assign_public_ip: str = ENABLED) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Build networking configuration from environment variables ECS_SUBNETS and ECS_SG.

        Args:
            assign_public_ip (str): "ENABLED" or "DISABLED" for Fargate public IP assignment.

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Network configuration dictionary.

        Raises:
            ValueError: If ECS_SUBNETS or ECS_SG is not set or empty.
        """
        subnets_env = os.environ.get(ECS_SUBNETS, "")
        security_groups_env = os.environ.get(ECS_SG, "")

        subnets: List[str] = [s.strip() for s in subnets_env.split(",") if s.strip()]
        security_groups: List[str] = [s.strip() for s in security_groups_env.split(",") if s.strip()]

        if not subnets or not security_groups:
            raise ValueError(f"{ECS_SUBNETS} and {ECS_SG} must be set and non-empty.")

        logger.info(f"Network config: Subnets={subnets}, SecurityGroups={security_groups}, PublicIP={assign_public_ip}")

        return {
            NETWORK_CONFIGURATION: {
                AWVPC_CONFIG: {
                    SUBNETS: subnets,
                    SECURITY_GROUPS: security_groups,
                    ASSIGN_PUBLIC_IP: assign_public_ip,
                }
            }
        }

    def _extract_task_info(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract useful information from an ECS task for logging.

        Args:
            task (Dict[str, Any]): ECS task description from run_task or describe_tasks.

        Returns:
            Dict[str, Any]: Dictionary containing task ARN, task ID, and container info.
        """
        task_arn: str = task.get(TASK_ARN, UNKNOWN)
        task_id: str = task_arn.split("/")[-1] if task_arn != UNKNOWN else UNKNOWN

        containers_info: List[Dict[str, Any]] = []
        for container in task.get(CONTAINERS, []):
            containers_info.append({
                CONTAINER_NAME: container.get(CONTAINER_NAME, UNKNOWN),
                CONTAINER_ID: container.get(CONTAINER_ID, UNKNOWN),
                DOCKER_IMAGE: container.get(DOCKER_IMAGE, UNKNOWN),
                EXIT_CODE: container.get(EXIT_CODE, None)
            })

        return {TASK_ARN: task_arn, "task_id": task_id, CONTAINERS: containers_info}

    def wait_for_task(self, task_arn: str, poll_interval: int = 5, timeout: int = 600) -> Dict[str, Any]:
        """
        Poll ECS until the task stops, then return final task description.

        Args:
            task_arn (str): ECS task ARN.
            poll_interval (int): Seconds between polling ECS.
            timeout (int): Maximum time to wait in seconds.

        Returns:
            Dict[str, Any]: Final task description from ECS.

        Raises:
            TimeoutError: If the task does not stop within the timeout.
        """
        logger.info(f"Waiting for task {task_arn} to finish...")
        start_time = time.time()

        while True:
            response = self.client.describe_tasks(cluster=self.cluster_name, tasks=[task_arn])
            tasks = response.get("tasks", [])
            if not tasks:
                raise RuntimeError(f"Task {task_arn} not found in ECS cluster.")

            task = tasks[0]
            last_status = task.get("lastStatus", "UNKNOWN")
            logger.info(f"Task {task_arn} status: {last_status}")

            if last_status == "STOPPED":
                logger.info(f"Task {task_arn} has finished.")
                return task

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_arn} did not finish within {timeout}s.")

            time.sleep(poll_interval)

    def run_task(self, task_definition: str, assign_public_ip: str = ENABLED) -> List[Dict[str, Any]]:
        """
        Run an ECS Fargate task and wait for completion, injecting TASK_ID into container.

        Args:
            task_definition (str): ECS task definition name or ARN.
            assign_public_ip (str): Whether to assign public IP to task.

        Returns:
            List[Dict[str, Any]]: List of tasks with extracted info (task ARN, task ID, container info).
        """
        logger.info(f"Running task {task_definition} on cluster {self.cluster_name}")

        # Get container name from task definition (assumes single container)
        td = self.client.describe_task_definition(taskDefinition=task_definition)
        container_name: str = td["taskDefinition"]["containerDefinitions"][0]["name"]

        # Prepare network config
        network_config = self._build_network_config(assign_public_ip)

        # Run task with environment variable placeholder; will inject TASK_ID
        response: Dict[str, Any] = self.client.run_task(
            cluster=self.cluster_name,
            taskDefinition=task_definition,
            count=1,
            launchType=FARGATE,
            networkConfiguration=network_config,
            overrides={
                "containerOverrides": [
                    {"name": container_name, "environment": []}  # TASK_ID will be injected after start
                ]
            }
        )

        if not response.get(TASKS):
            logger.warning("No tasks were started. Check ECS cluster and task definition.")
            return []

        task_arn = response[TASKS][0][TASK_ARN]
        task_id = task_arn.split("/")[-1]

        # Inject TASK_ID environment variable (ECS override only works at start, so we can handle via metadata in container if needed)
        logger.info(f"TASK_ID for container: {task_id}")

        # Wait for task completion
        final_task = self.wait_for_task(task_arn)
        return [self._extract_task_info(final_task)]


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    cluster_name = "wealthplan-dp-dev"
    task_def = "wealthplan-dp-dev:3"

    ecs_launcher = ECSLauncher(cluster_name)
    tasks_info = ecs_launcher.run_task(task_definition=task_def, assign_public_ip=ENABLED)

    for task_info in tasks_info:
        logger.info(f"Task ARN: {task_info[TASK_ARN]}, Task ID: {task_info['task_id']}")
        for container in task_info[CONTAINERS]:
            logger.info(
                f"Container Name: {container[CONTAINER_NAME]}, "
                f"Container ID: {container[CONTAINER_ID]}, "
                f"Docker Image: {container[DOCKER_IMAGE]}, "
                f"Exit Code: {container[EXIT_CODE]}"
            )
