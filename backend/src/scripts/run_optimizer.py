import os

from config.deterministic.bellman_config_mapper import BellmanConfigMapper
from io_handler.s3_io_handler import S3IOHandler
from wealthplan.optimizer.deterministic.deterministic_bellman_optimizer import (
    DeterministicBellmanOptimizer,
)


PARAMS_FILE_ENV = "PARAMS_FILE"
RUN_TASK_ID_ENV = "RUN_TASK_ID"

run_task_id = "run_task_id"


def main(params_file_name: str, run_task_id: str = "") -> None:
    """
    Run deterministic Bellman optimizer for lifecycle planning.

    Args:
        params_file_name (str): Name of the YAML file in S3.
        run_task_id (str): Optional task ID to tag outputs. Defaults to empty string.
    """
    # Use environment variables for bucket and prefixes
    s3_handler = S3IOHandler(params_file_name=params_file_name)

    yaml_dict = s3_handler.load_params()
    params = BellmanConfigMapper.map_yaml_to_params(yaml_dict)

    optimizer = DeterministicBellmanOptimizer(**params)
    optimizer.solve()

    s3_handler.save_results(
        results=optimizer.opt_results,
        run_id=optimizer.run_config_id,
        run_task_id=run_task_id,
    )


if __name__ == "__main__":
    params_file_name = os.environ.get(PARAMS_FILE_ENV, "lifecycle_params.yaml")
    run_task_id = os.environ.get(RUN_TASK_ID_ENV, "")

    main(params_file_name=params_file_name, run_task_id=run_task_id)
