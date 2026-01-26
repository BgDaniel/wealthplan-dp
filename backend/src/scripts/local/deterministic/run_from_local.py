import uuid

from io_handler.local_io_handler import LocalIOHandler
from config.config_mapper import ConfigMapper, KEY_RUN_TASK_ID
from wealthplan.optimizer.deterministic.deterministic_bellman_optimizer import (
    DeterministicBellmanOptimizer,
)


def main(
    params_file_name: str = "lifecycle_params.yaml",
    run_task_id: str = "",
    plot: bool = True
) -> None:
    """
    Run a deterministic Bellman optimizer for a lifecycle consumption–wealth
    planning problem using parameters loaded from a local YAML file.

    Args:
        params_file_name: Name of the YAML file to load from the local parameters path.
        run_task_id (str): Optional task ID to tag outputs. Defaults to empty string.
        plot: If True, will plot the results after solving.
    """
    # ----------------------------
    # Load configuration from local YAML
    # ----------------------------
    io_handler = LocalIOHandler(params_file_name=params_file_name)

    yaml_dict = io_handler.load_params()

    params = ConfigMapper.map_yaml_to_params(yaml_dict)

    # ----------------------------
    # Run deterministic Bellman solver
    # ----------------------------
    deterministic_optimizer: DeterministicBellmanOptimizer = (
        DeterministicBellmanOptimizer(**params)
    )

    deterministic_optimizer.solve()

    io_handler.save_results(results=deterministic_optimizer.opt_results,
                            run_config_id=deterministic_optimizer.run_config_id,
                            run_task_id=run_task_id)

    if plot:
        deterministic_optimizer.plot()


if __name__ == "__main__":
    run_task_id = uuid.uuid4().hex

    main(params_file_name='lifecycle_params.yaml', run_task_id=run_task_id, plot=True)
