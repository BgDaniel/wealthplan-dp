import uuid


from config.deterministic.bellman_config_mapper import BellmanConfigMapper
from io_handler.local_io_handler import LocalIOHandler
from wealthplan.optimizer.deterministic.bellman_optimizer import BellmanOptimizer


def main(
    params_file_name: str = "deterministic/lifecycle_params.yaml",
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

    params = BellmanConfigMapper.map_yaml_to_params(yaml_dict)

    # ----------------------------
    # Run deterministic Bellman solver
    # ----------------------------
    bellman_optimizer: BellmanOptimizer = (
        BellmanOptimizer(**params)
    )

    bellman_optimizer.solve()

    io_handler.save_results(results=bellman_optimizer.opt_results,
                            run_config_id=bellman_optimizer.run_config_id,
                            run_task_id=run_task_id)

    if plot:
        bellman_optimizer.plot()


if __name__ == "__main__":
    run_task_id = uuid.uuid4().hex

    main(params_file_name='deterministic/lifecycle_params.yaml', run_task_id=run_task_id, plot=True)
