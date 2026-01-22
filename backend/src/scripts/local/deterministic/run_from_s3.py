import os

from io_handler.s3_io_handler import S3IOHandler, S3_BUCKET_ENV, S3_PARAMS_PREFIX_ENV, S3_OUTPUT_PREFIX_ENV
from config.config_mapper import ConfigMapper
from wealthplan.optimizer.deterministic.deterministic_bellman_optimizer import (
    DeterministicBellmanOptimizer,
)


def main(
    params_file_name: str = "lifecycle_params.yaml",
    plot: bool = False
) -> None:
    """
    Run a deterministic Bellman optimizer for a lifecycle consumption–wealth
    planning problem using parameters loaded from a local YAML file.

    Args:
        params_file_name: Name of the YAML file to load from the local parameters path.
        plot: If True, will plot the results after solving.
    """
    # ----------------------------
    # Load configuration from local YAML
    # ----------------------------
    s3_handler = S3IOHandler(params_file_name=params_file_name)

    yaml_dict = s3_handler.load_params()

    params = ConfigMapper.map_yaml_to_params(yaml_dict)

    # ----------------------------
    # Run deterministic Bellman solver
    # ----------------------------
    deterministic_optimizer: DeterministicBellmanOptimizer = (
        DeterministicBellmanOptimizer(**params)
    )

    deterministic_optimizer.solve()

    s3_handler.save_results(results=deterministic_optimizer.opt_results, run_id=deterministic_optimizer.run_id)

    if plot:
        deterministic_optimizer.plot()


if __name__ == "__main__":
    os.environ[S3_BUCKET_ENV] = "wealthplan-dp-dev"
    os.environ[S3_PARAMS_PREFIX_ENV] = "params"
    os.environ[S3_OUTPUT_PREFIX_ENV] = "output"

    main(plot=False)
