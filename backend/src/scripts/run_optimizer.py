import argparse

from io_handler.s3_io_handler import S3IOHandler
from config.config_mapper import ConfigMapper
from wealthplan.optimizer.deterministic.deterministic_bellman_optimizer import (
    DeterministicBellmanOptimizer,
)


def main(params_file_name: str) -> None:
    """
    Run deterministic Bellman optimizer for lifecycle planning.

    Args:
        params_file_name: Name of the YAML file in S3.
    """
    # Use environment variables for bucket and prefixes
    s3_handler = S3IOHandler(params_file_name=params_file_name)

    yaml_dict = s3_handler.load_params()
    params = ConfigMapper.map_yaml_to_params(yaml_dict)

    optimizer = DeterministicBellmanOptimizer(**params)
    optimizer.solve()

    s3_handler.save_results(results=optimizer.opt_results, run_id=optimizer.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deterministic Bellman optimizer")
    parser.add_argument(
        "--params-file",
        type=str,
        default="lifecycle_params.yaml",
        help="YAML parameteriztion file in S3",
    )
    args = parser.parse_args()

    main(params_file_name=args.params_file)
