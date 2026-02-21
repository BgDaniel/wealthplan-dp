import datetime as dt
import uuid
from typing import List

from config.stochastic.bin_tree.bin_tree_config_mapper import BinTreeConfigMapper

from io_handler.local_io_handler import LocalIOHandler
from wealthplan.cashflows.cashflow_base import CashflowBase



from wealthplan.optimizer.stochastic.binomial_tree.bin_tree_optimizer import BinTreeOptimizer
from wealthplan.optimizer.stochastic.survival_process.survival_model import (
    SurvivalModel,
)


def main(
    params_file_name: str = "stochastic/bin_tree/lifecycle_params.yaml",
    run_task_id: str = "",
    plot: bool = True
) -> None:
    """
    Run a deterministic Bellman optimizer for a lifecycle consumption–wealth
    planning problem using parameters loaded from a local_training YAML file.

    Args:
        params_file_name: Name of the YAML file to load from the local_training parameters path.
        run_task_id (str): Optional task ID to tag outputs. Defaults to empty string.
        plot: If True, will plot the results after solving.
    """
    # ----------------------------
    # Load configuration from local_training YAML
    # ----------------------------
    io_handler = LocalIOHandler(params_file_name=params_file_name)

    yaml_dict = io_handler.load_params()

    params = BinTreeConfigMapper.map_yaml_to_params(yaml_dict)

    # ----------------------------
    # Run stochastic Binomial Tree Bellman solver
    # ----------------------------
    bin_tree_optimizer: BinTreeOptimizer = (
        BinTreeOptimizer(**params)
    )

    bin_tree_optimizer.backward_induction()
    bin_tree_optimizer.roll_forward()

    #io_handler.save_results(results=bin_tree_optimizer.opt_results,
    #                        run_config_id=bin_tree_optimizer.run_config_id,
    #                        run_task_id=run_task_id)

    if plot:
        bin_tree_optimizer.plot()


if __name__ == "__main__":
    run_task_id = uuid.uuid4().hex

    main(params_file_name='stochastic/bin_tree/lifecycle_params_test.yaml', run_task_id=run_task_id, plot=True)
