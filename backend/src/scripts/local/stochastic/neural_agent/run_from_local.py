import uuid
import torch

from config.stochastic.neural_agent.neural_agent_config_mapper import NeuralAgentConfigMapper
from io_handler.local_io_handler import LocalIOHandler
from wealthplan.optimizer.stochastic.neural_agent.neural_agent_optimizer import (
    NeuralAgentWealthOptimizer,
)


def main(
    params_file_name: str = "stochastic/neural_agent/lifecycle_params.yaml",
    run_task_id: str = "",
    train: bool = True,
    plot: bool = True
) -> None:
    """
    Run a stochastic neural-agent optimizer for lifecycle
    consumption–investment planning.

    Args:
        params_file_name: YAML parameter file name.
        run_task_id: Optional run identifier.
        train: If True, trains the agent and saves outputs.
               If False, only initializes the optimizer.
        plot: If True, will plot the results after solving.
    """
    # ----------------------------
    # Load configuration
    # ----------------------------
    io_handler = LocalIOHandler(params_file_name=params_file_name)
    yaml_dict = io_handler.load_params()

    params = NeuralAgentConfigMapper.map_yaml_to_params(yaml_dict)

    # ----------------------------
    # Initialize optimizer
    # ----------------------------
    neural_agent_optimizer = NeuralAgentWealthOptimizer(**params)

    # ----------------------------
    # Train agent (optional)
    # ----------------------------
    if train:
        print("Starting training...")
        neural_agent_optimizer.train(n_epochs=500)
        print("Training completed.")

        # ----------------------------
        # Save outputs
        # ----------------------------
        torch.save(
            neural_agent_optimizer.policy_net.state_dict(),
            f"trained_policy_net_{run_task_id}.pth",
        )

        neural_agent_optimizer.opt_savings.to_csv(f"opt_savings_{run_task_id}.csv")
        neural_agent_optimizer.opt_stocks.to_csv(f"opt_stocks_{run_task_id}.csv")
        neural_agent_optimizer.opt_consumption.to_csv(f"opt_consumption_{run_task_id}.csv")

        print("Saved trained model and results.")
    else:
        print("Training skipped (train=False). No outputs saved.")

    if plot:
        neural_agent_optimizer.plot()


if __name__ == "__main__":
    run_task_id = uuid.uuid4().hex
    main(run_task_id=run_task_id, train=True)
