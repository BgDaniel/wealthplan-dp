# run_local_training.py
import os
import uuid
import datetime as dt
from typing import Optional, Tuple, Dict, Any

from io_handler.local_io_handler import LocalIOHandler
from scripts.local.stochastic.neural_agent.cache.base_cache import TrainingAgentCache
from scripts.local.stochastic.neural_agent.cache.local_cache import LocalCache
from scripts.local.stochastic.neural_agent.core.hyper_params import HyperParameters
from scripts.local.stochastic.neural_agent.core.train_agent import plot_training_progress, train_agent


EnvVar = str
ENV_OUTPUT_FOLDER: EnvVar = "OUTPUT_FOLDER"


def main() -> None:
    """
    Entry point for running local_training NeuralAgent training.

    This script sets up:
    - Unique run ID
    - Output directory
    - Optional local_training cache
    - Hyperparameters
    - Lifecycle parameters loaded from YAML

    Training results are stored locally, and the agent's wealth evolution
    and neuronal net can be plotted after training.
    """
    # -------------------------
    # Run configuration
    # -------------------------
    run_id: str = uuid.uuid5(uuid.NAMESPACE_DNS, "local_test_run").hex
    params_file: str = "stochastic/neural_agent/lifecycle_params_test.yaml"
    use_cache: bool = True

    hyperparams: HyperParameters = HyperParameters(
        hidden_layers=[64, 128, 64],
        activation="Softplus",
        dropout=0.1,
        lr=0.001,
        n_epochs=50,
        n_episodes=10000,
        lambda_penalty=1.0,
        terminal_penalty=0.05
    )

    # -------------------------
    # Output directory
    # -------------------------
    base_output: Optional[str] = os.getenv(ENV_OUTPUT_FOLDER)

    if base_output is None:
        raise EnvironmentError(f"Environment variable '{ENV_OUTPUT_FOLDER}' not set.")

    output_dir: str = os.path.join(base_output, "neural_agent", run_id)
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Optional cache
    # -------------------------
    cache: Optional[TrainingAgentCache] = None

    if use_cache:
        cache_root: str = os.path.join(base_output, "neural_agent")
        cache = LocalCache(cache_root)

    # -------------------------
    # Load lifecycle parameters
    # -------------------------
    io_handler = LocalIOHandler(params_file_name=params_file)
    life_cycle_dict: Dict[str, Any] = io_handler.load_params()

    # -------------------------
    # Train agent
    # -------------------------
    mean_opt_consumption, trained_agent, epoch_rewards = train_agent(
        run_id=run_id,
        life_cycle_dict=life_cycle_dict,
        hyperparams=hyperparams,
        cache=cache
    )

    print(f"Local training completed. Optimized mean consumption: {mean_opt_consumption:.1f}")

    # -------------------------
    # Plot results
    # -------------------------
    trained_agent.plot_wealth_evolution()
    trained_agent.plot_neuronal_net(inspection_date=dt.date(2026, 7, 1))
    plot_training_progress(epoch_rewards)


if __name__ == "__main__":
    main()