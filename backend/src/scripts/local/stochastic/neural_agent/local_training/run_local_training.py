import uuid
import datetime as dt

from scripts.local.stochastic.neural_agent.hyper_params import HyperParameters
from scripts.local.stochastic.neural_agent.local_training.local_trainer import LocalTrainer
from scripts.local.stochastic.neural_agent.train_agent import plot_training_progress

if __name__ == "__main__":
    RUN_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "local_test_run").hex
    PARAMS_FILE = "stochastic/neural_agent/lifecycle_params_test.yaml"

    hyperparams = HyperParameters(
        hidden_layers=[64, 128, 64],
        activation="Softplus",
        dropout=0.1,
        lr=0.001,
        n_epochs=50,
        n_episodes=10000,
        lambda_penalty=1.0,
        terminal_penalty=0.05
    )

    local_trainer = LocalTrainer(
        run_id=RUN_ID,
        config_yaml=PARAMS_FILE,
        hyperparams=hyperparams,
        use_cache=False
    )

    mean_opt_consumption, trained_agent, epoch_rewards = local_trainer.train_agent()
    print(f"Local training completed. Optimized, mean consumption: {mean_opt_consumption:.1f}")

    trained_agent.plot_wealth_evolution()
    trained_agent.plot_neuronal_net(inspection_date=dt.date(2026, 7, 1))

    plot_training_progress(epoch_rewards)
