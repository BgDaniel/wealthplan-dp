# optuna_hyperparam_search.py
import uuid
import optuna
from train_agent import train_agent

# ----------------------------
# String constants
# ----------------------------
ACT_RELU = "ReLU"
ACT_LEAKY_RELU = "LeakyReLU"
ACT_TANH = "Tanh"

HIDDEN_LAYER_OPTIONS = [[32,32], [64,64], [128,64], [128,128]]
BATCH_SIZE_OPTIONS = [256, 512, 1024]

PARAMS_FILE = "stochastic/neural_agent/lifecycle_params.yaml"


def optimize_hyperparameters(
    params_file_name: str,
    n_trials: int = 20,
    plot_best: bool = False,
    device: str = "cpu"
) -> optuna.study.Study:
    """
    Run Optuna hyperparameter optimization for NeuralAgentWealthOptimizer.

    Args:
        params_file_name: YAML parameter file for simulation
        n_trials: Number of Optuna trials
        plot_best: If True, plot results of the best trial
        device: Torch device ("cpu" or "cuda")

    Returns:
        optuna.study.Study: Completed Optuna study object
    """

    def objective(trial: optuna.Trial) -> float:
        hyperparams = {
            "hidden_layers": trial.suggest_categorical("hidden_layers", HIDDEN_LAYER_OPTIONS),
            "activation": trial.suggest_categorical("activation", [ACT_RELU, ACT_LEAKY_RELU, ACT_TANH]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
            "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZE_OPTIONS)
        }

        run_task_id = uuid.uuid4().hex
        return train_agent(
            hyperparams=hyperparams,
            params_file_name=params_file_name,
            run_task_id=run_task_id,
            plot=False,
            save=False,
            device=device
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:", study.best_trial.params)
    print("Best objective (mean total consumption):", study.best_value)
    return study


# ----------------------
# Main section
# ----------------------
if __name__ == "__main__":
    device = "cpu"
    n_trials = 10
    plot_best = True

    study = optimize_hyperparameters(
        params_file_name=PARAMS_FILE,
        n_trials=n_trials,
        plot_best=plot_best,
        device=device
    )
