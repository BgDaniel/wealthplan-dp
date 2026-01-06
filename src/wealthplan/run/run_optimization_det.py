from parameter_loader import ParametersLoader
from wealthplan.optimizer.deterministic.deterministic_bellman_optimizer import (
    DeterministicBellmanOptimizer,
)


def main() -> None:
    """
    Run a deterministic Bellman optimizer for a lifecycle consumption–wealth
    planning problem using parameters loaded from a YAML file.
    """
    # ----------------------------
    # Load parameters
    # ----------------------------
    loader = ParametersLoader(filename="lifecycle_params.yaml")
    params = loader.load()

    # ----------------------------
    # Deterministic Bellman solver
    # ----------------------------
    deterministic_optimizer: DeterministicBellmanOptimizer = (
        DeterministicBellmanOptimizer(**params)
    )

    deterministic_optimizer.solve()
    deterministic_optimizer.plot()


if __name__ == "__main__":
    main()
