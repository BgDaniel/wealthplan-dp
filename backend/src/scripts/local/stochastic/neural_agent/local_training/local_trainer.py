import os
from typing import Optional, Dict

from config.stochastic.neural_agent.neural_agent_config_mapper import NeuralAgentConfigMapper
from io_handler.local_io_handler import LocalIOHandler
from scripts.local.stochastic.neural_agent.train_agent import train_agent
from scripts.local.stochastic.neural_agent.cache.local_cache import LocalTrainingCache
from scripts.local.stochastic.neural_agent.cache.base_cache import TrainingAgentCache

ENV_OUTPUT_FOLDER = "OUTPUT_FOLDER"  # Env var for main output folder


class LocalTrainer:
    """
    Trainer for running NeuralAgent training locally.

    Handles:
    - YAML config loading
    - Output directory creation
    - Optional caching of trained models
    """

    def __init__(
        self,
        run_id: str,
        config_yaml: str,
        hyperparams: Dict,
        device: str = "cpu",
        use_cache: bool = True,
    ) -> None:
        """
        Initialize LocalTrainer.

        Parameters
        ----------
        run_id : str
            Unique identifier for this run.
        config_yaml : str
            Path to lifecycle YAML configuration file.
        hyperparams : Dict
            Hyperparameter dictionary for training.
        device : str, default="cpu"
            Torch device.
        use_cache : bool, default=True
            If True, enable loading/saving models via cache.
        """
        self.run_id: str = run_id
        self.device: str = device
        self.use_cache: bool = use_cache

        base_output: Optional[str] = os.getenv(ENV_OUTPUT_FOLDER)

        if base_output is None:
            raise EnvironmentError(
                f"Environment variable '{ENV_OUTPUT_FOLDER}' not set."
            )

        # ------------------------------------------------------------
        # Output directory
        # ------------------------------------------------------------
        self.output_dir: str = os.path.join(base_output, "neural_agent", self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)

        # ------------------------------------------------------------
        # Optional cache instance
        # ------------------------------------------------------------
        self.cache: Optional[TrainingAgentCache] = None

        if self.use_cache:
            # cache root is the same base output folder
            cache_root = os.path.join(base_output, "neural_agent")
            self.cache = LocalTrainingCache(cache_root)

        # ------------------------------------------------------------
        # Load lifecycle parameters from YAML
        # ------------------------------------------------------------
        io_handler = LocalIOHandler(params_file_name=config_yaml)
        yaml_dict = io_handler.load_params()
        self.life_cycle_params = NeuralAgentConfigMapper.map_yaml_to_params(
            yaml_dict
        )

        self.hyperparams: Dict = hyperparams

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------
    def train_agent(self) -> float:
        """
        Train the agent locally, optionally using cache.

        Returns
        -------
        float
            Mean optimal consumption objective.
        """
        return train_agent(
            run_id=self.run_id,
            life_cycle_params=self.life_cycle_params,
            hyperparams=self.hyperparams,
            device=self.device,
            cache=self.cache
        )