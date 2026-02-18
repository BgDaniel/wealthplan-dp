# run_sagemaker_training.py
import uuid
from config.hyperparameters import HyperParameters
from trainer.sagemaker_trainer import SageMakerTrainer

# ----------------------------
# Run ID and YAML config
# ----------------------------
RUN_ID = uuid.uuid4().hex
PARAMS_FILE = "stochastic/neural_agent/lifecycle_params_test.yaml"

# ----------------------------
# Define hyperparameters
# ----------------------------
hyperparams = HyperParameters(
    hidden_layers=[64, 128, 64],
    activation="Softplus",
    dropout=0.1,
    lr=0.001,
    batch_size=5000,
    n_epochs=200,
    n_episodes=10000,
    lambda_penalty=1.0
)

# ----------------------------
# SageMaker role (replace with your own)
# ----------------------------
ROLE_ARN = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

# ----------------------------
# SageMaker training
# ----------------------------
sm_trainer = SageMakerTrainer(
    run_id=RUN_ID,
    config_yaml=PARAMS_FILE,
    hyperparams=hyperparams,
    role=ROLE_ARN
)

sm_trainer.train()
print("SageMaker training job submitted. Check S3 for outputs.")