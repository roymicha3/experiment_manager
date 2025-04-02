import os
from omegaconf import OmegaConf
from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment

# Get the current directory and load environment configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
env_config = OmegaConf.load(os.path.join(current_dir, "env.yaml"))

# Initialize environment with workspace relative to example directory
workspace = os.path.join(current_dir, env_config.workspace)
env = Environment(workspace=workspace, config=env_config)

# Create experiment configurations
base_config = OmegaConf.create({
    "model": {
        "type": "resnet18",
        "params": {
            "num_classes": 10,
            "pretrained": True
        }
    },
    "dataset": {
        "name": "cifar10",
        "batch_size": "?"  # Placeholder to be filled by trials
    }
})

trials_config = OmegaConf.create([
    {"learning_rate": 0.1, "batch_size": 32},
    {"learning_rate": 0.01, "batch_size": 64}
])

exp_config = OmegaConf.create({
    "name": "cifar10_resnet",
    "description": "Training ResNet18 on CIFAR-10"
})

# Save configurations in the environment's config directory
OmegaConf.save(base_config, os.path.join(env.config_dir, Experiment.BASE_CONFIG))
OmegaConf.save(trials_config, os.path.join(env.config_dir, Experiment.TRIALS_CONFIG))
OmegaConf.save(exp_config, os.path.join(env.config_dir, Experiment.CONFIG_FILE))

# Create experiment
experiment = Experiment(
    name="cifar10_resnet",
    id=1,
    desc="Training ResNet18 on CIFAR-10 with different learning rates",
    env=env
)

print(f"\nExperiment '{experiment.name}' created successfully!")
print(f"Workspace directory: {experiment.env.workspace}")
print(f"Config directory: {experiment.config_dir_path}")
print("\nBase configuration:")
print(OmegaConf.to_yaml(experiment.base_config))
print("\nTrials configuration:")
print(OmegaConf.to_yaml(experiment.trials_config))
