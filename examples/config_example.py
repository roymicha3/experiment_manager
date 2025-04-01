from omegaconf import OmegaConf
from experiment_manager.common.yaml_utils import insert_value, multiply

# Example 1: Using insert_value
base_config = OmegaConf.create({
    "model": {
        "learning_rate": "?",
        "hidden_size": "?"
    }
})

# Insert the same value for all placeholders
insert_value(base_config, 0.001)
print("Config after insert_value:", base_config)

# Example 2: Using multiply to create experiment combinations
model_configs = OmegaConf.create([
    {"model": "resnet18", "batch_size": 32},
    {"model": "resnet34", "batch_size": 16}
])

dataset_configs = OmegaConf.create([
    {"dataset": "cifar10", "augment": True},
    {"dataset": "cifar100", "augment": False}
])

# Generate all combinations
experiment_configs = multiply(model_configs, dataset_configs)
print("\nAll experiment combinations:")
for idx, config in enumerate(experiment_configs):
    print(f"\nExperiment {idx + 1}:")
    print(OmegaConf.to_yaml(config))
