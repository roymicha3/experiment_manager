# Experiment Manager

A flexible and extensible framework for managing machine learning experiments and trials.

## Features

- **Experiment Management**: Organize and track multiple experiments
- **Trial Management**: Run multiple trials with different configurations and repeats
- **Configuration Management**: YAML-based configuration with inheritance and merging
- **Workspace Organization**: Structured directory layout for artifacts, logs, and configs
- **Logging System**: Hierarchical logging for experiments and trials
- **Resume Support**: Ability to resume experiments from partial completion
- **Error Handling**: Graceful handling of configuration errors and invalid states
- **Pipeline System**: Extensible pipeline architecture with callback support
- **Training Callbacks**: Built-in callbacks for:
  - Early stopping with configurable patience
  - Checkpointing at specified intervals
  - Metric tracking and logging
  - Custom callback support

## Installation

```bash
pip install -e .
```

## Quick Start

1. Create configuration files:

```yaml
# experiment.yaml
name: my_experiment
id: 1
desc: Training a model with different hyperparameters
settings:
  model_type: mlp
  batch_size: 32
  epochs: 10

# base.yaml
settings:
  model_type: mlp
  batch_size: 32
  log_level: INFO

# trials.yaml
- name: small_network
  id: 1
  repeat: 3
  settings:
    hidden_layers: [64, 32]
    learning_rate: 0.001

- name: large_network
  id: 2
  repeat: 3
  settings:
    hidden_layers: [256, 128, 64]
    learning_rate: 0.0001
```

2. Run an experiment:

```python
from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment
from experiment_manager.pipelines import Pipeline
from experiment_manager.pipelines.callbacks import EarlyStopping, CheckpointCallback, MetricsTracker
from omegaconf import OmegaConf

# Create environment
env_config = OmegaConf.create({
    "workspace": "outputs",
    "settings": {"debug": True}
})
env = Environment(workspace="outputs", config=env_config)
env.setup_environment()

# Create experiment
experiment = Experiment(
    name="my_experiment",
    id=1,
    desc="Training with different hyperparameters",
    env=env,
    config_dir_path="path/to/configs"
)

# Set up pipeline with callbacks
pipeline = Pipeline(env)
pipeline.register_callback(EarlyStopping(env, patience=5, min_delta_percent=0.1))
pipeline.register_callback(CheckpointCallback(interval=10, env=env))
pipeline.register_callback(MetricsTracker(env))

# Run experiment
experiment.run()
```

## Project Structure

```
experiment_manager/
├── experiment_manager/
│   ├── __init__.py
│   ├── environment.py    # Environment management
│   ├── experiment.py     # Experiment execution
│   ├── trial.py         # Trial execution
│   ├── logger.py        # Logging utilities
│   ├── pipelines/       # Pipeline and callback system
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── callbacks/
│   │       ├── __init__.py
│   │       ├── callback.py
│   │       ├── early_stopping.py
│   │       ├── checkpoint.py
│   │       └── metric_tracker.py
│   └── common/
│       ├── __init__.py
│       ├── serializable.py  # YAML serialization
│       └── yaml_utils.py    # YAML utilities
├── tests/
│   ├── __init__.py
│   ├── test_experiment.py
│   ├── test_experiment_integration.py
│   └── test_trial.py
├── examples/
│   ├── configs/
│   │   ├── experiment.yaml
│   │   ├── base.yaml
│   │   └── trials.yaml
│   └── run_experiment.py
├── setup.py
├── requirements.txt
└── README.md
```

## Core Components

### Environment

The `Environment` class manages workspace directories and configurations:
- Handles log, artifact, and config directories
- Stores environment configuration
- Provides logging capabilities
- Supports nested environments for trials

### Experiment

The `Experiment` class manages experiment execution:
- Loads configurations from YAML files
- Creates and manages trials
- Handles configuration inheritance and merging
- Supports resuming from partial completion
- Validates configuration integrity

### Trial

The `Trial` class handles individual trial execution:
- Manages trial-specific workspace
- Executes trial with specified configuration
- Handles trial repetitions with unique outputs
- Supports nested trial environments

### Pipeline

The `Pipeline` class provides an extensible training pipeline:
- Manages training workflow
- Supports multiple callbacks
- Tracks training metrics
- Handles early stopping and checkpointing

### Callbacks

Built-in callbacks for common training tasks:
- `EarlyStopping`: Stops training if metrics don't improve
- `CheckpointCallback`: Saves model checkpoints at intervals
- `MetricsTracker`: Tracks and logs training metrics
- Support for custom callbacks via base `Callback` class

## Configuration System

The configuration system uses OmegaConf and supports:
- Base configurations shared across trials
- Experiment-level settings
- Trial-specific configurations
- Configuration inheritance and merging
- Validation of required fields (name, id, etc.)
- Error handling for invalid configurations
- YAML serialization and deserialization
- Dynamic configuration generation:
  - Placeholder replacement with `insert_value()`
  - Cartesian product of configurations with `multiply()`

### YAML Serialization

The framework provides a `YAMLSerializable` base class that enables:
- Automatic registration of serializable classes
- Loading/saving objects from/to YAML
- Factory pattern for creating objects from configuration
- Support for custom serialization logic

Example:
```python
from experiment_manager.common.serializable import YAMLSerializable

@YAMLSerializable.register("MyCallback")
class MyCallback(YAMLSerializable):
    def __init__(self, config: DictConfig = None):
        super().__init__(config)
        # Custom initialization

    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(config)
```

### Configuration Generation

The framework provides utilities for dynamic configuration:

```python
from experiment_manager.common.yaml_utils import insert_value, multiply
from omegaconf import OmegaConf

# Replace placeholders
base_config = OmegaConf.create({
    "learning_rate": "?",
    "batch_size": 32
})
config = insert_value(base_config, 0.001)  # learning_rate becomes 0.001

# Create cartesian product
model_configs = [
    {"model": "mlp", "layers": [64, 32]},
    {"model": "cnn", "filters": [32, 64]}
]
lr_configs = [
    {"learning_rate": 0.1},
    {"learning_rate": 0.01}
]
# Creates 4 configurations combining each model with each learning rate
configs = multiply(model_configs, lr_configs)
```

## Directory Structure

Each experiment creates the following structure:
```
workspace/
├── experiment_name/
│   ├── configs/
│   │   ├── experiment.yaml
│   │   ├── base.yaml
│   │   └── trials.yaml
│   ├── logs/
│   │   └── metrics.log     # Training metrics log
│   ├── artifacts/
│   │   └── checkpoint-*    # Model checkpoints
│   └── trials/
│       ├── trial_1/
│       │   ├── configs/
│       │   │   └── trial.yaml
│       │   ├── logs/
│       │   │   └── trial.log
│       │   ├── artifacts/
│       │   └── run_1/
│       │       ├── logs/
│       │       │   └── run.log
│       │       └── artifacts/
│       └── trial_2/
│           ├── configs/
│           │   └── trial.yaml
│           ├── logs/
│           │   └── trial.log
│           ├── artifacts/
│           └── run_1/
│               ├── logs/
│               │   └── run.log
│               └── artifacts/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## License

MIT License
