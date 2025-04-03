# Experiment Manager

A flexible and extensible framework for managing machine learning experiments and trials.

## Features

- **Experiment Management**: Organize and track multiple experiments
- **Trial Management**: Run multiple trials with different configurations
- **Configuration Management**: YAML-based configuration with inheritance
- **Workspace Organization**: Structured directory layout for artifacts, logs, and configs
- **Logging System**: Hierarchical logging for experiments and trials

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
from omegaconf import OmegaConf

# Create environment
env_config = OmegaConf.create({
    "workspace": "outputs",
    "settings": {"debug": True}
})
env = Environment(workspace="outputs", config=env_config)
env.setup_environment()

# Create and run experiment
exp_config = OmegaConf.create({
    "name": "my_experiment",
    "id": 1,
    "desc": "Training with different hyperparameters",
    "config_dir_path": "path/to/configs"
})
experiment = Experiment.from_config(exp_config, env)
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
│   └── common/
│       ├── __init__.py
│       ├── serializable.py  # YAML serialization
│       └── yaml_utils.py    # YAML utilities
├── tests/
│   ├── __init__.py
│   ├── test_experiment.py
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

### Experiment

The `Experiment` class manages experiment execution:
- Loads configurations from YAML files
- Creates and manages trials
- Handles configuration inheritance

### Trial

The `Trial` class handles individual trial execution:
- Manages trial-specific workspace
- Executes trial with specified configuration
- Handles trial repetitions

## Configuration System

The configuration system uses OmegaConf and supports:
- Base configurations shared across trials
- Experiment-level settings
- Trial-specific configurations
- Configuration inheritance and merging

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
│   ├── artifacts/
│   └── trials/
│       ├── trial_1/
│       │   ├── configs/
│       │   ├── logs/
│       │   └── artifacts/
│       └── trial_2/
│           ├── configs/
│           ├── logs/
│           └── artifacts/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## License

MIT License
