# Experiment Manager

A Python package for managing machine learning experiments and trials. This package provides a structured way to organize experiments, handle configurations, and manage outputs.

## Features

- **Environment Management**: Organize workspace directories for logs, artifacts, and configurations
- **Experiment Configuration**: Support for YAML-based configuration management using OmegaConf
- **Trial Management**: Run multiple trials with different parameters
- **Serialization Support**: Built-in YAML serialization for experiment components

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
experiment_manager/
├── experiment_manager/     # Main package directory
│   ├── __init__.py
│   ├── environment.py     # Environment management
│   ├── experiment.py      # Experiment handling
│   └── common/
│       ├── __init__.py
│       ├── serializable.py # YAML serialization utilities
│       └── yaml_utils.py   # YAML configuration utilities
├── examples/              # Example usage
│   ├── experiment_example.py
│   └── config_example.py
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_environment.py
│   └── test_experiment.py
├── requirements.txt      # Package dependencies
└── setup.py             # Package configuration
```

## Usage

### Basic Example

```python
from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment
from omegaconf import OmegaConf

# Create environment configuration
env_config = OmegaConf.create({
    "workspace": "experiments",
    "log_level": "INFO"
})

# Initialize environment
env = Environment(workspace="./experiments", config=env_config)
env.setup_environment()

# Create experiment
experiment = Experiment(
    name="my_experiment",
    id=1,
    desc="Test experiment",
    env=env
)

# Run experiment
experiment.run()
```

### Configuration Files

The package uses three main configuration files:

1. `experiment.yaml`: Main experiment configuration
2. `base.yaml`: Base model and training parameters
3. `trials.yaml`: Trial-specific parameters

Example configuration structure:

```yaml
# experiment.yaml
name: my_experiment
description: Test experiment
settings:
  batch_size: 32
  epochs: 100

# base.yaml
model:
  type: resnet50
  params:
    num_classes: 10

# trials.yaml
- learning_rate: 0.1
  settings:
    optimizer: adam
- learning_rate: 0.01
  settings:
    optimizer: sgd
```

## Directory Structure

When running experiments, the following directory structure is created:

```
workspace/
├── experiment_name/
│   ├── logs/        # Experiment logs
│   ├── artifacts/   # Model checkpoints and outputs
│   └── configs/     # Configuration files
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
