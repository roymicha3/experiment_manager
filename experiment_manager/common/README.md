# YAMLSerializable and Factory

## Overview

This module provides a framework for registering, serializing, and deserializing classes using YAML configurations. The `YAMLSerializable` class acts as a base class that supports YAML-based persistence, while the `Factory` class provides a convenient method to instantiate registered classes dynamically.

## Features

- **Class Registration**: Use `@YAMLSerializable.register(name)` to register a class by a unique identifier.
- **Serialization & Deserialization**: Save and load configurations using YAML via `OmegaConf`.
- **Factory Pattern**: Dynamically create instances of registered classes based on a given configuration.

## Usage

### 1. Define and Register a Serializable Class
```python
from omegaconf import DictConfig
from settings.serializable import YAMLSerializable

@YAMLSerializable.register("example_model")
class ExampleModel(YAMLSerializable):
    def __init__(self, param1, param2, ...):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        ...


    @classmethod
    def from_config(cls, config: DictConfig, ...):
        """
        Create an instance from a DictConfig.
        """
        return cls(config.param1, config.param2, ...)
```

### 2. Save and Load Configurations
```python
# Save example configuration
config = OmegaConf.create({"param1": "value", "param2": "value"})
example_instance = ExampleModel(config)

# Load instance from file
loaded_instance = ExampleModel.from_config(example_instance)
```

### 3. Instantiate a Registered Model Using Factory
```python
from settings.factory import Factory
from settings.serializable import YAMLSerializable

# import all classes the factory is responsible for
import ExampleModel

class ExampleFactory(Factory):
    """
    Factory class for creating a serializable object.
    """

    @staticmethod
    def create(name: str, config: DictConfig):
        """
        Create an instance of a registered model.
        """
        class_ = YAMLSerializable.get_by_name(name)
        return class_.from_config(config)
```

## Classes

### `YAMLSerializable`
**Description:**
A base class that enables YAML serialization and deserialization for models.

#### Methods:
- `register(name: str)`: Registers a class under a given name.
- `get_by_name(name: str)`: Retrieves a registered class by name.
- `save(file_path)`: Saves an instance configuration to YAML.
- `load(file_path)`: Loads an instance configuration from a YAML file.
- `from_config(config: DictConfig)`: Creates an instance from a given configuration.

### `Factory`
**Description:**
A helper class to instantiate registered classes dynamically.

#### Methods:
- `create(name: str, config: DictConfig)`: Creates an instance of a registered class using a provided configuration.

## Dependencies
- `omegaconf`

## Notes
- Ensure that all serializable classes are registered using `@YAMLSerializable.register("name")` before attempting to instantiate them using `Factory.create()`.
- Configurations must be in the form of an `OmegaConf.DictConfig` object.
