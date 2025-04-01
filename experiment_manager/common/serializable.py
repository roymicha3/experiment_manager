from omegaconf import OmegaConf, DictConfig
from typing import Type, Dict


class YAMLSerializable:
    """
    Base class for YAML serialization support.
    """
    _registry: Dict[str, Type] = {}

    def __init__(self, config: DictConfig = None):
        self.config = config

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a serializable class with a given name.
        """
        def decorator(class_ : Type):
            cls._registry[name] = class_
            return class_
        return decorator
    
    @classmethod
    def get_by_name(cls, name: str):
        """
        return an instance of a registered serializable object.
        """
        if name not in cls._registry:
            raise ValueError(f"'{name}' is not registered.")
        return cls._registry[name]
    

    def save(self, file_path):
        """
        Save model architecture to YAML using OmegaConf.
        """
        pass

    @classmethod
    def load(cls, file_path):
        """
        Load model architecture from YAML.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            config = OmegaConf.load(f)
        
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: DictConfig):
        """
        Create an instance from a DictConfig.
        """
        return cls(config)