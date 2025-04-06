from omegaconf import DictConfig
from experiment_manager.common.serializable import YAMLSerializable

class Factory:
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