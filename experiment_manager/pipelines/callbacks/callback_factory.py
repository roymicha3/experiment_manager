from omegaconf import DictConfig

from experiment_manager.environment import Environment
from experiment_manager.common.factory import Factory
from experiment_manager.common.serializable import YAMLSerializable

# import the callbacks classes
from experiment_manager.pipelines.callbacks.metric_tracker import MetricsTracker
from experiment_manager.pipelines.callbacks.early_stopping import EarlyStopping
from experiment_manager.pipelines.callbacks.checkpoint import CheckpointCallback


class CallbackFactory(Factory):
    """
    Factory class for creating callbacks.
    """
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment):
        """
        Create an instance of a registered callback.
        """
        return YAMLSerializable.get_by_name(name).from_config(config, env)

