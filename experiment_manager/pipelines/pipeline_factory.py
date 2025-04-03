from omegaconf import DictConfig
from experiment_manager.common.factory import Factory
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable

class PipelineFactory(Factory):
    """Factory class for creating pipeline instances."""
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        """
        Create a pipeline instance from configuration.
        """
        pipeline_class = YAMLSerializable.get_by_name(name)
        return pipeline_class.from_config(config, env, id)