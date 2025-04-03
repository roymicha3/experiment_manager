from omegaconf import DictConfig

from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment

# include the pipelines
from examples.pipelines.pipeline_example import TrainingPipeline


class ExamplePipelineFactory(PipelineFactory):
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None):
        return YAMLSerializable.get_by_name(name).from_config(config, env, id)
