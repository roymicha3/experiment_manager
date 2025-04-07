from omegaconf import DictConfig

from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment
from tests.pipelines.dummy_pipeline import DummyPipeline


class DummyPipelineFactory(PipelineFactory):
    """Factory for creating dummy pipelines."""
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        pipeline_class = YAMLSerializable.get_by_name(name)
        return pipeline_class.from_config(config, env, id)
