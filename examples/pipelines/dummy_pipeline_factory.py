from omegaconf import DictConfig

from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.callback_factory import CallbackFactory

# import pipelines
from examples.pipelines.dummy_pipeline import DummyPipeline
from examples.pipelines.simple_classifier import SimpleClassifierPipeline


class DummyPipelineFactory(PipelineFactory):
    """Factory for creating dummy pipelines."""
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        return PipelineFactory.create(name, config, env, id)
