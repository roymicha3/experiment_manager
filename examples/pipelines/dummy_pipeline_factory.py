from omegaconf import DictConfig

from experiment_manager import Environment, Pipeline
from experiment_manager.pipelines import PipelineFactory

# import pipelines
from examples.pipelines.dummy_pipeline import DummyPipeline
from examples.pipelines.simple_classifier import SimpleClassifierPipeline


class DummyPipelineFactory(PipelineFactory):
    """Factory for creating dummy pipelines."""
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        return PipelineFactory.create(name, config, env, id)
