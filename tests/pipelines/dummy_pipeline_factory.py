from omegaconf import DictConfig

from experiment_manager import Environment, Pipeline
from experiment_manager.pipelines import PipelineFactory

# import pipelines
from tests.pipelines.dummy_pipeline import DummyPipeline
from tests.pipelines.simple_classifier import SimpleClassifierPipeline
from tests.pipelines.env_args_check_pipeline import EnvArgsCheckPipeline


class DummyPipelineFactory(PipelineFactory):
    """
    Factory for creating dummy pipelines.
    """
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        return PipelineFactory.create(name, config, env, id)
