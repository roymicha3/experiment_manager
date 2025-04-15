from omegaconf import DictConfig

from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.callback_factory import CallbackFactory

# import pipelines
from tests.pipelines.dummy_pipeline import DummyPipeline
from tests.pipelines.simple_classifier import SimpleClassifierPipeline


class DummyPipelineFactory(PipelineFactory):
    """Factory for creating dummy pipelines."""
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        pipeline_class = YAMLSerializable.get_by_name(name)
        pipeline = pipeline_class.from_config(config, env, id)
        
        callbacks = config.get("callbacks", [])
        for callback in callbacks:
            callback = CallbackFactory.create(callback.type, callback, env)
            pipeline.register_callback(callback)
            
        return pipeline
