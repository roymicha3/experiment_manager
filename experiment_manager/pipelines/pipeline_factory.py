from omegaconf import DictConfig
from experiment_manager.common.factory import Factory
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.pipelines.callbacks.callback_factory import CallbackFactory

class PipelineFactory(Factory):
    """Factory class for creating pipeline instances."""
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        """
        Create a pipeline instance from configuration.
        """
        pipeline_class = YAMLSerializable.get_by_name(name)
        pipeline = pipeline_class.from_config(config, env, id)
        
        callbacks = config.get("callbacks", [])
        for callback in callbacks:
            callback = CallbackFactory.create(callback.type, callback, env)
            pipeline.register_callback(callback)
            
        return pipeline