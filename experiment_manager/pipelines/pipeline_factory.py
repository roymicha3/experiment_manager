from omegaconf import DictConfig
from experiment_manager.common.factory import Factory
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.common.factory_registry import FactoryType

class PipelineFactory(Factory):
    """Factory class for creating pipeline instances."""
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        """
        Create a pipeline instance from configuration.
        """
        pipeline_class = YAMLSerializable.get_by_name(name)
        pipeline = pipeline_class.from_config(config, env, id)
        
        callbacks = config.pipeline.get("callbacks", [])
        if not callbacks:
            env.logger.error("No callbacks found in configuration")
            return pipeline
        
        for callback in callbacks:
            factory = env.factory_registry.get(FactoryType.CALLBACK)
            callback = factory.create(callback.type, callback, env)
            pipeline.register_callback(callback)
            
        return pipeline