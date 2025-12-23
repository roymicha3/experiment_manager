from omegaconf import DictConfig

from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment

# Import test pipelines
from tests.pipelines.mnist_perceptron_pipeline import MNISTPerceptronPipeline


class TestPipelineFactory(PipelineFactory):
    """Factory for creating test pipelines."""
    
    @staticmethod  
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        """Create pipeline instance from configuration."""
        if name == "MNISTPerceptronPipeline":
            return MNISTPerceptronPipeline(config, env)
        else:
            # Fall back to the default factory for other pipeline types
            return PipelineFactory.create(name, config, env, id) 