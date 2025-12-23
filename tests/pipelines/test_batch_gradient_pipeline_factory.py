"""
Pipeline factory for MNIST batch gradient tracking tests.
"""

from omegaconf import DictConfig
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.common.serializable import YAMLSerializable
from tests.pipelines.mnist_batch_gradient_pipeline import MNISTBatchGradientPipeline


@YAMLSerializable.register("TestBatchGradientPipelineFactory")
class TestBatchGradientPipelineFactory(PipelineFactory, YAMLSerializable):
    """Factory for creating MNIST batch gradient tracking pipelines."""
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls()
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> MNISTBatchGradientPipeline:
        """Create and return a configured MNIST batch gradient tracking pipeline."""
        if name == "MNISTBatchGradientPipeline":
            return MNISTBatchGradientPipeline.from_config(config, env, id)
        else:
            raise ValueError(f"Unknown pipeline type: {name}")
