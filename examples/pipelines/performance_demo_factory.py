from omegaconf import DictConfig
from experiment_manager import Environment, Pipeline
from experiment_manager.pipelines import PipelineFactory
from examples.pipelines.performance_demo_pipeline import PerformanceDemoPipeline


class PerformanceDemoFactory(PipelineFactory):
    """Factory for creating PerformanceDemoPipeline instances."""
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        """Create a PerformanceDemoPipeline from configuration."""
        return PipelineFactory.create(name, config, env, id) 