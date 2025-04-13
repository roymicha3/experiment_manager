from omegaconf import DictConfig
import time

from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.common import Metric
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("DummyPipeline")
class DummyPipeline(Pipeline, YAMLSerializable):
    """A dummy pipeline that simulates training progress by tracking metrics."""
    
    def __init__(self, env: Environment, id: int = None):
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self)
        self.name = "DummyPipeline"
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        """Create a dummy pipeline from configuration."""
        return cls(env, id)
    
    def run(self, config: DictConfig):
        """Run the dummy pipeline, simulating a training process."""
        self.env.logger.info(f"Starting {self.name}")
        
        # Simulate epochs
        for epoch in range(config.pipeline.epochs):
            self.env.logger.info(f"Epoch {epoch + 1}/{config.pipeline.epochs}")
            
            # Simulate training progress
            train_acc = 0.5 + (epoch * 0.05)  # Starts at 0.5, increases by 0.05 each epoch
            train_loss = 1.0 - (epoch * 0.1)  # Starts at 1.0, decreases by 0.1 each epoch
            
            # Track training metrics
            self.env.tracker_manager.track(
                metric=Metric.TEST_ACC,  # Using TEST_ACC since TRAIN_ACC not available
                value=train_acc,
                step=epoch
            )
            self.env.tracker_manager.track(
                metric=Metric.TEST_LOSS,  # Using TEST_LOSS since TRAIN_LOSS not available
                value=train_loss,
                step=epoch
            )
            
            # Simulate validation
            val_acc = train_acc - 0.05  # Slightly worse than training
            val_loss = train_loss + 0.1  # Slightly worse than training
            
            # Track validation metrics
            self.env.tracker_manager.track(
                metric=Metric.VAL_ACC,
                value=val_acc,
                step=epoch
            )
            self.env.tracker_manager.track(
                metric=Metric.VAL_LOSS,
                value=val_loss,
                step=epoch
            )
            
            # Small delay to simulate computation
            time.sleep(0.1)
        
        # Track final test metrics
        self.env.tracker_manager.track(
            metric=Metric.TEST_ACC,
            value=train_acc + 0.02,  # Slightly better than training
            step=config.pipeline.epochs
        )
        self.env.tracker_manager.track(
            metric=Metric.TEST_LOSS,
            value=train_loss - 0.05,  # Slightly better than training
            step=config.pipeline.epochs
        )
        
        self.env.logger.info(f"Completed {self.name}")
        return "completed"
