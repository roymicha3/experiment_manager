from omegaconf import DictConfig
import time
from typing import Dict, Any

from experiment_manager.common.common import Level, RunStatus
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.common import Metric
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("DummyPipeline")
class DummyPipeline(Pipeline, YAMLSerializable):
    """A dummy pipeline that simulates training progress using proper decorators."""
    
    def __init__(self, env: Environment, id: int = None):
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self)
        self.name = "DummyPipeline"
        
        # Initialize training components
        self.model = None
        self.epochs = 3
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        """Create a dummy pipeline from configuration."""
        return cls(env, id)
    
    @Pipeline.run_wrapper  # CRITICAL: This decorator is required!
    def run(self, config: DictConfig) -> Dict[str, Any]:
        """Run the dummy pipeline with proper lifecycle management."""
        
        # Get configuration
        self.epochs = config.pipeline.get('epochs', 3)
        
        # Log parameters - tracked by all trackers
        self.env.tracker_manager.log_params({
            "pipeline_type": "DummyPipeline",
            "epochs": self.epochs,
            "model_type": "Dummy",
            "learning_rate": 0.001,
            "batch_size": 32
        })
        
        # Training loop using run_epoch with @epoch_wrapper
        for epoch in range(self.epochs):
            self.env.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Call run_epoch with @epoch_wrapper - handles lifecycle automatically
            self.run_epoch(epoch, self.model)
        
        # Final test evaluation (after all epochs)
        final_test_acc = 0.85
        final_test_loss = 0.25
        
        # Store final metrics in run_metrics for automatic tracking
        self.run_metrics[Metric.TEST_ACC] = final_test_acc
        self.run_metrics[Metric.TEST_LOSS] = final_test_loss
        
        # Also track directly for immediate availability
        self.env.tracker_manager.track(Metric.TEST_ACC, final_test_acc)
        self.env.tracker_manager.track(Metric.TEST_LOSS, final_test_loss)
        
        self.env.logger.info(f"Final Results - Loss: {final_test_loss:.4f}, Accuracy: {final_test_acc:.4f}")
        
        return {
            "status": "completed",
            "final_test_loss": final_test_loss,
            "final_test_accuracy": final_test_acc,
            "epochs_completed": self.epochs
        }
    
    @Pipeline.epoch_wrapper  # CRITICAL: This manages epoch lifecycle and metrics automatically
    def run_epoch(self, epoch_idx: int, model, *args, **kwargs) -> RunStatus:
        """Run one epoch with proper epoch lifecycle management."""
        
        # Simulate training progress
        train_acc = 0.5 + (epoch_idx * 0.05)  # Starts at 0.5, increases by 0.05 each epoch
        train_loss = 1.0 - (epoch_idx * 0.1)  # Starts at 1.0, decreases by 0.1 each epoch
        
        # Track training metrics with step
        self.env.tracker_manager.track(Metric.TRAIN_ACC, train_acc, step=epoch_idx)
        self.env.tracker_manager.track(Metric.TRAIN_LOSS, train_loss, step=epoch_idx)
        
        # Simulate validation
        val_acc = train_acc - 0.05  # Slightly worse than training
        val_loss = train_loss + 0.1  # Slightly worse than training
        
        # Track validation metrics with step
        self.env.tracker_manager.track(Metric.VAL_ACC, val_acc, step=epoch_idx)
        self.env.tracker_manager.track(Metric.VAL_LOSS, val_loss, step=epoch_idx)
        
        # CRITICAL: Store in epoch_metrics - these will be automatically tracked by @epoch_wrapper!
        self.epoch_metrics[Metric.TRAIN_ACC] = train_acc
        self.epoch_metrics[Metric.TRAIN_LOSS] = train_loss
        self.epoch_metrics[Metric.VAL_ACC] = val_acc
        self.epoch_metrics[Metric.VAL_LOSS] = val_loss
        
        # Small delay to simulate computation (reduced for faster execution)
        time.sleep(0.05)  # Reduced from 0.1 to 0.05
        
        self.env.logger.info(f"  Epoch {epoch_idx + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        self.env.logger.info(f"  Epoch {epoch_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return RunStatus.SUCCESS
