import os
import pytest
import tempfile
from unittest.mock import Mock, patch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.early_stopping import EarlyStopping
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.common import Metric, Level, RunStatus
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("TestPipelineForEarlyStopping")
class TestPipelineForEarlyStopping(Pipeline, YAMLSerializable):
    """Test pipeline specifically designed for early stopping tests."""
    
    def __init__(self, env: Environment, epochs: int = 10, patience_test_mode: str = "improve"):
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self)
        self.epochs = epochs
        self.patience_test_mode = patience_test_mode  # "improve", "no_improve", "plateau"
        self.current_epoch = 0
        
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls(
            env, 
            epochs=config.pipeline.get('epochs', 10),
            patience_test_mode=config.pipeline.get('patience_test_mode', 'improve')
        )
    
    @Pipeline.run_wrapper
    def run(self, config: DictConfig):
        """Run the test pipeline with early stopping."""
        self.env.logger.info("Starting test pipeline for early stopping")
        
        # The @Pipeline.run_wrapper will catch StopIteration and return RunStatus.STOPPED
        # When early stopping triggers, the epoch_wrapper will raise StopIteration  
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.run_epoch(epoch, model=None)  # Pass None as model placeholder
                
        return {"status": "completed", "epochs_completed": self.epochs}
    
    @Pipeline.epoch_wrapper
    def run_epoch(self, epoch_idx: int, model, *args, **kwargs):
        """Run a single epoch with different loss patterns for testing."""
        
        if self.patience_test_mode == "improve":
            # Consistently improving loss
            val_loss = 1.0 - (epoch_idx * 0.1)
        elif self.patience_test_mode == "no_improve":
            # Loss gets worse after epoch 2
            val_loss = 1.0 - (epoch_idx * 0.1) if epoch_idx <= 2 else 1.0 + (epoch_idx * 0.05)
        elif self.patience_test_mode == "plateau":
            # Loss plateaus after epoch 3
            val_loss = 1.0 - (epoch_idx * 0.1) if epoch_idx <= 3 else 0.7
        else:
            val_loss = 1.0
            
        train_loss = val_loss - 0.1
        val_acc = 1.0 - val_loss  # Inverse relationship for testing
        
        # Store metrics in epoch_metrics for automatic processing
        self.epoch_metrics = {
            Metric.TRAIN_LOSS: train_loss,
            Metric.VAL_LOSS: val_loss,
            Metric.VAL_ACC: val_acc,
            Metric.CUSTOM: ("epoch", epoch_idx)
        }
        
        self.env.logger.info(f"Epoch {epoch_idx}: val_loss={val_loss:.3f}, val_acc={val_acc:.3f}")


@pytest.fixture
def test_env():
    """Create a test environment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = OmegaConf.create({"workspace": tmp_dir, "verbose": False})
        env = Environment(workspace=tmp_dir, config=config)
        yield env
        # Clean up all resources properly
        env.close()


@pytest.fixture
def early_stopping_callback():
    """Create an early stopping callback for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = OmegaConf.create({"workspace": tmp_dir, "verbose": False})
        env = Environment(workspace=tmp_dir, config=config)
        callback = EarlyStopping(
            env=env,
            metric=Metric.VAL_LOSS,
            patience=3,
            min_delta_percent=1.0
        )
        yield callback
        # Clean up all resources properly
        env.close()


class TestEarlyStopping:
    """Comprehensive tests for early stopping functionality."""
    
    def test_early_stopping_initialization(self, test_env):
        """Test early stopping callback initialization."""
        callback = EarlyStopping(
            env=test_env,
            metric=Metric.VAL_LOSS,
            patience=5,
            min_delta_percent=2.0
        )
        
        assert callback.metric == Metric.VAL_LOSS
        assert callback.patience == 5
        assert callback.min_delta_percent == 2.0
        assert callback.counter == 0
        assert callback.best_metric is None
        assert callback.early_stop is False
    
    def test_early_stopping_from_config(self, test_env):
        """Test creating early stopping from config."""
        config = OmegaConf.create({
            "metric": "val_acc",
            "patience": 10,
            "min_delta_percent": 5.0
        })
        
        callback = EarlyStopping.from_config(config, test_env)
        
        assert callback.metric == Metric.VAL_ACC
        assert callback.patience == 10
        assert callback.min_delta_percent == 5.0
    
    def test_on_start_resets_state(self, early_stopping_callback):
        """Test that on_start properly resets callback state."""
        # Set some initial state
        early_stopping_callback.counter = 5
        early_stopping_callback.best_metric = 0.5
        
        # Call on_start
        early_stopping_callback.on_start()
        
        # Verify reset
        assert early_stopping_callback.counter == 0
        assert early_stopping_callback.best_metric is None
    
    def test_improvement_detection_loss_metric(self, early_stopping_callback):
        """Test improvement detection for loss metrics (lower is better)."""
        # First epoch - should set baseline
        result = early_stopping_callback.on_epoch_end(0, {Metric.VAL_LOSS: 1.0})
        assert result is True
        assert early_stopping_callback.best_metric == 1.0
        assert early_stopping_callback.counter == 0
        
        # Second epoch - improvement
        result = early_stopping_callback.on_epoch_end(1, {Metric.VAL_LOSS: 0.8})
        assert result is True
        assert early_stopping_callback.best_metric == 0.8
        assert early_stopping_callback.counter == 0
        
        # Third epoch - no improvement
        result = early_stopping_callback.on_epoch_end(2, {Metric.VAL_LOSS: 0.85})
        assert result is True
        assert early_stopping_callback.best_metric == 0.8
        assert early_stopping_callback.counter == 1
    
    def test_improvement_detection_accuracy_metric(self, test_env):
        """Test improvement detection for accuracy metrics (higher is better)."""
        callback = EarlyStopping(
            env=test_env,
            metric=Metric.VAL_ACC,
            patience=3,
            min_delta_percent=1.0,
            mode="max"  # Higher accuracy is better
        )
        
        # First epoch - baseline
        result = callback.on_epoch_end(0, {Metric.VAL_ACC: 0.8})
        assert result is True
        assert callback.best_metric == 0.8
        assert callback.counter == 0
        
        # Second epoch - improvement
        result = callback.on_epoch_end(1, {Metric.VAL_ACC: 0.85})
        assert result is True
        assert callback.best_metric == 0.85
        assert callback.counter == 0
        
        # Third epoch - no improvement
        result = callback.on_epoch_end(2, {Metric.VAL_ACC: 0.84})
        assert result is True
        assert callback.best_metric == 0.85
        assert callback.counter == 1
    
    def test_early_stopping_trigger(self, early_stopping_callback):
        """Test that early stopping triggers after patience epochs."""
        # Setup baseline
        early_stopping_callback.on_epoch_end(0, {Metric.VAL_LOSS: 1.0})
        
        # No improvement for patience epochs
        for i in range(1, 4):  # patience = 3
            result = early_stopping_callback.on_epoch_end(i, {Metric.VAL_LOSS: 1.1})
            if i < 3:
                assert result is True  # Should continue
            else:
                assert result is False  # Should stop
        
        assert early_stopping_callback.counter == 3
    
    def test_missing_metric_handling(self, early_stopping_callback):
        """Test handling when monitored metric is missing."""
        result = early_stopping_callback.on_epoch_end(0, {Metric.TRAIN_LOSS: 1.0})
        assert result is True  # Should continue when metric is missing
    
    def test_min_delta_percentage_threshold(self, test_env):
        """Test minimum delta percentage for improvement detection."""
        callback = EarlyStopping(
            env=test_env,
            metric=Metric.VAL_LOSS,
            patience=3,
            min_delta_percent=10.0  # Require 10% improvement
        )
        
        # Baseline
        callback.on_epoch_end(0, {Metric.VAL_LOSS: 1.0})
        
        # Small improvement (5% - below threshold)
        result = callback.on_epoch_end(1, {Metric.VAL_LOSS: 0.95})
        assert result is True
        assert callback.counter == 1  # Should count as no improvement
        
        # Large improvement (15% - above threshold)
        result = callback.on_epoch_end(2, {Metric.VAL_LOSS: 0.85})
        assert result is True
        assert callback.counter == 0  # Should reset counter
        assert callback.best_metric == 0.85
    
    def test_on_end_logging(self, early_stopping_callback):
        """Test on_end method logging."""
        # Set some best metric
        early_stopping_callback.best_metric = 0.75
        
        with patch.object(early_stopping_callback.env.logger, 'info') as mock_logger:
            early_stopping_callback.on_end({Metric.VAL_LOSS: 0.8})
            mock_logger.assert_called()
    
    def test_early_stopping_with_pipeline_integration(self, test_env):
        """Test early stopping integration with a complete pipeline."""
        # Create pipeline configured for early stopping
        pipeline = TestPipelineForEarlyStopping(
            env=test_env, 
            epochs=10, 
            patience_test_mode="no_improve"
        )
        
        # Create early stopping callback
        early_stopping = EarlyStopping(
            env=test_env,
            metric=Metric.VAL_LOSS,
            patience=2,
            min_delta_percent=1.0
        )
        
        # Register callback
        pipeline.register_callback(early_stopping)
        
        # Create config
        config = OmegaConf.create({
            "pipeline": {
                "epochs": 10,
                "patience_test_mode": "no_improve"
            }
        })
        
        # Run pipeline
        result = pipeline.run(config)
        
        # Should have stopped early
        assert result == RunStatus.STOPPED
    
    def test_early_stopping_with_improving_metrics(self, test_env):
        """Test that early stopping doesn't trigger when metrics keep improving."""
        pipeline = TestPipelineForEarlyStopping(
            env=test_env, 
            epochs=8, 
            patience_test_mode="improve"
        )
        
        early_stopping = EarlyStopping(
            env=test_env,
            metric=Metric.VAL_LOSS,
            patience=3,
            min_delta_percent=1.0
        )
        
        pipeline.register_callback(early_stopping)
        
        config = OmegaConf.create({
            "pipeline": {
                "epochs": 8,
                "patience_test_mode": "improve"
            }
        })
        
        result = pipeline.run(config)
        
        # Should complete all epochs - pipeline returns what the run function returns
        assert isinstance(result, dict)
        assert result["status"] == "completed"
        assert result["epochs_completed"] == 8
    
    def test_early_stopping_with_plateau(self, test_env):
        """Test early stopping with metric plateau."""
        pipeline = TestPipelineForEarlyStopping(
            env=test_env, 
            epochs=10, 
            patience_test_mode="plateau"
        )
        
        early_stopping = EarlyStopping(
            env=test_env,
            metric=Metric.VAL_LOSS,
            patience=2,
            min_delta_percent=1.0
        )
        
        pipeline.register_callback(early_stopping)
        
        config = OmegaConf.create({
            "pipeline": {
                "epochs": 10,
                "patience_test_mode": "plateau"
            }
        })
        
        result = pipeline.run(config)
        
        # Should stop early due to plateau
        assert result == RunStatus.STOPPED
    
    def test_early_stopping_logs_analysis(self, test_env):
        """Test that early stopping events are properly logged for analysis."""
        pipeline = TestPipelineForEarlyStopping(
            env=test_env, 
            epochs=10, 
            patience_test_mode="no_improve"
        )
        
        early_stopping = EarlyStopping(
            env=test_env,
            metric=Metric.VAL_LOSS,
            patience=2,
            min_delta_percent=1.0
        )
        
        pipeline.register_callback(early_stopping)
        
        config = OmegaConf.create({
            "pipeline": {
                "epochs": 10,
                "patience_test_mode": "no_improve"
            }
        })
        
        result = pipeline.run(config)
        
        # Check that early stopping was logged
        log_files = []
        for root, dirs, files in os.walk(test_env.workspace):
            for f in files:
                if f.endswith(".log"):
                    log_files.append(os.path.join(root, f))
        
        found_early_stopping = False
        for log_file in log_files:
            with open(log_file, 'r') as f:
                content = f.read()
                if "Early stopping triggered" in content:
                    found_early_stopping = True
                    break
        
        assert found_early_stopping, "Early stopping should be logged"
        assert result == RunStatus.STOPPED


if __name__ == "__main__":
    pytest.main([__file__]) 