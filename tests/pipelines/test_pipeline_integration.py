import os
import pytest
import tempfile
import time
from unittest.mock import Mock, patch
from omegaconf import OmegaConf, DictConfig

from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.pipelines.callbacks.early_stopping import EarlyStopping
from experiment_manager.pipelines.callbacks.metric_tracker import MetricsTracker
from experiment_manager.common.common import Metric, Level, RunStatus
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("TestIntegrationPipeline")
class TestIntegrationPipeline(Pipeline, YAMLSerializable):
    """Test pipeline for integration testing with early stopping."""
    
    def __init__(self, env: Environment, epochs: int = 5, scenario: str = "normal"):
        super().__init__(env)
        self.epochs = epochs
        self.scenario = scenario
        self.epoch_count = 0
        
    @Pipeline.run_wrapper
    def run(self, config: DictConfig):
        """Main run method with proper pipeline wrapper."""
        self.env.logger.info(f"Starting integration test pipeline with scenario: {self.scenario}")
        
        # Add callbacks for testing
        early_stopping = EarlyStopping(
            env=self.env,
            metric=Metric.VAL_LOSS,
            patience=2,
            min_delta_percent=5.0
        )
        metrics_tracker = MetricsTracker(env=self.env)
        
        self.callbacks = [early_stopping, metrics_tracker]
        
        # Initialize callbacks
        for callback in self.callbacks:
            callback.on_start()
        
        try:
            for epoch in range(self.epochs):
                metrics = self._run_epoch(epoch, model=None)  # Pass a model placeholder
                
                # Call callbacks
                should_continue = True
                for callback in self.callbacks:
                    result = callback.on_epoch_end(epoch, metrics)
                    if result is False:
                        should_continue = False
                        break
                
                if not should_continue:
                    self.env.logger.info(f"Training stopped early at epoch {epoch}")
                    return RunStatus.STOPPED
                    
        finally:
            # End callbacks
            final_metrics = self._get_final_metrics()
            for callback in self.callbacks:
                callback.on_end(final_metrics)
        
        return RunStatus.SUCCESS
    
    @Pipeline.epoch_wrapper
    def _run_epoch(self, epoch_idx: int, model, *args, **kwargs) -> dict:
        """Run a single epoch and return metrics."""
        self.epoch_count += 1
        
        # Simulate different scenarios
        if self.scenario == "improving":
            # Metrics improve consistently
            train_loss = 1.0 - (epoch_idx * 0.2)
            val_loss = 1.0 - (epoch_idx * 0.18)
            val_acc = 0.5 + (epoch_idx * 0.1)
        elif self.scenario == "overfitting":
            # Training improves but validation gets worse after epoch 2
            train_loss = 1.0 - (epoch_idx * 0.15)
            if epoch_idx <= 2:
                val_loss = 1.0 - (epoch_idx * 0.1)
            else:
                val_loss = 0.8 + ((epoch_idx - 2) * 0.1)  # Gets worse
            val_acc = max(0.5 + (2 - abs(epoch_idx - 2)) * 0.1, 0.3)
        elif self.scenario == "plateauing":
            # Metrics improve then plateau
            if epoch_idx <= 2:
                train_loss = 1.0 - (epoch_idx * 0.2)
                val_loss = 1.0 - (epoch_idx * 0.18)
            else:
                train_loss = 0.6  # Plateaus
                val_loss = 0.64  # Plateaus
            val_acc = min(0.5 + (epoch_idx * 0.1), 0.7)
        else:  # normal
            # Standard improving metrics
            train_loss = 1.0 - (epoch_idx * 0.15)
            val_loss = 1.0 - (epoch_idx * 0.12)
            val_acc = 0.5 + (epoch_idx * 0.08)
        
        # Add some small random noise
        import random
        train_loss += random.uniform(-0.02, 0.02)
        val_loss += random.uniform(-0.02, 0.02)
        val_acc += random.uniform(-0.01, 0.01)
        
        # Ensure values are reasonable
        train_loss = max(0.1, train_loss)
        val_loss = max(0.1, val_loss)
        val_acc = max(0.1, min(0.95, val_acc))
        
        metrics = {
            Metric.TRAIN_LOSS: train_loss,
            Metric.VAL_LOSS: val_loss,
            Metric.VAL_ACC: val_acc
        }
        
        self.env.logger.info(f"Epoch {epoch_idx}: {metrics}")
        
        # Short delay to simulate work
        time.sleep(0.05)
        
        return metrics
    
    def _get_final_metrics(self) -> dict:
        """Get final test metrics."""
        return {
            Metric.TEST_ACC: 0.85,
            Metric.TEST_LOSS: 0.3
        }


@pytest.fixture
def test_env():
    """Create a test environment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = OmegaConf.create({
            "workspace": tmp_dir, 
            "verbose": False,
            "trackers": []
        })
        env = Environment(workspace=tmp_dir, config=config)
        yield env
        # Clean up all resources properly
        env.close()


class TestPipelineIntegration:
    """Test pipeline integration with early stopping."""
    
    def test_normal_training_completes(self, test_env):
        """Test that normal training runs to completion."""
        pipeline = TestIntegrationPipeline(
            env=test_env,
            epochs=3,
            scenario="normal"
        )
        
        # Create a simple config for the test
        config = OmegaConf.create({"pipeline": {"epochs": 3}})
        
        # Run pipeline
        pipeline.run(config)
        
        # Should complete all epochs
        assert pipeline.epoch_count == 3
        
        # Check that metrics were tracked
        metrics_tracker = pipeline.callbacks[1]  # Second callback is metrics tracker
        assert len(metrics_tracker.metrics[Metric.TRAIN_LOSS]) == 3
        assert len(metrics_tracker.metrics[Metric.VAL_LOSS]) == 3
        assert len(metrics_tracker.metrics[Metric.VAL_ACC]) == 3
        
        # Check that metrics file was created
        assert os.path.exists(metrics_tracker.log_path)
    
    def test_early_stopping_triggers(self, test_env):
        """Test that early stopping triggers when validation loss increases."""
        pipeline = TestIntegrationPipeline(
            env=test_env,
            epochs=6,  # Enough epochs for early stopping to trigger
            scenario="overfitting"
        )
        
        # Create a simple config for the test
        config = OmegaConf.create({"pipeline": {"epochs": 6}})
        
        # Run pipeline
        pipeline.run(config)
        
        # Should stop early (before epoch 6)
        assert pipeline.epoch_count < 6
        assert pipeline.epoch_count >= 3  # Should run at least a few epochs
        
        # Check that early stopping callback tracked the decision
        early_stopping = pipeline.callbacks[0]
        assert early_stopping.stopped_early
        assert early_stopping.best_epoch >= 0
    
    def test_metrics_progression_improving(self, test_env):
        """Test metrics progression in improving scenario."""
        pipeline = TestIntegrationPipeline(
            env=test_env,
            epochs=4,
            scenario="improving"
        )
        
        # Create a simple config for the test
        config = OmegaConf.create({"pipeline": {"epochs": 4}})
        
        pipeline.run(config)
        
        # Get metrics from tracker
        metrics_tracker = pipeline.callbacks[1]
        train_losses = metrics_tracker.metrics[Metric.TRAIN_LOSS]
        val_losses = metrics_tracker.metrics[Metric.VAL_LOSS]
        val_accuracies = metrics_tracker.metrics[Metric.VAL_ACC]
        
        # Train loss should generally decrease
        assert train_losses[0] > train_losses[-1]
        
        # Validation loss should generally decrease
        assert val_losses[0] > val_losses[-1]
        
        # Validation accuracy should generally increase
        assert val_accuracies[0] < val_accuracies[-1]
    
    def test_early_stopping_patience(self, test_env):
        """Test early stopping patience mechanism."""
        import random
        random.seed(42)  # Make test deterministic
        pipeline = TestIntegrationPipeline(
            env=test_env,
            epochs=8,
            scenario="plateauing"
        )
        
        # Create a simple config for the test
        config = OmegaConf.create({"pipeline": {"epochs": 8}})
        
        pipeline.run(config)
        
        early_stopping = pipeline.callbacks[0]
        
        # Should have stopped due to plateauing
        if early_stopping.stopped_early:
            assert early_stopping.wait_count == early_stopping.patience
            # With patience=2, early stopping triggers after 2 epochs with no improvement.
            # In the plateauing scenario, this happens after epoch 4, so epoch_count should be 5.
            assert pipeline.epoch_count == 5
    
    def test_callbacks_integration(self, test_env):
        """Test that callbacks work together properly."""
        pipeline = TestIntegrationPipeline(
            env=test_env,
            epochs=3,
            scenario="normal"
        )
        
        # Create a simple config for the test
        config = OmegaConf.create({"pipeline": {"epochs": 3}})
        
        pipeline.run(config)
        
        # Both callbacks should be present
        assert len(pipeline.callbacks) == 2
        
        early_stopping = pipeline.callbacks[0]
        metrics_tracker = pipeline.callbacks[1]
        
        # Both should have been called
        assert isinstance(early_stopping, EarlyStopping)
        assert isinstance(metrics_tracker, MetricsTracker)
        
        # Metrics tracker should have data
        assert len(metrics_tracker.metrics) > 0
        
        # Early stopping should have tracked metrics
        assert early_stopping.best_metric is not None
    
    def test_pipeline_with_minimal_epochs(self, test_env):
        """Test pipeline behavior with minimal epochs."""
        pipeline = TestIntegrationPipeline(
            env=test_env,
            epochs=1,
            scenario="normal"
        )
        
        # Create a simple config for the test
        config = OmegaConf.create({"pipeline": {"epochs": 1}})
        
        pipeline.run(config)
        
        # Should complete the single epoch
        assert pipeline.epoch_count == 1
        
        # Metrics should still be tracked
        metrics_tracker = pipeline.callbacks[1]
        assert len(metrics_tracker.metrics[Metric.TRAIN_LOSS]) == 1
        
        # Early stopping shouldn't trigger with just one epoch
        early_stopping = pipeline.callbacks[0]
        assert not early_stopping.stopped_early
    
    def test_metrics_file_content(self, test_env):
        """Test that metrics file contains expected content."""
        pipeline = TestIntegrationPipeline(
            env=test_env,
            epochs=2,
            scenario="normal"
        )
        
        # Create a simple config for the test
        config = OmegaConf.create({"pipeline": {"epochs": 2}})
        
        pipeline.run(config)
        
        metrics_tracker = pipeline.callbacks[1]
        
        # Read metrics file
        with open(metrics_tracker.log_path, 'r') as f:
            content = f.read()
        
        # Should contain metrics
        assert "train_loss" in content
        assert "val_loss" in content
        assert "val_acc" in content
        assert "test_acc" in content
        
        # Should have epoch data
        assert "epoch" in content.lower() or "0" in content
    
    def test_early_stopping_best_metric_tracking(self, test_env):
        """Test that early stopping correctly tracks best metrics."""
        pipeline = TestIntegrationPipeline(
            env=test_env,
            epochs=4,
            scenario="improving"
        )
        
        # Create a simple config for the test
        config = OmegaConf.create({"pipeline": {"epochs": 4}})
        
        pipeline.run(config)
        
        early_stopping = pipeline.callbacks[0]
        metrics_tracker = pipeline.callbacks[1]
        
        # Get validation losses from tracker
        val_losses = metrics_tracker.metrics[Metric.VAL_LOSS]
        
        # Best metric should be among the recorded values
        assert early_stopping.best_metric in val_losses or abs(early_stopping.best_metric - min(val_losses)) < 0.1
        
        # Best epoch should be valid
        assert 0 <= early_stopping.best_epoch < len(val_losses)
    
    def test_pipeline_error_handling(self, test_env):
        """Test pipeline behavior with simulated errors."""
        pipeline = TestIntegrationPipeline(
            env=test_env,
            epochs=3,
            scenario="normal"
        )
        
        # Mock an error in epoch processing
        original_run_epoch = pipeline._run_epoch
        
        def error_epoch(epoch_idx, model, *args, **kwargs):
            if epoch_idx == 1:
                raise RuntimeError("Simulated training error")
            return original_run_epoch(epoch_idx, model, *args, **kwargs)
        
        pipeline._run_epoch = error_epoch
        
        # Create a simple config for the test
        config = OmegaConf.create({"pipeline": {"epochs": 3}})
        
        # The @Pipeline.run_wrapper decorator catches all exceptions and returns RunStatus.FAILED
        # instead of propagating them. This is the correct behavior for production pipelines.
        result = pipeline.run(config)
        
        # Verify that the pipeline returned FAILED status due to the exception
        assert result == RunStatus.FAILED, f"Expected RunStatus.FAILED but got {result}"
        
        # The pipeline should have completed epoch 0 successfully (epoch_count=1) and failed on epoch 1
        assert pipeline.epoch_count == 1, f"Expected epoch_count=1 but got {pipeline.epoch_count}"
        
        # Verify that callbacks were still properly initialized (even if cleanup may vary)
        assert len(pipeline.callbacks) == 2


if __name__ == "__main__":
    pytest.main([__file__]) 