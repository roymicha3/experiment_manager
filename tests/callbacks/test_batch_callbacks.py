"""
Tests for batch-level callback functionality.

Tests the on_batch_end callback method added to the Callback base class
and its integration with the Pipeline batch_wrapper decorator.
"""
import pytest
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch
from omegaconf import DictConfig, OmegaConf

from experiment_manager.pipelines.callbacks.callback import Callback
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.environment import Environment
from experiment_manager.common.common import Metric, RunStatus


# =============================================================================
# Test Callback Implementations
# =============================================================================

class BatchCountingCallback(Callback):
    """A callback that counts batch_end calls."""
    
    def __init__(self):
        self.batch_end_count = 0
        self.batch_indices: List[int] = []
        self.batch_metrics: List[Dict[str, Any]] = []
    
    def on_start(self) -> None:
        self.batch_end_count = 0
        self.batch_indices.clear()
        self.batch_metrics.clear()
    
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        return True
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any]) -> bool:
        self.batch_end_count += 1
        self.batch_indices.append(batch_idx)
        self.batch_metrics.append(metrics.copy())
        return True  # Continue training
    
    def on_end(self, metrics: Dict[str, Any]) -> None:
        pass


class BatchStoppingCallback(Callback):
    """A callback that stops training after a certain number of batches."""
    
    def __init__(self, stop_after_batch: int):
        self.stop_after_batch = stop_after_batch
        self.stopped = False
    
    def on_start(self) -> None:
        self.stopped = False
    
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        return True
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any]) -> bool:
        if batch_idx >= self.stop_after_batch:
            self.stopped = True
            return False  # Stop training
        return True
    
    def on_end(self, metrics: Dict[str, Any]) -> None:
        pass


class DefaultBatchCallback(Callback):
    """A callback that uses the default on_batch_end implementation."""
    
    def on_start(self) -> None:
        pass
    
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        return True
    
    # Note: on_batch_end is NOT overridden - uses default implementation
    
    def on_end(self, metrics: Dict[str, Any]) -> None:
        pass


# =============================================================================
# Unit Tests for Callback Base Class
# =============================================================================

class TestCallbackBatchEndMethod:
    """Tests for the on_batch_end method in the Callback base class."""
    
    def test_default_implementation_returns_true(self):
        """Default on_batch_end should return True (continue training)."""
        callback = DefaultBatchCallback()
        result = callback.on_batch_end(0, {})
        assert result is True
    
    def test_default_implementation_accepts_any_metrics(self):
        """Default on_batch_end should accept any metrics dict."""
        callback = DefaultBatchCallback()
        
        # Empty metrics
        assert callback.on_batch_end(0, {}) is True
        
        # With metrics
        metrics = {Metric.TRAIN_LOSS: 0.5, Metric.TRAIN_ACC: 0.8}
        assert callback.on_batch_end(1, metrics) is True
    
    def test_custom_implementation_can_return_false(self):
        """Custom on_batch_end can return False to stop training."""
        callback = BatchStoppingCallback(stop_after_batch=5)
        
        # Should continue for batches 0-4
        for i in range(5):
            assert callback.on_batch_end(i, {}) is True
        
        # Should stop at batch 5
        assert callback.on_batch_end(5, {}) is False
        assert callback.stopped is True
    
    def test_batch_counting_callback_tracks_calls(self):
        """BatchCountingCallback should correctly track all batch_end calls."""
        callback = BatchCountingCallback()
        callback.on_start()
        
        # Simulate 10 batches
        for i in range(10):
            metrics = {Metric.TRAIN_LOSS: 1.0 - i * 0.1}
            callback.on_batch_end(i, metrics)
        
        assert callback.batch_end_count == 10
        assert callback.batch_indices == list(range(10))
        assert len(callback.batch_metrics) == 10


# =============================================================================
# Integration Tests with Pipeline
# =============================================================================

class BatchTestPipeline(Pipeline):
    """A test pipeline that uses the batch_wrapper decorator."""
    
    def __init__(self, env: Environment):
        super().__init__(env)
        self.batches_executed = 0
    
    @Pipeline.run_wrapper
    def run(self, config: DictConfig) -> RunStatus:
        num_epochs = config.get("epochs", 2)
        batches_per_epoch = config.get("batches_per_epoch", 5)
        
        for epoch in range(num_epochs):
            self.run_epoch(epoch, None, batches_per_epoch=batches_per_epoch)
        
        return RunStatus.COMPLETED
    
    @Pipeline.epoch_wrapper
    def run_epoch(self, epoch_idx: int, model, batches_per_epoch: int = 5) -> RunStatus:
        for batch_idx in range(batches_per_epoch):
            self.run_batch(batch_idx)
        return RunStatus.COMPLETED
    
    @Pipeline.batch_wrapper
    def run_batch(self, batch_idx: int) -> RunStatus:
        self.batches_executed += 1
        self.batch_metrics[Metric.TRAIN_LOSS] = 1.0 - batch_idx * 0.1
        return RunStatus.COMPLETED


class TestPipelineBatchCallbacks:
    """Integration tests for batch callbacks with Pipeline."""
    
    @pytest.fixture
    def mock_env(self, tmp_path):
        """Create a mock environment for testing."""
        # Create a proper mock that handles properties
        env = MagicMock(spec=Environment)
        env.workspace = str(tmp_path)
        env.config = OmegaConf.create({
            "workspace": str(tmp_path),
            "verbose": False,
            "debug": False
        })
        env.logger = MagicMock()
        env.tracker_manager = MagicMock()
        env.factory_registry = MagicMock()
        
        return env
    
    def test_batch_callback_called_for_each_batch(self, mock_env):
        """on_batch_end should be called for each batch."""
        pipeline = BatchTestPipeline(mock_env)
        counting_callback = BatchCountingCallback()
        pipeline.register_callback(counting_callback)
        
        config = OmegaConf.create({"epochs": 2, "batches_per_epoch": 5})
        counting_callback.on_start()
        
        # Run the pipeline
        try:
            pipeline.run(config)
        except StopIteration:
            pass  # Expected when epoch_wrapper raises StopIteration
        
        # Should have 2 epochs * 5 batches = 10 batch_end calls
        assert counting_callback.batch_end_count == 10
    
    def test_batch_callback_receives_metrics(self, mock_env):
        """on_batch_end should receive the batch metrics."""
        pipeline = BatchTestPipeline(mock_env)
        counting_callback = BatchCountingCallback()
        pipeline.register_callback(counting_callback)
        
        config = OmegaConf.create({"epochs": 1, "batches_per_epoch": 3})
        counting_callback.on_start()
        
        try:
            pipeline.run(config)
        except StopIteration:
            pass
        
        # Check that metrics were received
        assert len(counting_callback.batch_metrics) == 3
        for metrics in counting_callback.batch_metrics:
            assert Metric.TRAIN_LOSS in metrics
    
    def test_batch_callback_can_stop_training(self, mock_env):
        """Returning False from on_batch_end should stop training."""
        pipeline = BatchTestPipeline(mock_env)
        stopping_callback = BatchStoppingCallback(stop_after_batch=2)
        pipeline.register_callback(stopping_callback)
        
        config = OmegaConf.create({"epochs": 5, "batches_per_epoch": 10})
        
        # Run should complete with STOPPED status (run_wrapper catches StopIteration)
        status = pipeline.run(config)
        
        assert status == RunStatus.STOPPED
        assert stopping_callback.stopped is True
        # Only batches 0, 1, 2 should have been processed before stopping
        assert pipeline.batches_executed == 3
    
    def test_multiple_callbacks_all_called(self, mock_env):
        """Multiple callbacks should all have on_batch_end called."""
        pipeline = BatchTestPipeline(mock_env)
        
        callback1 = BatchCountingCallback()
        callback2 = BatchCountingCallback()
        callback3 = DefaultBatchCallback()
        
        pipeline.register_callback(callback1)
        pipeline.register_callback(callback2)
        pipeline.register_callback(callback3)
        
        config = OmegaConf.create({"epochs": 1, "batches_per_epoch": 5})
        callback1.on_start()
        callback2.on_start()
        
        try:
            pipeline.run(config)
        except StopIteration:
            pass
        
        assert callback1.batch_end_count == 5
        assert callback2.batch_end_count == 5
    
    def test_one_callback_stops_all_continue(self, mock_env):
        """If one callback returns False, training should stop even if others return True."""
        pipeline = BatchTestPipeline(mock_env)
        
        continuing_callback = BatchCountingCallback()
        stopping_callback = BatchStoppingCallback(stop_after_batch=1)
        
        pipeline.register_callback(continuing_callback)
        pipeline.register_callback(stopping_callback)
        
        config = OmegaConf.create({"epochs": 5, "batches_per_epoch": 10})
        continuing_callback.on_start()
        
        # Run should complete with STOPPED status (run_wrapper catches StopIteration)
        status = pipeline.run(config)
        
        assert status == RunStatus.STOPPED
        # Both callbacks called for batches 0 and 1, then stopped
        assert continuing_callback.batch_end_count == 2
        assert stopping_callback.stopped is True


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing callbacks."""
    
    def test_existing_callbacks_work_without_on_batch_end(self):
        """Callbacks without on_batch_end override should still work."""
        callback = DefaultBatchCallback()
        
        # Should be able to call all lifecycle methods
        callback.on_start()
        assert callback.on_epoch_end(0, {}) is True
        assert callback.on_batch_end(0, {}) is True  # Default implementation
        callback.on_end({})
    
    def test_callback_is_still_abstract(self):
        """Callback should still require on_start, on_epoch_end, on_end to be implemented."""
        
        # This should raise TypeError because abstract methods aren't implemented
        class IncompleteCallback(Callback):
            pass
        
        with pytest.raises(TypeError):
            IncompleteCallback()
    
    def test_on_batch_end_is_not_abstract(self):
        """on_batch_end should not be abstract - has default implementation."""
        
        # This should work because on_batch_end has a default implementation
        class MinimalCallback(Callback):
            def on_start(self) -> None:
                pass
            
            def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
                return True
            
            def on_end(self, metrics: Dict[str, Any]) -> None:
                pass
        
        callback = MinimalCallback()
        assert callback.on_batch_end(0, {}) is True

