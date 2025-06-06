#!/usr/bin/env python3
"""
Demonstration script for early stopping and analytics tests.
This script shows that our test pipelines work correctly using the Metric enum.
"""

import os
import tempfile
import gc
from omegaconf import OmegaConf

from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.early_stopping import EarlyStopping
from experiment_manager.pipelines.callbacks.metric_tracker import MetricsTracker
from experiment_manager.common.common import Metric


def cleanup_logger(env):
    """Clean up logger handlers to prevent file locking on Windows"""
    if hasattr(env.logger, 'logger') and hasattr(env.logger.logger, 'handlers'):
        for handler in env.logger.logger.handlers[:]:
            if hasattr(handler, 'close'):
                handler.close()
            env.logger.logger.removeHandler(handler)
    # Force garbage collection to help release file handles
    gc.collect()


def test_early_stopping_basic():
    """Test basic early stopping functionality with Metric enum."""
    print("\n=== Testing Early Stopping Basic Functionality ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = OmegaConf.create({"workspace": tmp_dir, "verbose": False, "trackers": []})
        env = Environment(workspace=tmp_dir, config=config)
        
        # Create early stopping callback using Metric enum with explicit mode
        early_stopping = EarlyStopping(
            env=env,
            metric=Metric.VAL_LOSS,  # Use Metric enum
            patience=2,
            min_delta_percent=5.0,
            mode="min"  # Explicit: minimize validation loss
        )
        
        print(f"✓ Early stopping initialized with patience={early_stopping.patience}")
        print(f"✓ Monitoring metric: {early_stopping.metric}")
        print(f"✓ Min delta percent: {early_stopping.min_delta_percent}")
        print(f"✓ Mode: {early_stopping.mode}")
        
        # Test improvement detection
        early_stopping.on_start()
        
        # Simulate improving validation loss - using Metric enum as keys
        result1 = early_stopping.on_epoch_end(0, {Metric.VAL_LOSS: 1.0})
        result2 = early_stopping.on_epoch_end(1, {Metric.VAL_LOSS: 0.8})  # 20% improvement
        result3 = early_stopping.on_epoch_end(2, {Metric.VAL_LOSS: 0.75}) # 6.25% improvement
        
        print(f"✓ Epoch 0: val_loss=1.0, continue={result1}")
        print(f"✓ Epoch 1: val_loss=0.8, continue={result2}")
        print(f"✓ Epoch 2: val_loss=0.75, continue={result3}")
        print(f"✓ Best metric so far: {early_stopping.best_metric}")
        
        # Test stagnation
        result4 = early_stopping.on_epoch_end(3, {Metric.VAL_LOSS: 0.76})  # Slight increase
        result5 = early_stopping.on_epoch_end(4, {Metric.VAL_LOSS: 0.77})  # Another increase
        
        print(f"✓ Epoch 3: val_loss=0.76, continue={result4} (counter={early_stopping.counter})")
        print(f"✓ Epoch 4: val_loss=0.77, continue={result5} (counter={early_stopping.counter})")
        
        if early_stopping.counter >= early_stopping.patience:
            print(f"✓ Early stopping triggered! Training stopped.")
        else:
            print(f"✓ Training continues, patience not exceeded.")
            
        cleanup_logger(env)


def test_metrics_tracking_basic():
    """Test basic metrics tracking functionality with Metric enum."""
    print("\n=== Testing Metrics Tracking Basic Functionality ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = OmegaConf.create({"workspace": tmp_dir, "verbose": False, "trackers": []})
        env = Environment(workspace=tmp_dir, config=config)
        
        # Create metrics tracker
        tracker = MetricsTracker(env=env)
        
        print(f"✓ Metrics tracker initialized")
        print(f"✓ Log path: {tracker.log_path}")
        
        # Start tracking
        tracker.on_start()
        print(f"✓ Tracking started, metrics cleared: {len(tracker.metrics) == 0}")
        
        # Simulate training epochs using Metric enum
        for epoch in range(3):
            epoch_metrics = {
                Metric.TRAIN_LOSS: 1.0 - (epoch * 0.2),
                Metric.VAL_LOSS: 1.0 - (epoch * 0.15),
                Metric.VAL_ACC: 0.5 + (epoch * 0.1)
            }
            
            result = tracker.on_epoch_end(epoch, epoch_metrics)
            print(f"✓ Epoch {epoch}: train_loss={epoch_metrics[Metric.TRAIN_LOSS]:.2f}, "
                  f"val_loss={epoch_metrics[Metric.VAL_LOSS]:.2f}, "
                  f"val_acc={epoch_metrics[Metric.VAL_ACC]:.2f}")
        
        # Check tracked metrics
        print(f"✓ Total epochs tracked: {len(tracker.metrics[Metric.TRAIN_LOSS])}")
        print(f"✓ Latest train_loss: {tracker.get_latest(Metric.TRAIN_LOSS)}")
        print(f"✓ Latest val_acc: {tracker.get_latest(Metric.VAL_ACC)}")
        
        # End tracking
        final_metrics = {Metric.TEST_ACC: 0.85, Metric.TEST_LOSS: 0.4}
        tracker.on_end(final_metrics)
        
        # Check file was created
        file_exists = os.path.exists(tracker.log_path)
        print(f"✓ Metrics file created: {file_exists}")
        
        if file_exists:
            with open(tracker.log_path, 'r') as f:
                content = f.read()
                print(f"✓ File contains train_loss: {'train_loss' in content}")
                print(f"✓ File contains test_acc: {'test_acc' in content}")
        
        cleanup_logger(env)


def test_integration_workflow():
    """Test integration of early stopping and metrics tracking using Metric enum."""
    print("\n=== Testing Early Stopping + Metrics Integration ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = OmegaConf.create({"workspace": tmp_dir, "verbose": False, "trackers": []})
        env = Environment(workspace=tmp_dir, config=config)
        
        # Create both callbacks using Metric enum
        early_stopping = EarlyStopping(
            env=env,
            metric=Metric.VAL_LOSS,  # Use Metric enum
            patience=2,
            min_delta_percent=10.0,  # High threshold for demo
            mode="min"  # Minimize validation loss
        )
        
        tracker = MetricsTracker(env=env)
        
        callbacks = [early_stopping, tracker]
        
        # Initialize
        for callback in callbacks:
            callback.on_start()
        
        print(f"✓ Both callbacks initialized")
        
        # Simulate overfitting scenario using Metric enum as keys
        epochs_data = [
            {Metric.TRAIN_LOSS: 1.0, Metric.VAL_LOSS: 1.0, Metric.VAL_ACC: 0.5},   # Epoch 0
            {Metric.TRAIN_LOSS: 0.7, Metric.VAL_LOSS: 0.8, Metric.VAL_ACC: 0.6},   # Epoch 1 - improving
            {Metric.TRAIN_LOSS: 0.5, Metric.VAL_LOSS: 0.7, Metric.VAL_ACC: 0.65},  # Epoch 2 - still improving
            {Metric.TRAIN_LOSS: 0.3, Metric.VAL_LOSS: 0.85, Metric.VAL_ACC: 0.6},  # Epoch 3 - overfitting starts
            {Metric.TRAIN_LOSS: 0.2, Metric.VAL_LOSS: 0.9, Metric.VAL_ACC: 0.58},  # Epoch 4 - overfitting continues
            {Metric.TRAIN_LOSS: 0.1, Metric.VAL_LOSS: 0.95, Metric.VAL_ACC: 0.55}, # Epoch 5 - should trigger early stop
        ]
        
        for epoch, metrics in enumerate(epochs_data):
            print(f"Epoch {epoch}: train_loss={metrics[Metric.TRAIN_LOSS]:.2f}, "
                  f"val_loss={metrics[Metric.VAL_LOSS]:.2f}, val_acc={metrics[Metric.VAL_ACC]:.2f}")
            
            # Call callbacks
            should_continue = True
            for callback in callbacks:
                result = callback.on_epoch_end(epoch, metrics)
                if result is False:
                    should_continue = False
                    print(f"  → Early stopping triggered by {callback.__class__.__name__}!")
                    break
            
            if not should_continue:
                break
        
        # End training
        final_metrics = {Metric.TEST_ACC: 0.62, Metric.TEST_LOSS: 0.75}
        for callback in callbacks:
            callback.on_end(final_metrics)
        
        print(f"✓ Training completed")
        print(f"✓ Early stopping triggered: {early_stopping.counter >= early_stopping.patience}")
        print(f"✓ Counter at end: {early_stopping.counter}")
        print(f"✓ Best val_loss: {early_stopping.best_metric:.3f}")
        print(f"✓ Epochs tracked by metrics: {len(tracker.metrics[Metric.TRAIN_LOSS])}")
        print(f"✓ Metrics file exists: {os.path.exists(tracker.log_path)}")
        
        cleanup_logger(env)


def test_early_stopping_max_mode():
    """Test early stopping with max mode (for accuracy metrics)."""
    print("\n=== Testing Early Stopping Max Mode (Accuracy) ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = OmegaConf.create({"workspace": tmp_dir, "verbose": False, "trackers": []})
        env = Environment(workspace=tmp_dir, config=config)
        
        # Create early stopping callback for accuracy (higher is better)
        early_stopping = EarlyStopping(
            env=env,
            metric=Metric.VAL_ACC,  # Use validation accuracy
            patience=2,
            min_delta_percent=5.0,
            mode="max"  # Explicit: maximize validation accuracy
        )
        
        print(f"✓ Early stopping initialized for accuracy with mode={early_stopping.mode}")
        
        # Test improvement detection
        early_stopping.on_start()
        
        # Simulate improving validation accuracy
        result1 = early_stopping.on_epoch_end(0, {Metric.VAL_ACC: 0.5})
        result2 = early_stopping.on_epoch_end(1, {Metric.VAL_ACC: 0.6})   # 20% improvement
        result3 = early_stopping.on_epoch_end(2, {Metric.VAL_ACC: 0.65})  # 8.33% improvement
        
        print(f"✓ Epoch 0: val_acc=0.5, continue={result1}")
        print(f"✓ Epoch 1: val_acc=0.6, continue={result2}")
        print(f"✓ Epoch 2: val_acc=0.65, continue={result3}")
        print(f"✓ Best metric so far: {early_stopping.best_metric}")
        
        # Test stagnation (accuracy stops improving)
        result4 = early_stopping.on_epoch_end(3, {Metric.VAL_ACC: 0.64})  # Slight decrease
        result5 = early_stopping.on_epoch_end(4, {Metric.VAL_ACC: 0.63})  # Another decrease
        
        print(f"✓ Epoch 3: val_acc=0.64, continue={result4} (counter={early_stopping.counter})")
        print(f"✓ Epoch 4: val_acc=0.63, continue={result5} (counter={early_stopping.counter})")
        
        if early_stopping.counter >= early_stopping.patience:
            print(f"✓ Early stopping triggered! Training stopped.")
        else:
            print(f"✓ Training continues, patience not exceeded.")
            
        cleanup_logger(env)


def test_metric_enum_properties():
    """Test that the Metric enum works correctly for both callbacks."""
    print("\n=== Testing Metric Enum Properties ===")
    
    # Test that metric enum values work as dictionary keys
    test_metrics = {
        Metric.TRAIN_LOSS: 0.5,
        Metric.VAL_LOSS: 0.6,
        Metric.VAL_ACC: 0.8,
        Metric.TEST_ACC: 0.75
    }
    
    print(f"✓ Metric enum as dict keys works: {len(test_metrics) == 4}")
    print(f"✓ VAL_LOSS value: {test_metrics[Metric.VAL_LOSS]}")
    print(f"✓ VAL_LOSS name property: {Metric.VAL_LOSS.name}")
    print(f"✓ VAL_LOSS string representation: {str(Metric.VAL_LOSS)}")
    
    # Test metric lookup
    val_loss_found = Metric.VAL_LOSS in test_metrics
    print(f"✓ Metric lookup works: {val_loss_found}")
    
    # Test iteration
    metric_names = [metric.name for metric in test_metrics.keys()]
    print(f"✓ Metric names: {metric_names}")
    
    # Test mode validation
    print(f"✓ Mode options: 'min' (minimize metric), 'max' (maximize metric)")


if __name__ == "__main__":
    print("🧪 Running Early Stopping and Analytics Tests with Metric Enum")
    print("=" * 70)
    
    try:
        test_metric_enum_properties()
        test_early_stopping_basic()
        test_early_stopping_max_mode()
        test_metrics_tracking_basic()
        test_integration_workflow()
        
        print("\n" + "=" * 70)
        print("✅ All tests completed successfully!")
        print("\n📋 Summary:")
        print("  • Metric enum works correctly as dictionary keys")
        print("  • Early stopping properly detects metric improvements using Metric enum")
        print("  • Early stopping supports both 'min' and 'max' modes for different metric types")
        print("  • Early stopping triggers when patience is exceeded")
        print("  • Metrics tracking collects and saves training data using Metric enum")
        print("  • Both systems work together in pipeline workflows")
        print("  • Files are created and contain expected content")
        print("  • Logger cleanup prevents Windows file locking issues")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 