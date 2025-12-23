# Pipeline Development Guide

This guide demonstrates how to properly implement pipelines in the Experiment Manager framework, covering essential patterns, decorators, and best practices.

## Overview

A pipeline in the Experiment Manager is a structured component that orchestrates machine learning experiments with proper tracking, metrics recording, and lifecycle management. This guide covers the essential elements you need to implement a pipeline correctly.

## Essential Components

### 1. Required Imports

```python
import time
import numpy as np
from omegaconf import DictConfig
from typing import Dict, Any

from experiment_manager.common.common import Level, Metric, RunStatus
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable
```

### 2. Pipeline Class Structure

Every pipeline must:
- Inherit from `Pipeline` and `YAMLSerializable`
- Be registered with `@YAMLSerializable.register("YourPipelineName")`
- Implement required methods with proper decorators

```python
@YAMLSerializable.register("YourPipelineName")
class YourPipeline(Pipeline, YAMLSerializable):
    """Your pipeline description."""
    
    def __init__(self, env: Environment, id: int = None):
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self)
        self.name = "YourPipelineName"
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls(env, id)
```

## Critical Pattern: Using Decorators

### @Pipeline.run_wrapper Decorator

**CRITICAL**: The `run` method MUST use the `@Pipeline.run_wrapper` decorator:

```python
@Pipeline.run_wrapper
def run(self, config: DictConfig) -> Dict[str, Any]:
    """Run the pipeline with proper lifecycle management."""
    # Your implementation here
    return {"status": "completed"}
```

**What the decorator does:**
- Automatically calls `_on_run_start()` before execution
- Handles exceptions and status management
- Automatically calls `_on_run_end()` after execution
- Ensures proper tracker lifecycle management

### @Pipeline.epoch_wrapper Decorator (Optional)

If you implement epoch-level training, use the epoch wrapper:

```python
@Pipeline.epoch_wrapper
def run_epoch(self, epoch_idx: int, model, *args, **kwargs) -> RunStatus:
    """Run one epoch with proper epoch lifecycle management."""
    # Your epoch implementation here
    return RunStatus.SUCCESS
```

## Proper Metrics Recording

### Using self.run_metrics and self.epoch_metrics

The Pipeline base class provides dictionaries for metrics that are automatically tracked:

```python
@Pipeline.run_wrapper
def run(self, config: DictConfig) -> Dict[str, Any]:
    # Log parameters at the start
    self.env.tracker_manager.log_params({
        "model_type": "ResNet50",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": config.pipeline.epochs
    })
    
    for epoch in range(config.pipeline.epochs):
        # Manual epoch lifecycle management
        self.env.tracker_manager.on_create(Level.EPOCH, epoch_id=epoch)
        self.env.tracker_manager.on_start(Level.EPOCH)
        
        # Your training code here...
        train_loss = 0.5  # Your actual loss
        train_acc = 0.8   # Your actual accuracy
        
        # Store metrics in epoch_metrics for automatic tracking
        self.epoch_metrics[Metric.TRAIN_LOSS] = train_loss
        self.epoch_metrics[Metric.TRAIN_ACC] = train_acc
        
        # Or track directly with step information
        self.env.tracker_manager.track(Metric.TRAIN_LOSS, train_loss, step=epoch)
        self.env.tracker_manager.track(Metric.TRAIN_ACC, train_acc, step=epoch)
        
        # Custom metrics
        self.env.tracker_manager.track(
            Metric.CUSTOM, 
            ("batch_processing_time", 1.2), 
            step=epoch
        )
        
        self.env.tracker_manager.on_end(Level.EPOCH)
    
    # Store final metrics in run_metrics
    self.run_metrics[Metric.TEST_ACC] = final_test_accuracy
    self.run_metrics[Metric.TEST_LOSS] = final_test_loss
    
    return {"test_accuracy": final_test_accuracy}
```

### Available Metrics

Use the predefined `Metric` enum values:

```python
# Standard metrics
Metric.TRAIN_LOSS      # Training loss
Metric.TRAIN_ACC       # Training accuracy
Metric.VAL_LOSS        # Validation loss
Metric.VAL_ACC         # Validation accuracy
Metric.TEST_LOSS       # Test loss
Metric.TEST_ACC        # Test accuracy
Metric.LEARNING_RATE   # Learning rate

# Custom metrics
Metric.CUSTOM          # Format: (metric_name, value)
```

## Checkpoint Management

Properly save checkpoints using the tracker manager:

```python
# During training
if epoch % checkpoint_interval == 0:
    checkpoint_path = f"checkpoints/epoch_{epoch}.pth"
    
    # Save actual model if you have one
    if hasattr(self, 'model'):
        torch.save(self.model.state_dict(), checkpoint_path)
    
    # Record checkpoint with metrics
    self.env.tracker_manager.on_checkpoint(
        network=self.model if hasattr(self, 'model') else None,
        checkpoint_path=checkpoint_path,
        metrics={
            Metric.TRAIN_LOSS: current_loss,
            Metric.TRAIN_ACC: current_accuracy,
            "epoch": epoch
        }
    )
```

## Complete Example: Corrected Performance Demo Pipeline

There are two approaches to implementing a pipeline. Let's show both patterns:

### Approach 1: Using @Pipeline.epoch_wrapper (Recommended)

```python
@YAMLSerializable.register("PerformanceDemoPipelineFixed")
class PerformanceDemoPipelineFixed(Pipeline, YAMLSerializable):
    """Performance monitoring demo with proper metrics tracking using epoch_wrapper."""
    
    def __init__(self, env: Environment, id: int = None):
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self)
        self.name = "PerformanceDemoPipelineFixed"
        
        # Initialize training components (would be actual model in real implementation)
        self.model = None  # In real implementation, initialize your model here
        self.batches_per_epoch = 8
        self.epochs = 3
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls(env, id)
    
    @Pipeline.run_wrapper  # CRITICAL: This decorator is required!
    def run(self, config: DictConfig) -> Dict[str, Any]:
        """Run the pipeline with proper lifecycle management."""
        
        # Get configuration
        self.epochs = config.pipeline.get('epochs', 3)
        self.batches_per_epoch = config.pipeline.get('batches_per_epoch', 8)
        
        # Log parameters - tracked by all trackers
        self.env.tracker_manager.log_params({
            "pipeline_type": "PerformanceDemoPipelineFixed",
            "epochs": self.epochs,
            "batches_per_epoch": self.batches_per_epoch,
            "model_type": "ResNet50_Demo",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 64
        })
        
        # Training loop using run_epoch with @epoch_wrapper
        for epoch in range(self.epochs):
            self.env.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Call run_epoch with @epoch_wrapper - handles lifecycle automatically
            self.run_epoch(epoch, self.model)
        
        # Final test evaluation (after all epochs)
        final_test_loss = 0.15 + np.random.normal(0, 0.01)
        final_test_acc = 0.92 + np.random.normal(0, 0.01)
        final_test_loss = max(0.1, final_test_loss)
        final_test_acc = min(0.95, max(0.0, final_test_acc))
        
        # Store final metrics in run_metrics for automatic tracking
        self.run_metrics[Metric.TEST_LOSS] = final_test_loss
        self.run_metrics[Metric.TEST_ACC] = final_test_acc
        
        # Also track directly for immediate availability
        self.env.tracker_manager.track(Metric.TEST_LOSS, final_test_loss)
        self.env.tracker_manager.track(Metric.TEST_ACC, final_test_acc)
        
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
        
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        
        # Batch training simulation
        for batch in range(self.batches_per_epoch):
            # Simulate varying workload
            if batch % 3 == 0:
                self._simulate_cpu_work(1.0)
            elif batch % 3 == 1:
                self._simulate_memory_work(200)
            else:
                self._simulate_mixed_work(0.8, 150)
            
            # Realistic training metrics with learning progress
            base_loss = 1.5
            base_acc = 0.4
            progress = (epoch_idx * self.batches_per_epoch + batch) / (self.epochs * self.batches_per_epoch)
            
            batch_loss = base_loss * (1 - progress * 0.7) + np.random.normal(0, 0.02)
            batch_acc = base_acc + progress * 0.5 + np.random.normal(0, 0.01)
            
            # Clamp values
            batch_loss = max(0.1, batch_loss)
            batch_acc = min(0.95, max(0.0, batch_acc))
            
            epoch_train_loss += batch_loss
            epoch_train_acc += batch_acc
            
            # Track batch-level metrics with step
            step = epoch_idx * self.batches_per_epoch + batch
            self.env.tracker_manager.track(Metric.TRAIN_LOSS, batch_loss, step=step)
            self.env.tracker_manager.track(Metric.TRAIN_ACC, batch_acc, step=step)
            
            # Custom metrics
            self.env.tracker_manager.track(
                Metric.CUSTOM, 
                ("batch_processing_time", 1.0), 
                step=step
            )
        
        # Epoch averages
        avg_train_loss = epoch_train_loss / self.batches_per_epoch
        avg_train_acc = epoch_train_acc / self.batches_per_epoch
        
        # Validation simulation
        val_loss = avg_train_loss + 0.05 + np.random.normal(0, 0.01)
        val_acc = avg_train_acc - 0.02 + np.random.normal(0, 0.01)
        val_loss = max(0.1, val_loss)
        val_acc = min(0.95, max(0.0, val_acc))
        
        # CRITICAL: Store in epoch_metrics - these will be automatically tracked by @epoch_wrapper!
        self.epoch_metrics[Metric.TRAIN_LOSS] = avg_train_loss
        self.epoch_metrics[Metric.TRAIN_ACC] = avg_train_acc
        self.epoch_metrics[Metric.VAL_LOSS] = val_loss
        self.epoch_metrics[Metric.VAL_ACC] = val_acc
        
        # Also track validation directly with epoch step
        self.env.tracker_manager.track(Metric.VAL_LOSS, val_loss, step=epoch_idx)
        self.env.tracker_manager.track(Metric.VAL_ACC, val_acc, step=epoch_idx)
        
        # Checkpoint every few epochs
        if epoch_idx % 2 == 0:
            checkpoint_path = f"checkpoints/epoch_{epoch_idx}.pth"
            self.env.tracker_manager.on_checkpoint(
                network=model,  # Pass the actual model if available
                checkpoint_path=checkpoint_path,
                metrics={
                    Metric.TRAIN_LOSS: avg_train_loss,
                    Metric.TRAIN_ACC: avg_train_acc,
                    Metric.VAL_LOSS: val_loss,
                    Metric.VAL_ACC: val_acc
                }
            )
            self.env.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return RunStatus.SUCCESS
    
    def _simulate_cpu_work(self, duration: float):
        """Simulate CPU-intensive work."""
        start_time = time.time()
        while time.time() - start_time < duration:
            _ = sum(i * i for i in range(1000))
    
    def _simulate_memory_work(self, mb: int):
        """Simulate memory-intensive work."""
        arrays = []
        for _ in range(mb // 10):
            arrays.append(np.random.random((1000, 100)))
        time.sleep(0.1)  # Simulate processing
        del arrays
    
    def _simulate_mixed_work(self, duration: float, mb: int):
        """Simulate mixed CPU and memory work."""
        self._simulate_cpu_work(duration * 0.6)
        self._simulate_memory_work(mb)
```

### Approach 2: Manual Epoch Management (If you can't use run_epoch)

```python
@YAMLSerializable.register("PerformanceDemoPipelineManual") 
class PerformanceDemoPipelineManual(Pipeline, YAMLSerializable):
    """Performance monitoring demo with manual epoch lifecycle management."""
    
    @Pipeline.run_wrapper  # Still required!
    def run(self, config: DictConfig) -> Dict[str, Any]:
        epochs = config.pipeline.get('epochs', 3)
        batches_per_epoch = config.pipeline.get('batches_per_epoch', 8)
        
        # Log parameters
        self.env.tracker_manager.log_params({
            "pipeline_type": "PerformanceDemoPipelineManual",
            "epochs": epochs,
            "batches_per_epoch": batches_per_epoch
        })
        
        # Training loop with manual epoch management
        for epoch in range(epochs):
            self.env.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Manual epoch lifecycle management
            self.env.tracker_manager.on_create(Level.EPOCH, epoch_id=epoch)
            self.env.tracker_manager.on_start(Level.EPOCH)
            
            # CRITICAL: Clear epoch metrics at start of each epoch
            self.epoch_metrics.clear()
            
            # ... training code here ...
            
            # Store epoch metrics
            self.epoch_metrics[Metric.TRAIN_LOSS] = avg_train_loss
            self.epoch_metrics[Metric.TRAIN_ACC] = avg_train_acc
            
            # Manual end of epoch - this tracks epoch_metrics automatically
            self.env.tracker_manager.on_end(Level.EPOCH)
            
            # Manual clearing (since we're not using @epoch_wrapper)
            self.epoch_metrics.clear()
        
        # Store final results
        self.run_metrics[Metric.TEST_LOSS] = final_test_loss
        self.run_metrics[Metric.TEST_ACC] = final_test_acc
        
        return {"status": "completed"}
```

## 🐛 Bug Fix Applied

**IMPORTANT**: We discovered and fixed a bug in the `@Pipeline.epoch_wrapper` decorator! The wrapper was not clearing `self.epoch_metrics` after each epoch, causing metrics to accumulate across epochs.

**Fixed in**: `experiment_manager/pipelines/pipeline.py` line 121
**Change**: Added `self.epoch_metrics.clear()` in the `finally` block after `_on_epoch_end()`

This means:
- ✅ **If you use `@Pipeline.epoch_wrapper`**: Metrics are now automatically cleared between epochs
- ⚠️ **If you use manual epoch management**: You MUST clear `self.epoch_metrics.clear()` manually after each epoch

## Common Mistakes to Avoid

### ❌ WRONG: Missing @Pipeline.run_wrapper

```python
def run(self, config: DictConfig):  # Missing decorator!
    # This will not properly integrate with the framework
    pass
```

### ❌ WRONG: Not using epoch_metrics or run_metrics

```python
@Pipeline.run_wrapper
def run(self, config: DictConfig):
    loss = 0.5
    # Only tracking directly, metrics won't be captured by all trackers
    self.env.tracker_manager.track(Metric.TRAIN_LOSS, loss)
```

### ❌ WRONG: Not using proper Metric enums

```python
# Don't do this
self.env.tracker_manager.track("train_loss", loss)  # Wrong!

# Do this instead
self.env.tracker_manager.track(Metric.TRAIN_LOSS, loss)  # Correct!
```

### ❌ WRONG: Not managing epoch lifecycle

```python
for epoch in range(epochs):
    # Missing epoch lifecycle management
    train_loss = self.train_one_epoch()
    # Won't properly track epoch boundaries
```

## ✅ Correct Patterns Summary

1. **Always use `@Pipeline.run_wrapper`** on your `run` method
2. **Use `self.epoch_metrics` and `self.run_metrics`** dictionaries for automatic tracking
3. **Use proper `Metric` enum values** for all tracking calls
4. **Manage epoch lifecycle** with `on_create(Level.EPOCH)`, `on_start(Level.EPOCH)`, `on_end(Level.EPOCH)`
5. **Log parameters** at the start with `log_params()`
6. **Track checkpoints** with `on_checkpoint()`
7. **Use step parameters** for time-series metrics tracking

## Custom Metrics: Tracked vs Untracked

The framework provides two types of custom metrics to give you fine-grained control over what gets persisted to trackers:

### Metric.CUSTOM (Tracked)

**Use for:** Important metrics you want to visualize, analyze, and compare across experiments.

**Goes to:** All trackers (Database, MLflow, TensorBoard, Log files, etc.)

```python
@Pipeline.epoch_wrapper
def run_epoch(self, epoch_idx: int, model, *args, **kwargs):
    # ... training code ...
    
    # These metrics will be saved to all trackers
    self.epoch_metrics[Metric.CUSTOM] = [
        ("model_accuracy", accuracy_value),
        ("learning_rate", current_lr),
        ("gradient_norm", grad_norm),
        ("validation_score", val_score),
    ]
    
    return RunStatus.FINISHED
```

### Metric.CUSTOM_UNTRACKED (Untracked)

**Use for:** Debug information, temporary calculations, and internal state that you don't want cluttering your tracking databases.

**Goes to:** Only accessible in callbacks via the `epoch_metrics` dictionary. NOT saved to any trackers.

```python
@Pipeline.epoch_wrapper
def run_epoch(self, epoch_idx: int, model, *args, **kwargs):
    # ... training code ...
    
    # These metrics are only accessible in callbacks, not tracked
    self.epoch_metrics[Metric.CUSTOM_UNTRACKED] = [
        ("debug_tensor_norm", debug_value),
        ("temp_calculation", temp_value),
        ("internal_counter", counter_value),
        ("intermediate_state", state_info),
    ]
    
    return RunStatus.FINISHED
```

### Using Both Together

You can use both metric types in the same epoch:

```python
@Pipeline.epoch_wrapper
def run_epoch(self, epoch_idx: int, model, *args, **kwargs):
    # ... training code ...
    
    # Tracked: Important metrics for analysis
    self.epoch_metrics[Metric.CUSTOM] = [
        ("val_accuracy", val_acc),
        ("learning_rate", lr),
    ]
    
    # Untracked: Debug info
    self.epoch_metrics[Metric.CUSTOM_UNTRACKED] = [
        ("debug_gradient_max", grad_max),
        ("debug_memory_usage", mem_usage),
    ]
    
    # Standard metrics work as always
    self.epoch_metrics[Metric.TRAIN_LOSS] = train_loss
    self.epoch_metrics[Metric.VAL_ACC] = val_acc
    
    return RunStatus.FINISHED
```

### Accessing Untracked Metrics in Callbacks

Create a custom callback to process untracked metrics:

```python
class DebugMetricsCallback(Callback):
    """Process untracked debug metrics."""
    
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        # Access untracked metrics
        if Metric.CUSTOM_UNTRACKED in metrics:
            for name, value in metrics[Metric.CUSTOM_UNTRACKED]:
                # Conditional logging based on debug values
                if "debug_gradient_max" in name and value > 10.0:
                    self.env.logger.warning(f"High gradient detected: {value}")
        
        return False  # Continue training
```

### Comparison

| Feature | `Metric.CUSTOM` | `Metric.CUSTOM_UNTRACKED` |
|---------|----------------|---------------------------|
| **Saved to Database** | ✅ Yes | ❌ No |
| **Saved to MLflow** | ✅ Yes | ❌ No |
| **Saved to TensorBoard** | ✅ Yes | ❌ No |
| **Available in Callbacks** | ✅ Yes | ✅ Yes |
| **Use Case** | Important metrics | Debug/temporary data |
| **Example** | Validation accuracy, learning rate | Debug info, intermediate calculations |

See `examples/pipelines/untracked_metrics_demo.py` for a complete working example.

---

## Factory Implementation

Don't forget to implement the corresponding factory:

```python
class YourPipelineFactory(PipelineFactory):
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment) -> Pipeline:
        return YourPipeline.from_config(config, env)
```

This guide ensures your pipeline integrates properly with all trackers (MLflow, TensorBoard, Database, etc.) and follows the framework's architecture correctly. 