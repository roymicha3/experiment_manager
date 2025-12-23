# Pipeline Development: Critical Fixes & Improvements

## 🚨 Critical Bug Fixed: Epoch Metrics Accumulation

### Problem Discovered
The `@Pipeline.epoch_wrapper` decorator in `experiment_manager/pipelines/pipeline.py` was **not clearing `self.epoch_metrics`** after each epoch, causing metrics to accumulate across epochs.

### Root Cause
```python
# BEFORE (buggy):
finally:
    should_stop = self._on_epoch_end(epoch_idx, self.epoch_metrics)
    # Missing: self.epoch_metrics.clear()
    if should_stop:
        raise StopIteration("Stopping pipeline execution")
```

### Fix Applied
```python
# AFTER (fixed):
finally:
    should_stop = self._on_epoch_end(epoch_idx, self.epoch_metrics)
    
    # BUG FIX: Clear epoch metrics after each epoch to prevent accumulation
    self.epoch_metrics.clear()
    
    if should_stop:
        raise StopIteration("Stopping pipeline execution")
```

### Impact
- ✅ **Fixed**: Epoch metrics no longer accumulate across epochs when using `@Pipeline.epoch_wrapper`
- ⚠️ **Action Required**: Manual epoch management still requires explicit `self.epoch_metrics.clear()`

## 📚 Guide Improvements: Proper Pipeline Patterns

### Original Issues in Example Pipeline
1. **Missing `@Pipeline.run_wrapper`** - Critical for lifecycle management
2. **Not using `run_epoch` with `@Pipeline.epoch_wrapper`** - Missed automatic epoch lifecycle
3. **Not utilizing `self.epoch_metrics` and `self.run_metrics`** - Framework's automatic tracking system
4. **Manual epoch management without proper clearing** - Led to discovering the bug
5. **Not using proper `Metric` enum values** - Framework expects specific constants

### Correct Pattern: Using @epoch_wrapper (Recommended)

```python
@YAMLSerializable.register("YourPipeline")
class YourPipeline(Pipeline, YAMLSerializable):
    
    @Pipeline.run_wrapper  # REQUIRED for lifecycle management
    def run(self, config: DictConfig) -> Dict[str, Any]:
        # Log parameters for all trackers
        self.env.tracker_manager.log_params({...})
        
        # Training loop using run_epoch
        for epoch in range(epochs):
            self.run_epoch(epoch, model)  # Handles lifecycle automatically
        
        # Store final metrics in run_metrics
        self.run_metrics[Metric.TEST_LOSS] = final_loss
        self.run_metrics[Metric.TEST_ACC] = final_acc
        
        return {"status": "completed"}
    
    @Pipeline.epoch_wrapper  # Handles epoch lifecycle + metrics automatically
    def run_epoch(self, epoch_idx: int, model, *args, **kwargs) -> RunStatus:
        # Training code here...
        
        # Store metrics in epoch_metrics - automatically tracked!
        self.epoch_metrics[Metric.TRAIN_LOSS] = avg_loss
        self.epoch_metrics[Metric.TRAIN_ACC] = avg_acc
        self.epoch_metrics[Metric.VAL_LOSS] = val_loss
        self.epoch_metrics[Metric.VAL_ACC] = val_acc
        
        return RunStatus.SUCCESS
```

### Alternative Pattern: Manual Epoch Management

```python
@Pipeline.run_wrapper
def run(self, config: DictConfig) -> Dict[str, Any]:
    for epoch in range(epochs):
        # Manual epoch lifecycle
        self.env.tracker_manager.on_create(Level.EPOCH, epoch_id=epoch)
        self.env.tracker_manager.on_start(Level.EPOCH)
        
        # CRITICAL: Clear at start of each epoch
        self.epoch_metrics.clear()
        
        # Training code...
        self.epoch_metrics[Metric.TRAIN_LOSS] = loss
        
        # End epoch - tracks epoch_metrics automatically
        self.env.tracker_manager.on_end(Level.EPOCH)
        
        # CRITICAL: Clear after tracking (manual management)
        self.epoch_metrics.clear()
```

## 🔧 Framework Integration Benefits

### Using Proper Patterns Enables:
1. **Automatic MLflow tracking** - All metrics logged to MLflow automatically
2. **Database persistence** - Metrics stored in experiment database 
3. **TensorBoard integration** - Real-time visualization
4. **Performance monitoring** - PerformanceTracker gets full lifecycle events
5. **Checkpoint management** - Proper model checkpointing
6. **Callback system** - EarlyStopping, MetricsTracker, etc.

### What Gets Automatically Tracked:
- `self.epoch_metrics` → Tracked at end of each epoch
- `self.run_metrics` → Tracked at end of the run
- Parameters from `log_params()` → Tracked by all trackers
- Direct `tracker_manager.track()` calls → Immediate tracking

## 📋 Verification Checklist

When implementing a pipeline, verify:

- [ ] `@Pipeline.run_wrapper` decorator on `run()` method
- [ ] Either `@Pipeline.epoch_wrapper` on `run_epoch()` OR manual epoch lifecycle
- [ ] Using `Metric` enum constants (not strings)
- [ ] Storing metrics in `self.epoch_metrics` and `self.run_metrics`
- [ ] Calling `log_params()` at start of training
- [ ] Proper checkpoint handling with `on_checkpoint()`
- [ ] Manual `epoch_metrics.clear()` if not using `@epoch_wrapper`

## 🎯 Next Steps

1. **Update existing pipelines** to use proper decorator patterns
2. **Test epoch metrics clearing** in your pipelines
3. **Verify all trackers** receive metrics correctly
4. **Check performance monitoring** integrates properly
5. **Review checkpoint management** follows the patterns

---
*Fixed: December 5, 2024 - Critical bug in epoch_wrapper + pipeline guide improvements* 

# Pipeline Implementation Fixes Summary

This document summarizes the fixes applied to example pipelines to align with the Pipeline Development Guide.

## Fixed Issues

### 1. **Performance Demo Pipeline** - `examples/pipelines/performance_demo_pipeline.py`

**Issues Fixed:**
- ✅ **Missing `@Pipeline.run_wrapper` decorator** - Added to `run()` method
- ✅ **Missing `@Pipeline.epoch_wrapper` decorator** - Added `run_epoch()` method with decorator
- ✅ **Manual epoch lifecycle management** - Replaced with proper decorator-based approach
- ✅ **No epoch_metrics clearing** - Now handled automatically by `@epoch_wrapper`
- ✅ **Performance optimizations** - Reduced timing and resource usage

**Key Changes:**
```python
@Pipeline.run_wrapper  # ADDED
def run(self, config: DictConfig) -> Dict[str, Any]:
    # ... config processing ...
    for epoch in range(self.epochs):
        self.run_epoch(epoch, self.model)  # CHANGED: Use run_epoch
    
    # Store final metrics in run_metrics
    self.run_metrics[Metric.TEST_LOSS] = final_test_loss
    self.run_metrics[Metric.TEST_ACC] = final_test_acc
    return results

@Pipeline.epoch_wrapper  # ADDED
def run_epoch(self, epoch_idx: int, model, *args, **kwargs) -> RunStatus:
    # ... epoch implementation ...
    
    # Store in epoch_metrics for automatic tracking
    self.epoch_metrics[Metric.TRAIN_LOSS] = avg_train_loss
    self.epoch_metrics[Metric.TRAIN_ACC] = avg_train_acc
    self.epoch_metrics[Metric.VAL_LOSS] = val_loss
    self.epoch_metrics[Metric.VAL_ACC] = val_acc
    
    return RunStatus.SUCCESS
```

**Performance Optimizations:**
- Reduced `work_duration_seconds` from 1.2 to 0.2
- Reduced `cpu_intensity` from 1.5 to 0.5  
- Reduced `memory_usage_mb` from 300 to 100
- Reduced `batches_per_epoch` from 8 to 5
- Optimized simulation methods for faster execution

### 2. **Dummy Pipeline** - `examples/pipelines/dummy_pipeline.py`

**Issues Fixed:**
- ✅ **Missing `@Pipeline.run_wrapper` decorator** - Added to `run()` method
- ✅ **Missing `@Pipeline.epoch_wrapper` decorator** - Added `run_epoch()` method with decorator
- ✅ **Manual epoch lifecycle management** - Replaced with proper decorator-based approach
- ✅ **No epoch_metrics clearing** - Now handled automatically by `@epoch_wrapper`

**Key Changes:**
```python
@Pipeline.run_wrapper  # ADDED
def run(self, config: DictConfig) -> Dict[str, Any]:
    # ... parameter logging ...
    for epoch in range(self.epochs):
        self.run_epoch(epoch, self.model)  # CHANGED: Use run_epoch
    
    # Store final metrics in run_metrics
    self.run_metrics[Metric.TEST_ACC] = final_test_acc
    self.run_metrics[Metric.TEST_LOSS] = final_test_loss
    return results

@Pipeline.epoch_wrapper  # ADDED
def run_epoch(self, epoch_idx: int, model, *args, **kwargs) -> RunStatus:
    # ... training simulation ...
    
    # Store in epoch_metrics for automatic tracking
    self.epoch_metrics[Metric.TRAIN_ACC] = train_acc
    self.epoch_metrics[Metric.TRAIN_LOSS] = train_loss
    self.epoch_metrics[Metric.VAL_ACC] = val_acc
    self.epoch_metrics[Metric.VAL_LOSS] = val_loss
    
    return RunStatus.SUCCESS
```

### 3. **Simple Classifier Pipeline** - `examples/pipelines/simple_classifier.py`

**Status:** ✅ **Already Correctly Implemented**
- Already uses `@Pipeline.run_wrapper` decorator
- Already uses `@Pipeline.epoch_wrapper` decorator  
- Properly handles epoch metrics

**Configuration Optimization:**
- Reduced epochs from 15 to 5 in `examples/configs/simple_experiment/base.yaml`

### 4. **Configuration Optimizations**

**Performance Demo Config** - `examples/configs/performance_demo/base.yaml`:
- `work_duration_seconds`: 1.2 → 0.2
- `cpu_intensity`: 1.5 → 0.5
- `memory_usage_mb`: 300 → 100
- `batches_per_epoch`: 8 → 5
- `monitoring_interval`: 0.5s → 1.0s (reduced overhead)
- `history_size`: 2000 → 1000 (reduced memory usage)

**Simple Experiment Config** - `examples/configs/simple_experiment/base.yaml`:
- `epochs`: 15 → 5

## Benefits of These Fixes

### 1. **Proper Lifecycle Management**
- Automatic epoch creation, start, and end handling
- Proper metrics tracking and clearing
- Exception safety with `finally` blocks

### 2. **Automatic Metrics Handling**
- `epoch_metrics` automatically tracked and cleared
- `run_metrics` automatically tracked at pipeline end
- Prevents metric accumulation between epochs

### 3. **Better Performance**
- Significantly reduced execution time (from ~30+ seconds to ~5-10 seconds)
- Optimized resource usage while maintaining demo functionality
- Reduced monitoring overhead

### 4. **Consistency**
- All pipelines now follow the same patterns
- Consistent error handling and status management
- Proper return types and structures

## Usage Notes

### For New Pipelines:
- ⚠️ **Always use `@Pipeline.run_wrapper` on your `run()` method**
- ⚠️ **Use `@Pipeline.epoch_wrapper` if implementing epoch-based training**
- ⚠️ **Store metrics in `self.epoch_metrics` and `self.run_metrics`**
- ⚠️ **Return proper types: `Dict[str, Any]` for run, `RunStatus` for run_epoch**

### Migration Checklist:
- [ ] Add `@Pipeline.run_wrapper` to `run()` method
- [ ] Add `@Pipeline.epoch_wrapper` to `run_epoch()` method if applicable
- [ ] Remove manual epoch lifecycle calls (`on_create`, `on_start`, `on_end`)
- [ ] Use `self.epoch_metrics` instead of direct tracking in epochs
- [ ] Use `self.run_metrics` for final pipeline metrics
- [ ] Update return types and add proper typing imports
- [ ] Test that epoch metrics are properly cleared between epochs

### Performance Guidelines:
- Keep simulation work under 0.5 seconds per batch for demos
- Use reasonable epoch counts (3-5 for demos)
- Optimize monitoring intervals based on demo requirements
- Consider lightweight mode for performance trackers in production 