# Custom Factory Implementation Guide

This guide shows you how to create and use custom factories in your experiments, giving you full control over how pipelines, callbacks, and trackers are instantiated.

---

## Why Use Custom Factories?

Custom factories allow you to:

- **Control object instantiation** - Add custom logic before/after creating objects
- **Override default behavior** - Replace framework defaults with your own implementations
- **Add validation** - Ensure configurations meet your requirements before instantiation
- **Implement plugins** - Load external modules or classes dynamically

---

## Quick Start: 3 Steps

### 1. Create Your Custom Factory

Inherit from the appropriate factory class and override the `create` method:

```python
from omegaconf import DictConfig
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.pipelines.pipeline import Pipeline

class MyCustomPipelineFactory(PipelineFactory):
    """Custom factory with additional logic."""

    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        # Add your custom logic here
        env.logger.info(f"🎯 Creating pipeline '{name}' with custom factory")

        # Optional: Add validation
        if "required_field" not in config.pipeline:
            raise ValueError("Missing required_field in pipeline config")

        # Call parent create to handle standard pipeline creation + callbacks
        pipeline = PipelineFactory.create(name, config, env, id)

        # Optional: Additional post-creation setup
        env.logger.info(f"✅ Pipeline '{name}' created and customized")

        return pipeline
```

### 2. Register Your Factory

Use the `FactoryRegistry` to register your custom factory:

```python
from experiment_manager.experiment import Experiment
from experiment_manager.common.factory_registry import FactoryRegistry, FactoryType
from my_custom.factories import MyCustomPipelineFactory

# Create a registry
registry = FactoryRegistry()

# Register your custom factory
registry.register(FactoryType.PIPELINE, MyCustomPipelineFactory())

# You can also register multiple custom factories
# registry.register(FactoryType.CALLBACK, MyCustomCallbackFactory())
# registry.register(FactoryType.TRACKER, MyCustomTrackerFactory())
```

### 3. Use in Your Experiment

Pass the registry to your experiment:

```python
# Create experiment with custom factories
experiment = Experiment.create(
    config_dir="path/to/config",
    factory_registry=registry
)

# Run as normal - your custom factory will be used automatically
experiment.run()
```

---

## Complete Working Example

Here's a complete example showing a custom pipeline factory that adds automatic error recovery:

**File: `my_project/factories/resilient_factory.py`**

```python
from omegaconf import DictConfig
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.pipelines.pipeline import Pipeline

class ResilientPipelineFactory(PipelineFactory):
    """
    Pipeline factory that adds automatic retry logic and error recovery.
    """

    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        env.logger.info(f"🛡️ Creating resilient pipeline: {name}")

        # Validate configuration
        if config.pipeline.get("enable_retry", False):
            max_retries = config.pipeline.get("max_retries", 3)
            env.logger.info(f"   Retry enabled: max {max_retries} attempts")

        # Create pipeline with standard logic
        pipeline = PipelineFactory.create(name, config, env, id)

        # Store custom attributes
        pipeline.max_retries = config.pipeline.get("max_retries", 3)
        pipeline.retry_count = 0

        env.logger.info(f"✅ Resilient pipeline created: {name}")
        return pipeline
```

**File: `my_project/run_experiment.py`**

```python
#!/usr/bin/env python
"""Run experiment with custom resilient factory."""
import os
from pathlib import Path

from experiment_manager.experiment import Experiment
from experiment_manager.common.factory_registry import FactoryRegistry, FactoryType
from my_project.factories.resilient_factory import ResilientPipelineFactory

def main():
    # Configuration directory
    config_dir = Path(__file__).parent / "configs" / "my_experiment"

    # Create custom factory registry
    registry = FactoryRegistry()
    registry.register(FactoryType.PIPELINE, ResilientPipelineFactory())

    # Create and run experiment
    print("🚀 Starting experiment with custom factory...")
    experiment = Experiment.create(str(config_dir), factory_registry=registry)
    experiment.run()
    print("✅ Experiment completed!")

if __name__ == "__main__":
    main()
```

---

## Advanced: Custom Callback Factory

You can also create custom callback factories:

```python
from omegaconf import DictConfig
from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.callback_factory import CallbackFactory

class MyCallbackFactory(CallbackFactory):
    """Custom callback factory with dependency injection."""

    def __init__(self, external_service=None):
        """
        Initialize with optional external dependencies.

        Args:
            external_service: External service to inject into callbacks
        """
        self.external_service = external_service

    @staticmethod
    def create(name: str, config: DictConfig, env: Environment):
        # Get callback from registry
        callback = CallbackFactory.create(name, config, env)

        # Inject dependencies if available
        if hasattr(self, 'external_service') and self.external_service:
            callback.external_service = self.external_service

        return callback

# Usage:
from my_services import NotificationService

service = NotificationService()
callback_factory = MyCallbackFactory(external_service=service)

registry = FactoryRegistry()
registry.register(FactoryType.CALLBACK, callback_factory)
```

---

## Advanced: Custom Tracker Factory

Create custom tracker factories for specialized tracking logic:

```python
from omegaconf import DictConfig
from experiment_manager.trackers.tracker_factory import TrackerFactory
from experiment_manager.trackers.tracker import Tracker

class CloudTrackerFactory(TrackerFactory):
    """Factory for creating cloud-based trackers."""

    @staticmethod
    def create(name: str, config: DictConfig, workspace: str) -> Tracker:
        # Standard creation
        tracker = TrackerFactory.create(name, config, workspace)

        # Add cloud-specific configuration
        if hasattr(config, 'cloud_bucket'):
            tracker.cloud_bucket = config.cloud_bucket
            tracker.enable_cloud_sync = True

        return tracker

# Usage:
registry = FactoryRegistry()
registry.register(FactoryType.TRACKER, CloudTrackerFactory())
```

---

## Factory Types Reference

The framework supports three factory types:

| Factory Type           | Purpose                    | Default Factory   |
| ---------------------- | -------------------------- | ----------------- |
| `FactoryType.PIPELINE` | Creates pipeline instances | `PipelineFactory` |
| `FactoryType.CALLBACK` | Creates callback instances | `CallbackFactory` |
| `FactoryType.TRACKER`  | Creates tracker instances  | `TrackerFactory`  |

---

## Best Practices

1. **Always call parent `create()`** - Unless you have a very specific reason, call the parent's `create()` method to leverage existing logic

2. **Add logging** - Log factory operations to help with debugging

3. **Validate early** - Check configuration requirements in the factory before instantiation

4. **Keep it simple** - Factories should be thin wrappers; complex logic belongs in the objects themselves

5. **Document your factories** - Clearly document what customizations your factory adds

6. **Test thoroughly** - Ensure your custom factory works with all expected configurations

---

## Troubleshooting

### Issue: `TypeError: factory must be an instance of Factory`

**Solution:** Make sure you're passing an **instance** (with parentheses):

```python
# ❌ Wrong - passing the class
registry.register(FactoryType.PIPELINE, MyFactory)

# ✅ Correct - passing an instance
registry.register(FactoryType.PIPELINE, MyFactory())
```

### Issue: Circular import errors

**Solution:** Import factories only where needed, not at module level:

```python
def create_registry():
    # Import here, not at top of file
    from my_custom.factories import MyFactory

    registry = FactoryRegistry()
    registry.register(FactoryType.PIPELINE, MyFactory())
    return registry
```

### Issue: Custom factory not being used

**Solution:** Verify the registry is passed to `Experiment.create()`:

```python
# ❌ Wrong - using default factories
experiment = Experiment.create(config_dir)

# ✅ Correct - using custom registry
experiment = Experiment.create(config_dir, factory_registry=registry)
```

---

## Next Steps

- See `examples/` directory for more factory examples
- Read `experiment_manager/pipelines/pipeline_factory.py` to understand the base implementation
- Check `experiment_manager/common/factory_registry.py` for API details

---

**Questions?** Check the main documentation or examine the factory implementations in the framework source code.
