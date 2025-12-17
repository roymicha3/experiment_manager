from typing import Dict, TYPE_CHECKING
from enum import Enum

from experiment_manager.common.factory import Factory

# Avoid circular imports by using lazy imports
if TYPE_CHECKING:
    from experiment_manager.pipelines.pipeline_factory import PipelineFactory
    from experiment_manager.pipelines.callbacks.callback_factory import CallbackFactory
    from experiment_manager.trackers.tracker_factory import TrackerFactory

class FactoryType(Enum):
    """
    Enum class for factory types supported by the FactoryRegistry.
    
    Attributes:
        PIPELINE: Factory type for creating pipeline instances
        CALLBACK: Factory type for creating callback instances
        TRACKER: Factory type for creating tracker instances
    """
    PIPELINE = "pipeline"
    CALLBACK = "callback"
    TRACKER = "tracker"


class FactoryRegistry:
    """
    Central registry for managing factory instances across the experiment framework.
    
    This class provides a single source of truth for all factory objects used throughout
    the experiment lifecycle. Users can override default factories by registering custom
    implementations, enabling full control over object instantiation.
    
    The registry is designed to support dependency injection patterns, allowing custom
    factories to be passed to experiments without modifying core framework code.
    
    Example:
        # Using default factories
        registry = FactoryRegistry()
        
        # Overriding with custom factory
        custom_factory = MyCustomPipelineFactory()
        registry.register(FactoryType.PIPELINE, custom_factory)
        
        # Pass to experiment
        experiment = Experiment.create(config_dir, factory_registry=registry)
    
    Attributes:
        factories: Dictionary mapping FactoryType enum values to Factory instances
    """
    def __init__(self):
        """
        Initialize the registry with default factory instances.
        
        Default factories:
            - PIPELINE: PipelineFactory for creating pipeline instances
            - CALLBACK: CallbackFactory for creating callback instances
            - TRACKER: TrackerFactory for creating tracker instances
        """
        # Lazy import to avoid circular dependency
        from experiment_manager.pipelines.pipeline_factory import PipelineFactory
        from experiment_manager.pipelines.callbacks.callback_factory import CallbackFactory
        from experiment_manager.trackers.tracker_factory import TrackerFactory
        
        self.factories: Dict[FactoryType, Factory] = {
            FactoryType.PIPELINE: PipelineFactory(),
            FactoryType.CALLBACK: CallbackFactory(),
            FactoryType.TRACKER: TrackerFactory(),
        }

    def register(self, factory_type: FactoryType, factory: Factory) -> None:
        """
        Register or override a factory instance for a specific factory type.
        
        This method allows users to customize the factory used for creating specific
        types of objects. The registered factory will be used for all subsequent
        object creation requests of that type.
        
        Args:
            factory_type: The type of factory to register (from FactoryType enum)
            factory: The factory instance to register. Must be an instance of Factory
                    or a subclass thereof.
        
        Raises:
            TypeError: If factory_type is not a FactoryType enum value
            TypeError: If factory is not an instance of Factory or its subclass
        
        Example:
            registry = FactoryRegistry()
            custom_factory = CustomPipelineFactory()
            registry.register(FactoryType.PIPELINE, custom_factory)
        """
        # Validate factory_type
        if not isinstance(factory_type, FactoryType):
            raise TypeError(
                f"factory_type must be a FactoryType enum value, got {type(factory_type).__name__}"
            )
        
        # Validate factory instance
        if not isinstance(factory, Factory):
            raise TypeError(
                f"factory must be an instance of Factory or its subclass, got {type(factory).__name__}"
            )
        
        self.factories[factory_type] = factory

    def get(self, factory_type: FactoryType) -> Factory:
        """
        Retrieve a factory instance for a specific factory type.
        
        Args:
            factory_type: The type of factory to retrieve (from FactoryType enum)
        
        Returns:
            The factory instance registered for the given type
        
        Raises:
            TypeError: If factory_type is not a FactoryType enum value
            KeyError: If no factory is registered for the given type (should not
                     occur with default initialization)
        
        Example:
            registry = FactoryRegistry()
            pipeline_factory = registry.get(FactoryType.PIPELINE)
            pipeline = pipeline_factory.create("MyPipeline", config, env)
        """
        # Validate factory_type
        if not isinstance(factory_type, FactoryType):
            raise TypeError(
                f"factory_type must be a FactoryType enum value, got {type(factory_type).__name__}"
            )
        
        return self.factories[factory_type]