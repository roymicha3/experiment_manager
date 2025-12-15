from typing import Dict
from enum import Enum

from experiment_manager.common.factory import Factory
from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.pipelines.callbacks.callback_factory import CallbackFactory
from experiment_manager.trackers.tracker_factory import TrackerFactory

class FactoryType(Enum):
    """
    Enum class for factory types.
    """
    PIPELINE = "pipeline"
    CALLBACK = "callback"
    TRACKER = "tracker"


class FactoryRegistry:
    """
    Registry class for managing factory instances.
    """
    def __init__(self):
        self.factories: Dict[FactoryType, Factory] = {
            FactoryType.PIPELINE: PipelineFactory(),
            FactoryType.CALLBACK: CallbackFactory(),
            FactoryType.TRACKER: TrackerFactory(),
        }

    def register(self, factory_type: FactoryType, factory: Factory):
        """Register a factory instance."""
        self.factories[factory_type] = factory

    def get(self, factory_type: FactoryType) -> Factory:
        """Get a factory instance."""
        return self.factories[factory_type]