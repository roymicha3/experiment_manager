from abc import ABC, abstractmethod
from experiment_manager.common.common import Level, Metric

from experiment_manager.common.serializable import YAMLSerializable


class Tracker(YAMLSerializable, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def track(self, metric: Metric, step: int):
        pass
    
    @abstractmethod
    def on_create(self, level: Level, *args, **kwargs):
        pass
    
    @abstractmethod
    def on_start(self, level: Level, *args, **kwargs):
        pass
    
    @abstractmethod
    def on_end(self, level: Level, *args, **kwargs):
        pass
    
    @abstractmethod
    def save(self):
        pass