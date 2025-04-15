from abc import ABC, abstractmethod
from typing import Dict, Any
from experiment_manager.common.common import Level, Metric

from experiment_manager.common.serializable import YAMLSerializable


class Tracker(YAMLSerializable, ABC):
    def __init__(self, workspace: str = None):
        super().__init__()
        self.workspace = workspace

    def set_workspace(self, workspace: str):
        self.workspace = workspace
        
    @abstractmethod
    def track(self, metric: Metric, value, step: int, *args, **kwargs):
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
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
    def on_add_artifact(self, level: Level, artifact_path: str, *args, **kwargs):
        pass
    
    @abstractmethod
    def create_child(self, workspace: str=None) -> "Tracker":
        pass
    
    @abstractmethod
    def save(self):
        pass