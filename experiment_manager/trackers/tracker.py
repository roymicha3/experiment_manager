import os
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from experiment_manager.common.common import Level, Metric

from experiment_manager.common.serializable import YAMLSerializable


class Tracker(YAMLSerializable, ABC):
    def __init__(self, workspace: str = None):
        super().__init__()
        if not os.path.basename(workspace) == "artifacts":
            workspace = os.path.join(workspace, "artifacts")
        
        self.workspace = workspace

    def set_workspace(self, workspace: str):
        self.workspace = workspace
        
    @abstractmethod
    def track(self, metric: Metric, value, step: int, *args, **kwargs):
        pass
    
    @abstractmethod
    def on_checkpoint(self, 
                    network: torch.nn.Module, 
                    checkpoint_path: str, 
                    metrics: Optional[Dict[Metric, Any]] = {},
                    *args,
                    **kwargs):
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