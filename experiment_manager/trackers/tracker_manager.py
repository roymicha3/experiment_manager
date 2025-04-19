import torch
from typing import Dict, Any, List
from omegaconf import DictConfig

from experiment_manager.trackers.tracker import Tracker
from experiment_manager.common.common import Level, Metric
from experiment_manager.trackers.tracker_factory import TrackerFactory


    

class TrackerManager(Tracker):
    def __init__(self, workspace: str = None) -> None:
        super().__init__(workspace)
        self.trackers: List[Tracker] = []
    
    def add_tracker(self, tracker: Tracker) -> None:
        self.trackers.append(tracker)
    
    def track(self, metric: Metric, value, step: int = None, *args, **kwargs):
        for tracker in self.trackers:
            tracker.track(metric, value, step, *args, **kwargs)
            
    def on_checkpoint(self, network: torch.nn.Module, checkpoint_path: str, *args, **kwargs):
        for tracker in self.trackers:
            tracker.on_checkpoint(network, checkpoint_path, *args, **kwargs)
    
    def log_params(self, params: Dict[str, Any]):
        for tracker in self.trackers:
            tracker.log_params(params)
    
    def on_create(self, level: Level, *args, **kwargs):
        for tracker in self.trackers:
            tracker.on_create(level, *args, **kwargs)
    
    def on_start(self, level: Level, *args, **kwargs):
        for tracker in self.trackers:
            tracker.on_start(level, *args, **kwargs)
    
    def on_end(self, level: Level, *args, **kwargs):    
        for tracker in self.trackers:
            tracker.on_end(level, *args, **kwargs)
            
    def on_add_artifact(self, level: Level, artifact_path: str, *args, **kwargs):
        for tracker in self.trackers:
            tracker.on_add_artifact(level, artifact_path, *args, **kwargs)
            
    def create_child(self, workspace: str = None):
        new_workspace = workspace if workspace else self.workspace
        manager = TrackerManager(workspace = new_workspace)
        
        for tracker in self.trackers:
            manager.add_tracker(tracker.create_child(new_workspace))
        return manager
    
    def save(self):
        for tracker in self.trackers:
            tracker.save()

    @classmethod
    def from_config(cls, config: DictConfig, workspace: str):
        manager = cls(workspace)
        for tracker_conf in config.get("trackers", []):
            if "type" not in tracker_conf:
                raise ValueError("missing required 'type' field")
            
            tracker = TrackerFactory.create(tracker_conf.type, tracker_conf, workspace)
            manager.add_tracker(tracker)
        
        return manager


class TrackScope:
    
    def __init__(self, tracker_manager: TrackerManager, level: Level, *args, **kwargs):
        self.tracker_manager = tracker_manager
        self.level = level
        self.args = args
        self.kwargs = kwargs
    
    def __enter__(self):
        self.tracker_manager.on_start(self.level, *self.args, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tracker_manager.on_end(self.level)
    
    
