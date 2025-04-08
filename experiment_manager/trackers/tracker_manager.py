from omegaconf import DictConfig
from typing import Dict, Any

from experiment_manager.trackers.tracker import Tracker
from experiment_manager.common.common import Level, Metric
from experiment_manager.trackers.tracker_factory import TrackerFactory


class TrackerManager(Tracker):
    def __init__(self, workspace: str = None) -> None:
        super().__init__(workspace)
        self.trackers = []
    
    def add_tracker(self, tracker: Tracker) -> None:
        self.trackers.append(tracker)
    
    def track(self, metric: Metric, value, step: int = None, *args):
        for tracker in self.trackers:
            tracker.track(metric, value, step, *args)
    
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


    
    
