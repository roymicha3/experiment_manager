from omegaconf import DictConfig

from experiment_manager.trackers.tracker import Tracker
from experiment_manager.common.common import Level, Metric
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.trackers.tracker_factory import TrackerFactory


class TrackerManager(YAMLSerializable):
    def __init__(self, workspace: str = None) -> None:
        self.trackers = []
        self.workspace = workspace
    
    def add_tracker(self, tracker: Tracker) -> None:
        self.trackers.append(tracker)

    def track(self, metric: Metric, step: int):
        for tracker in self.trackers:
            tracker.track(metric, step)

    def on_create(self, level: Level, *args, **kwargs):
        for tracker in self.trackers:
            tracker.on_create(level, *args, **kwargs)
    
    def on_start(self, level: Level, *args, **kwargs):
        for tracker in self.trackers:
            tracker.on_start(level, *args, **kwargs)
    
    def on_end(self, level: Level, *args, **kwargs):    
        for tracker in self.trackers:
            tracker.on_end(level, *args, **kwargs)
    
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


    
    
