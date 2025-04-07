from omegaconf import DictConfig
from experiment_manager.common.factory import Factory
from experiment_manager.common.serializable import YAMLSerializable

# import the tracker classes
from experiment_manager.trackers.tracker import Tracker
from experiment_manager.trackers.log_tracker import LogTracker

class TrackerFactory(Factory):
    
    @staticmethod
    def create(name: str, config: DictConfig, workspace: str) -> Tracker:
        return YAMLSerializable.get_by_name(name).from_config(config, workspace)