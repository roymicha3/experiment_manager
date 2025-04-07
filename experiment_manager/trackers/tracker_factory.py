from omegaconf import DictConfig
from experiment_manager.common.factory import Factory
from experiment_manager.common.serializable import YAMLSerializable

# import the tracker classes
from experiment_manager.trackers.tracker import Tracker
from experiment_manager.trackers.log_tracker import LogTracker

class TrackerFactory(Factory):

    def create(self, name: str, config: DictConfig, workspace: str) -> Tracker:
        return super().create(name, config, workspace)