from omegaconf import DictConfig
from experiment_manager.common.factory import Factory
from experiment_manager.common.serializable import YAMLSerializable

# import the tracker classes
from experiment_manager.trackers.tracker import Tracker

class TrackerFactory(Factory):

    def create(self, name: str, config: DictConfig) -> Tracker:
        return super().create(name, config)