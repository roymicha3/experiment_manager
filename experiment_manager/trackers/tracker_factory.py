from omegaconf import DictConfig
from experiment_manager.common.factory import Factory
from experiment_manager.common.serializable import YAMLSerializable

# import the tracker classes
from experiment_manager.trackers.tracker import Tracker

from experiment_manager.trackers.plugins.db_tracker import DBTracker
from experiment_manager.trackers.plugins.log_tracker import LogTracker
from experiment_manager.trackers.plugins.mlflow_tracker import MLflowTracker
from experiment_manager.trackers.plugins.tensorboard_tracker import TensorBoardTracker
from experiment_manager.trackers.plugins.performance_tracker import PerformanceTracker

class TrackerFactory(Factory):
    
    @staticmethod
    def create(name: str, config: DictConfig, workspace: str) -> Tracker:
        return YAMLSerializable.get_by_name(name).from_config(config, workspace)