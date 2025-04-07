from experiment_manager.common.common import Metric, MetricCategory, Level
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.trackers.tracker import Tracker
from experiment_manager.logger import FileLogger, CompositeLogger



class LogTracker(Tracker):
    LOG_NAME = "tracker.log"

    def __init__(self, log_path: str, name: str = LogTracker.LOG_NAME, verbose: bool = False):
        super().__init__()
        self.log_path = log_path
        self.name = name
        if verbose:
            self.logger = CompositeLogger(name=self.name, log_path=self.log_path)
        else:
            self.logger = FileLogger(name=self.name, log_path=self.log_path)

    def log(self, message: str) -> None:
        self.logger.log(message)

    def track(self, metric: Metric, step: int):
        self.log(f"{metric} at step {step}")

    def on_create(self, level: Level, *args, **kwargs):
        self.log(f"Creating {level}")
        self.log(args)
        self.log(kwargs)

    def on_start(self, level: Level, *args, **kwargs):
        self.log(f"Starting {level}")
        self.log(args)
        self.log(kwargs)

    def on_end(self, level: Level, *args, **kwargs):
        self.log(f"Ending {level}")
        self.log(args)
        self.log(kwargs)

    def save(self):
        pass