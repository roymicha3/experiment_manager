import os
import logging
from typing import Dict, Any
from omegaconf import DictConfig

from experiment_manager.trackers.tracker import Tracker
from experiment_manager.common.common import Level, Metric
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("LogTracker")
class LogTracker(Tracker):
    LOG_NAME = "tracker.log"

    def __init__(self, workspace: str, name: str = LOG_NAME, verbose: bool = False):
        super().__init__(workspace)
        self.name = name
        self.verbose = verbose
        self.current_level = None
        self._setup_logger()

    def _setup_logger(self):
        os.makedirs(self.workspace, exist_ok=True)
        log_path = os.path.join(self.workspace, self.name)
        self.logger = logging.getLogger("experiment_tracker")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Add file handler
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Add console handler if verbose
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def _get_indent(self, level: Level) -> str:
        """Get indentation based on level."""
        indent_map = {
            Level.EXPERIMENT: "",
            Level.TRIAL: "  ",
            Level.TRIAL_RUN: "    ",
            Level.PIPELINE: "      ",
            Level.EPOCH: "        "
        }
        return indent_map.get(level, "")
    
    def log(self, level: Level, message: str):
        indent = self._get_indent(level)
        self.logger.info(f"{indent}{message}")
    
    def track(self, metric: Metric, value, step: int = None, *args):
        level = self.current_level or Level.EXPERIMENT
        step_str = f" at step {step}" if step is not None else ""
        self.log(level, f"{metric.name}: {value}{step_str}")
        if args:
            self.log(level, f"Additional info: {args}")
    
    def log_params(self, params: Dict[str, Any]):
        level = self.current_level or Level.EXPERIMENT
        self.log(level, "Parameters:")
        for key, value in params.items():
            self.log(level, f"  {key}: {value}")
    
    def on_create(self, level: Level, *args, **kwargs):
        self.current_level = level
        self.log(level, f"Creating {level.name}")
        if args:
            self.log(level, f"Args: {args}")
        if kwargs:
            self.log(level, f"Kwargs: {kwargs}")
    
    def on_start(self, level: Level, *args, **kwargs):
        self.current_level = level
        self.log(level, f"Starting {level.name}")
        if args:
            self.log(level, f"Args: {args}")
        if kwargs:
            self.log(level, f"Kwargs: {kwargs}")
    
    def on_end(self, level: Level, *args, **kwargs):
        self.log(level, f"Ending {level.name}")
        if args:
            self.log(level, f"Args: {args}")
        if kwargs:
            self.log(level, f"Kwargs: {kwargs}")
        self.current_level = None
    
    def on_metric(self, level: Level, metric: Dict[str, Any], *args, **kwargs):
        self.log(level, f"Metric:")
        for key, value in metric.items():
            self.log(level, f"  {key}: {value}")
        if args:
            self.log(level, f"Args: {args}")
        if kwargs:
            self.log(level, f"Kwargs: {kwargs}")
    
    def on_add_artifact(self, level: Level, artifact_path: str, *args, **kwargs):
        self.log(level, f"Adding artifact:")
        self.log(level, f"  Path: {artifact_path}")
        if args:
            self.log(level, f"  Type: {args[0] if args else 'unknown'}")
        if kwargs:
            self.log(level, f"  Kwargs: {kwargs}")
    
    def create_child(self, workspace: str = None) -> "Tracker":
        return self
    
    def save(self):
        pass
    
    @classmethod
    def from_config(cls, config: DictConfig, workspace: str) -> "LogTracker":
        return cls(workspace, config.type, config.verbose)