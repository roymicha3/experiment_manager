import os
import time
import torch
import logging
from omegaconf import DictConfig
from typing import Dict, Any, Optional

from experiment_manager.trackers.tracker import Tracker
from experiment_manager.common.common import Level, Metric
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("PerformanceTracker")
class PerformanceTracker(Tracker):
    LOG_NAME = "performance.log"

    def __init__(self, workspace: str, name: str = LOG_NAME, verbose: bool = False):
        super().__init__(workspace)
        self.name = name
        self.verbose = verbose
        self.current_level = None
        self.start_time = None
        self._setup_logger()

    def _setup_logger(self):
        os.makedirs(self.workspace, exist_ok=True)
        self.log_path = os.path.join(self.workspace, self.LOG_NAME)
        self.logger = logging.getLogger("performance_tracker")
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_path)
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
        return '' + "  " * level.value
    
    def log(self, level: Level, message: str):
        indent = self._get_indent(level)
        self.logger.info(f"{indent}{message}")
    
    def track(self, metric: Metric, value, step: int = None, *args, **kwargs):
        pass
            
    def on_checkpoint(self, 
                      network: torch.nn.Module, 
                      checkpoint_path: str, 
                      metrics: Optional[Dict[Metric, Any]] = {},
                      *args,
                      **kwargs):
        pass
    
    def log_params(self, params: Dict[str, Any]):
        pass
    
    def on_create(self, level: Level, *args, **kwargs):
        self.current_level = level
    
    def on_start(self, level: Level, *args, **kwargs):
        self.current_level = level
        self.start_time = time.time()
        self.log(level, f"Starting {level.name}")
        self.log(level, f"+ Start time: {self.start_time}")
    
    def on_end(self, level: Level, *args, **kwargs):
        self.log(level, f"Ending {level.name}")
        self.log(level, f"+ End time: {time.time()}")
        self.log(level, f"+ Duration: {time.time() - self.start_time:.2f} seconds")
        self.current_level = None
    
    def on_metric(self, level: Level, metric: Dict[str, Any], *args, **kwargs):
        pass
    
    def on_add_artifact(self, level: Level, artifact_path: str, *args, **kwargs):
        pass
    
    def create_child(self, workspace: str = None) -> "Tracker":
        return self
    
    def save(self):
        pass
    
    @classmethod
    def from_config(cls, config: DictConfig, workspace: str) -> "PerformanceTracker":
        name = config.get("name", PerformanceTracker.LOG_NAME)
        verbose = config.get("verbose", False)
        return cls(workspace, name, verbose)
