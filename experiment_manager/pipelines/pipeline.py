from abc import ABC
from typing import Dict, Any, List
from omegaconf import DictConfig

from experiment_manager.common.common import Level
from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.callback import Callback

class Pipeline(ABC):
    
    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.callbacks: List[Callback] = []
        self.env.tracker_manager.on_create(level=Level.PIPELINE)
        
    
    def register_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)
        self.env.logger.info(f"Registered callback: {callback.__class__.__name__}")
        
    
    def on_start(self) -> None:
        self.env.logger.info("Starting pipeline execution")
        self.env.tracker_manager.on_start(level=Level.PIPELINE)
        for callback in self.callbacks:
            self.env.logger.debug(f"Executing on_start for {callback.__class__.__name__}")
            callback.on_start()
            
    def on_epoch_start(self) -> None:
        self.env.tracker_manager.on_create(Level.EPOCH)
        self.env.tracker_manager.on_start(Level.EPOCH)
            
    
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        self.env.tracker_manager.on_end(Level.EPOCH)
        
        stop_flag = False
        for callback in self.callbacks:
            self.env.logger.debug(f"Executing on_epoch_end for {callback.__class__.__name__}")
            should_continue = callback.on_epoch_end(epoch_idx, metrics)
            if not should_continue:
                stop_flag = True
        return stop_flag
    
    def on_end(self, metrics: Dict[str, Any]) -> None:
        self.env.logger.info("Pipeline execution completed")
        self.env.tracker_manager.on_end(Level.PIPELINE)
        for callback in self.callbacks:
            self.env.logger.debug(f"Executing on_end for {callback.__class__.__name__}")
            callback.on_end(metrics)
            
    def run(self, config: DictConfig) -> None:
        """
        Run the pipeline with the given configuration.
        """
        raise NotImplementedError("Pipeline.run() must be implemented by subclasses")