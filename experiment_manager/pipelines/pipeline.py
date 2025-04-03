from typing import List, Dict, Any
from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.callback import Callback
from abc import ABC, abstractmethod

class Pipeline(ABC):
    
    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.callbacks: List[Callback] = []
        
    
    def register_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)
        self.env.logger.info(f"Registered callback: {callback.__class__.__name__}")
        
    
    def on_start(self) -> None:
        self.env.logger.info("Starting pipeline execution")
        for callback in self.callbacks:
            self.env.logger.debug(f"Executing on_start for {callback.__class__.__name__}")
            callback.on_start()
            
    
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            self.env.logger.debug(f"Executing on_epoch_end for {callback.__class__.__name__}")
            should_continue = callback.on_epoch_end(epoch_idx, metrics)
            if not should_continue:
                self.env.logger.info(f"Callback {callback.__class__.__name__} requested early termination")
                break
            
    
    def on_end(self, metrics: Dict[str, Any]) -> None:
        self.env.logger.info("Pipeline execution completed")
        for callback in self.callbacks:
            self.env.logger.debug(f"Executing on_end for {callback.__class__.__name__}")
            callback.on_end(metrics)
            
    
    def run(self, config: DictConfig) -> None:
        raise NotImplementedError
    