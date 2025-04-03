
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
        
    
    def on_start(self) -> None:
        for callback in self.callbacks:
            callback.on_start()
            
    
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch_idx, metrics)
            
    
    def on_end(self, metrics: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_end(metrics)
            
    
    def run(self, config: DictConfig) -> None:
        raise NotImplementedError