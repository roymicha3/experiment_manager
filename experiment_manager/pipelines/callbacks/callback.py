from typing import Dict, Any
from abc import ABC, abstractmethod


class Callback(ABC):
    """Base class for all callbacks."""
    
    @abstractmethod
    def on_start(self) -> None:
        """Called when training starts."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_end(self, metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
        pass
