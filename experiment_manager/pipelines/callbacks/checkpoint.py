import os
from typing import Dict, Any
from omegaconf import DictConfig

from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.callback import Callback, Metric
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("CheckpointCallback")
class CheckpointCallback(Callback, YAMLSerializable):
    """
    Metrics tracker for storing training statistics
    """
    CHECKPOINT_NAME = "checkpoint"
    
    def __init__(self, interval: int, env: Environment):
        super(CheckpointCallback, self).__init__()
        super(YAMLSerializable, self).__init__()
        
        self.index = 0
        self.current_checkpoint = 0
        
        self.interval = interval
        self.env = env
        
        self.checkpoint_path = os.path.join(self.env.artifact_dir, CheckpointCallback.CHECKPOINT_NAME)
        
    def on_epoch_end(self, epoch_idx, metrics: Dict[str, Any]) -> bool:
        """Called at the end of each epoch."""
        self.index += 1
        if self.index % self.interval == 0:
            file_path = f"{self.checkpoint_path}-{self.parent_id}-{self.current_checkpoint}"
            self.env.logger.info(f"Saving checkpoint {self.current_checkpoint} to {file_path}")
            metrics[Metric.NETWORK].save(file_path)
            self.current_checkpoint += 1
            self.env.logger.info(f"Checkpoint {self.current_checkpoint-1} saved successfully")
        
        return True

    def on_end(self, metrics: Dict[str, Any]):
        """Called at the end of training."""
        self.env.logger.info("Saving final checkpoint")
        file_path = f"{self.checkpoint_path}-{self.parent_id}-final"
        metrics[Metric.NETWORK].save(file_path)
        self.env.logger.info(f"Final checkpoint saved to {file_path}")

    def get_latest(self, key: str, default: Any = None) -> Any:
        pass
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        """
        Create an instance from a DictConfig.
        """
        return cls(config.interval, env)
