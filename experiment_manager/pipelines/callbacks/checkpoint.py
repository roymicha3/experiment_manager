import os
from typing import Dict, Any
from omegaconf import DictConfig

from experiment_manager.environment import Environment
from experiment_manager.common.common import Metric, Level
from experiment_manager.pipelines.callbacks.callback import Callback
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
        
        self.env.logger.info(f"Checkpoint callback initialized with interval={interval}")
        
        self.checkpoint_path = os.path.join(self.env.artifact_dir, CheckpointCallback.CHECKPOINT_NAME)
        
    def on_epoch_end(self, epoch_idx, metrics: Dict[str, Any]) -> bool:
        """Called at the end of each epoch."""
        self.index += 1
        
        if self.index % self.interval == 0:
            file_path = f"{self.checkpoint_path}-{self.current_checkpoint}"
            self.env.logger.info(f"Saving checkpoint {self.current_checkpoint} to {file_path}")
            
            if not Metric.NETWORK in metrics.keys():
                self.env.logger.warning(f"Metric {Metric.NETWORK.name} is not in metrics, skipping checkpoint save")
                return True
            
            try:
                metrics[Metric.NETWORK].save(file_path)
                self.env.tracker_manager.on_add_artifact(
                    level=Level.TRIAL_RUN,
                    artifact_path=file_path)
                
                self.current_checkpoint += 1
                self.env.logger.info(f"Checkpoint {self.current_checkpoint-1} saved successfully")
            except Exception as e:
                self.env.logger.error(f"Failed to save checkpoint {self.current_checkpoint}: {e}")
        
        return True
    
    def on_start(self) -> None:
        pass

    def on_end(self, metrics: Dict[str, Any]):
        """Called at the end of training."""
        self.env.logger.info("Saving final checkpoint")
        file_path = f"{self.checkpoint_path}-final"
        if not Metric.NETWORK in metrics.keys():
                self.env.logger.warning(f"Metric {Metric.NETWORK.name} is not in metrics, skipping final checkpoint save")
                return
        
        try:
            metrics[Metric.NETWORK].save(file_path)
            self.env.tracker_manager.on_add_artifact(
                level=Level.TRIAL_RUN,
                artifact_path=file_path,
                artifact_type="checkpoint")
            
            self.env.logger.info(f"Final checkpoint saved to {file_path}")
        except Exception as e:
            self.env.logger.error(f"Failed to save final checkpoint: {e}")
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        """
        Create an instance from a DictConfig.
        """
        return cls(config.interval, env)
