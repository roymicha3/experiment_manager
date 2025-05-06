import os
import time
import torch
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter

from experiment_manager.trackers.tracker import Tracker
from experiment_manager.common.common import Metric, Level
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("TensorBoardTracker")
class TensorBoardTracker(Tracker, YAMLSerializable):
    
    LOG_DIR_NAME = "tensorboard"
    CONFIG_FILE = "tensorboard_tracker.yaml"
    
    def __init__(self, workspace: str, name: str, root: bool = False, run_id=None):
        super(TensorBoardTracker, self).__init__(workspace)
        super(YAMLSerializable, self).__init__()
        self.name = name
        self.run_id = run_id or str(time.time())
        self.epoch = 0

        if root:
            self.workspace = os.path.join(self.workspace, TensorBoardTracker.LOG_DIR_NAME, name)
        
        os.makedirs(self.workspace, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.workspace)
        self.start_times = {}

    def track(self, metric: Metric, value, step: int, *args, **kwargs):
        if metric == Metric.CUSTOM:
            self.writer.add_scalar(value[0], value[1], global_step = self.epoch)
        else:
            self.writer.add_scalar(metric.name, value, global_step = self.epoch)

    def on_checkpoint(self, 
                      network: torch.nn.Module, 
                      checkpoint_path: str, 
                      metrics: Optional[Dict[Metric, Any]] = {},
                      *args,
                      **kwargs):
        
        if Metric.DATA in metrics.keys():
            self.writer.add_graph(network, metrics[Metric.DATA]) # TODO: might be redundant
        
    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            self.writer.add_text(f"param/{key}", str(value))

    def on_create(self, level: Level, *args, **kwargs):
        pass

    def on_start(self, level: Level, *args, **kwargs):
        if level in {Level.TRIAL, Level.TRIAL_RUN}:
            self.start_times[level] = time.time()

    def on_end(self, level: Level, *args, **kwargs):
        if level in {Level.TRIAL, Level.TRIAL_RUN}:
            end_time = time.time()
            start_time = self.start_times.get(level)
            if start_time:
                duration = end_time - start_time
                self.writer.add_scalar(f"{level.name.lower()}/duration", duration, self.epoch)

        elif level == Level.EPOCH:
            self.epoch += 1

    def on_add_artifact(self, level: Level, artifact_path: str, *args, **kwargs):
        # TensorBoard doesn't support general artifact logging
        pass

    def create_child(self, workspace: str = None) -> "Tracker":
        # require a workspace to build tensorboard trial hierarchy
        if not workspace:
            raise FileNotFoundError
        
        name = os.path.basename(workspace)
        child_workspace = os.path.join(self.workspace, name)
        os.makedirs(child_workspace, exist_ok=True)
        return TensorBoardTracker(child_workspace, 
                                  self.name, root=False, 
                                  run_id=self.run_id)

    def save(self):
        OmegaConf.save(self, os.path.join(self.workspace, TensorBoardTracker.CONFIG_FILE))

    @classmethod
    def from_config(cls, config: DictConfig, workspace: str) -> "Tracker":
        return cls(workspace, config.name, root=True)
