import os
import time
import torch
import mlflow
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig

from experiment_manager.trackers.tracker import Tracker
from experiment_manager.common.common import Metric, Level
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("MLflowTracker")
class MLflowTracker(Tracker, YAMLSerializable):
    
    def __init__(self, workspace: str, name: str, root: bool = False, run_id = None):
        super(MLflowTracker, self).__init__()
        super(YAMLSerializable, self).__init__()
        self.workspace = workspace
        self.name = name
        self.run_id = run_id
        self.epoch = 0
        if root:
            mlflow.set_tracking_uri(f"file:////{workspace}/mlruns")
            mlflow.set_experiment(self.name)
        
    def track(self, metric: Metric, value, step: int, *args, **kwargs):
        mlflow.log_metric(metric.name, value, step = self.epoch)
        
    def on_checkpoint(self, network: torch.nn.Module, checkpoint_path: str, *args, **kwargs):
        mlflow.pytorch.log_model(network, os.path.basename(checkpoint_path))
    

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)
    
    
    def on_create(self, level: Level, *args, **kwargs):
        pass
    
    
    def on_start(self, level: Level, *args, **kwargs):
        if level == Level.EXPERIMENT:
            return

        elif level == Level.TRIAL:
            trial_name = kwargs.get("trial_name", None)
            mlflow.start_run(run_name = trial_name)
            mlflow.set_tag("start_time", time.time())
            

            self.run_id = mlflow.active_run().info.run_id
            
        elif level == Level.TRIAL_RUN:
            mlflow.start_run(nested=True)
            mlflow.set_tag("start_time", time.time())
            

            self.run_id = mlflow.active_run().info.run_id
            
        elif level == Level.PIPELINE:
            pass
        
        elif level == Level.EPOCH:
            pass
        
        elif level == Level.BATCH:
            pass
    

    def on_end(self, level: Level, *args, **kwargs):
        if level == Level.EXPERIMENT:
            return

        elif level == Level.TRIAL:
            end_time = time.time()
            mlflow.set_tag("end_time", end_time)
            start_time = mlflow.get_run(mlflow.active_run().info.run_id).data.tags.get("start_time")
            
            if start_time:
                duration = end_time - float(start_time)
                mlflow.log_metric("duration", duration)
            
            mlflow.end_run()
            
        elif level == Level.TRIAL_RUN:
            end_time = time.time()
            mlflow.set_tag("end_time", end_time)
            
            start_time = mlflow.get_run(mlflow.active_run().info.run_id).data.tags.get("start_time")
            if start_time:
                duration = end_time - float(start_time)
                mlflow.log_metric("duration", duration)
            
            mlflow.end_run()
    
        elif level == Level.PIPELINE:
            pass
        
        elif level == Level.EPOCH:
            self.epoch += 1
            
        elif level == Level.BATCH:
            pass
        
        
    
    def on_add_artifact(self, level: Level, artifact_path: str, *args, **kwargs):
        # TODO: log the artifact path here
        pass
    
    def create_child(self, workspace: str=None) -> "Tracker":
        return MLflowTracker(self.workspace, self.name, root=False, run_id=self.run_id)
    
    def save(self):
        OmegaConf.save(self, os.path.join(self.workspace, "mlflow_tracker.yaml"))
        
    
    @classmethod
    def from_config(cls, config: DictConfig, workspace: str) -> "Tracker":
        return cls(workspace, config.name, root=True)