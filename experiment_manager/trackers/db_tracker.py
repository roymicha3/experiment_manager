import os
import json
from typing import Dict, Any
from omegaconf import DictConfig

from experiment_manager.trackers.tracker import Tracker
from experiment_manager.db.manager import DatabaseManager
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.common.common import Metric, MetricCategory, Level



@YAMLSerializable.register("DBTracker")
class DBTracker(Tracker, YAMLSerializable):
    DB_NAME = "tracker.db"
    
    def __init__(self, workspace: str, name: str = DB_NAME):
        super().__init__(workspace)
        self.name = name
        self.id = None
        self.parent = None
        self.db_manager = \
            DatabaseManager(
                database_path=os.path.join(self.workspace, self.name),
                use_sqlite=True,
                recreate=True)
            
        self.epoch_idx = None
    
    @classmethod
    def from_config(cls, config: DictConfig, workspace: str) -> "DBTracker":
        return cls(workspace, config.name)
    
    def track(self, metric: Metric, value, step: int, *args):
        self.db_manager.create_metric(
            metric_type=metric.value,
            total_val=value,
            per_label_val=args)
    
    def log_params(self, params: Dict[str, Any]):
        params_path = os.path.join(self.workspace, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f)
        
        self.db_manager.create_artifact(
            artifact_type="params",
            location=params_path)
    
    def on_create(self, level: Level, *args, **kwargs):
        res = None
        if level == Level.EXPERIMENT:
            res = self.db_manager.create_experiment(
                title=args[0],
                description=kwargs.get("description", ""))
        
        elif level == Level.TRIAL:
            if not self.parent:
                raise ValueError("Parent tracker must be created first")
            res = self.db_manager.create_trial(
                experiment_id=self.parent.id,
                name=args[0])
        
        elif level == Level.TRIAL_RUN:
            if not self.parent:
                raise ValueError("Parent tracker must be created first")
            res = self.db_manager.create_trial_run(
                trial_id=self.parent.id,
                status=kwargs.get("status", "started"))
            
        if not res:
            raise ValueError("Invalid level")
        
        self.id = res.id
    
    def on_start(self, level: Level, *args, **kwargs):
        if level == Level.EXPERIMENT:
            pass
        elif level == Level.TRIAL:
            pass
        elif level == Level.TRIAL_RUN:
            self.epoch_idx = 0
            pass
        elif level == Level.EPOCH:
            self.db_manager.create_epoch(
                experiment_id=self.id,
                trial_id=self.parent.id,
                trial_run_id=self.parent.parent.id,
                epoch=self.epoch_idx + 1,
                status=kwargs.get("status", "started"))
        
    
    def on_end(self, level: Level, *args, **kwargs):
        if level == Level.EXPERIMENT:
            pass
        elif level == Level.TRIAL:
            pass
        elif level == Level.TRIAL_RUN:
            self.epoch_idx = 0
            pass
        elif level == Level.EPOCH:
            self.epoch_idx += 1
    
    def on_add_artifact(self, level: Level, artifact_path:str, *args, **kwargs):
        
        artifact = self.db_manager.record_artifact(
            artifact_type=args[0],
            location=artifact_path)
        
        if level == Level.EXPERIMENT:
            self.db_manager.link_experiment_artifact(self.id, artifact.id)
        elif level == Level.TRIAL:
            self.db_manager.link_trial_artifact(self.id, artifact.id)
        elif level == Level.TRIAL_RUN:
            self.db_manager.link_trial_run_artifact(self.id, artifact.id)
        elif level == Level.EPOCH:
            if not self.epoch_idx:
                raise ValueError("Epoch must be created first")
            self.db_manager.link_epoch_artifact(self.epoch_idx, self.id, artifact.id)
        else:
            raise ValueError(f"Invalid level: {level}")
    
    
    def create_child(self, workspace: str = None) -> "Tracker":
        if not self.id:
            raise ValueError("Parent tracker must be created first")
        
        tracker = DBTracker(workspace, self.name)
        tracker.parent = self
        return tracker
        