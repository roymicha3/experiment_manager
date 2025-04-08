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
    
    def __init__(self, workspace: str, name: str = DB_NAME, recreate: bool = None):
        super().__init__(workspace)
        self.name = name
        self.id = None
        self.parent = None
        
        # Auto-detect if database exists if recreate not specified
        db_path = os.path.join(self.workspace, self.name)
        if recreate is None:
            recreate = not os.path.exists(db_path)
        
        self.db_manager = \
            DatabaseManager(
                database_path=db_path,
                use_sqlite=True,
                recreate=recreate)
            
        self.epoch_idx = None
    
    @classmethod
    def from_config(cls, config: DictConfig, workspace: str) -> "DBTracker":
        return cls(workspace, config.name, config.get("recreate", False))
    
    def track(self, metric: Metric, value, step: int = None, *args):
        """Track a metric value.
        
        Args:
            metric: The metric to track
            value: The value to track (can be scalar or dict)
            step: Optional step number
            *args: Additional args treated as per_label_val
        """
        if not self.id:
            raise ValueError("Tracker must be created first")
        
        if isinstance(value, dict):
            metric_record = self.db_manager.record_metric(
                metric_type=metric.name,
                total_val=None,
                per_label_val=value)
        else:
            metric_record = self.db_manager.record_metric(
                metric_type=metric.name,
                total_val=value,
                per_label_val=args[0] if args else None)
        
        # Link metric to current trial run if we're in an epoch
        if self.epoch_idx is not None and self.id:
            self.db_manager.add_epoch_metric(self.epoch_idx, self.id, metric_record.id)
    
    def log_params(self, params: Dict[str, Any]):
        params_path = os.path.join(self.workspace, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f)
        
        self.db_manager.record_artifact(
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
        
        tracker = DBTracker(workspace or self.workspace, self.name, recreate=False)
        tracker.id = self.id
        tracker.parent = self
        return tracker
    
    def save(self):
        """Save any pending changes to the database."""
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.connection.commit()