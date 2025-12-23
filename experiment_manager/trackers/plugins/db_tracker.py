import os
import json
import torch
from datetime import datetime
from omegaconf import DictConfig
from typing import Dict, Any, Optional

from experiment_manager.trackers.tracker import Tracker
from experiment_manager.db.manager import DatabaseManager
from experiment_manager.common.common import Metric, Level
from experiment_manager.common.serializable import YAMLSerializable



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
        self.batch_idx = None
        
    
    @classmethod
    def from_config(cls, config: DictConfig, workspace: str) -> "DBTracker":
        return cls(workspace, config.name, config.get("recreate", False))
    
    def track(self, metric: Metric, value, step: int = None, *args, **kwargs):
        """Track a metric value.
        
        Args:
            metric: The metric to track
            value: The value to track (can be scalar, dict, or list for custom metrics)
            step: Optional step number
            *args: Additional args treated as per_label_val
        """
        if metric == Metric.CUSTOM:
            # Handle list of custom metrics
            if isinstance(value, list):
                for custom_metric in value:
                    self._track_single_metric(custom_metric[0], custom_metric[1], step, *args, **kwargs)
            else:
                # Handle single custom metric (backward compatibility)
                metric_name, metric_value = value
                self._track_single_metric(metric_name, metric_value, step, *args, **kwargs)
        else:
            # Handle regular metrics (unchanged)
            self._track_single_metric(metric.name, value, step, *args, **kwargs)
    
    def _track_single_metric(self, metric_name, metric_value, step: int = None, *args, **kwargs):
        """Track a single metric value.
        
        Args:
            metric_name: The name of the metric
            metric_value: The value to track (can be scalar or dict)
            step: Optional step number
            *args: Additional args treated as per_label_val
        """
        if not self.id:
            raise ValueError("Tracker must be created first")
        
        if isinstance(metric_value, dict):
            metric_record = self.db_manager.record_metric(
                metric_type=metric_name,
                total_val=None,
                per_label_val=metric_value)
        else:
            metric_record = self.db_manager.record_metric(
                metric_type=metric_name,
                total_val=metric_value,
                per_label_val=kwargs.get("per_label_val", None))
        
        # Link metric based on current tracking level
        if self.batch_idx is not None and self.epoch_idx is not None and self.id:
            # Link metric to batch (most granular level)
            self.db_manager.add_batch_metric(
                batch_idx=self.batch_idx,
                epoch_idx=self.epoch_idx,
                trial_run_id=self.id,
                metric_id=metric_record.id)
        elif self.epoch_idx is not None and self.id:
            # Link metric to epoch
            self.db_manager.add_epoch_metric(
                epoch_idx=self.epoch_idx,
                trial_run_id=self.id,
                metric_id=metric_record.id)
        else:
            # Create results entry if it doesn't exist
            self.db_manager._execute_query(
                "INSERT OR IGNORE INTO RESULTS (trial_run_id, time) VALUES (?, ?)",
                (self.id, datetime.now().isoformat()))
            # Link metric to results
            self.db_manager.link_results_metric(
                trial_run_id=self.id,
                metric_id=metric_record.id)
            
    def on_checkpoint(self, 
                      network: torch.nn.Module, 
                      checkpoint_path: str, 
                      metrics: Optional[Dict[Metric, Any]] = {},
                      *args,
                      **kwargs):
        self.on_add_artifact(Level.EPOCH, checkpoint_path)
    
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
            self.epoch_idx = 0
            
        elif level == Level.PIPELINE:
            return
            
        elif level == Level.EPOCH:
            if not self.id:
                raise ValueError("Trial run must be created first")
            # Extract epoch_id from kwargs if provided, otherwise use current epoch_idx
            epoch_id = kwargs.get("epoch_id", self.epoch_idx)
            if epoch_id is not None:
                self.epoch_idx = epoch_id
            self.db_manager.create_epoch(
                epoch_idx=self.epoch_idx,
                trial_run_id=self.id)
            self.batch_idx = None  # Reset batch index for new epoch (not tracking batches by default)
            return
        
        elif level == Level.BATCH:
            if not self.id:
                raise ValueError("Trial run must be created first")
            if self.epoch_idx is None:
                raise ValueError("Epoch must be created before batch")
            # Extract batch_id from kwargs if provided
            batch_id = kwargs.get("batch_id")
            if batch_id is not None:
                self.batch_idx = batch_id
            if self.batch_idx is None:
                raise ValueError("batch_id must be provided when creating batch")
            self.db_manager.create_batch(
                batch_idx=self.batch_idx,
                epoch_idx=self.epoch_idx,
                trial_run_id=self.id)
            return
            
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
            self.batch_idx = None
            pass
        elif level == Level.EPOCH:
            # Extract epoch_id from kwargs if provided, otherwise use current epoch_idx
            epoch_id = kwargs.get("epoch_id", self.epoch_idx)
            if epoch_id is not None:
                self.epoch_idx = epoch_id
            self.batch_idx = None  # Reset batch index for new epoch (not tracking batches by default)
            pass
        elif level == Level.BATCH:
            # Extract batch_id from kwargs if provided
            batch_id = kwargs.get("batch_id")
            if batch_id is not None:
                self.batch_idx = batch_id
            pass
        
    
    def on_end(self, level: Level, *args, **kwargs):
        if level == Level.EXPERIMENT:
            pass
        elif level == Level.TRIAL:
            pass
        elif level == Level.TRIAL_RUN:
            # Reset epoch_idx and batch_idx so final metrics are linked to RESULTS
            self.epoch_idx = None
            self.batch_idx = None
        elif level == Level.PIPELINE:
            # Also reset indices at pipeline level for child trackers
            self.epoch_idx = None
            self.batch_idx = None
        elif level == Level.EPOCH:
            self.epoch_idx += 1
            self.batch_idx = None  # Reset batch index when epoch ends
        elif level == Level.BATCH:
            if self.batch_idx is not None:
                self.batch_idx += 1  # Increment for next batch
    

    def on_add_artifact(self, level: Level, artifact_path:str, *args, **kwargs):
        artifact_type = kwargs.get("artifact_type", "unknown")
        
        artifact = self.db_manager.record_artifact(
            artifact_type=artifact_type,
            location=artifact_path)
        
        if level == Level.EXPERIMENT:
            self.db_manager.link_experiment_artifact(self.id, artifact.id)
        elif level == Level.TRIAL:
            self.db_manager.link_trial_artifact(self.id, artifact.id)
        elif level == Level.TRIAL_RUN:
            self.db_manager.link_trial_run_artifact(self.id, artifact.id)
        elif level == Level.EPOCH:
            if self.epoch_idx is None:
                raise ValueError("Epoch must be created first")
            self.db_manager.link_epoch_artifact(self.epoch_idx, self.id, artifact.id)
        elif level == Level.BATCH:
            if self.batch_idx is None or self.epoch_idx is None:
                raise ValueError("Batch and epoch must be created first")
            self.db_manager.link_batch_artifact(self.batch_idx, self.epoch_idx, self.id, artifact.id)
        else:
            raise ValueError(f"Invalid level: {level}")
    
    
    def create_child(self, workspace: str = None) -> "Tracker":
        if not self.id:
            raise ValueError("Parent tracker must be created first")
        
        tracker = DBTracker(self.workspace, self.name, recreate=False)
        tracker.id = self.id
        tracker.parent = self
        # Inherit the parent's epoch_idx and batch_idx state
        tracker.epoch_idx = self.epoch_idx
        tracker.batch_idx = self.batch_idx
        return tracker
    
    def save(self):
        """Save any pending changes to the database."""
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.connection.commit()