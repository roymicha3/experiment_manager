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
        super().__init__()
        self.workspace = workspace
        self.name = name
        self._setup_db()

    def _setup_db(self):
        self.db_manager = DatabaseManager(
            database_path=os.path.join(self.workspace, self.name),
            use_sqlite=True,
            recreate=True)
    
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
        if level == Level.EXPERIMENT:
            self.db_manager.create_experiment(
                title=args[0],
                description=kwargs.get("description", ""))
        elif level == Level.TRIAL:
            self.db_manager.create_trial(
                experiment_id=args[0],
                name=args[1])
        elif level == Level.TRIAL_RUN:
            self.db_manager.create_trial_run(
                trial_id=args[0],
                status=kwargs.get("status", "started"))
    
    def on_start(self, level: Level, *args, **kwargs):
        pass
            
    
    def on_end(self, level: Level, *args, **kwargs):
        pass
    
    def on_add_artifact(self, level: Level, *args, **kwargs):
        self.db_manager.create_artifact(
            artifact_type=args[0],
            location=args[1])
        
        self.db_manager
    
    def save(self):
        pass