"""Database manager for experiment tracking."""
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
import json
import logging
import sqlite3
import mysql.connector
from datetime import datetime
from pathlib import Path

from experiment_manager.db.db import init_sqlite_db, init_mysql_db
from experiment_manager.db.tables import Experiment, Trial, TrialRun, Metric, Artifact

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DatabaseError(Exception):
    """Base class for database-related errors."""
    pass

class ConnectionError(DatabaseError):
    """Error connecting to the database."""
    pass

class QueryError(DatabaseError):
    """Error executing a database query."""
    pass

class DatabaseManager:
    """Manages database operations for experiment tracking.
    
    This class provides a high-level interface for interacting with the experiment
    tracking database. It supports both SQLite and MySQL backends and provides
    methods for managing experiments, trials, metrics, and artifacts.
    
    Attributes:
        use_sqlite (bool): Whether SQLite is being used as the backend
        connection: Database connection object (SQLite or MySQL)
        cursor: Database cursor object
    """
    
    def __init__(self, database_path: Union[str, Path] = "experiment_manager.db", 
                 use_sqlite: bool = False, host: str = "localhost", 
                 user: str = "root", password: str = "", recreate: bool = False):
        """
        Initialize database connection.
        """
        self.use_sqlite = use_sqlite
        try:
            if use_sqlite:
                self.connection = init_sqlite_db(database_path, recreate=recreate)
                # Make SQLite return dictionaries for rows
                self.connection.row_factory = sqlite3.Row
                self.cursor = self.connection.cursor()
            else:
                self.connection = init_mysql_db(host, user, password, database_path, recreate=recreate)
                self.cursor = self.connection.cursor(dictionary=True)
        except (sqlite3.Error, mysql.connector.Error) as e:
            raise ConnectionError(f"Failed to connect to database: {e}") from e
        
    def __del__(self):
        """
        Close database connection on object destruction.
        """
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            if hasattr(self, 'connection') and self.connection:
                self.connection.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def _execute_query(self, query: str, params: tuple = None) -> Any:
        
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor
        except (sqlite3.Error, mysql.connector.Error) as e:
            raise QueryError(f"Query execution failed: {e}") from e
    
    def _get_placeholder(self) -> str:
        """Get the appropriate parameter placeholder for the current database."""
        return "?" if self.use_sqlite else "%s"
            
    def create_experiment(self, title: str, description: str = "") -> Experiment:
        
        now = datetime.now().isoformat()
        ph = self._get_placeholder()
        desc_field = "desc" if self.use_sqlite else "`desc`"
        
        query = f"""
        INSERT INTO EXPERIMENT (title, {desc_field}, start_time, update_time)
        VALUES ({ph}, {ph}, {ph}, {ph})
        """
        
        cursor = self._execute_query(query, (title, description, now, now))
        self.connection.commit()
        
        return Experiment(
            id=cursor.lastrowid,
            title=title,
            description=description,
            start_time=datetime.fromisoformat(now),
            update_time=datetime.fromisoformat(now)
        )
    
    def create_trial(self, experiment_id: int, name: str) -> Trial:
        
        now = datetime.now().isoformat()
        ph = self._get_placeholder()
        
        query = f"""
        INSERT INTO TRIAL (experiment_id, name, start_time, update_time)
        VALUES ({ph}, {ph}, {ph}, {ph})
        """
        
        cursor = self._execute_query(query, (experiment_id, name, now, now))
        self.connection.commit()
        
        return Trial(
            id=cursor.lastrowid,
            name=name,
            experiment_id=experiment_id,
            start_time=datetime.fromisoformat(now),
            update_time=datetime.fromisoformat(now)
        )
    
    def create_trial_run(self, trial_id: int, status: str = "started") -> TrialRun:
        
        now = datetime.now().isoformat()
        ph = self._get_placeholder()
        
        query = f"""
        INSERT INTO TRIAL_RUN (trial_id, status, start_time, update_time)
        VALUES ({ph}, {ph}, {ph}, {ph})
        """
        
        cursor = self._execute_query(query, (trial_id, status, now, now))
        self.connection.commit()
        
        return TrialRun(
            id=cursor.lastrowid,
            trial_id=trial_id,
            status=status,
            start_time=datetime.fromisoformat(now),
            update_time=datetime.fromisoformat(now)
        )
    
    def record_metric(self, total_val: float, metric_type: str, 
                     per_label_val: Optional[Dict] = None) -> Metric:
        
        ph = self._get_placeholder()
        per_label_json = json.dumps(per_label_val) if per_label_val else None
        
        query = f"""
        INSERT INTO METRIC (type, total_val, per_label_val)
        VALUES ({ph}, {ph}, {ph})
        """
        
        cursor = self._execute_query(query, (metric_type, total_val, per_label_json))
        self.connection.commit()
        
        return Metric(
            id=cursor.lastrowid,
            type=metric_type,
            total_val=total_val,
            per_label_val=per_label_val
        )
    
    def add_epoch_metric(self, epoch_idx: int, trial_run_id: int, metric_id: int) -> None:
        
        ph = self._get_placeholder()
        
        query = f"""
        INSERT INTO EPOCH_METRIC (epoch_idx, epoch_trial_run_id, metric_id)
        VALUES ({ph}, {ph}, {ph})
        """
        
        self._execute_query(query, (epoch_idx, trial_run_id, metric_id))
        self.connection.commit()
    
    def record_artifact(self, artifact_type: str, location: str) -> Artifact:
        
        ph = self._get_placeholder()
        
        query = f"""
        INSERT INTO ARTIFACT (type, loc)
        VALUES ({ph}, {ph})
        """
        
        cursor = self._execute_query(query, (artifact_type, location))
        self.connection.commit()
        
        return Artifact(
            id=cursor.lastrowid,
            type=artifact_type,
            location=location
        )
    
    def create_epoch(self, epoch_idx: int, trial_run_id: int) -> None:
        """Create a new epoch entry."""
        ph = self._get_placeholder()
        
        query = f"""
        INSERT INTO EPOCH (idx, trial_run_id, time)
        VALUES ({ph}, {ph}, {ph})
        """
        
        self._execute_query(query, (epoch_idx, trial_run_id, datetime.now().isoformat()))
        self.connection.commit()

    def get_experiment_metrics(self, experiment_id: int) -> List[Metric]:
        
        ph = self._get_placeholder()
        
        query = f"""
        SELECT DISTINCT m.* FROM METRIC m
        LEFT JOIN EPOCH_METRIC em ON em.metric_id = m.id
        LEFT JOIN EPOCH e ON e.idx = em.epoch_idx AND e.trial_run_id = em.epoch_trial_run_id
        LEFT JOIN TRIAL_RUN tr ON tr.id = e.trial_run_id
        LEFT JOIN RESULTS_METRIC rm ON rm.metric_id = m.id
        LEFT JOIN RESULTS r ON r.trial_run_id = rm.results_id
        JOIN TRIAL t ON t.id = tr.trial_id OR t.id = (SELECT trial_id FROM TRIAL_RUN WHERE id = r.trial_run_id)
        WHERE t.experiment_id = {ph}
        """
        
        cursor = self._execute_query(query, (experiment_id,))
        rows = cursor.fetchall()
        
        return [
            Metric(
                id=row["id"],
                type=row["type"],
                total_val=row["total_val"],
                per_label_val=json.loads(row["per_label_val"]) if row["per_label_val"] else None
            )
            for row in rows
        ]
    
    def get_trial_artifacts(self, trial_id: int) -> List[Artifact]:
        ph = self._get_placeholder()
        
        query = f"""
        SELECT a.* FROM ARTIFACT a
        JOIN TRIAL_ARTIFACT ta ON ta.artifact_id = a.id
        WHERE ta.trial_id = {ph}
        """
        
        cursor = self._execute_query(query, (trial_id,))
        rows = cursor.fetchall()
        
        return [
            Artifact(
                id=row["id"],
                type=row["type"],
                location=row["loc"]
            )
            for row in rows
        ]

    def link_experiment_artifact(self, experiment_id: int, artifact_id: int) -> None:
        """Link an artifact to an experiment."""
        ph = self._get_placeholder()
        
        query = f"""
        INSERT INTO EXPERIMENT_ARTIFACT (experiment_id, artifact_id)
        VALUES ({ph}, {ph})
        """
        
        self._execute_query(query, (experiment_id, artifact_id))
        self.connection.commit()

    def link_trial_artifact(self, trial_id: int, artifact_id: int) -> None:
        """Link an artifact to a trial."""
        ph = self._get_placeholder()
        
        query = f"""
        INSERT INTO TRIAL_ARTIFACT (trial_id, artifact_id)
        VALUES ({ph}, {ph})
        """
        
        self._execute_query(query, (trial_id, artifact_id))
        self.connection.commit()

    def link_epoch_artifact(self, epoch_idx: int, trial_run_id: int, artifact_id: int) -> None:
        """Link an artifact to an epoch."""
        ph = self._get_placeholder()
        
        query = f"""
        INSERT INTO EPOCH_ARTIFACT (epoch_idx, epoch_trial_run_id, artifact_id)
        VALUES ({ph}, {ph}, {ph})
        """
        
        self._execute_query(query, (epoch_idx, trial_run_id, artifact_id))
        self.connection.commit()

    def link_trial_run_artifact(self, trial_run_id: int, artifact_id: int) -> None:
        """Link an artifact to a trial run."""
        ph = self._get_placeholder()
        
        query = f"""
        INSERT INTO TRIAL_RUN_ARTIFACT (trial_run_id, artifact_id)
        VALUES ({ph}, {ph})
        """
        
        self._execute_query(query, (trial_run_id, artifact_id))
        self.connection.commit()

    def link_results_metric(self, trial_run_id: int, metric_id: int) -> None:
        """Link a metric to results (trial run)."""
        ph = self._get_placeholder()
        
        query = f"""
        INSERT INTO RESULTS_METRIC (results_id, metric_id)
        VALUES ({ph}, {ph})
        """
        
        self._execute_query(query, (trial_run_id, metric_id))
        self.connection.commit()

    def get_experiment_artifacts(self, experiment_id: int) -> List[Artifact]:
        """Get all artifacts associated with an experiment."""
        ph = self._get_placeholder()
        
        query = f"""
        SELECT a.* FROM ARTIFACT a
        JOIN EXPERIMENT_ARTIFACT ea ON ea.artifact_id = a.id
        WHERE ea.experiment_id = {ph}
        """
        
        cursor = self._execute_query(query, (experiment_id,))
        rows = cursor.fetchall()
        
        return [
            Artifact(
                id=row["id"],
                type=row["type"],
                location=row["loc"]
            )
            for row in rows
        ]

    def get_trial_run_artifacts(self, trial_run_id: int) -> List[Artifact]:
        """Get all artifacts associated with a trial run."""
        ph = self._get_placeholder()
        
        query = f"""
        SELECT a.* FROM ARTIFACT a
        JOIN TRIAL_RUN_ARTIFACT tra ON tra.artifact_id = a.id
        WHERE tra.trial_run_id = {ph}
        """
        
        cursor = self._execute_query(query, (trial_run_id,))
        rows = cursor.fetchall()
        
        return [
            Artifact(
                id=row["id"],
                type=row["type"],
                location=row["loc"]
            )
            for row in rows
        ]

    def get_epoch_artifacts(self, epoch_idx: int, trial_run_id: int) -> List[Artifact]:
        """Get all artifacts associated with an epoch."""
        ph = self._get_placeholder()
        
        query = f"""
        SELECT a.* FROM ARTIFACT a
        JOIN EPOCH_ARTIFACT ea ON ea.artifact_id = a.id
        WHERE ea.epoch_idx = {ph} AND ea.epoch_trial_run_id = {ph}
        """
        
        cursor = self._execute_query(query, (epoch_idx, trial_run_id))
        rows = cursor.fetchall()
        
        return [
            Artifact(
                id=row["id"],
                type=row["type"],
                location=row["loc"]
            )
            for row in rows
        ]
