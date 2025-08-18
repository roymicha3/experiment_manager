"""Database manager for experiment tracking."""
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
import json
import logging
import sqlite3
import mysql.connector
from datetime import datetime
from pathlib import Path
import pandas as pd

from experiment_manager.db.db import init_sqlite_db, init_mysql_db
from experiment_manager.db.tables import Experiment, Trial, TrialRun, Metric, Artifact, Epoch

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
                 user: str = "root", password: str = "", recreate: bool = False, readonly: bool = False):
        """
        Initialize database connection.
        """
        self.use_sqlite = use_sqlite
        self.readonly = readonly
        try:
            if use_sqlite:
                self.connection = init_sqlite_db(database_path, recreate=recreate, readonly=readonly)
                self.connection.row_factory = sqlite3.Row
                self.cursor = self.connection.cursor()
            else:
                self.connection = init_mysql_db(host, user, password, database_path, recreate=recreate)
                self.cursor = self.connection.cursor(dictionary=True)
        except (sqlite3.Error, OSError, mysql.connector.Error) as e:
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
        # Check if experiment exists
        ph = self._get_placeholder()
        check_query = f"SELECT id FROM EXPERIMENT WHERE id = {ph}"
        cursor = self._execute_query(check_query, (experiment_id,))
        if not cursor.fetchone():
            raise QueryError(f"Experiment with id {experiment_id} does not exist")
        
        now = datetime.now().isoformat()
        
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
        # Check if trial exists
        ph = self._get_placeholder()
        check_query = f"SELECT id FROM TRIAL WHERE id = {ph}"
        cursor = self._execute_query(check_query, (trial_id,))
        if not cursor.fetchone():
            raise QueryError(f"Trial with id {trial_id} does not exist")
        
        now = datetime.now().isoformat()
        
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
        
        cursor = self._execute_query(query, (epoch_idx, trial_run_id, datetime.now().isoformat()))
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
        
        # Check if experiment exists
        check_exp_query = f"SELECT id FROM EXPERIMENT WHERE id = {ph}"
        cursor = self._execute_query(check_exp_query, (experiment_id,))
        if not cursor.fetchone():
            raise QueryError(f"Experiment with id {experiment_id} does not exist")
        
        # Check if artifact exists
        check_art_query = f"SELECT id FROM ARTIFACT WHERE id = {ph}"
        cursor = self._execute_query(check_art_query, (artifact_id,))
        if not cursor.fetchone():
            raise QueryError(f"Artifact with id {artifact_id} does not exist")
        
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
        
    
    def update_trial_run_status(self, trial_run_id: int, status: str) -> None:
        ph = self._get_placeholder()
        
        query = f"""
        UPDATE TRIAL_RUN
        SET status = {ph}
        WHERE id = {ph}
        """
        
        self._execute_query(query, (status, trial_run_id))
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

    # ========================
    # Data retrieval methods
    # ========================
    
    def get_experiment_data(self, 
                          experiment_ids: Optional[List[int]] = None,
                          filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Get comprehensive hierarchical experiment data.
        
        Single optimized query that joins across the experiment hierarchy:
        experiment → trial → trial_run → metrics, with optional configuration data.
        
        Args:
            experiment_ids: List of experiment IDs to include (if None, include all)
            filters: Additional filters dict with keys:
                - trial_names: List of trial names to include
                - run_status: List of run statuses to include
                - metric_types: List of metric types to include
                - date_range: Dict with 'start' and 'end' datetime
                - include_configs: Whether to include configuration data
                
        Returns:
            pd.DataFrame: Structured data with columns:
                experiment_id, experiment_title, trial_id, trial_name, 
                trial_run_id, run_status, run_start_time, run_update_time,
                metric_id, metric_type, metric_total_val, metric_per_label_val
        """
        if filters is None:
            filters = {}
            
        ph = self._get_placeholder()
        
        # Build the base query
        desc_field = "desc" if self.use_sqlite else "`desc`"
        query_parts = [f"""
        SELECT 
            e.id as experiment_id,
            e.title as experiment_title,
            e.{desc_field} as experiment_description,
            e.start_time as experiment_start_time,
            t.id as trial_id,
            t.name as trial_name,
            t.start_time as trial_start_time,
            tr.id as trial_run_id,
            tr.status as run_status,
            tr.start_time as run_start_time,
            tr.update_time as run_update_time,
            m.id as metric_id,
            m.type as metric_type,
            m.total_val as metric_total_val,
            m.per_label_val as metric_per_label_val,
            ep.idx as epoch_idx,
            ep.time as epoch_time
        FROM EXPERIMENT e
        LEFT JOIN TRIAL t ON t.experiment_id = e.id
        LEFT JOIN TRIAL_RUN tr ON tr.trial_id = t.id
        LEFT JOIN RESULTS_METRIC rm ON rm.results_id = tr.id
        LEFT JOIN METRIC m ON m.id = rm.metric_id
        LEFT JOIN EPOCH_METRIC em ON em.metric_id = m.id
        LEFT JOIN EPOCH ep ON ep.idx = em.epoch_idx AND ep.trial_run_id = em.epoch_trial_run_id
        """]
        
        where_conditions = []
        params = []
        
        # Filter by experiment IDs
        if experiment_ids:
            placeholders = ', '.join([ph] * len(experiment_ids))
            where_conditions.append(f"e.id IN ({placeholders})")
            params.extend(experiment_ids)
        
        # Filter by trial names
        if 'trial_names' in filters and filters['trial_names']:
            placeholders = ', '.join([ph] * len(filters['trial_names']))
            where_conditions.append(f"t.name IN ({placeholders})")
            params.extend(filters['trial_names'])
        
        # Filter by run status
        if 'run_status' in filters and filters['run_status']:
            placeholders = ', '.join([ph] * len(filters['run_status']))
            where_conditions.append(f"tr.status IN ({placeholders})")
            params.extend(filters['run_status'])
        
        # Filter by metric types
        if 'metric_types' in filters and filters['metric_types']:
            placeholders = ', '.join([ph] * len(filters['metric_types']))
            where_conditions.append(f"m.type IN ({placeholders})")
            params.extend(filters['metric_types'])
        
        # Filter by date range
        if 'date_range' in filters and filters['date_range']:
            date_range = filters['date_range']
            if 'start' in date_range:
                where_conditions.append(f"tr.start_time >= {ph}")
                params.append(date_range['start'].isoformat() if hasattr(date_range['start'], 'isoformat') else date_range['start'])
            if 'end' in date_range:
                where_conditions.append(f"tr.start_time <= {ph}")
                params.append(date_range['end'].isoformat() if hasattr(date_range['end'], 'isoformat') else date_range['end'])
        
        # Add WHERE clause if we have conditions
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        # Add ordering for consistent results
        query_parts.append("ORDER BY e.id, t.id, tr.id, m.id")
        
        query = " ".join(query_parts)
        
        try:
            cursor = self._execute_query(query, tuple(params) if params else None)
            rows = cursor.fetchall()
            
            # Convert to DataFrame
            if rows:
                # Convert rows to list of dicts for pandas
                data = []
                for row in rows:
                    row_dict = dict(row) if hasattr(row, 'keys') else {
                        'experiment_id': row[0], 'experiment_title': row[1], 'experiment_description': row[2],
                        'experiment_start_time': row[3], 'trial_id': row[4], 'trial_name': row[5],
                        'trial_start_time': row[6], 'trial_run_id': row[7], 'run_status': row[8],
                        'run_start_time': row[9], 'run_update_time': row[10], 'metric_id': row[11],
                        'metric_type': row[12], 'metric_total_val': row[13], 'metric_per_label_val': row[14],
                        'epoch_idx': row[15], 'epoch_time': row[16]
                    }
                    
                    # Parse JSON for per_label_val
                    if row_dict.get('metric_per_label_val'):
                        try:
                            row_dict['metric_per_label_val'] = json.loads(row_dict['metric_per_label_val'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['metric_per_label_val'] = None
                    
                    data.append(row_dict)
                
                return pd.DataFrame(data)
            else:
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=[
                    'experiment_id', 'experiment_title', 'experiment_description', 'experiment_start_time',
                    'trial_id', 'trial_name', 'trial_start_time', 'trial_run_id', 'run_status',
                    'run_start_time', 'run_update_time', 'metric_id', 'metric_type', 
                    'metric_total_val', 'metric_per_label_val', 'epoch_idx', 'epoch_time'
                ])
                
        except Exception as e:
            logger.error(f"Error executing experiment data query: {e}")
            raise QueryError(f"Failed to retrieve experiment data: {e}") from e

    def get_aggregated_metrics(self, 
                              experiment_ids: Optional[List[int]] = None,
                              group_by: str = 'trial',
                              functions: Optional[List[str]] = None) -> pd.DataFrame:
        """Get pre-aggregated metrics to reduce data transfer for large experiments.
        
        Args:
            experiment_ids: List of experiment IDs (if None, include all)
            group_by: Grouping level - 'experiment', 'trial', or 'trial_run'
            functions: List of aggregation functions ['mean', 'std', 'min', 'max', 'count']
                      (if None, uses ['mean', 'std', 'count'])
                      
        Returns:
            pd.DataFrame: Aggregated metrics with columns based on group_by level
        """
        if functions is None:
            functions = ['mean', 'std', 'count']
        
        # Validate inputs
        valid_group_by = ['experiment', 'trial', 'trial_run']
        if group_by not in valid_group_by:
            raise ValueError(f"group_by must be one of {valid_group_by}")
        
        valid_functions = ['mean', 'std', 'min', 'max', 'count', 'sum']
        invalid_functions = [f for f in functions if f not in valid_functions]
        if invalid_functions:
            raise ValueError(f"Invalid aggregation functions: {invalid_functions}")
        
        ph = self._get_placeholder()
        
        # Map aggregation functions to SQL
        sql_functions = {
            'mean': 'AVG(m.total_val)',
            'std': 'STDDEV(m.total_val)' if not self.use_sqlite else 'AVG((m.total_val - sub.avg_val) * (m.total_val - sub.avg_val))',
            'min': 'MIN(m.total_val)',
            'max': 'MAX(m.total_val)',
            'count': 'COUNT(m.total_val)',
            'sum': 'SUM(m.total_val)'
        }
        
        # Build SELECT clause for aggregations
        agg_selects = []
        for func in functions:
            if func == 'std' and self.use_sqlite:
                # SQLite doesn't have STDDEV, so we need a subquery approach
                continue  # Handle SQLite std separately
            agg_selects.append(f"{sql_functions[func]} as {func}")
        
        agg_select_str = ', '.join(agg_selects)
        
        # Build GROUP BY and SELECT based on grouping level
        if group_by == 'experiment':
            group_fields = ['e.id', 'e.title', 'm.type']
            select_fields = ['e.id as experiment_id', 'e.title as experiment_title', 'm.type as metric_type']
            group_by_str = 'e.id, e.title, m.type'
        elif group_by == 'trial':
            group_fields = ['e.id', 'e.title', 't.id', 't.name', 'm.type']
            select_fields = ['e.id as experiment_id', 'e.title as experiment_title', 
                           't.id as trial_id', 't.name as trial_name', 'm.type as metric_type']
            group_by_str = 'e.id, e.title, t.id, t.name, m.type'
        else:  # trial_run
            group_fields = ['e.id', 'e.title', 't.id', 't.name', 'tr.id', 'tr.status', 'm.type']
            select_fields = ['e.id as experiment_id', 'e.title as experiment_title',
                           't.id as trial_id', 't.name as trial_name',
                           'tr.id as trial_run_id', 'tr.status as run_status', 'm.type as metric_type']
            group_by_str = 'e.id, e.title, t.id, t.name, tr.id, tr.status, m.type'
        
        select_str = ', '.join(select_fields + [agg_select_str])
        
        query_parts = [f"""
        SELECT {select_str}
        FROM EXPERIMENT e
        JOIN TRIAL t ON t.experiment_id = e.id
        JOIN TRIAL_RUN tr ON tr.trial_id = t.id
        JOIN RESULTS_METRIC rm ON rm.results_id = tr.id
        JOIN METRIC m ON m.id = rm.metric_id
        """]
        
        where_conditions = []
        params = []
        
        # Filter by experiment IDs
        if experiment_ids:
            placeholders = ', '.join([ph] * len(experiment_ids))
            where_conditions.append(f"e.id IN ({placeholders})")
            params.extend(experiment_ids)
        
        # Add WHERE clause if we have conditions
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        query_parts.append(f"GROUP BY {group_by_str}")
        query_parts.append("ORDER BY e.id, t.id")
        
        query = " ".join(query_parts)
        
        try:
            cursor = self._execute_query(query, tuple(params) if params else None)
            rows = cursor.fetchall()
            
            if rows:
                data = [dict(row) if hasattr(row, 'keys') else 
                       {key: row[i] for i, key in enumerate([
                           field.split(' as ')[1] if ' as ' in field else field 
                           for field in select_fields + functions
                       ])} for row in rows]
                return pd.DataFrame(data)
            else:
                # Return empty DataFrame with expected columns
                expected_columns = [field.split(' as ')[1] if ' as ' in field else field 
                                  for field in select_fields] + functions
                return pd.DataFrame(columns=expected_columns)
                
        except Exception as e:
            logger.error(f"Error executing aggregated metrics query: {e}")
            raise QueryError(f"Failed to retrieve aggregated metrics: {e}") from e

    def get_failure_data(self, 
                        experiment_ids: Optional[List[int]] = None,
                        include_configs: bool = False) -> pd.DataFrame:
        """Get specialized data for failure analysis.
        
        Joins run status with timing and error information, optionally including
        configuration parameters for failure pattern analysis.
        
        Args:
            experiment_ids: List of experiment IDs (if None, include all)
            include_configs: Whether to include configuration data (artifacts)
            
        Returns:
            pd.DataFrame: Failure analysis data with columns:
                experiment_id, trial_id, trial_run_id, status, start_time, 
                update_time, duration_seconds, failure_reason (if available),
                config_data (if include_configs=True)
        """
        ph = self._get_placeholder()
        
        # Base query for failure data
        select_parts = ["""
            e.id as experiment_id,
            e.title as experiment_title,
            t.id as trial_id,
            t.name as trial_name,
            tr.id as trial_run_id,
            tr.status as run_status,
            tr.start_time as run_start_time,
            tr.update_time as run_update_time
        """]
        
        from_parts = ["""
        FROM EXPERIMENT e
        JOIN TRIAL t ON t.experiment_id = e.id
        JOIN TRIAL_RUN tr ON tr.trial_id = t.id
        """]
        
        # Add configuration data if requested
        if include_configs:
            select_parts.append("""
            , a.location as config_location,
            a.type as config_type
            """)
            from_parts.append("""
            LEFT JOIN TRIAL_RUN_ARTIFACT tra ON tra.trial_run_id = tr.id
            LEFT JOIN ARTIFACT a ON a.id = tra.artifact_id AND a.type LIKE '%config%'
            """)
        
        query_parts = ["SELECT " + " ".join(select_parts)] + from_parts
        
        where_conditions = []
        params = []
        
        # Filter by experiment IDs
        if experiment_ids:
            placeholders = ', '.join([ph] * len(experiment_ids))
            where_conditions.append(f"e.id IN ({placeholders})")
            params.extend(experiment_ids)
        
        # Add WHERE clause if we have conditions
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        query_parts.append("ORDER BY e.id, t.id, tr.id")
        
        query = " ".join(query_parts)
        
        try:
            cursor = self._execute_query(query, tuple(params) if params else None)
            rows = cursor.fetchall()
            
            if rows:
                data = []
                for row in rows:
                    row_dict = dict(row) if hasattr(row, 'keys') else {
                        'experiment_id': row[0], 'experiment_title': row[1],
                        'trial_id': row[2], 'trial_name': row[3],
                        'trial_run_id': row[4], 'run_status': row[5],
                        'run_start_time': row[6], 'run_update_time': row[7]
                    }
                    
                    # Add config data if included
                    if include_configs and len(row) > 8:
                        row_dict['config_location'] = row[8] if len(row) > 8 else None
                        row_dict['config_type'] = row[9] if len(row) > 9 else None
                    
                    # Calculate duration if both timestamps are available
                    if row_dict.get('run_start_time') and row_dict.get('run_update_time'):
                        try:
                            start_time = datetime.fromisoformat(row_dict['run_start_time']) if isinstance(row_dict['run_start_time'], str) else row_dict['run_start_time']
                            update_time = datetime.fromisoformat(row_dict['run_update_time']) if isinstance(row_dict['run_update_time'], str) else row_dict['run_update_time']
                            duration = (update_time - start_time).total_seconds()
                            row_dict['duration_seconds'] = duration
                        except (ValueError, TypeError):
                            row_dict['duration_seconds'] = None
                    else:
                        row_dict['duration_seconds'] = None
                    
                    data.append(row_dict)
                
                return pd.DataFrame(data)
            else:
                # Return empty DataFrame with expected columns
                base_columns = [
                    'experiment_id', 'experiment_title', 'trial_id', 'trial_name',
                    'trial_run_id', 'run_status', 'run_start_time', 'run_update_time',
                    'duration_seconds'
                ]
                if include_configs:
                    base_columns.extend(['config_location', 'config_type'])
                return pd.DataFrame(columns=base_columns)
                
        except Exception as e:
            logger.error(f"Error executing failure data query: {e}")
            raise QueryError(f"Failed to retrieve failure data: {e}") from e

    def get_epoch_series(self, 
                        trial_run_ids: List[int],
                        metric_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Get time series data for training curve analysis.
        
        Optimized for epoch-level metric extraction to analyze training progression.
        
        Args:
            trial_run_ids: List of trial run IDs to analyze
            metric_types: List of metric types to include (if None, include all)
            
        Returns:
            pd.DataFrame: Time series data with columns:
                trial_run_id, epoch_idx, epoch_time, metric_type, 
                metric_total_val, metric_per_label_val
        """
        if not trial_run_ids:
            return pd.DataFrame(columns=[
                'trial_run_id', 'epoch_idx', 'epoch_time', 
                'metric_type', 'metric_total_val', 'metric_per_label_val'
            ])
        
        ph = self._get_placeholder()
        
        query_parts = ["""
        SELECT 
            tr.id as trial_run_id,
            ep.idx as epoch_idx,
            ep.time as epoch_time,
            m.type as metric_type,
            m.total_val as metric_total_val,
            m.per_label_val as metric_per_label_val
        FROM TRIAL_RUN tr
        JOIN EPOCH ep ON ep.trial_run_id = tr.id
        JOIN EPOCH_METRIC em ON em.epoch_idx = ep.idx AND em.epoch_trial_run_id = ep.trial_run_id
        JOIN METRIC m ON m.id = em.metric_id
        """]
        
        where_conditions = []
        params = []
        
        # Filter by trial run IDs
        placeholders = ', '.join([ph] * len(trial_run_ids))
        where_conditions.append(f"tr.id IN ({placeholders})")
        params.extend(trial_run_ids)
        
        # Filter by metric types
        if metric_types:
            placeholders = ', '.join([ph] * len(metric_types))
            where_conditions.append(f"m.type IN ({placeholders})")
            params.extend(metric_types)
        
        query_parts.append("WHERE " + " AND ".join(where_conditions))
        query_parts.append("ORDER BY tr.id, ep.idx, m.type")
        
        query = " ".join(query_parts)
        
        try:
            cursor = self._execute_query(query, tuple(params))
            rows = cursor.fetchall()
            
            if rows:
                data = []
                for row in rows:
                    row_dict = dict(row) if hasattr(row, 'keys') else {
                        'trial_run_id': row[0],
                        'epoch_idx': row[1],
                        'epoch_time': row[2],
                        'metric_type': row[3],
                        'metric_total_val': row[4],
                        'metric_per_label_val': row[5]
                    }
                    
                    # Parse JSON for per_label_val
                    if row_dict.get('metric_per_label_val'):
                        try:
                            row_dict['metric_per_label_val'] = json.loads(row_dict['metric_per_label_val'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['metric_per_label_val'] = None
                    
                    data.append(row_dict)
                
                return pd.DataFrame(data)
            else:
                return pd.DataFrame(columns=[
                    'trial_run_id', 'epoch_idx', 'epoch_time', 
                    'metric_type', 'metric_total_val', 'metric_per_label_val'
                ])
                
        except Exception as e:
            logger.error(f"Error executing epoch series query: {e}")
            raise QueryError(f"Failed to retrieve epoch series data: {e}") from e

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute a custom SQL query and return results as DataFrame.
        
        This method is used for custom data queries.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            cursor = self._execute_query(query, params)
            rows = cursor.fetchall()
            
            if rows:
                # Get column names
                if hasattr(cursor, 'description') and cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    data = [dict(zip(columns, row)) if not hasattr(row, 'keys') else dict(row) for row in rows]
                else:
                    # Fallback for rows that are already dicts
                    data = [dict(row) if hasattr(row, 'keys') else row for row in rows]
                
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error executing custom query: {e}")
            raise QueryError(f"Failed to execute query: {e}") from e

    def create_optimized_indexes(self) -> None:
        """Create database indexes optimized for data queries.
        
        This method creates indexes that improve performance for the data
        methods above. Should be called after database initialization.
        """
        indexes = [
            # Experiment-based queries
            "CREATE INDEX IF NOT EXISTS idx_trial_experiment_id ON TRIAL(experiment_id)",
            "CREATE INDEX IF NOT EXISTS idx_trial_run_trial_id ON TRIAL_RUN(trial_id)",
            "CREATE INDEX IF NOT EXISTS idx_trial_run_status ON TRIAL_RUN(status)",
            
            # Metric-based queries
            "CREATE INDEX IF NOT EXISTS idx_results_metric_results_id ON RESULTS_METRIC(results_id)",
            "CREATE INDEX IF NOT EXISTS idx_results_metric_metric_id ON RESULTS_METRIC(metric_id)",
            "CREATE INDEX IF NOT EXISTS idx_metric_type ON METRIC(type)",
            
            # Epoch-based queries
            "CREATE INDEX IF NOT EXISTS idx_epoch_trial_run_id ON EPOCH(trial_run_id)",
            "CREATE INDEX IF NOT EXISTS idx_epoch_metric_epoch ON EPOCH_METRIC(epoch_idx, epoch_trial_run_id)",
            "CREATE INDEX IF NOT EXISTS idx_epoch_metric_metric_id ON EPOCH_METRIC(metric_id)",
            
            # Time-based queries
            "CREATE INDEX IF NOT EXISTS idx_trial_run_start_time ON TRIAL_RUN(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_experiment_start_time ON EXPERIMENT(start_time)",
            
            # Artifact-based queries (for configuration analysis)
            "CREATE INDEX IF NOT EXISTS idx_trial_run_artifact_trial_run_id ON TRIAL_RUN_ARTIFACT(trial_run_id)",
            "CREATE INDEX IF NOT EXISTS idx_artifact_type ON ARTIFACT(type)",
        ]
        
        for index_sql in indexes:
            try:
                self._execute_query(index_sql)
                self.connection.commit()
                logger.info(f"Created index: {index_sql}")
            except Exception as e:
                logger.warning(f"Failed to create index: {index_sql}. Error: {e}")
                # Continue with other indexes even if one fails
