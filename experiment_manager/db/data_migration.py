"""Data migration utilities for analysts working with experiment data."""
import json
import logging
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
from threading import Lock
import time

from experiment_manager.db.manager import DatabaseManager, DatabaseError
from experiment_manager.db.migration_manager import MigrationManager, MigrationError
from experiment_manager.db.tables import Experiment, Trial, TrialRun, Metric, Artifact, Epoch
from experiment_manager.db.version_utils import get_next_version

logger = logging.getLogger(__name__)

class DataMigrationError(Exception):
    """Error in data migration operations."""
    pass

class MigrationStrategy(Enum):
    """Strategy for data migration."""
    CONSERVATIVE = "conservative"  # Strict validation, slow but safe
    BALANCED = "balanced"         # Moderate validation, good balance
    AGGRESSIVE = "aggressive"     # Minimal validation, fast but risky

@dataclass
class MigrationProgress:
    """Progress tracking for migration operations."""
    total_items: int
    processed_items: int
    failed_items: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    current_operation: str = ""
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.processed_items == 0:
            return 100.0
        return ((self.processed_items - self.failed_items) / self.processed_items) * 100.0
    
    def update_eta(self):
        """Update estimated time of completion."""
        if self.processed_items > 0:
            elapsed = datetime.now() - self.start_time
            elapsed_seconds = elapsed.total_seconds()
            
            # Avoid division by zero for very fast operations
            if elapsed_seconds > 0:
                rate = self.processed_items / elapsed_seconds
                remaining_items = self.total_items - self.processed_items
                if rate > 0:
                    remaining_seconds = remaining_items / rate
                    self.estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)

@dataclass
class DataSnapshot:
    """Represents a data snapshot for rollback purposes."""
    snapshot_id: str
    created_at: datetime
    description: str
    file_path: Path
    metadata: Dict[str, Any]
    size_bytes: int

class MetricTransformer:
    """Utilities for transforming JSON per-label metrics."""
    
    @staticmethod
    def validate_metric_json(json_data: Union[str, Dict]) -> Tuple[bool, Optional[str]]:
        """Validate metric JSON structure.
        
        Args:
            json_data: JSON string or dict to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data
            
            if not isinstance(data, dict):
                return False, "Metric JSON must be a dictionary"
            
            # Check for valid metric structure
            for key, value in data.items():
                if not isinstance(key, str):
                    return False, f"Metric keys must be strings, got {type(key)}"
                if not isinstance(value, (int, float)):
                    return False, f"Metric values must be numeric, got {type(value)} for key '{key}'"
            
            return True, None
        
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    @staticmethod
    def transform_metric_format(metric_data: Dict, transformation_rules: Dict[str, Callable]) -> Dict:
        """Transform metric data according to provided rules.
        
        Args:
            metric_data: Original metric data
            transformation_rules: Dict mapping field names to transformation functions
            
        Returns:
            Transformed metric data
        """
        transformed = metric_data.copy()
        
        for field, transform_func in transformation_rules.items():
            if field in transformed:
                try:
                    transformed[field] = transform_func(transformed[field])
                except Exception as e:
                    logger.warning(f"Failed to transform field '{field}': {e}")
        
        return transformed
    
    @staticmethod
    def aggregate_per_label_metrics(metrics: List[Dict], aggregation_func: str = "mean") -> Dict:
        """Aggregate multiple per-label metrics into a single metric.
        
        Args:
            metrics: List of per-label metric dictionaries
            aggregation_func: Aggregation function ('mean', 'sum', 'max', 'min')
            
        Returns:
            Aggregated metric dictionary
        """
        if not metrics:
            return {}
        
        # Get all unique labels
        all_labels = set()
        for metric in metrics:
            if isinstance(metric, dict):
                all_labels.update(metric.keys())
        
        aggregated = {}
        
        for label in all_labels:
            values = []
            for metric in metrics:
                if isinstance(metric, dict) and label in metric:
                    values.append(metric[label])
            
            if values:
                if aggregation_func == "mean":
                    aggregated[label] = sum(values) / len(values)
                elif aggregation_func == "sum":
                    aggregated[label] = sum(values)
                elif aggregation_func == "max":
                    aggregated[label] = max(values)
                elif aggregation_func == "min":
                    aggregated[label] = min(values)
                else:
                    raise ValueError(f"Unsupported aggregation function: {aggregation_func}")
        
        return aggregated

class HierarchyPreserver:
    """Utilities for preserving experiment hierarchy during migration."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize hierarchy preserv with database manager.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db_manager = db_manager
    
    def get_experiment_hierarchy(self, experiment_id: int) -> Dict[str, Any]:
        """Get complete hierarchy for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary containing full hierarchy data
        """
        ph = self.db_manager._get_placeholder()
        
        # Get experiment
        exp_cursor = self.db_manager._execute_query(
            f"SELECT * FROM EXPERIMENT WHERE id = {ph}",
            (experiment_id,)
        )
        experiment_data = exp_cursor.fetchone()
        
        if not experiment_data:
            raise DataMigrationError(f"Experiment {experiment_id} not found")
        
        # Get trials
        trials_cursor = self.db_manager._execute_query(
            f"SELECT * FROM TRIAL WHERE experiment_id = {ph}",
            (experiment_id,)
        )
        trials_data = trials_cursor.fetchall()
        
        # Get trial runs and associated data
        hierarchy = {
            "experiment": dict(experiment_data),
            "trials": []
        }
        
        for trial_data in trials_data:
            trial_id = trial_data["id"]
            
            # Get trial runs
            runs_cursor = self.db_manager._execute_query(
                f"SELECT * FROM TRIAL_RUN WHERE trial_id = {ph}",
                (trial_id,)
            )
            runs_data = runs_cursor.fetchall()
            
            trial_hierarchy = {
                "trial": dict(trial_data),
                "runs": []
            }
            
            for run_data in runs_data:
                run_id = run_data["id"]
                
                # Get epochs
                epochs_cursor = self.db_manager._execute_query(
                    f"SELECT * FROM EPOCH WHERE trial_run_id = {ph} ORDER BY idx",
                    (run_id,)
                )
                epochs_data = epochs_cursor.fetchall()
                
                # Get results
                results_cursor = self.db_manager._execute_query(
                    f"SELECT * FROM RESULTS WHERE trial_run_id = {ph}",
                    (run_id,)
                )
                results_data = results_cursor.fetchone()
                
                run_hierarchy = {
                    "run": dict(run_data),
                    "epochs": [dict(epoch) for epoch in epochs_data],
                    "results": dict(results_data) if results_data else None
                }
                
                trial_hierarchy["runs"].append(run_hierarchy)
            
            hierarchy["trials"].append(trial_hierarchy)
        
        return hierarchy
    
    def validate_hierarchy_integrity(self, experiment_id: int) -> Tuple[bool, List[str]]:
        """Validate the integrity of experiment hierarchy.
        
        Args:
            experiment_id: ID of the experiment to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        ph = self.db_manager._get_placeholder()
        
        try:
            # Check experiment exists
            exp_cursor = self.db_manager._execute_query(
                f"SELECT COUNT(*) as count FROM EXPERIMENT WHERE id = {ph}",
                (experiment_id,)
            )
            if exp_cursor.fetchone()["count"] == 0:
                issues.append(f"Experiment {experiment_id} does not exist")
                return False, issues
            
            # Check trials reference valid experiment
            orphan_trials_cursor = self.db_manager._execute_query(
                f"""SELECT t.id FROM TRIAL t 
                   LEFT JOIN EXPERIMENT e ON t.experiment_id = e.id 
                   WHERE t.experiment_id = {ph} AND e.id IS NULL""",
                (experiment_id,)
            )
            orphan_trials = orphan_trials_cursor.fetchall()
            for trial in orphan_trials:
                issues.append(f"Trial {trial['id']} references non-existent experiment")
            
            # Check trial runs reference valid trials
            orphan_runs_cursor = self.db_manager._execute_query(
                f"""SELECT tr.id FROM TRIAL_RUN tr 
                   LEFT JOIN TRIAL t ON tr.trial_id = t.id 
                   WHERE t.experiment_id = {ph} AND t.id IS NULL""",
                (experiment_id,)
            )
            orphan_runs = orphan_runs_cursor.fetchall()
            for run in orphan_runs:
                issues.append(f"Trial run {run['id']} references non-existent trial")
            
            # Check epochs reference valid trial runs
            orphan_epochs_cursor = self.db_manager._execute_query(
                f"""SELECT e.idx, e.trial_run_id FROM EPOCH e 
                   LEFT JOIN TRIAL_RUN tr ON e.trial_run_id = tr.id 
                   LEFT JOIN TRIAL t ON tr.trial_id = t.id
                   WHERE t.experiment_id = {ph} AND tr.id IS NULL""",
                (experiment_id,)
            )
            orphan_epochs = orphan_epochs_cursor.fetchall()
            for epoch in orphan_epochs:
                issues.append(f"Epoch {epoch['idx']} references non-existent trial run {epoch['trial_run_id']}")
            
            # Check results reference valid trial runs
            orphan_results_cursor = self.db_manager._execute_query(
                f"""SELECT r.trial_run_id FROM RESULTS r 
                   LEFT JOIN TRIAL_RUN tr ON r.trial_run_id = tr.id 
                   LEFT JOIN TRIAL t ON tr.trial_id = t.id
                   WHERE t.experiment_id = {ph} AND tr.id IS NULL""",
                (experiment_id,)
            )
            orphan_results = orphan_results_cursor.fetchall()
            for result in orphan_results:
                issues.append(f"Results references non-existent trial run {result['trial_run_id']}")
            
            return len(issues) == 0, issues
        
        except Exception as e:
            issues.append(f"Error during validation: {e}")
            return False, issues

class DataValidator:
    """Validation utilities for data migration."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize validator with database manager.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db_manager = db_manager
    
    def validate_foreign_keys(self) -> Dict[str, List[str]]:
        """Validate all foreign key constraints.
        
        Returns:
            Dictionary mapping table names to lists of constraint violations
        """
        violations = {}
        ph = self.db_manager._get_placeholder()
        
        # Define foreign key relationships to check
        fk_checks = [
            ("TRIAL", "experiment_id", "EXPERIMENT", "id"),
            ("TRIAL_RUN", "trial_id", "TRIAL", "id"),
            ("RESULTS", "trial_run_id", "TRIAL_RUN", "id"),
            ("EPOCH", "trial_run_id", "TRIAL_RUN", "id"),
            ("EXPERIMENT_ARTIFACT", "experiment_id", "EXPERIMENT", "id"),
            ("EXPERIMENT_ARTIFACT", "artifact_id", "ARTIFACT", "id"),
            ("TRIAL_ARTIFACT", "trial_id", "TRIAL", "id"),
            ("TRIAL_ARTIFACT", "artifact_id", "ARTIFACT", "id"),
            ("TRIAL_RUN_ARTIFACT", "trial_run_id", "TRIAL_RUN", "id"),
            ("TRIAL_RUN_ARTIFACT", "artifact_id", "ARTIFACT", "id"),
            ("RESULTS_METRIC", "results_id", "RESULTS", "trial_run_id"),
            ("RESULTS_METRIC", "metric_id", "METRIC", "id"),
            ("RESULTS_ARTIFACT", "results_id", "RESULTS", "trial_run_id"),
            ("RESULTS_ARTIFACT", "artifact_id", "ARTIFACT", "id"),
            ("EPOCH_METRIC", "epoch_trial_run_id", "TRIAL_RUN", "id"),
            ("EPOCH_METRIC", "metric_id", "METRIC", "id"),
            ("EPOCH_ARTIFACT", "epoch_trial_run_id", "TRIAL_RUN", "id"),
            ("EPOCH_ARTIFACT", "artifact_id", "ARTIFACT", "id"),
        ]
        
        for child_table, child_col, parent_table, parent_col in fk_checks:
            try:
                query = f"""
                SELECT {child_col} FROM {child_table} c
                WHERE NOT EXISTS (
                    SELECT 1 FROM {parent_table} p 
                    WHERE p.{parent_col} = c.{child_col}
                )
                """
                cursor = self.db_manager._execute_query(query)
                orphans = cursor.fetchall()
                
                if orphans:
                    table_violations = [
                        f"{child_table}.{child_col} = {row[child_col]} references non-existent {parent_table}.{parent_col}"
                        for row in orphans
                    ]
                    violations[child_table] = violations.get(child_table, []) + table_violations
            
            except Exception as e:
                logger.error(f"Error checking foreign key {child_table}.{child_col}: {e}")
                violations[child_table] = violations.get(child_table, []) + [f"Check failed: {e}"]
        
        return violations
    
    def validate_json_metrics(self) -> List[Dict[str, Any]]:
        """Validate JSON structure in metric per_label_val fields.
        
        Returns:
            List of validation issues
        """
        issues = []
        
        try:
            cursor = self.db_manager._execute_query(
                "SELECT id, type, per_label_val FROM METRIC WHERE per_label_val IS NOT NULL"
            )
            metrics = cursor.fetchall()
            
            for metric in metrics:
                metric_id = metric["id"]
                metric_type = metric["type"]
                json_data = metric["per_label_val"]
                
                is_valid, error_msg = MetricTransformer.validate_metric_json(json_data)
                if not is_valid:
                    issues.append({
                        "metric_id": metric_id,
                        "metric_type": metric_type,
                        "error": error_msg,
                        "data": json_data[:100] + "..." if len(str(json_data)) > 100 else json_data
                    })
        
        except Exception as e:
            issues.append({
                "metric_id": None,
                "metric_type": None,
                "error": f"Failed to validate metrics: {e}",
                "data": None
            })
        
        return issues
    
    def validate_data_consistency(self) -> Dict[str, Any]:
        """Perform comprehensive data consistency checks.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "foreign_key_violations": self.validate_foreign_keys(),
            "json_metric_issues": self.validate_json_metrics(),
            "summary": {}
        }
        
        # Calculate summary statistics
        total_fk_violations = sum(len(violations) for violations in results["foreign_key_violations"].values())
        total_json_issues = len(results["json_metric_issues"])
        
        results["summary"] = {
            "total_foreign_key_violations": total_fk_violations,
            "total_json_metric_issues": total_json_issues,
            "overall_status": "PASS" if total_fk_violations == 0 and total_json_issues == 0 else "FAIL"
        }
        
        return results

class SnapshotManager:
    """Manages data snapshots for migration rollback."""
    
    def __init__(self, db_manager: DatabaseManager, snapshot_dir: Union[str, Path] = "snapshots"):
        """Initialize snapshot manager.
        
        Args:
            db_manager: DatabaseManager instance
            snapshot_dir: Directory to store snapshots
        """
        self.db_manager = db_manager
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    def create_snapshot(self, description: str = "Data migration snapshot") -> DataSnapshot:
        """Create a full database snapshot.
        
        Args:
            description: Description of the snapshot
            
        Returns:
            DataSnapshot object
        """
        snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        snapshot_path = self.snapshot_dir / f"{snapshot_id}.db"
        
        try:
            if self.db_manager.use_sqlite:
                # For SQLite, copy the database file
                if hasattr(self.db_manager, 'database_path') and self.db_manager.database_path:
                    source_path = Path(self.db_manager.database_path)
                    if source_path.exists():
                        shutil.copy2(source_path, snapshot_path)
                    else:
                        # Database might be in-memory, dump to file
                        self._dump_sqlite_to_file(snapshot_path)
                else:
                    self._dump_sqlite_to_file(snapshot_path)
            else:
                # For MySQL, create a SQLite dump
                self._dump_mysql_to_sqlite(snapshot_path)
            
            size_bytes = snapshot_path.stat().st_size if snapshot_path.exists() else 0
            
            # Get metadata about the database state
            metadata = self._collect_database_metadata()
            
            snapshot = DataSnapshot(
                snapshot_id=snapshot_id,
                created_at=datetime.now(),
                description=description,
                file_path=snapshot_path,
                metadata=metadata,
                size_bytes=size_bytes
            )
            
            # Save snapshot metadata
            self._save_snapshot_metadata(snapshot)
            
            logger.info(f"Created snapshot {snapshot_id} at {snapshot_path}")
            return snapshot
        
        except Exception as e:
            if snapshot_path.exists():
                snapshot_path.unlink()
            raise DataMigrationError(f"Failed to create snapshot: {e}") from e
    
    def _dump_sqlite_to_file(self, output_path: Path):
        """Dump SQLite database to file."""
        with sqlite3.connect(str(output_path)) as backup_conn:
            self.db_manager.connection.backup(backup_conn)
    
    def _dump_mysql_to_sqlite(self, output_path: Path):
        """Dump MySQL database to SQLite file."""
        # This is a simplified implementation
        # In production, you might want to use mysqldump or similar tools
        
        with sqlite3.connect(str(output_path)) as sqlite_conn:
            sqlite_conn.row_factory = sqlite3.Row
            
            # Get all table names
            tables_cursor = self.db_manager._execute_query("SHOW TABLES")
            table_names = [row[0] for row in tables_cursor.fetchall()]
            
            for table_name in table_names:
                # Get table structure (simplified)
                desc_cursor = self.db_manager._execute_query(f"DESCRIBE {table_name}")
                columns = desc_cursor.fetchall()
                
                # Create table in SQLite (simplified column mapping)
                create_sql = f"CREATE TABLE {table_name} ("
                col_defs = []
                for col in columns:
                    col_name = col['Field']
                    col_type = col['Type']
                    # Simple type mapping
                    if 'int' in col_type.lower():
                        sqlite_type = 'INTEGER'
                    elif 'float' in col_type.lower() or 'double' in col_type.lower():
                        sqlite_type = 'REAL'
                    else:
                        sqlite_type = 'TEXT'
                    col_defs.append(f"{col_name} {sqlite_type}")
                
                create_sql += ", ".join(col_defs) + ")"
                sqlite_conn.execute(create_sql)
                
                # Copy data
                data_cursor = self.db_manager._execute_query(f"SELECT * FROM {table_name}")
                rows = data_cursor.fetchall()
                
                if rows:
                    placeholders = ", ".join(["?" for _ in range(len(rows[0]))])
                    insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
                    sqlite_conn.executemany(insert_sql, [tuple(row) for row in rows])
            
            sqlite_conn.commit()
    
    def _collect_database_metadata(self) -> Dict[str, Any]:
        """Collect metadata about current database state."""
        metadata = {}
        
        try:
            # Count records in each table
            tables = [
                "EXPERIMENT", "TRIAL", "TRIAL_RUN", "RESULTS", "EPOCH", 
                "METRIC", "ARTIFACT", "SCHEMA_VERSION"
            ]
            
            for table in tables:
                try:
                    cursor = self.db_manager._execute_query(f"SELECT COUNT(*) as count FROM {table}")
                    count = cursor.fetchone()["count"]
                    metadata[f"{table.lower()}_count"] = count
                except Exception:
                    metadata[f"{table.lower()}_count"] = 0
            
            # Get current schema version
            current_version = self.db_manager.get_current_schema_version()
            metadata["schema_version"] = current_version.version if current_version else None
            
        except Exception as e:
            logger.warning(f"Failed to collect some metadata: {e}")
        
        return metadata
    
    def _save_snapshot_metadata(self, snapshot: DataSnapshot):
        """Save snapshot metadata to JSON file."""
        metadata_path = self.snapshot_dir / f"{snapshot.snapshot_id}_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(asdict(snapshot), f, indent=2, default=str)
    
    def list_snapshots(self) -> List[DataSnapshot]:
        """List all available snapshots.
        
        Returns:
            List of DataSnapshot objects
        """
        snapshots = []
        
        for metadata_file in self.snapshot_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Convert string dates back to datetime objects
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['file_path'] = Path(data['file_path'])
                
                snapshot = DataSnapshot(**data)
                if snapshot.file_path.exists():
                    snapshots.append(snapshot)
                else:
                    logger.warning(f"Snapshot file missing: {snapshot.file_path}")
            
            except Exception as e:
                logger.error(f"Failed to load snapshot metadata from {metadata_file}: {e}")
        
        return sorted(snapshots, key=lambda s: s.created_at, reverse=True)
    
    def restore_snapshot(self, snapshot_id: str):
        """Restore database from a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to restore
            
        Raises:
            DataMigrationError: If restoration fails
        """
        snapshots = self.list_snapshots()
        target_snapshot = None
        
        for snapshot in snapshots:
            if snapshot.snapshot_id == snapshot_id:
                target_snapshot = snapshot
                break
        
        if not target_snapshot:
            raise DataMigrationError(f"Snapshot {snapshot_id} not found")
        
        if not target_snapshot.file_path.exists():
            raise DataMigrationError(f"Snapshot file not found: {target_snapshot.file_path}")
        
        try:
            if self.db_manager.use_sqlite:
                # For SQLite, replace the database file or restore to current connection
                if hasattr(self.db_manager, 'database_path') and self.db_manager.database_path and self.db_manager.database_path != ":memory:":
                    target_path = Path(self.db_manager.database_path)
                    # Close current connection
                    self.db_manager.connection.close()
                    # Replace file
                    shutil.copy2(target_snapshot.file_path, target_path)
                    # Reconnect
                    self.db_manager._connect()
                else:
                    # For in-memory or test databases, restore by recreating data
                    self._restore_sqlite_from_file(target_snapshot.file_path)
            else:
                # For MySQL, would need to implement MySQL restoration
                raise DataMigrationError("MySQL snapshot restoration not yet implemented")
            
            logger.info(f"Successfully restored snapshot {snapshot_id}")
        
        except Exception as e:
            raise DataMigrationError(f"Failed to restore snapshot: {e}") from e
    
    def _restore_sqlite_from_file(self, snapshot_path: Path):
        """Restore SQLite database from snapshot file to current connection."""
        # Drop all existing tables
        tables_cursor = self.db_manager._execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        table_names = [row[0] for row in tables_cursor.fetchall()]
        
        for table_name in table_names:
            self.db_manager._execute_query(f"DROP TABLE IF EXISTS {table_name}")
        
        self.db_manager.connection.commit()
        
        # Restore from snapshot
        with sqlite3.connect(str(snapshot_path)) as source_conn:
            source_conn.backup(self.db_manager.connection)
        
        self.db_manager.connection.commit()

class DataMigrationManager:
    """Main manager for data migration operations."""
    
    def __init__(self, db_manager: DatabaseManager, 
                 migration_dir: Union[str, Path] = "migrations",
                 snapshot_dir: Union[str, Path] = "snapshots"):
        """Initialize data migration manager.
        
        Args:
            db_manager: DatabaseManager instance
            migration_dir: Directory for migration files
            snapshot_dir: Directory for snapshots
        """
        self.db_manager = db_manager
        self.migration_manager = MigrationManager(db_manager, migration_dir)
        self.snapshot_manager = SnapshotManager(db_manager, snapshot_dir)
        self.validator = DataValidator(db_manager)
        self.hierarchy_preserver = HierarchyPreserver(db_manager)
        
        # Progress tracking
        self._progress_lock = Lock()
        self._current_progress: Optional[MigrationProgress] = None
        self._progress_callbacks: List[Callable[[MigrationProgress], None]] = []
    
    def add_progress_callback(self, callback: Callable[[MigrationProgress], None]):
        """Add a callback to receive progress updates.
        
        Args:
            callback: Function to call with progress updates
        """
        self._progress_callbacks.append(callback)
    
    def _update_progress(self, **kwargs):
        """Update migration progress and notify callbacks."""
        with self._progress_lock:
            if self._current_progress:
                for key, value in kwargs.items():
                    if hasattr(self._current_progress, key):
                        setattr(self._current_progress, key, value)
                
                self._current_progress.update_eta()
                
                # Notify callbacks
                for callback in self._progress_callbacks:
                    try:
                        callback(self._current_progress)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
    
    def migrate_experiment_data(self, 
                               source_experiment_id: int,
                               target_experiment_id: int = None,
                               strategy: MigrationStrategy = MigrationStrategy.BALANCED,
                               create_snapshot: bool = True,
                               batch_size: int = 1000,
                               transformation_rules: Dict[str, Callable] = None) -> MigrationProgress:
        """Migrate experiment data with full hierarchy preservation.
        
        Args:
            source_experiment_id: ID of source experiment
            target_experiment_id: ID of target experiment (creates new if None)
            strategy: Migration strategy
            create_snapshot: Whether to create a snapshot before migration
            batch_size: Batch size for processing large datasets
            transformation_rules: Optional transformation rules for data
            
        Returns:
            MigrationProgress object with final status
        """
        # Initialize progress tracking
        self._current_progress = MigrationProgress(
            total_items=0,
            processed_items=0,
            failed_items=0,
            start_time=datetime.now(),
            current_operation="Initializing migration"
        )
        
        try:
            # Create snapshot if requested
            if create_snapshot:
                self._update_progress(current_operation="Creating pre-migration snapshot")
                snapshot = self.snapshot_manager.create_snapshot(
                    f"Pre-migration snapshot for experiment {source_experiment_id}"
                )
                logger.info(f"Created snapshot: {snapshot.snapshot_id}")
            
            # Validate source experiment
            self._update_progress(current_operation="Validating source experiment")
            is_valid, issues = self.hierarchy_preserver.validate_hierarchy_integrity(source_experiment_id)
            if not is_valid and strategy == MigrationStrategy.CONSERVATIVE:
                raise DataMigrationError(f"Source experiment validation failed: {issues}")
            elif issues and strategy != MigrationStrategy.AGGRESSIVE:
                logger.warning(f"Source experiment has issues: {issues}")
            
            # Get source hierarchy
            self._update_progress(current_operation="Loading source experiment hierarchy")
            source_hierarchy = self.hierarchy_preserver.get_experiment_hierarchy(source_experiment_id)
            
            # Calculate total items for progress tracking
            total_items = 1  # experiment
            total_items += len(source_hierarchy["trials"])
            for trial in source_hierarchy["trials"]:
                total_items += len(trial["runs"])
                for run in trial["runs"]:
                    total_items += len(run["epochs"])
                    if run["results"]:
                        total_items += 1
            
            self._update_progress(total_items=total_items, current_operation="Starting data migration")
            
            # Create or use target experiment
            if target_experiment_id is None:
                exp_data = source_hierarchy["experiment"]
                target_experiment = self.db_manager.create_experiment(
                    title=f"Migrated_{exp_data['title']}",
                    description=f"Migrated from experiment {source_experiment_id}: {exp_data.get('desc', '')}"
                )
                target_experiment_id = target_experiment.id
                logger.info(f"Created target experiment {target_experiment_id}")
            
            # Process migration in batches
            processed_items = 0
            failed_items = 0
            
            # Migrate experiment-level data
            self._update_progress(
                processed_items=processed_items + 1,
                current_operation=f"Migrating experiment {source_experiment_id}"
            )
            processed_items += 1
            
            # Migrate trials
            for trial_data in source_hierarchy["trials"]:
                try:
                    self._migrate_trial_data(
                        trial_data, target_experiment_id, transformation_rules
                    )
                    processed_items += 1
                    self._update_progress(
                        processed_items=processed_items,
                        current_operation=f"Migrated trial {trial_data['trial']['name']}"
                    )
                except Exception as e:
                    failed_items += 1
                    error_msg = f"Failed to migrate trial {trial_data['trial']['id']}: {e}"
                    logger.error(error_msg)
                    self._current_progress.errors.append(error_msg)
                    
                    if strategy == MigrationStrategy.CONSERVATIVE:
                        raise DataMigrationError(error_msg) from e
            
            # Final validation
            if strategy != MigrationStrategy.AGGRESSIVE:
                self._update_progress(current_operation="Performing final validation")
                validation_results = self.validator.validate_data_consistency()
                if validation_results["summary"]["overall_status"] == "FAIL":
                    error_msg = f"Post-migration validation failed: {validation_results['summary']}"
                    if strategy == MigrationStrategy.CONSERVATIVE:
                        raise DataMigrationError(error_msg)
                    else:
                        logger.warning(error_msg)
            
            self._update_progress(
                processed_items=processed_items,
                failed_items=failed_items,
                current_operation="Migration completed successfully"
            )
            
            logger.info(f"Migration completed: {processed_items} items processed, {failed_items} failed")
            return self._current_progress
        
        except Exception as e:
            if self._current_progress:
                self._current_progress.errors.append(str(e))
                self._update_progress(current_operation=f"Migration failed: {e}")
            raise
    
    def _migrate_trial_data(self, trial_data: Dict, target_experiment_id: int, 
                           transformation_rules: Dict[str, Callable] = None):
        """Migrate a single trial and all its data.
        
        Args:
            trial_data: Trial hierarchy data
            target_experiment_id: Target experiment ID
            transformation_rules: Optional transformation rules
        """
        # Create trial
        trial_info = trial_data["trial"]
        new_trial = self.db_manager.create_trial(
            experiment_id=target_experiment_id,
            name=trial_info["name"]
        )
        
        # Migrate trial runs
        for run_data in trial_data["runs"]:
            try:
                self._migrate_trial_run_data(run_data, new_trial.id, transformation_rules)
            except Exception as e:
                logger.error(f"Failed to migrate trial run {run_data['run']['id']}: {e}")
                raise
    
    def _migrate_trial_run_data(self, run_data: Dict, target_trial_id: int,
                               transformation_rules: Dict[str, Callable] = None):
        """Migrate a single trial run and all its data.
        
        Args:
            run_data: Trial run hierarchy data
            target_trial_id: Target trial ID
            transformation_rules: Optional transformation rules
        """
        # Create trial run
        run_info = run_data["run"]
        new_trial_run = self.db_manager.create_trial_run(
            trial_id=target_trial_id,
            status=run_info["status"]
        )
        
        # Migrate epochs
        for epoch_data in run_data["epochs"]:
            self.db_manager.create_epoch(
                epoch_idx=epoch_data["idx"],
                trial_run_id=new_trial_run.id
            )
        
        # Migrate results if present
        if run_data["results"]:
            # Create results entry
            ph = self.db_manager._get_placeholder()
            query = f"""
            INSERT INTO RESULTS (trial_run_id, time)
            VALUES ({ph}, {ph})
            """
            self.db_manager._execute_query(
                query, 
                (new_trial_run.id, run_data["results"]["time"])
            )
            self.db_manager.connection.commit()
    
    def batch_transform_metrics(self,
                               experiment_ids: List[int] = None,
                               transformation_rules: Dict[str, Callable] = None,
                               batch_size: int = 1000,
                               create_snapshot: bool = True) -> MigrationProgress:
        """Batch transform metrics across experiments.
        
        Args:
            experiment_ids: List of experiment IDs to process (all if None)
            transformation_rules: Transformation rules for metrics
            batch_size: Batch size for processing
            create_snapshot: Whether to create snapshot before transformation
            
        Returns:
            MigrationProgress object with final status
        """
        if transformation_rules is None:
            transformation_rules = {}
        
        # Create snapshot if requested
        if create_snapshot:
            snapshot = self.snapshot_manager.create_snapshot(
                "Pre-metric-transformation snapshot"
            )
            logger.info(f"Created snapshot: {snapshot.snapshot_id}")
        
        # Get metrics to transform
        if experiment_ids:
            exp_placeholders = ", ".join([self.db_manager._get_placeholder() for _ in experiment_ids])
            metrics_query = f"""
            SELECT DISTINCT m.* FROM METRIC m
            JOIN EPOCH_METRIC em ON em.metric_id = m.id
            JOIN EPOCH e ON e.idx = em.epoch_idx AND e.trial_run_id = em.epoch_trial_run_id
            JOIN TRIAL_RUN tr ON tr.id = e.trial_run_id
            JOIN TRIAL t ON t.id = tr.trial_id
            WHERE t.experiment_id IN ({exp_placeholders})
            AND m.per_label_val IS NOT NULL
            """
            cursor = self.db_manager._execute_query(metrics_query, experiment_ids)
        else:
            cursor = self.db_manager._execute_query(
                "SELECT * FROM METRIC WHERE per_label_val IS NOT NULL"
            )
        
        metrics = cursor.fetchall()
        
        # Initialize progress
        self._current_progress = MigrationProgress(
            total_items=len(metrics),
            processed_items=0,
            failed_items=0,
            start_time=datetime.now(),
            current_operation="Starting metric transformation"
        )
        
        processed = 0
        failed = 0
        
        # Process metrics in batches
        for i in range(0, len(metrics), batch_size):
            batch = metrics[i:i + batch_size]
            
            for metric in batch:
                try:
                    if metric["per_label_val"]:
                        # Parse JSON
                        per_label_data = json.loads(metric["per_label_val"])
                        
                        # Apply transformations
                        if transformation_rules:
                            transformed_data = MetricTransformer.transform_metric_format(
                                per_label_data, transformation_rules
                            )
                        else:
                            transformed_data = per_label_data
                        
                        # Update metric
                        ph = self.db_manager._get_placeholder()
                        update_query = f"UPDATE METRIC SET per_label_val = {ph} WHERE id = {ph}"
                        self.db_manager._execute_query(
                            update_query, 
                            (json.dumps(transformed_data), metric["id"])
                        )
                    
                    processed += 1
                    
                except Exception as e:
                    failed += 1
                    error_msg = f"Failed to transform metric {metric['id']}: {e}"
                    logger.error(error_msg)
                    self._current_progress.errors.append(error_msg)
            
            # Commit batch
            self.db_manager.connection.commit()
            
            # Update progress
            self._update_progress(
                processed_items=processed,
                failed_items=failed,
                current_operation=f"Processed {processed}/{len(metrics)} metrics"
            )
        
        self._update_progress(current_operation="Metric transformation completed")
        logger.info(f"Metric transformation completed: {processed} processed, {failed} failed")
        
        return self._current_progress
    
    def get_migration_status(self) -> Optional[MigrationProgress]:
        """Get current migration progress.
        
        Returns:
            Current MigrationProgress or None if no migration in progress
        """
        return self._current_progress 