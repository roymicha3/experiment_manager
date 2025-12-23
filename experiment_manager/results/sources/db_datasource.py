import json
from typing import List, Union, Optional
import pandas as pd
from omegaconf import DictConfig

from experiment_manager.db import tables
from experiment_manager.db.manager import DatabaseManager
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.results.sources.datasource import ExperimentDataSource
from experiment_manager.results.data_models import Experiment, Trial, TrialRun, MetricRecord, Artifact
from experiment_manager.common.common import ArtifactType, RunStatus, Level

@YAMLSerializable.register("DBDataSource")
class DBDataSource(ExperimentDataSource, YAMLSerializable):
    def __init__(self, 
                 db_path: str, 
                 use_sqlite: bool = True, 
                 host: str = "localhost", 
                 user: str = "root", 
                 password: str = "", 
                 config: DictConfig = None, 
                 readonly: bool = True):
        
        ExperimentDataSource.__init__(self)
        YAMLSerializable.__init__(self, config)
        
        self.db_path = db_path
        self.db_manager = DatabaseManager(
            database_path=db_path,
            use_sqlite=use_sqlite,
            host=host,
            user=user,
            password=password,
            readonly=readonly
        )
        self.readonly = readonly
        
    @classmethod
    def from_config(cls, config: DictConfig):
        # Validate configuration before attempting instantiation
        cls._validate_config(config)

        return cls(
            db_path=config.db_path,
            use_sqlite=config.get('use_sqlite', True),
            host=config.get('host', 'localhost'),
            user=config.get('user', 'root'),
            password=config.get('password', ''),
            config=config,
            readonly=config.get('readonly', True)
        )

    @staticmethod
    def _validate_config(config: DictConfig):
        """Validate YAML/DictConfig for required fields and correct types."""
        # Required field: db_path must be a string
        if not hasattr(config, "db_path"):
            raise AttributeError("'db_path' is a required configuration field for DBDataSource")
        if not isinstance(config.db_path, str):
            raise TypeError("'db_path' must be a string path")

        # Optional bool field use_sqlite
        if "use_sqlite" in config and not isinstance(config.use_sqlite, bool):
            raise TypeError("'use_sqlite' must be a boolean")

        # Optional string fields host, user, password
        for field in ("host", "user", "password"):
            if field in config and not isinstance(config.get(field), str):
                raise TypeError(f"'{field}' must be a string")
    

    def get_experiment(self, experiment_id: Optional[Union[str, int]] = None) -> Experiment:
        """
        Fetch the experiment and all associated trials and runs.
        """
        # Get all experiments from the database
        experiments_tables = self._get_experiments()
        
        if not experiments_tables:
            raise ValueError("No experiments found in database")
        
        # Find the specific experiment if ID provided
        if experiment_id is not None:
            experiment_table = self._find_experiment(experiments_tables, experiment_id)
            if experiment_table is None:
                raise ValueError(f"Experiment not found: {experiment_id}")
        else:
            # Return first experiment if no ID specified
            experiment_table = experiments_tables[0]
        
        experiment = Experiment(
            id=experiment_table.id,
            name=experiment_table.title,
            description=experiment_table.description,
            dir_path=None
        )

        return experiment


    def _find_experiment(self, experiments: List[tables.Experiment], 
                        experiment_id: Union[str, int]) -> Optional[tables.Experiment]:
        """Find experiment by ID (int) or title (str)."""
        for exp in experiments:
            # Try to match by ID (if experiment_id is numeric)
            try:
                if int(experiment_id) == exp.id:
                    return exp
            except (ValueError, TypeError):
                pass
            
            # Try to match by title (if experiment_id is string)
            if str(experiment_id) == exp.title:
                return exp
        
        return None

    def _get_experiments(self) -> List[tables.Experiment]:
        """Get all experiments from the database."""
        desc_field = "desc" if self.db_manager.use_sqlite else "`desc`"
        
        query = f"""
        SELECT id, title, {desc_field} as description, start_time, update_time
        FROM EXPERIMENT
        ORDER BY id
        """
        cursor = self.db_manager._execute_query(query)
        rows = cursor.fetchall()
        return [tables.Experiment(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            start_time=row["start_time"],
            update_time=row["update_time"]
        ) for row in rows]

    def _get_experiment_by_id(self, experiment_id: int) -> Optional[tables.Experiment]:
        """Get experiment by ID."""
        experiments = self._get_experiments()
        return next((exp for exp in experiments if exp.id == experiment_id), None)

    def _get_experiment_by_title(self, title: str) -> Optional[tables.Experiment]:
        """Get experiment by title."""
        experiments = self._get_experiments()
        return next((exp for exp in experiments if exp.title == title), None)

    def get_trials(self, experiment: Experiment) -> List[Trial]:
        """List all trials for an experiment."""
        ph = self.db_manager._get_placeholder()
        
        query = f"""
        SELECT id, name, experiment_id, start_time, update_time
        FROM TRIAL 
        WHERE experiment_id = {ph}
        ORDER BY id
        """
        
        cursor = self.db_manager._execute_query(query, (experiment.id,))
        rows = cursor.fetchall()
        
        trials = []
        for row in rows:
            trial = Trial(
                id=row["id"],
                name=row["name"],
                experiment_id=row["experiment_id"],
                dir_path=None
            )
            
            trials.append(trial)
        
        return trials

    def get_trial_runs(self, trial: Trial) -> List[TrialRun]:
        """List all runs for a trial."""
        ph = self.db_manager._get_placeholder()
        
        query = f"""
        SELECT id, trial_id, status, start_time, update_time
        FROM TRIAL_RUN 
        WHERE trial_id = {ph}
        ORDER BY id
        """
        
        cursor = self.db_manager._execute_query(query, (trial.id,))
        rows = cursor.fetchall()
        
        trial_runs = []
        for row in rows:
            trial_run = TrialRun(
                id=row["id"],
                trial_id=row["trial_id"],
                status=RunStatus(row["status"]) if isinstance(row["status"], int) else row["status"],
                num_epochs=self._get_num_epochs(row["id"]),
                dir_path=None
            )

            trial_runs.append(trial_run)
        
        return trial_runs

    def _get_num_epochs(self, trial_run_id: int) -> int:
        """Get the number of epochs for a trial run."""
        ph = self.db_manager._get_placeholder()
        
        query = f"""
        SELECT COUNT(DISTINCT idx) as epoch_count
        FROM EPOCH 
        WHERE trial_run_id = {ph}
        """
        
        cursor = self.db_manager._execute_query(query, (trial_run_id,))
        row = cursor.fetchone()
        
        return row["epoch_count"] if row else 0

    def get_metrics(self, trial_run: TrialRun) -> List[MetricRecord]:
        """Fetch all metrics for a run (across all epochs)."""
        ph = self.db_manager._get_placeholder()
        
        # Get metrics from both RESULTS and EPOCH tables
        results_query = f"""
        SELECT NULL as epoch, m.type, m.total_val, m.per_label_val
        FROM RESULTS r
        JOIN RESULTS_METRIC rm ON rm.results_id = r.trial_run_id
        JOIN METRIC m ON m.id = rm.metric_id
        WHERE r.trial_run_id = {ph}
        """
        
        epoch_query = f"""
        SELECT e.idx as epoch, m.type, m.total_val, m.per_label_val
        FROM EPOCH e
        JOIN EPOCH_METRIC em ON em.epoch_idx = e.idx AND em.epoch_trial_run_id = e.trial_run_id
        JOIN METRIC m ON m.id = em.metric_id
        WHERE e.trial_run_id = {ph}
        ORDER BY e.idx
        """
        
        metrics = []
        
        # Get final results metrics
        cursor = self.db_manager._execute_query(results_query, (trial_run.id,))
        for row in cursor.fetchall():
            metric_dict = {row["type"]: row["total_val"]}
            
            # Add per-label values if they exist
            if row["per_label_val"]:
                try:
                    per_label = json.loads(row["per_label_val"]) if isinstance(row["per_label_val"], str) else row["per_label_val"]
                    metric_dict[f"{row['type']}_per_label"] = per_label
                except (json.JSONDecodeError, TypeError):
                    pass
            
            metrics.append(MetricRecord(
                trial_run_id=trial_run.id,
                epoch=row["epoch"],
                metrics=metric_dict,
                is_custom=False,
                granularity='results'
            ))
        
        # Get epoch-level metrics
        cursor = self.db_manager._execute_query(epoch_query, (trial_run.id,))
        for row in cursor.fetchall():
            metric_dict = {row["type"]: row["total_val"]}
            
            # Add per-label values if they exist
            if row["per_label_val"] is not None:
                try:
                    per_label = json.loads(row["per_label_val"]) if isinstance(row["per_label_val"], str) else row["per_label_val"]
                    metric_dict[f"{row['type']}_per_label"] = per_label
                except (json.JSONDecodeError, TypeError):
                    pass
            
            metrics.append(MetricRecord(
                trial_run_id=trial_run.id,
                epoch=row["epoch"],
                metrics=metric_dict,
                is_custom=False,
                granularity='epoch'
            ))
        
        return metrics

    def get_batch_metrics(self, trial_run: TrialRun) -> List[MetricRecord]:
        """Fetch all batch-level metrics for a trial run."""
        ph = self.db_manager._get_placeholder()
        
        batch_query = f"""
        SELECT b.idx as batch, b.epoch_idx as epoch, b.time, 
               m.type, m.total_val, m.per_label_val
        FROM BATCH b
        JOIN BATCH_METRIC bm ON bm.batch_idx = b.idx 
                            AND bm.epoch_idx = b.epoch_idx 
                            AND bm.trial_run_id = b.trial_run_id
        JOIN METRIC m ON m.id = bm.metric_id
        WHERE b.trial_run_id = {ph}
        ORDER BY b.epoch_idx, b.idx
        """
        
        metrics = []
        cursor = self.db_manager._execute_query(batch_query, (trial_run.id,))
        
        for row in cursor.fetchall():
            metric_dict = {row["type"]: row["total_val"]}
            
            # Add per-label values if they exist
            if row["per_label_val"] is not None:
                try:
                    per_label = json.loads(row["per_label_val"]) if isinstance(row["per_label_val"], str) else row["per_label_val"]
                    metric_dict[f"{row['type']}_per_label"] = per_label
                except (json.JSONDecodeError, TypeError):
                    pass
            
            metrics.append(MetricRecord(
                trial_run_id=trial_run.id,
                epoch=row["epoch"],
                batch=row["batch"],
                metrics=metric_dict,
                is_custom=False,
                timestamp=row["time"] if "time" in row.keys() else None,
                granularity='batch'
            ))
        
        return metrics

    def get_artifacts(self, entity_level: Level, entity: Union[Experiment, Trial, TrialRun]) -> List[Artifact]:
        """
        Fetch artifacts attached to experiment, trial, or trial_run.
        entity_level: "experiment", "trial", "trial_run"
        """
        ph = self.db_manager._get_placeholder()
        
        if entity_level == Level.EXPERIMENT.value:
            junction_table = "EXPERIMENT_ARTIFACT"
            id_field = "experiment_id"
            entity_id = entity.id
        elif entity_level == Level.TRIAL.value:
            junction_table = "TRIAL_ARTIFACT"
            id_field = "trial_id"
            entity_id = entity.id
        elif entity_level == Level.TRIAL_RUN.value:
            junction_table = "TRIAL_RUN_ARTIFACT"
            id_field = "trial_run_id"
            entity_id = entity.id
        else:
            raise ValueError(f"Unknown entity_level: {entity_level}")
        
        query = f"""
        SELECT a.id, a.type, a.loc
        FROM ARTIFACT a
        JOIN {junction_table} ja ON ja.artifact_id = a.id
        WHERE ja.{id_field} = {ph}
        """
        
        cursor = self.db_manager._execute_query(query, (entity_id,))
        rows = cursor.fetchall()
        
        return [
            Artifact(
                id=row["id"],
                type=row["type"],
                path=row["loc"]
            )
            for row in rows
        ]

    def metrics_dataframe(self, experiment: Experiment) -> pd.DataFrame:
        """Return a flattened pandas DataFrame of all metrics for the given experiment.

        The returned DataFrame follows the structure expected by the test helper
        `tests.conftest.create_metrics_dataframe` so that external callers can
        rely on a stable schema:

        ```
        experiment_id | experiment_name | trial_id | trial_name | trial_run_id | trial_run_status | epoch | metric | value | is_custom
        ```

        Args:
            experiment: Experiment object retrieved via ``get_experiment``.

        Returns:
            pd.DataFrame containing one row per metric value.
        """
        data: list[dict] = []

        # Fetch trials explicitly to avoid relying on nested structures that may
        # change over time. This mirrors the logic in the test helper
        # ``create_metrics_dataframe``.
        trials = self.get_trials(experiment)

        for trial in trials:
            trial_runs = self.get_trial_runs(trial)
            for run in trial_runs:
                metrics_records = self.get_metrics(run)
                for record in metrics_records:
                    for metric_name, metric_value in record.metrics.items():
                        # Skip per-label metrics in the flattened view – callers
                        # can access them directly from the underlying record
                        if metric_name.endswith("_per_label"):
                            continue

                        data.append({
                            "experiment_id": experiment.id,
                            "experiment_name": experiment.name,
                            "trial_id": trial.id,
                            "trial_name": trial.name,
                            "trial_run_id": run.id,
                            "trial_run_status": run.status,
                            "epoch": record.epoch,
                            "metric": metric_name,
                            "value": metric_value,
                            "is_custom": record.is_custom,
                        })

        return pd.DataFrame(data)

    def close(self):
        """Close database connection."""
        if hasattr(self.db_manager, 'connection') and self.db_manager.connection:
            self.db_manager.connection.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()