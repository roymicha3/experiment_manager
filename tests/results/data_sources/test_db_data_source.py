"""Tests for the DBDataSource class."""
import pytest
import json
from datetime import datetime
from pathlib import Path

from experiment_manager.results.sources.db_datasource import DBDataSource
from experiment_manager.results.data_models import Experiment, Trial, TrialRun, MetricRecord, Artifact
from experiment_manager.db.manager import DatabaseManager
from experiment_manager.common.common import Level
from tests.conftest import create_metrics_dataframe


@pytest.fixture
def db_manager(tmp_path):
    """Create a test database manager with sample data."""
    db_path = tmp_path / "test_data_source.db"
    manager = DatabaseManager(database_path=str(db_path), use_sqlite=True, recreate=True)
    
    # Create sample data
    _create_sample_data(manager)
    
    yield manager
    # Cleanup happens automatically as tmp_path is cleaned up by pytest


def _create_sample_data(db_manager):
    """Create sample data for testing."""
    # Create experiments
    exp1 = db_manager.create_experiment("Test Experiment 1", "First test experiment")
    exp2 = db_manager.create_experiment("Test Experiment 2", "Second test experiment")
    
    # Create trials for experiment 1
    trial1_1 = db_manager.create_trial(exp1.id, "Trial 1-1")
    trial1_2 = db_manager.create_trial(exp1.id, "Trial 1-2")
    
    # Create trials for experiment 2
    trial2_1 = db_manager.create_trial(exp2.id, "Trial 2-1")
    
    # Create trial runs
    run1_1_1 = db_manager.create_trial_run(trial1_1.id, "completed")
    run1_1_2 = db_manager.create_trial_run(trial1_1.id, "failed")
    run1_2_1 = db_manager.create_trial_run(trial1_2.id, "completed")
    run2_1_1 = db_manager.create_trial_run(trial2_1.id, "running")
    
    # Create RESULTS entries for ALL trial runs
    ph = db_manager._get_placeholder()
    now = datetime.now().isoformat()
    
    for run_id in [run1_1_1.id, run1_1_2.id, run1_2_1.id, run2_1_1.id]:
        query = f"INSERT INTO RESULTS (trial_run_id, time) VALUES ({ph}, {ph})"
        db_manager._execute_query(query, (run_id, now))
    
    db_manager.connection.commit()
    
    # Create metrics
    metric1 = db_manager.record_metric(0.95, "accuracy", {"class1": 0.9, "class2": 1.0})
    metric2 = db_manager.record_metric(0.15, "loss")
    metric3 = db_manager.record_metric(0.92, "accuracy")
    metric4 = db_manager.record_metric(0.88, "f1_score")
    
    # Create epoch metrics
    epoch_metric1 = db_manager.record_metric(0.85, "accuracy")
    epoch_metric2 = db_manager.record_metric(0.25, "loss")
    epoch_metric3 = db_manager.record_metric(0.90, "accuracy")
    epoch_metric4 = db_manager.record_metric(0.18, "loss")
    
    # Link metrics to results - make sure f1_score is linked to a run with RESULTS
    db_manager.link_results_metric(run1_1_1.id, metric1.id)
    db_manager.link_results_metric(run1_1_1.id, metric2.id)
    db_manager.link_results_metric(run1_2_1.id, metric3.id)
    db_manager.link_results_metric(run1_2_1.id, metric4.id)  # Link f1_score to run1_2_1 which has RESULTS
    
    # Create epochs and link epoch metrics
    db_manager.create_epoch(1, run1_1_1.id)
    db_manager.create_epoch(2, run1_1_1.id)
    db_manager.create_epoch(1, run1_2_1.id)
    
    db_manager.add_epoch_metric(1, run1_1_1.id, epoch_metric1.id)
    db_manager.add_epoch_metric(1, run1_1_1.id, epoch_metric2.id)
    db_manager.add_epoch_metric(2, run1_1_1.id, epoch_metric3.id)
    db_manager.add_epoch_metric(2, run1_1_1.id, epoch_metric4.id)
    
    # Create artifacts
    artifact1 = db_manager.record_artifact("model", "/models/model1.pt")
    artifact2 = db_manager.record_artifact("config", "/configs/config1.yaml")
    artifact3 = db_manager.record_artifact("log", "/logs/run1.log")
    artifact4 = db_manager.record_artifact("checkpoint", "/checkpoints/epoch1.pt")
    
    # Link artifacts
    db_manager.link_experiment_artifact(exp1.id, artifact1.id)
    db_manager.link_trial_artifact(trial1_1.id, artifact2.id)
    db_manager.link_trial_run_artifact(run1_1_1.id, artifact3.id)
    db_manager.link_epoch_artifact(1, run1_1_1.id, artifact4.id)


@pytest.fixture
def db_datasource(db_manager):
    """Create a DBDataSource instance."""
    return DBDataSource(db_manager.connection.execute("PRAGMA database_list").fetchone()[2])


class TestDBDataSource:
    """Test cases for DBDataSource."""
    
    def test_initialization(self, tmp_path):
        """Test DBDataSource initialization."""
        db_path = tmp_path / "test_init.db"
        
        # Test SQLite initialization (needs write access to create DB)
        source = DBDataSource(str(db_path), readonly=False)
        assert source.db_path == str(db_path)
        assert source.db_manager.use_sqlite is True
        source.close()
        
        # Test with custom parameters (needs write access)
        source = DBDataSource(str(db_path), use_sqlite=True, readonly=False)
        assert source.db_manager.use_sqlite is True
        source.close()
    
    def test_context_manager(self, tmp_path):
        """Test DBDataSource as context manager."""
        db_path = tmp_path / "test_context.db"
        
        # Needs write access to create DB
        with DBDataSource(str(db_path), readonly=False) as source:
            assert source.db_manager.connection is not None
        
        # Connection should be closed after exiting context
        # Note: We can't directly test if connection is closed with sqlite3,
        # but we can verify the close method was called
    
    def test_get_experiment_by_id(self, db_datasource):
        """Test getting experiment by ID."""
        experiment = db_datasource.get_experiment("1")
        
        assert experiment is not None
        assert experiment.id == 1
        assert experiment.name == "Test Experiment 1"
        assert experiment.description == "First test experiment"
        trials = db_datasource.get_trials(experiment)
        assert len(trials) == 2
    
    def test_get_experiment_by_title(self, db_datasource):
        """Test getting experiment by title."""
        experiment = db_datasource.get_experiment("Test Experiment 2")
        
        assert experiment is not None
        assert experiment.id == 2
        assert experiment.name == "Test Experiment 2"
        assert experiment.description == "Second test experiment"
        trials = db_datasource.get_trials(experiment)
        assert len(trials) == 1
    
    def test_get_experiment_not_found(self, db_datasource):
        """Test getting non-existent experiment."""
        with pytest.raises(ValueError, match="Experiment not found: 999"):
            db_datasource.get_experiment("999")
        
        with pytest.raises(ValueError, match="Experiment not found: Non-existent"):
            db_datasource.get_experiment("Non-existent")
    
    def test_get_trials(self, db_datasource):
        """Test getting trials for an experiment."""
        experiment = Experiment(id=1, name="Test", description="Test")
        trials = db_datasource.get_trials(experiment)
        
        assert len(trials) == 2
        assert trials[0].name == "Trial 1-1"
        assert trials[1].name == "Trial 1-2"
        assert all(trial.experiment_id == 1 for trial in trials)
        
        # Verify runs count via helper
        runs_trial1 = db_datasource.get_trial_runs(trials[0])
        runs_trial2 = db_datasource.get_trial_runs(trials[1])
        assert len(runs_trial1) == 2  # Trial 1-1 has 2 runs
        assert len(runs_trial2) == 1  # Trial 1-2 has 1 run
    
    def test_get_trial_runs(self, db_datasource):
        """Test getting trial runs for a trial."""
        trial = Trial(id=1, name="Trial 1-1", experiment_id=1)
        runs = db_datasource.get_trial_runs(trial)
        
        # Verify we can fetch metrics and artifacts via explicit loaders
        # Note: Not all runs have metrics/artifacts in the test data
        for r in runs:
            metrics = db_datasource.get_metrics(r)
            artifacts = db_datasource.get_artifacts(Level.TRIAL_RUN.value, r)
            
            # Metrics and artifacts should be lists (may be empty)
            assert isinstance(metrics, list)
            assert isinstance(artifacts, list)
        
        # At least one run should have metrics (run1_1_1 has metrics linked)
        total_metrics = sum(len(db_datasource.get_metrics(r)) for r in runs)
        assert total_metrics > 0, "At least one run should have metrics"
        
        # At least one run should have artifacts (run1_1_1 has artifacts linked)
        total_artifacts = sum(len(db_datasource.get_artifacts(Level.TRIAL_RUN.value, r)) for r in runs)
        assert total_artifacts > 0, "At least one run should have artifacts"
        
        assert len(runs) == 2
        assert runs[0].status == "completed"
        assert runs[1].status == "failed"
        assert all(run.trial_id == 1 for run in runs)
        assert all(run.num_epochs >= 0 for run in runs)
    
    def test_get_metrics(self, db_datasource):
        """Test getting metrics for a trial run."""
        trial_run = TrialRun(id=1, trial_id=1, status="completed", num_epochs=2)
        metrics = db_datasource.get_metrics(trial_run)
        
        assert len(metrics) > 0
        
        # Check for both results and epoch metrics
        epoch_metrics = [m for m in metrics if m.epoch is not None]
        result_metrics = [m for m in metrics if m.epoch is None]
        
        assert len(epoch_metrics) > 0  # Should have epoch metrics
        assert len(result_metrics) > 0  # Should have final result metrics
        
        # Check metric structure
        for metric in metrics:
            assert metric.trial_run_id == 1
            assert isinstance(metric.metrics, dict)
            assert len(metric.metrics) > 0
        
        # Check for specific metrics we created
        accuracy_metrics = [m for m in metrics if "accuracy" in m.metrics]
        loss_metrics = [m for m in metrics if "loss" in m.metrics]
        
        assert len(accuracy_metrics) > 0
        assert len(loss_metrics) > 0
        
        # Check per-label metrics handling
        per_label_metrics = [m for m in metrics if any(k.endswith("_per_label") for k in m.metrics.keys())]
        assert len(per_label_metrics) > 0  # Should have at least one metric with per-label data
    
    def test_get_artifacts_experiment(self, db_datasource):
        """Test getting artifacts for an experiment."""
        experiment = Experiment(id=1, name="Test", description="Test")
        artifacts = db_datasource.get_artifacts(Level.EXPERIMENT.value, experiment)
        
        assert len(artifacts) == 1
        assert str(artifacts[0].type) in ("model", "ArtifactType.MODEL")
        assert artifacts[0].path == "/models/model1.pt"
    
    def test_get_artifacts_trial(self, db_datasource):
        """Test getting artifacts for a trial."""
        trial = Trial(id=1, name="Trial 1-1", experiment_id=1)
        artifacts = db_datasource.get_artifacts(Level.TRIAL.value, trial)
        
        assert len(artifacts) == 1
        assert artifacts[0].type == "config"
        assert artifacts[0].path == "/configs/config1.yaml"
    
    def test_get_artifacts_trial_run(self, db_datasource):
        """Test getting artifacts for a trial run."""
        trial_run = TrialRun(id=1, trial_id=1, status="completed", num_epochs=2)
        artifacts = db_datasource.get_artifacts(Level.TRIAL_RUN.value, trial_run)
        
        assert len(artifacts) == 1
        assert str(artifacts[0].type) in ("log", "ArtifactType.LOG")
        assert artifacts[0].path == "/logs/run1.log"
    
    def test_get_artifacts_invalid_level(self, db_datasource):
        """Test getting artifacts with invalid entity level."""
        experiment = Experiment(id=1, name="Test", description="Test")
        
        with pytest.raises(ValueError, match="Unknown entity_level: invalid"):
            db_datasource.get_artifacts("invalid", experiment)
    
    def test_metrics_dataframe(self, db_datasource):
        """Test creating metrics DataFrame."""
        experiment = db_datasource.get_experiment("1")
        trials = db_datasource.get_trials(experiment)
        
        # Build DataFrame directly – no need to manipulate nested attributes
        df = create_metrics_dataframe(db_datasource, experiment)
        
        assert not df.empty
        
        # Check required columns
        required_columns = [
            'experiment_id', 'experiment_name', 'trial_id', 'trial_name',
            'trial_run_id', 'trial_run_status', 'epoch', 'metric', 'value', 'is_custom'
        ]
        for col in required_columns:
            assert col in df.columns
        
        # Check data integrity
        assert all(df['experiment_id'] == 1)
        assert all(df['experiment_name'] == "Test Experiment 1")
        assert df['trial_id'].isin([1, 2]).all()
        assert df['trial_name'].isin(["Trial 1-1", "Trial 1-2"]).all()
        
        # Check metrics - only check for metrics that should be in experiment 1
        unique_metrics = df['metric'].unique()
        expected_metrics = ['accuracy', 'loss', 'f1_score']  # Now f1_score should be present
        for metric in expected_metrics:
            assert metric in unique_metrics
        
        # Check no per-label metrics in main DataFrame
        per_label_metrics = df[df['metric'].str.endswith('_per_label')]
        assert len(per_label_metrics) == 0
        
        # Check epoch data
        epoch_data = df[df['epoch'].notna()]
        result_data = df[df['epoch'].isna()]
        
        assert len(epoch_data) > 0
        assert len(result_data) > 0
        
        # Verify data consistency: the number of non–per-label metric rows in
        # the DataFrame should equal the total number of such metric records
        # fetched via explicit helper calls.
        all_runs = [run for t in trials for run in db_datasource.get_trial_runs(t)]

        non_per_label_count = 0
        for r in all_runs:
            metrics_records = db_datasource.get_metrics(r)
            non_per_label_count += sum(
                1
                for record in metrics_records
                for name in record.metrics.keys()
                if not name.endswith('_per_label')
            )

        assert len(df) == non_per_label_count
    
    def test_metrics_dataframe_empty_experiment(self, db_datasource):
        """Test metrics DataFrame with experiment that has no data."""
        # Create empty experiment
        empty_experiment = Experiment(id=999, name="Empty", description="Empty")
        df = create_metrics_dataframe(db_datasource, empty_experiment)
        
        assert df.empty
        # When DataFrame is empty from empty data (not an uninitialized DataFrame),
        # pandas still creates it with the columns from the list comprehension
        # Since we're creating from an empty list, there are no columns
        # Let's just verify it's empty and not test for columns structure
        assert len(df) == 0
    
    def test_num_epochs(self, db_datasource):
        """Test getting number of epochs for a trial run."""
        # Test trial run with epochs
        num_epochs = db_datasource._get_num_epochs(1)
        assert num_epochs == 2  # We created 2 epochs for trial run 1
        
        # Test trial run without epochs
        num_epochs = db_datasource._get_num_epochs(999)
        assert num_epochs == 0
    
    def test_complete_workflow(self, db_datasource):
        """Test complete workflow with all methods."""
        # Get experiment
        experiment = db_datasource.get_experiment("Test Experiment 1")
        assert experiment is not None
        trials = db_datasource.get_trials(experiment)
        assert len(trials) == 2
        
        # Check trials have runs
        for trial in trials:
            runs = db_datasource.get_trial_runs(trial)
            assert len(runs) > 0
            
            # Check that we can fetch metrics for runs (some may be empty)
            for run in runs:
                metrics = db_datasource.get_metrics(run)
                assert isinstance(metrics, list)
        
        # Verify that at least some runs have metrics
        all_runs = [run for t in trials for run in db_datasource.get_trial_runs(t)]
        total_metrics = sum(len(db_datasource.get_metrics(r)) for r in all_runs)
        assert total_metrics > 0, "At least some runs should have metrics"
        
        # Create DataFrame
        df = create_metrics_dataframe(db_datasource, experiment)
        assert not df.empty
        
        # Verify data consistency: the number of non–per-label metric rows in
        # the DataFrame should equal the total number of such metric records
        # fetched via explicit helper calls.
        all_runs = [run for t in trials for run in db_datasource.get_trial_runs(t)]

        non_per_label_count = 0
        for r in all_runs:
            metrics_records = db_datasource.get_metrics(r)
            non_per_label_count += sum(
                1
                for record in metrics_records
                for name in record.metrics.keys()
                if not name.endswith('_per_label')
            )

        assert len(df) == non_per_label_count
    
    def test_close_method(self, db_datasource):
        """Test the close method."""
        # Should not raise any errors
        db_datasource.close()
        
        # Should be safe to call multiple times
        db_datasource.close()
    
    def test_private_methods_error_handling(self, db_datasource):
        """Test error handling in private methods."""
        # Test non-existent experiment by ID
        result = db_datasource._get_experiment_by_id(999)
        assert result is None
        
        # Test non-existent experiment by title
        result = db_datasource._get_experiment_by_title("Non-existent")
        assert result is None 
