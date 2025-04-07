"""Tests for the DatabaseManager class."""
import pytest
import sqlite3
from pathlib import Path
from datetime import datetime

from experiment_manager.db.manager import DatabaseManager, QueryError
from experiment_manager.db.tables import Experiment, Trial, TrialRun, Metric, Artifact

@pytest.fixture
def db_manager(tmp_path):
    """Create a test database manager."""
    db_path = tmp_path / "test_experiment_manager.db"
    manager = DatabaseManager(database_path=str(db_path), use_sqlite=True, recreate=True)
    yield manager
    # Cleanup happens automatically as tmp_path is cleaned up by pytest

def test_database_structure(db_manager):
    """Test that the database has the correct structure."""
    tables = [
        "EXPERIMENT", "TRIAL", "TRIAL_RUN", "RESULTS", "EPOCH",
        "METRIC", "ARTIFACT", "EXPERIMENT_ARTIFACT", "TRIAL_ARTIFACT",
        "RESULTS_METRIC", "RESULTS_ARTIFACT", "EPOCH_METRIC",
        "EPOCH_ARTIFACT", "TRIAL_RUN_ARTIFACT"
    ]
    
    cursor = db_manager._execute_query("SELECT name FROM sqlite_master WHERE type='table'")
    db_tables = [row["name"] for row in cursor.fetchall()]
    
    for table in tables:
        assert table in db_tables, f"Table {table} not found in database"

def test_experiment_creation(db_manager):
    """Test creating an experiment."""
    experiment = db_manager.create_experiment("Test Experiment", "Test Description")
    assert experiment.title == "Test Experiment"
    assert experiment.description == "Test Description"
    assert isinstance(experiment.start_time, datetime)
    assert isinstance(experiment.update_time, datetime)

def test_trial_creation(db_manager):
    """Test creating a trial."""
    experiment = db_manager.create_experiment("Test Experiment")
    trial = db_manager.create_trial(experiment.id, "Test Trial")
    assert trial.name == "Test Trial"
    assert trial.experiment_id == experiment.id

def test_trial_run_creation(db_manager):
    """Test creating a trial run."""
    experiment = db_manager.create_experiment("Test Experiment")
    trial = db_manager.create_trial(experiment.id, "Test Trial")
    trial_run = db_manager.create_trial_run(trial.id)
    assert trial_run.trial_id == trial.id
    assert trial_run.status == "started"

def test_metric_recording(db_manager):
    """Test recording a metric."""
    metric = db_manager.record_metric(0.95, "accuracy", {"class1": 0.9, "class2": 1.0})
    assert metric.type == "accuracy"
    assert metric.total_val == 0.95
    assert metric.per_label_val == {"class1": 0.9, "class2": 1.0}

def test_artifact_recording_and_linking(db_manager):
    """Test recording and linking artifacts."""
    # Create experiment structure
    experiment = db_manager.create_experiment("Test Experiment")
    trial = db_manager.create_trial(experiment.id, "Test Trial")
    trial_run = db_manager.create_trial_run(trial.id)
    
    # Create and link artifact to experiment
    artifact1 = db_manager.record_artifact("model", "model.pt")
    db_manager.link_experiment_artifact(experiment.id, artifact1.id)
    
    # Create and link artifact to trial
    artifact2 = db_manager.record_artifact("config", "config.yaml")
    db_manager.link_trial_artifact(trial.id, artifact2.id)
    
    # Create and link artifact to trial run
    artifact3 = db_manager.record_artifact("log", "run.log")
    db_manager.link_trial_run_artifact(trial_run.id, artifact3.id)
    
    # Create and link artifact to epoch
    artifact4 = db_manager.record_artifact("checkpoint", "epoch_1.pt")
    db_manager.link_epoch_artifact(1, trial_run.id, artifact4.id)
    
    # Verify links
    exp_artifacts = db_manager.get_experiment_artifacts(experiment.id)
    assert len(exp_artifacts) == 1
    assert exp_artifacts[0].type == "model"
    
    trial_artifacts = db_manager.get_trial_artifacts(trial.id)
    assert len(trial_artifacts) == 1
    assert trial_artifacts[0].type == "config"
    
    run_artifacts = db_manager.get_trial_run_artifacts(trial_run.id)
    assert len(run_artifacts) == 1
    assert run_artifacts[0].type == "log"
    
    epoch_artifacts = db_manager.get_epoch_artifacts(1, trial_run.id)
    assert len(epoch_artifacts) == 1
    assert epoch_artifacts[0].type == "checkpoint"

def test_metric_linking(db_manager):
    """Test linking metrics to results."""
    # Create experiment structure
    experiment = db_manager.create_experiment("Test Experiment")
    trial = db_manager.create_trial(experiment.id, "Test Trial")
    trial_run = db_manager.create_trial_run(trial.id)
    
    # Create and link metric
    metric = db_manager.record_metric(0.95, "accuracy")
    db_manager.link_results_metric(trial_run.id, metric.id)
    
    # Create and link epoch metric
    epoch_metric = db_manager.record_metric(0.90, "loss")
    db_manager.create_epoch(1, trial_run.id)
    db_manager.add_epoch_metric(1, trial_run.id, epoch_metric.id)
    
    # Create RESULTS entry for trial_run
    ph = db_manager._get_placeholder()
    query = f"""
    INSERT INTO RESULTS (trial_run_id, time)
    VALUES ({ph}, {ph})
    """
    db_manager._execute_query(query, (trial_run.id, datetime.now().isoformat()))
    db_manager.connection.commit()
    
    # Verify experiment metrics
    exp_metrics = db_manager.get_experiment_metrics(experiment.id)
    assert len(exp_metrics) == 2
    metric_types = {m.type for m in exp_metrics}
    assert "accuracy" in metric_types
    assert "loss" in metric_types

def test_error_handling(db_manager):
    """Test error handling for invalid operations."""
    with pytest.raises(QueryError):
        # Try to create a trial with non-existent experiment
        db_manager.create_trial(999, "Test Trial")
    
    with pytest.raises(QueryError):
        # Try to create a trial run with non-existent trial
        db_manager.create_trial_run(999)
    
    with pytest.raises(QueryError):
        # Try to link artifact to non-existent experiment
        artifact = db_manager.record_artifact("test", "test.txt")
        db_manager.link_experiment_artifact(999, artifact.id)
