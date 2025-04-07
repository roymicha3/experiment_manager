"""Tests for database manager."""
import pytest
import json
from datetime import datetime
from experiment_manager.db.manager import DatabaseManager
from experiment_manager.db.tables import Experiment, Trial, TrialRun, Metric, Artifact

@pytest.fixture
def test_db(tmp_path_factory):
    """Create test database and tables."""
    db_dir = tmp_path_factory.mktemp('db')
    db_path = db_dir / 'test.db'
    yield str(db_path)

@pytest.fixture
def db_manager(test_db):
    """Create database manager instance."""
    return DatabaseManager(database_path=test_db, use_sqlite=True, recreate=True)

def test_create_experiment(db_manager: DatabaseManager):
    """Test creating an experiment."""
    exp = db_manager.create_experiment("Test Experiment", "Test Description")
    assert exp.id is not None
    
    # Verify experiment was created
    db_manager.cursor.execute("SELECT * FROM EXPERIMENT WHERE id = ?", (exp.id,))
    row = db_manager.cursor.fetchone()
    assert row["title"] == "Test Experiment"
    assert row["desc"] == "Test Description"

def test_create_trial(db_manager: DatabaseManager):
    """Test creating a trial."""
    exp = db_manager.create_experiment("Test Experiment")
    trial = db_manager.create_trial(exp.id, "Test Trial")
    assert trial.id is not None
    
    # Verify trial was created
    db_manager.cursor.execute("SELECT * FROM TRIAL WHERE id = ?", (trial.id,))
    row = db_manager.cursor.fetchone()
    assert row["name"] == "Test Trial"
    assert row["experiment_id"] == exp.id

def test_create_trial_run(db_manager: DatabaseManager):
    """Test creating a trial run."""
    exp = db_manager.create_experiment("Test Experiment")
    trial = db_manager.create_trial(exp.id, "Test Trial")
    run = db_manager.create_trial_run(trial.id)
    assert run.id is not None
    
    # Verify trial run was created
    db_manager.cursor.execute("SELECT * FROM TRIAL_RUN WHERE id = ?", (run.id,))
    row = db_manager.cursor.fetchone()
    assert row["trial_id"] == trial.id
    assert row["status"] == "started"

def test_record_metric(db_manager: DatabaseManager):
    """Test recording a metric."""
    metric = db_manager.record_metric(0.95, "accuracy", {"class_1": 0.9, "class_2": 1.0})
    assert metric.id is not None
    
    # Verify metric was created
    db_manager.cursor.execute("SELECT * FROM METRIC WHERE id = ?", (metric.id,))
    row = db_manager.cursor.fetchone()
    assert row["type"] == "accuracy"
    assert row["total_val"] == 0.95
    per_label_val = json.loads(row["per_label_val"])
    assert "class_1" in per_label_val

def test_add_epoch_metric(db_manager: DatabaseManager):
    """Test adding a metric to an epoch."""
    exp = db_manager.create_experiment("Test Experiment")
    trial = db_manager.create_trial(exp.id, "Test Trial")
    run = db_manager.create_trial_run(trial.id)
    metric = db_manager.record_metric(0.95, "accuracy")
    
    # Create epoch
    db_manager.cursor.execute(
        "INSERT INTO EPOCH (idx, trial_run_id, time) VALUES (?, ?, ?)",
        (1, run.id, datetime.now().isoformat())
    )
    db_manager.connection.commit()
    
    # Add metric to epoch
    db_manager.add_epoch_metric(1, run.id, metric.id)
    
    # Verify association was created
    db_manager.cursor.execute(
        "SELECT * FROM EPOCH_METRIC WHERE epoch_idx = ? AND epoch_trial_run_id = ?",
        (1, run.id)
    )
    assoc = db_manager.cursor.fetchone()
    assert assoc["metric_id"] == metric.id

def test_record_artifact(db_manager: DatabaseManager):
    """Test recording an artifact."""
    artifact = db_manager.record_artifact("model", "/path/to/model.pt")
    assert artifact.id is not None
    
    # Verify artifact was created
    db_manager.cursor.execute("SELECT * FROM ARTIFACT WHERE id = ?", (artifact.id,))
    row = db_manager.cursor.fetchone()
    assert row["type"] == "model"
    assert row["loc"] == "/path/to/model.pt"

def test_get_experiment_metrics(db_manager: DatabaseManager):
    """Test getting experiment metrics."""
    exp = db_manager.create_experiment("Test Experiment")
    trial = db_manager.create_trial(exp.id, "Test Trial")
    run = db_manager.create_trial_run(trial.id)
    metric = db_manager.record_metric(0.95, "accuracy")
    
    # Create epoch
    db_manager.cursor.execute(
        "INSERT INTO EPOCH (idx, trial_run_id, time) VALUES (?, ?, ?)",
        (1, run.id, datetime.now().isoformat())
    )
    db_manager.connection.commit()
    
    # Add metric to epoch
    db_manager.add_epoch_metric(1, run.id, metric.id)
    
    # Get metrics
    metrics = db_manager.get_experiment_metrics(exp.id)
    assert len(metrics) == 1
    assert metrics[0].id == metric.id
    assert metrics[0].type == "accuracy"
    assert metrics[0].total_val == 0.95

def test_get_trial_artifacts(db_manager: DatabaseManager):
    """Test getting trial artifacts."""
    exp = db_manager.create_experiment("Test Experiment")
    trial = db_manager.create_trial(exp.id, "Test Trial")
    artifact = db_manager.record_artifact("model", "/path/to/model.pt")
    
    # Associate artifact with trial
    db_manager.cursor.execute(
        "INSERT INTO TRIAL_ARTIFACT (trial_id, artifact_id) VALUES (?, ?)",
        (trial.id, artifact.id)
    )
    db_manager.connection.commit()
    
    # Get artifacts
    artifacts = db_manager.get_trial_artifacts(trial.id)
    assert len(artifacts) == 1
    assert artifacts[0].id == artifact.id
    assert artifacts[0].type == "model"
    assert artifacts[0].location == "/path/to/model.pt"
