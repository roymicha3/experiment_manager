"""Tests for the DBTracker."""
import os
import pytest
from pathlib import Path
from omegaconf import OmegaConf

from experiment_manager.trackers.plugins.db_tracker import DBTracker
from experiment_manager.common.common import Level, Metric
from experiment_manager.results.sources.db_data_source import DBDataSource

@pytest.fixture
def workspace(tmp_path):
    return tmp_path / "test_workspace"

@pytest.fixture
def db_tracker(workspace):
    os.makedirs(workspace, exist_ok=True)
    tracker = DBTracker(str(workspace))
    yield tracker
    if hasattr(tracker, 'db_manager'):
        tracker.db_manager.__del__()

def test_db_reuse(workspace):
    """Test that DBTracker can reuse an existing database."""
    # Create initial tracker and add some data
    tracker1 = DBTracker(str(workspace), recreate=True)
    tracker1.on_create(Level.EXPERIMENT, "Test Experiment 1")
    tracker1.track(Metric.TEST_ACC, 0.95)
    
    # Clean up first tracker
    tracker1.db_manager.__del__()
    
    # Create second tracker with auto-detection (should reuse)
    tracker2 = DBTracker(str(workspace))  # No recreate specified
    tracker2.on_create(Level.EXPERIMENT, "Test Experiment 2")
    
    # Verify both experiments exist in database
    experiments = tracker2.db_manager._execute_query("SELECT title FROM EXPERIMENT").fetchall()
    assert len(experiments) == 2
    titles = [exp["title"] for exp in experiments]
    assert "Test Experiment 1" in titles
    assert "Test Experiment 2" in titles
    
    # Clean up
    tracker1.db_manager.__del__()
    tracker2.db_manager.__del__()

def test_child_tracker_reuse(workspace):
    """Test that child trackers reuse the parent database."""
    parent = DBTracker(str(workspace), recreate=True)
    parent.on_create(Level.EXPERIMENT, "Parent Experiment")
    
    child = parent.create_child()
    child.on_create(Level.TRIAL, "Child Trial")
    
    # Verify parent experiment exists in child's database
    result = child.db_manager._execute_query(
        "SELECT title FROM EXPERIMENT WHERE title = ?",
        ("Parent Experiment",)
    ).fetchone()
    assert result is not None
    assert result["title"] == "Parent Experiment"
    
    # Clean up
    parent.db_manager.__del__()
    child.db_manager.__del__()
    
    # Clean up
    parent.db_manager.__del__()
    child.db_manager.__del__()

def test_from_config_reuse(workspace):
    """Test database reuse through configuration."""
    # Create initial database
    tracker1 = DBTracker(str(workspace), recreate=True)
    tracker1.on_create(Level.EXPERIMENT, "Config Test 1")
    
    # Clean up first tracker
    tracker1.db_manager.__del__()
    
    # Create tracker from config with recreate=False
    config = OmegaConf.create({
        "name": DBTracker.DB_NAME,
        "recreate": False
    })
    tracker2 = DBTracker.from_config(config, str(workspace))
    tracker2.on_create(Level.EXPERIMENT, "Config Test 2")
    
    # Verify both experiments exist
    experiments = tracker2.db_manager._execute_query("SELECT title FROM EXPERIMENT").fetchall()
    assert len(experiments) == 2
    titles = [exp["title"] for exp in experiments]
    assert "Config Test 1" in titles
    assert "Config Test 2" in titles
    
    # Clean up
    tracker1.db_manager.__del__()
    tracker2.db_manager.__del__()

def test_recreate_removes_data(workspace):
    """Test that recreate=True removes existing data."""
    # Create initial data
    tracker1 = DBTracker(str(workspace))
    tracker1.on_create(Level.EXPERIMENT, "Should Be Deleted")
    tracker1.track(Metric.TEST_ACC, 0.95)
    
    # Clean up first tracker
    tracker1.db_manager.__del__()
    
    # Create new tracker with recreate=True
    tracker2 = DBTracker(str(workspace), recreate=True)
    experiments = tracker2.db_manager._execute_query("SELECT title FROM EXPERIMENT").fetchall()
    assert len(experiments) == 0  # Database should be empty
    metrics = tracker2.db_manager._execute_query("SELECT type FROM METRIC").fetchall()
    assert len(metrics) == 0  # Database should be empty
    
    # Clean up
    tracker2.db_manager.__del__()

def test_metric_tracking(workspace):
    """Test metric tracking with different value types."""
    tracker = DBTracker(str(workspace))
    tracker.on_create(Level.EXPERIMENT, "Metric Test")
    
    # Test scalar value
    tracker.track(Metric.TEST_ACC, 0.95, step=0)
    
    # Test dictionary value
    confusion = {"true_positive": 100, "false_positive": 5}
    tracker.track(Metric.CONFUSION, 0.0, 0, per_label_val=confusion)
    
    # Test with step
    tracker.track(Metric.TEST_LOSS, 0.1, 1)
    
    # Verify metrics were recorded
    metrics = tracker.db_manager._execute_query(
        "SELECT type, total_val, per_label_val FROM METRIC"
    ).fetchall()
    
    assert len(metrics) == 3
    metric_map = {m["type"]: m for m in metrics}
    
    assert metric_map["test_acc"]["total_val"] == 0.95
    assert metric_map["test_acc"]["per_label_val"] is None
    
    # compare dictionaries
    assert eval(metric_map["confusion"]["per_label_val"]) == confusion
    
    assert metric_map["test_loss"]["total_val"] == 0.1
    
    # Clean up
    tracker.db_manager.__del__()

def test_db_tracker_integration_with_real_data(experiment_db_only):
    """Test DBTracker integration with real MNIST experiment data.
    
    This test verifies that DBTracker database manager can successfully 
    read and query real experiment data, confirming integration with
    actual experiment database structures.
    """
    # experiment_db_only provides the path to real MNIST experiment database
    real_db_path = experiment_db_only
    
    # Create a database manager directly using the real database path
    from experiment_manager.db.manager import DatabaseManager
    db_manager = DatabaseManager(database_path=real_db_path, use_sqlite=True, recreate=False)
    
    try:
        # Test basic database connectivity and real data presence
        experiments = db_manager._execute_query("SELECT title FROM EXPERIMENT").fetchall()
        
        # Should have the real MNIST experiment
        experiment_titles = [exp["title"] for exp in experiments]
        assert "test_mnist_baseline" in experiment_titles
        assert len(experiment_titles) >= 1
        
        # Test querying real trial data
        trials = db_manager._execute_query("SELECT name FROM TRIAL").fetchall()
        trial_names = {trial["name"] for trial in trials}
        expected_names = {"small_lr", "medium_lr", "large_lr"}
        assert trial_names == expected_names
        assert len(trials) == 3  # Should have 3 trials
        
        # Test querying real trial runs
        trial_runs = db_manager._execute_query("SELECT id, status FROM TRIAL_RUN").fetchall()
        assert len(trial_runs) == 6  # 3 trials × 2 repetitions each
        
        # Verify all trial runs have a valid status (they should be started or completed)
        valid_statuses = ["started", "completed", "success", "finished", "running"]
        for run in trial_runs:
            assert run["status"] in valid_statuses, f"Unexpected status: {run['status']}"
        
        # Test querying real metrics data
        metrics = db_manager._execute_query("SELECT type, total_val FROM METRIC WHERE total_val IS NOT NULL").fetchall()
        assert len(metrics) > 0  # Should have many metrics from real experiment
        
        # Verify we have expected metric types from MNIST experiment
        metric_types = {metric["type"] for metric in metrics}
        expected_metric_types = {"train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"}
        
        # Should have at least some of the expected metric types
        found_metrics = metric_types.intersection(expected_metric_types)
        assert len(found_metrics) >= 3, f"Expected at least 3 metric types, found: {found_metrics}"
        
        # Test that metrics have reasonable values for MNIST (accuracy between 0-1, loss > 0)
        for metric in metrics:
            if "acc" in metric["type"]:
                assert 0.0 <= metric["total_val"] <= 1.0, f"Accuracy {metric['type']} should be 0-1, got {metric['total_val']}"
            elif "loss" in metric["type"]:
                assert metric["total_val"] >= 0.0, f"Loss {metric['type']} should be >= 0, got {metric['total_val']}"
        
        # Test epoch data query
        epochs = db_manager._execute_query("SELECT idx, trial_run_id FROM EPOCH").fetchall()
        assert len(epochs) > 0  # Should have epoch data from real experiment
        
        # Verify epoch indices are reasonable (0, 1, 2 for 3-epoch MNIST experiment)
        epoch_indices = {epoch["idx"] for epoch in epochs}
        assert epoch_indices == {0, 1, 2}, f"Expected epoch indices [0, 1, 2], got {epoch_indices}"
        
        print(f"✅ Successfully verified DBTracker integration with real MNIST data:")
        print(f"   - Experiments: {len(experiment_titles)}")
        print(f"   - Trials: {len(trials)}")
        print(f"   - Trial runs: {len(trial_runs)}")
        print(f"   - Metrics: {len(metrics)}")
        print(f"   - Metric types: {metric_types}")
        
    finally:
        # Clean up database manager
        if hasattr(db_manager, 'connection') and db_manager.connection:
            db_manager.connection.close()
