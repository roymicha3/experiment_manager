"""Tests for the DBTracker."""
import os
import pytest
from pathlib import Path
from omegaconf import OmegaConf

from experiment_manager.trackers.plugins.db_tracker import DBTracker
from experiment_manager.common.common import Level, Metric

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
