"""Tests for the LogTracker."""
import os
import pytest

from experiment_manager.trackers.plugins.log_tracker import LogTracker
from experiment_manager.common.common import Level

@pytest.fixture
def workspace(tmp_path):
    return tmp_path / "test_workspace"

@pytest.fixture
def log_tracker(workspace):
    os.makedirs(workspace, exist_ok=True)
    return LogTracker(str(workspace), verbose=True)

def test_log_indentation(log_tracker, workspace):
    """Test that logs are properly indented based on level."""
    # Create experiment
    log_tracker.on_create(Level.EXPERIMENT, "Test Experiment", description="Test")
    
    # Create trial
    log_tracker.on_create(Level.TRIAL, "Trial 1")
    
    # Create trial run
    log_tracker.on_create(Level.TRIAL_RUN)
    
    # Create epoch
    log_tracker.on_create(Level.EPOCH)
    
    # Add metric
    log_tracker.on_metric(Level.EPOCH, {"accuracy": 0.95, "loss": 0.1})
    
    # Add artifact
    log_tracker.on_add_artifact(Level.EPOCH, "model.pt", "checkpoint")
    
    # End everything
    log_tracker.on_end(Level.EPOCH)
    log_tracker.on_end(Level.TRIAL_RUN)
    log_tracker.on_end(Level.TRIAL)
    log_tracker.on_end(Level.EXPERIMENT)
    
    # Read log file
    log_path = workspace / "artifacts" / LogTracker.LOG_NAME
    assert log_path.exists()
    
    log_content = log_path.read_text()
    lines = log_content.splitlines()
    
    # Verify indentation
    found_patterns = {
        "experiment": False,
        "trial": False,
        "trial_run": False,
        "epoch": False,
        "metric": False,
        "artifact": False
    }
    
    for line in lines:
        content = line.split(' - ', 1)[1] if ' - ' in line else line
        if "Creating EXPERIMENT" in content:
            assert not content.startswith(" "), "EXPERIMENT should not be indented"
            found_patterns["experiment"] = True
        elif "Creating TRIAL" in content and "RUN" not in content:
            assert content.startswith("  "), "TRIAL should be indented with 2 spaces"
            found_patterns["trial"] = True
        elif "Creating TRIAL_RUN" in content:
            assert content.startswith("    "), "TRIAL_RUN should be indented with 4 spaces"
            found_patterns["trial_run"] = True
        elif "Creating EPOCH" in content:
            assert content.startswith("        "), "EPOCH should be indented with 8 spaces"
            found_patterns["epoch"] = True
        elif "accuracy: 0.95" in content:
            assert content.startswith("          "), "Metric should be indented with 10 spaces"
            found_patterns["metric"] = True
        elif "Path: model.pt" in content:
            assert content.startswith("          "), "Artifact path should be indented with 10 spaces"
            found_patterns["artifact"] = True
    
    # Verify all patterns were found
    assert all(found_patterns.values()), f"Missing patterns: {[k for k, v in found_patterns.items() if not v]}"

def test_metric_formatting(log_tracker, workspace):
    """Test that metrics are properly formatted."""
    metrics = {
        "accuracy": 0.95,
        "loss": 0.1,
        "confusion_matrix": {
            "true_positive": 100,
            "false_positive": 5
        }
    }
    
    log_tracker.on_create(Level.EXPERIMENT, "Test")
    log_tracker.on_metric(Level.EXPERIMENT, metrics)
    
    log_path = workspace / "artifacts" / LogTracker.LOG_NAME
    lines = log_path.read_text().splitlines()
    
    # Extract content after timestamps
    lines = [line.split(' - ', 1)[1] if ' - ' in line else line 
            for line in lines]
    content = '\n'.join(lines)
    
    assert "Metric:" in content
    assert "  accuracy: 0.95" in content
    assert "  loss: 0.1" in content
    assert "  confusion_matrix:" in content

def test_artifact_logging(log_tracker, workspace):
    """Test that artifacts are properly logged."""
    log_tracker.on_create(Level.EXPERIMENT, "Test")
    log_tracker.on_add_artifact(
        Level.EXPERIMENT,
        "config.yaml",
        "config",
        description="Configuration file"
    )
    
    log_path = workspace / "artifacts" / LogTracker.LOG_NAME
    log_content = log_path.read_text()
    
    # Extract content after timestamps
    lines = [line.split(' - ', 1)[1] if ' - ' in line else line 
            for line in log_content.splitlines()]
    content = '\n'.join(lines)
    
    assert "Adding artifact:" in content
    assert "  Path: config.yaml" in content
    assert "  Type: config" in content
    assert "  Kwargs: {'description': 'Configuration file'}" in content
