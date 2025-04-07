import os
import pytest
from omegaconf import OmegaConf, DictConfig

from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment
from experiment_manager.common.common import Level, Metric
from experiment_manager.trackers.log_tracker import LogTracker

@pytest.fixture
def env_config():
    return OmegaConf.create({
        "workspace": "test_workspace",
        "settings": {
            "debug": True,
            "verbose": True
        },
        "trackers": {
            "log": {
                "type": "LogTracker",
                "name": "test_tracker.log",
                "verbose": True
            }
        }
    })

@pytest.fixture
def experiment_config():
    return OmegaConf.create({
        "name": "test_experiment",
        "id": 1,
        "desc": "Test experiment for tracking",
        "settings": {
            "param1": "value1"
        }
    })

@pytest.fixture
def env(env_config, tmp_path):
    workspace = os.path.join(str(tmp_path), env_config.workspace)
    env = Environment(
        workspace=workspace,
        config=env_config,
        verbose=True,
        level=Level.EXPERIMENT
    )
    env.setup_environment()
    return env

@pytest.fixture
def experiment(env, experiment_config, tmp_path):
    config_dir = os.path.join(str(tmp_path), "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    experiment = Experiment(
        name=experiment_config.name,
        id=experiment_config.id,
        desc=experiment_config.desc,
        env=env,
        config_dir_path=config_dir
    )
    return experiment

def test_experiment_tracking(experiment):
    """Test that metrics are tracked at experiment level"""
    metric_value = 0.95
    metric_name = "accuracy"
    
    # Track a metric
    experiment.env.tracker_manager.track(
        name=metric_name,
        value=metric_value,
        metric_type=Metric.ACCURACY
    )
    
    # Verify in log tracker
    tracker = experiment.env.tracker_manager.trackers[0]
    assert isinstance(tracker, LogTracker)
    
    log_path = os.path.join(tracker.workspace, tracker.name)
    assert os.path.exists(log_path)
    
    # Read log file and verify content
    with open(log_path, 'r') as f:
        content = f.read()
        assert metric_name in content
        assert str(metric_value) in content
        assert Metric.ACCURACY.value in content

def test_hierarchical_tracking(experiment):
    """Test that metrics are tracked through the hierarchy"""
    metric_value = 0.85
    metric_name = "loss"
    
    # Create a trial environment
    trial_env = experiment.env.create_child("trial_1")
    
    # Track from trial level
    trial_env.tracker_manager.track(
        name=metric_name,
        value=metric_value,
        metric_type=Metric.LOSS
    )
    
    # Verify in both trial and experiment trackers
    for env in [trial_env, experiment.env]:
        tracker = env.tracker_manager.trackers[0]
        log_path = os.path.join(tracker.workspace, tracker.name)
        assert os.path.exists(log_path)
        
        with open(log_path, 'r') as f:
            content = f.read()
            assert metric_name in content
            assert str(metric_value) in content
            assert Metric.LOSS.value in content

def test_tracker_cleanup(experiment):
    """Test that trackers properly clean up resources"""
    metric_value = 0.75
    metric_name = "validation_loss"
    
    # Track some metrics
    experiment.env.tracker_manager.track(
        name=metric_name,
        value=metric_value,
        metric_type=Metric.VAL_LOSS
    )
    
    # Get log file path
    tracker = experiment.env.tracker_manager.trackers[0]
    log_path = os.path.join(tracker.workspace, tracker.name)
    
    # Verify file exists
    assert os.path.exists(log_path)
    
    # Clean up environment
    experiment.env.tracker_manager.close()
    
    # File should still exist but be properly closed
    assert os.path.exists(log_path)
    # Could add more specific checks for file handles if needed
