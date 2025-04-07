import os
import pytest
from omegaconf import OmegaConf, DictConfig

from experiment_manager.environment import Environment
from experiment_manager.common.common import Level
from experiment_manager.trackers.tracker_manager import TrackerManager
from experiment_manager.trackers.log_tracker import LogTracker

@pytest.fixture
def base_config():
    return OmegaConf.create({
        "workspace": "test_workspace",
        "settings": {
            "debug": True,
            "verbose": True
        },
        "trackers": [
            {
                "type": "LogTracker",
                "name": "test_tracker.log",
                "verbose": True
            },
        ]
    })

@pytest.fixture
def env(base_config, tmp_path):
    workspace = os.path.join(str(tmp_path), base_config.workspace)
    env = Environment(
        workspace=workspace,
        config=base_config,
        verbose=True,
        level=Level.EXPERIMENT
    )
    env.setup_environment()
    return env

def test_tracker_config_loading(env, base_config):
    """Test that tracker configuration is properly loaded"""
    assert env.tracker_manager is not None
    assert len(env.tracker_manager.trackers) == 1
    
    tracker = env.tracker_manager.trackers[0]
    assert isinstance(tracker, LogTracker)
    assert tracker.name == "test_tracker.log"
    assert tracker.verbose == True

def test_tracker_config_inheritance(env, base_config):
    """Test that child environments inherit tracker configuration"""
    child_env = env.create_child("child")
    assert child_env.tracker_manager is not None
    assert len(child_env.tracker_manager.trackers) == len(env.tracker_manager.trackers)
    
    parent_tracker = env.tracker_manager.trackers[0]
    child_tracker = child_env.tracker_manager.trackers[0]
    
    assert isinstance(child_tracker, type(parent_tracker))
    assert child_tracker.name == parent_tracker.name
    assert child_tracker.verbose == parent_tracker.verbose

def test_tracker_config_validation(base_config, tmp_path):
    """Test that invalid tracker configurations are caught"""
    # Test missing type
    invalid_config = OmegaConf.create(base_config)
    invalid_config.trackers[0].pop("type")
    
    workspace = os.path.join(str(tmp_path), invalid_config.workspace)
    with pytest.raises(ValueError, match="missing required 'type' field"):
        env = Environment(
            workspace=workspace,
            config=invalid_config,
            verbose=True,
            level=Level.EXPERIMENT
        )

def test_tracker_workspace_paths(env):
    """Test that tracker workspace paths are properly set"""
    tracker = env.tracker_manager.trackers[0]
    assert tracker.workspace == env.artifact_dir
