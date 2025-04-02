import os
import pytest
from omegaconf import OmegaConf
from experiment_manager.environment import Environment

@pytest.fixture
def env_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return OmegaConf.load(os.path.join(current_dir, "env.yaml"))

@pytest.fixture
def env(env_config, tmp_path):
    # Use pytest's tmp_path fixture to create a temporary directory
    workspace = os.path.join(str(tmp_path), env_config.workspace)
    env = Environment(workspace=workspace, config=env_config)
    env.setup_environment()  # Explicitly set up the environment
    return env

def test_environment_initialization(env):
    """Test that environment is initialized correctly"""
    assert os.path.exists(env.workspace)
    assert os.path.exists(env.log_dir)
    assert os.path.exists(env.artifact_dir)
    assert os.path.exists(env.config_dir)

def test_environment_paths(env):
    """Test that environment paths are correct"""
    assert env.log_dir == os.path.join(env.workspace, Environment.LOG_DIR)
    assert env.artifact_dir == os.path.join(env.workspace, Environment.ARTIFACT_DIR)
    assert env.config_dir == os.path.join(env.workspace, Environment.CONFIG_DIR)

def test_environment_from_config(env_config, tmp_path):
    """Test creating environment from config"""
    workspace = os.path.join(str(tmp_path), env_config.workspace)
    env_config.workspace = workspace
    env = Environment.from_config(env_config)
    env.setup_environment()  # Explicitly set up the environment
    assert env.workspace == workspace
    assert env.config == env_config

def test_environment_save(env):
    """Test saving environment config"""
    env.save()
    config_path = os.path.join(env.config_dir, Environment.CONFIG_FILE)
    assert os.path.exists(config_path)
    loaded_config = OmegaConf.load(config_path)
    assert loaded_config == env.config

def test_set_workspace(env, tmp_path):
    """Test setting new workspace"""
    new_workspace = os.path.join(str(tmp_path), "new_workspace")
    env.set_workspace(new_workspace)
    assert env.workspace == os.path.abspath(new_workspace)
    
    # Test inner workspace
    inner_workspace = "inner_workspace"
    env.set_workspace(inner_workspace, inner=True)
    assert env.workspace == os.path.abspath(os.path.join(new_workspace, inner_workspace))
