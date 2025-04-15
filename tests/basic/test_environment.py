import os
import glob
import pytest
from omegaconf import OmegaConf
from experiment_manager.environment import Environment, ProductPaths

@pytest.fixture
def env_config(tmp_path):
    return OmegaConf.create({
        "workspace": os.path.join(str(tmp_path), "test_outputs"),
        "verbose": True,
        "debug": True,
    })

@pytest.fixture
def env(env_config, tmp_path):
    env = Environment.from_config(env_config)
    return env

def test_environment_initialization(env):
    """Test that environment is initialized correctly"""
    assert os.path.exists(env.workspace)
    assert os.path.exists(env.log_dir)
    assert os.path.exists(env.config_dir)

def test_environment_paths(env):
    """Test that environment paths are correct"""
    assert env.log_dir == os.path.join(env.workspace, ProductPaths.LOG_DIR.value)
    assert env.artifact_dir == os.path.join(env.workspace, ProductPaths.ARTIFACT_DIR.value)
    assert env.config_dir == os.path.join(env.workspace, ProductPaths.CONFIG_DIR.value)

def test_environment_from_config(env_config, tmp_path):
    """Test creating environment from config"""
    workspace = os.path.join(str(tmp_path), env_config.workspace)
    env_config.workspace = workspace
    env = Environment.from_config(env_config)
    
    assert env.workspace == workspace
    assert env.config == env_config

def test_environment_save(env):
    """Test saving environment config"""
    config_path = os.path.join(env.config_dir, ProductPaths.CONFIG_FILE.value)
    assert os.path.exists(config_path)
    loaded_config = OmegaConf.load(config_path)
    assert loaded_config == env.config

def test_set_workspace(env, tmp_path):
    """Test setting new workspace"""
    new_workspace = os.path.join(str(tmp_path), "new_workspace")
    env.set_workspace(new_workspace)
      # Set up the new workspace
    assert env.workspace == os.path.abspath(new_workspace)
    
    # Test inner workspace
    inner_workspace = "inner_workspace"
    env.set_workspace(inner_workspace, inner=True)
      # Set up the inner workspace
    assert env.workspace == os.path.abspath(os.path.join(new_workspace, inner_workspace))

def test_environment_logging_file_only(env):
    """Test that logs are created in the correct directory with file-only logging"""
    # Setup environment with verbose=False for file-only logging
    
    
    # Check that log directory exists
    assert os.path.exists(env.log_dir)
    
    # Generate some log messages
    env.logger.info("Test info message")
    env.logger.debug("Test debug message")
    env.logger.warning("Test warning message")
    
    # Check that log file was created
    log_files = glob.glob(os.path.join(env.log_dir, "*.log"))
    assert len(log_files) == 1, "Expected exactly one log file"
    
    # Check log file content
    with open(log_files[0], 'r') as f:
        content = f.read()
        assert "Test info message" in content
        assert "Test warning message" in content

def test_environment_logging_composite(env):
    """Test that logs are created with both file and console logging"""
    # Setup environment with verbose=True for both console and file logging
    
    
    # Check that log directory exists
    assert os.path.exists(env.log_dir)
    
    # Generate some log messages
    env.logger.info("Test info message")
    env.logger.debug("Test debug message")
    env.logger.warning("Test warning message")
    
    # Check that log file was created
    log_files = glob.glob(os.path.join(env.log_dir, "*.log"))
    assert len(log_files) == 1, "Expected exactly one log file"
    
    # Check log file content
    with open(log_files[0], 'r') as f:
        content = f.read()
        assert "Test info message" in content
        assert "Test debug message" in content
        assert "Test warning message" in content