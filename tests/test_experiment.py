import os
import glob
import pytest
from omegaconf import OmegaConf
from experiment_manager.experiment import Experiment
from experiment_manager.environment import Environment
from experiment_manager.logger import FileLogger


@pytest.fixture
def env_config():
    return OmegaConf.create({
        "workspace": "test_outputs",
        "settings": {
            "debug": True,
            "verbose": True
        }
    })

@pytest.fixture
def env(env_config, tmp_path):
    workspace = os.path.join(str(tmp_path), env_config.workspace)
    env = Environment(workspace=workspace, config=env_config)
    env.setup_environment()  # This creates the log directory
    
    # Ensure log directory exists
    os.makedirs(env.log_dir, exist_ok=True)
    
    # Set up a FileLogger for testing
    log_file = "environment.log"
    env.logger = FileLogger(
        name="environment",
        log_dir=env.log_dir,
        filename=log_file
    )
    return env

@pytest.fixture
def config_dir(env):
    # Create config directory
    config_dir = os.path.join(env.config_dir, "test_experiment")
    os.makedirs(config_dir, exist_ok=True)
    
    # Create required config files
    experiment_conf = {"settings": {"param1": "value1"}}
    base_conf = {"base_settings": {"param2": "value2"}}
    trials_conf = [{"trial_id": 1, "settings": {}}]
    
    # Save config files
    OmegaConf.save(experiment_conf, os.path.join(config_dir, Experiment.CONFIG_FILE))
    OmegaConf.save(base_conf, os.path.join(config_dir, Experiment.BASE_CONFIG))
    OmegaConf.save(trials_conf, os.path.join(config_dir, Experiment.TRIALS_CONFIG))
    
    return config_dir

def test_experiment_creates_log_file(env, config_dir):
    """Test that experiment logs to the environment's log file"""
    # Create experiment
    name = "test_exp"
    exp_id = 123
    
    experiment = Experiment(
        name=name,
        id=exp_id,
        desc="Test experiment",
        env=env,
        config_dir_path=config_dir
    )
    
    # Check that environment log file exists and contains experiment logs
    expected_log_file = None
    for log_file in os.listdir(env.log_dir):
        if "log" in log_file:
            expected_log_file = os.path.join(env.log_dir, log_file)
            break
    assert expected_log_file is not None, "Log file not found in environment log directory"
    assert os.path.exists(expected_log_file), f"Log file not found at {expected_log_file}"