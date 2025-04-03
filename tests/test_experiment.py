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
            "verbose": False  # Disable console logging
        }
    })

@pytest.fixture
def env(env_config, tmp_path):
    workspace = os.path.join(str(tmp_path), env_config.workspace)
    env = Environment(
        workspace=workspace,
        config=env_config,
        verbose=False  # Disable console logging
    )
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
    experiment_conf = {
        "name": "test_experiment",
        "id": 123,
        "desc": "Test experiment",
        "settings": {"param1": "value1"}
    }
    base_conf = {"settings": {"param2": "value2"}}
    trials_conf = [
        {
            "name": "test_trial_1",
            "id": 1,
            "repeat": 2,
            "settings": {"trial_specific": "value1", "trial_param": "trial_value"}
        },
        {
            "name": "test_trial_2",
            "id": 2,
            "repeat": 1,
            "settings": {"trial_specific": "value2", "trial_param": "trial_value"}
        }
    ]
    
    # Save config files
    OmegaConf.save(experiment_conf, os.path.join(config_dir, Experiment.CONFIG_FILE))
    OmegaConf.save(base_conf, os.path.join(config_dir, Experiment.BASE_CONFIG))
    OmegaConf.save(trials_conf, os.path.join(config_dir, Experiment.TRIALS_CONFIG))
    
    return config_dir

def test_experiment_creates_log_file(env, config_dir):
    """Test that experiment logs to the environment's log file"""
    print("\nStarting test_experiment_creates_log_file")
    
    # Create and setup experiment
    name = "test_exp"
    exp_id = 123
    
    experiment = Experiment(
        name=name,
        id=exp_id,
        desc="Test experiment",
        env=env,
        config_dir_path=config_dir
    )
    
    print("\nCreated experiment")
    print(f"Experiment workspace: {experiment.env.workspace}")
    print(f"Experiment log directory: {experiment.env.log_dir}")
    
    print("\nSetup experiment")
    experiment.run()
    print("\nRan experiment")
    
    # Print directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(experiment.env.workspace):
        level = root.replace(experiment.env.workspace, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
    
    # Check log files in each expected location
    log_files = []
    print("\n=== Directory Contents ===")
    for root, dirs, files in os.walk(experiment.env.workspace):
        print(f"\nDirectory: {root}")
        print("Files:", files)
        print("Subdirs:", dirs)
        for f in files:
            if "log" in f:
                log_file = os.path.join(root, f)
                print(f"Found log file: {log_file}")
                log_files.append(log_file)
    
    print("\n=== All Log Files ===")
    for log_file in log_files:
        print(f"- {os.path.relpath(log_file, experiment.env.workspace)}")
    
    # Should have:
    # 1. Experiment log
    # 2-4. test_trial_1 logs (2 run logs + 1 environment log)
    # 5-6. test_trial_2 logs (1 run log + 1 environment log)
    print(f"\nFound {len(log_files)} log files")
    for log_file in log_files:
        print(f"- {log_file}")
    
    assert len(log_files) == 6, f"Expected 6 log files (experiment + 3 trial_1 logs + 2 trial_2 logs), found {len(log_files)}"
    
    # Verify each log file exists and is readable
    for log_file in log_files:
        assert os.path.exists(log_file), f"Log file not found at {log_file}"
        assert os.path.getsize(log_file) > 0, f"Log file is empty: {log_file}"
        
    # Print log files for debugging
    print("\nFound log files:")
    for log_file in log_files:
        print(f"- {os.path.relpath(log_file, experiment.env.workspace)}")