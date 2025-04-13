import os
import pytest
from omegaconf import OmegaConf
from experiment_manager.experiment import Experiment, ConfigPaths
from experiment_manager.environment import Environment
from tests.pipelines.dummy_pipeline_factory import DummyPipelineFactory

    
@pytest.fixture
def config_dir(tmp_path):
    config_dir = os.path.join("tests", "configs", "test_experiment")
    return config_dir


@pytest.fixture
def prepare_env(tmp_path, config_dir):
    env_path = os.path.join(config_dir, ConfigPaths.ENV_CONFIG.value)
    env = OmegaConf.load(env_path)
    env.workspace = os.path.join(tmp_path, "test_outputs")
    OmegaConf.save(env, env_path)
    return env


def test_experiment_creates_log_file(prepare_env, config_dir):
    """Test that experiment logs to the environment's log file"""
    print("\nStarting test_experiment_creates_log_file")
    
    experiment = Experiment.create(config_dir, DummyPipelineFactory)
    
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
    # 2. first trial log
    # 3-4.    two reapts - 2 run logs
    # 5. second trial log
    # 6.    one reapts - 1 run log
    print(f"\nFound {len(log_files)} log files")
    for log_file in log_files:
        print(f"- {log_file}")
    
    assert len(log_files) == 5, f"Expected 5 log files (experiment + 3 trial_1 logs + 2 trial_2 logs), found {len(log_files)}"
    
    # Verify each log file exists and is readable
    for log_file in log_files:
        assert os.path.exists(log_file), f"Log file not found at {log_file}"
        assert os.path.getsize(log_file) > 0, f"Log file is empty: {log_file}"
        
    # Print log files for debugging
    print("\nFound log files:")
    for log_file in log_files:
        print(f"- {os.path.relpath(log_file, experiment.env.workspace)}")