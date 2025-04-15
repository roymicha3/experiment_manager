import os
import pytest
from omegaconf import OmegaConf
from experiment_manager.experiment import Experiment, ConfigPaths
from experiment_manager.environment import Environment
from tests.pipelines.dummy_pipeline_factory import DummyPipelineFactory

    
@pytest.fixture
def config_dir(tmp_path):
    config_dir = os.path.join("tests", "configs", "test_callback")
    return config_dir


@pytest.fixture
def prepare_env(tmp_path, config_dir):
    env_path = os.path.join(config_dir, ConfigPaths.ENV_CONFIG.value)
    env = OmegaConf.load(env_path)
    env.workspace = os.path.join(tmp_path, "test_outputs")
    OmegaConf.save(env, env_path)
    return env


def test_experiment_creates_checkpoint(prepare_env, config_dir):
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
    checkpoint_files = []
    print("\n=== Directory Contents ===")
    for root, dirs, files in os.walk(experiment.env.workspace):
        print(f"\nDirectory: {root}")
        print("Files:", files)
        print("Subdirs:", dirs)
        for f in files:
            if "checkpoint" in f:
                checkpoint_file = os.path.join(root, f)
                print(f"Found checkpoint file: {checkpoint_file}")
                checkpoint_files.append(checkpoint_file)
    
    print("\n=== All Checkpoint Files ===")
    for checkpoint_file in checkpoint_files:
        print(f"- {os.path.relpath(checkpoint_file, experiment.env.workspace)}")
    
    # Should have:
    print(f"\nFound {len(checkpoint_files)} checkpoint files")
    for checkpoint_file in checkpoint_files:
        print(f"- {checkpoint_file}")
    
    assert len(checkpoint_files) == 6, f"Expected 6 checkpoint files (experiment + 3 trial_1 checkpoints + 2 trial_2 checkpoints), found {len(checkpoint_files)}"
    
    # Verify each checkpoint file exists and is readable
    for checkpoint_file in checkpoint_files:
        assert os.path.exists(checkpoint_file), f"Checkpoint file not found at {checkpoint_file}"
        assert os.path.getsize(checkpoint_file) > 0, f"Checkpoint file is empty: {checkpoint_file}"
        
    # Print checkpoint files for debugging
    print("\nFound checkpoint files:")
    for checkpoint_file in checkpoint_files:
        print(f"- {os.path.relpath(checkpoint_file, experiment.env.workspace)}")