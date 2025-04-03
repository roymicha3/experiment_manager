import os
import glob
import pytest
from omegaconf import OmegaConf
from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment


@pytest.fixture
def workspace_dir(tmp_path):
    """Create a temporary workspace directory"""
    return str(tmp_path / "workspace")


@pytest.fixture
def config_dir(tmp_path):
    """Create config directory with experiment files"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # Create experiment.yaml
    exp_config = {
        "name": "test_exp",
        "id": 1,
        "desc": "Test experiment",
        "settings": {
            "model_type": "mlp",
            "batch_size": 32
        }
    }
    OmegaConf.save(OmegaConf.create(exp_config), config_dir / "experiment.yaml")
    
    # Create base.yaml
    base_config = {
        "settings": {
            "model_type": "mlp",
            "batch_size": 32,
            "log_level": "INFO"
        }
    }
    OmegaConf.save(OmegaConf.create(base_config), config_dir / "base.yaml")
    
    # Create trials.yaml
    trials_config = [
        {
            "name": "trial_1",
            "id": 1,
            "repeat": 2,
            "settings": {"learning_rate": 0.1}
        },
        {
            "name": "trial_2",
            "id": 2,
            "repeat": 1,
            "settings": {"learning_rate": 0.01}
        }
    ]
    OmegaConf.save(OmegaConf.create(trials_config), config_dir / "trials.yaml")
    
    return str(config_dir)


@pytest.fixture
def env(workspace_dir):
    """Create environment"""
    env_config = OmegaConf.create({
        "workspace": workspace_dir,
        "settings": {
            "debug": True,
            "verbose": True
        }
    })
    env = Environment(workspace=workspace_dir, config=env_config)
    env.setup_environment()
    return env


class TestExperimentIntegration:
    def test_experiment_directory_structure(self, env, config_dir):
        """Test that experiment creates correct directory structure"""
        # Create and run experiment
        exp_config = OmegaConf.create({
            "name": "test_exp",
            "id": 1,
            "desc": "Test experiment",
            "config_dir_path": config_dir
        })
        experiment = Experiment.from_config(exp_config, env)
        experiment.run()
        
        # Check experiment directory structure
        exp_dir = experiment.env.workspace
        assert os.path.exists(exp_dir)
        assert os.path.exists(os.path.join(exp_dir, "configs"))
        assert os.path.exists(os.path.join(exp_dir, "logs"))
        assert os.path.exists(os.path.join(exp_dir, "trials"))
        
        # Check trial directories are at the correct level
        trials_dir = os.path.join(exp_dir, "trials")
        trial1_dir = os.path.join(trials_dir, "trial_1")
        trial2_dir = os.path.join(trials_dir, "trial_2")
        
        # These should exist
        assert os.path.exists(trial1_dir)
        assert os.path.exists(trial2_dir)
        
        # Each trial should have its own config and log directories
        for trial_dir in [trial1_dir, trial2_dir]:
            assert os.path.exists(os.path.join(trial_dir, "configs"))
            assert os.path.exists(os.path.join(trial_dir, "logs"))
            assert os.path.exists(os.path.join(trial_dir, "artifacts"))
    
    def test_experiment_logging(self, env, config_dir):
        """Test that experiment creates correct number of log files"""
        # Create and run experiment
        exp_config = OmegaConf.create({
            "name": "test_exp",
            "id": 1,
            "desc": "Test experiment",
            "config_dir_path": config_dir
        })
        experiment = Experiment.from_config(exp_config, env)
        experiment.run()
        
        def count_log_files(directory):
            """Count .log files in a directory"""
            return len(glob.glob(os.path.join(directory, "**/*.log"), recursive=True))
        
        # Check total number of log files
        total_logs = count_log_files(experiment.env.workspace)
        expected_logs = 6  # 1 experiment + 1 trials dir + 2 trial dirs + one trial with 2 repeat and another with 1 repeat
        assert total_logs == expected_logs, (
            f"Expected {expected_logs} log files in total "
            f"(1 experiment + 1 trials dir + 2 trial dirs), found {total_logs}"
        )
        
        # Check experiment logs
        exp_logs = count_log_files(os.path.join(experiment.env.workspace, "logs"))
        assert exp_logs == 1, f"Expected 1 experiment log file, found {exp_logs}"
        
        # Check trials dir logs
        trials_logs = count_log_files(os.path.join(experiment.env.workspace, "trials", "logs"))
        assert trials_logs == 1, f"Expected 1 trials dir log file, found {trials_logs}"
        
        # Check individual trial logs
        trial1_logs = count_log_files(os.path.join(experiment.env.workspace, "trials", "trial_1", "logs"))
        assert trial1_logs == 1, f"Expected 1 log file for trial_1, found {trial1_logs}"
        
        trial2_logs = count_log_files(os.path.join(experiment.env.workspace, "trials", "trial_2", "logs"))
        assert trial2_logs == 1, f"Expected 1 log file for trial_2, found {trial2_logs}"
    
    def test_trial_config_inheritance(self, env, config_dir):
        """Test that trials inherit and merge settings correctly"""
        # Create and run experiment
        exp_config = OmegaConf.create({
            "name": "test_exp",
            "id": 1,
            "desc": "Test experiment",
            "config_dir_path": config_dir
        })
        experiment = Experiment.from_config(exp_config, env)
        experiment.run()
        
        # Check trial 1 config
        trial1_config = OmegaConf.load(os.path.join(
            experiment.env.workspace, "trials", "trial_1", "configs", "trial.yaml"
        ))
        assert trial1_config.settings.model_type == "mlp"  # From base config
        assert trial1_config.settings.batch_size == 32     # From base config
        assert trial1_config.settings.learning_rate == 0.1  # From trial config
        
        # Check trial 2 config
        trial2_config = OmegaConf.load(os.path.join(
            experiment.env.workspace, "trials", "trial_2", "configs", "trial.yaml"
        ))
        assert trial2_config.settings.model_type == "mlp"   # From base config
        assert trial2_config.settings.batch_size == 32      # From base config
        assert trial2_config.settings.learning_rate == 0.01 # From trial config
    
    def test_trial_workspace_isolation(self, env, config_dir):
        """Test that each trial has its own isolated workspace"""
        # Create and run experiment
        exp_config = OmegaConf.create({
            "name": "test_exp",
            "id": 1,
            "desc": "Test experiment",
            "config_dir_path": config_dir
        })
        experiment = Experiment.from_config(exp_config, env)
        experiment.run()
        
        # Get trial directories
        trials_dir = os.path.join(experiment.env.workspace, "trials")
        trial1_dir = os.path.join(trials_dir, "trial_1")
        trial2_dir = os.path.join(trials_dir, "trial_2")
        
        # Each trial should have its own isolated workspace
        for trial_dir in [trial1_dir, trial2_dir]:
            # Should have all required directories
            assert os.path.exists(os.path.join(trial_dir, "configs"))
            assert os.path.exists(os.path.join(trial_dir, "logs"))
            assert os.path.exists(os.path.join(trial_dir, "artifacts"))
            
            # Should have trial config
            assert os.path.exists(os.path.join(trial_dir, "configs", "trial.yaml"))
            
            # Should have environment config
            assert os.path.exists(os.path.join(trial_dir, "configs", "env.yaml"))
            
            # Should have log file
            assert len(os.listdir(os.path.join(trial_dir, "logs"))) > 0
