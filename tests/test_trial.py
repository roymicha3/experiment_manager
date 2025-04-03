import os
import pytest
from omegaconf import OmegaConf
from experiment_manager.environment import Environment
from experiment_manager.trial import Trial
from experiment_manager.experiment import Experiment


# ============= Fixtures =============

@pytest.fixture
def base_config():
    """Base configuration with common settings"""
    return OmegaConf.create({
        "settings": {
            "debug": True,
            "verbose": True,
            "base_param": "base_value"
        }
    })

@pytest.fixture
def env_config(base_config):
    """Environment configuration"""
    env_conf = OmegaConf.create({
        "workspace": "test_outputs",
        "settings": base_config.settings
    })
    return env_conf

@pytest.fixture
def env(env_config, tmp_path):
    """Test environment"""
    workspace = os.path.join(str(tmp_path), env_config.workspace)
    env = Environment(workspace=workspace, config=env_config)
    env.setup_environment()
    return env

@pytest.fixture
def trial_config(base_config):
    """Single trial configuration"""
    return OmegaConf.create({
        "name": "test_trial",
        "id": 123,
        "repeat": 3,
        "settings": {
            "trial_param": "trial_value",
            "nested": {
                "param": "value"
            }
        }
    })

@pytest.fixture
def experiment_config(base_config, env):
    """Experiment configuration with multiple trials"""
    # Create required config files
    config = OmegaConf.create({
        "name": "test_experiment",
        "id": 456,
        "desc": "Test experiment with multiple trials",
        "settings": base_config.settings,
        "config_dir_path": os.path.join(env.config_dir, "experiment_configs"),  # Save to a subdirectory
        "trials": []
    })
    
    base_conf = OmegaConf.create({
        "settings": base_config.settings
    })
    
    trials_conf = OmegaConf.create([
        {
            "name": "test_trial_1",
            "id": 1,
            "repeat": 2,
            "settings": {"trial_specific": "value1"}
        },
        {
            "name": "test_trial_2",
            "id": 2,
            "repeat": 1,
            "settings": {"trial_specific": "value2"}
        }
    ])
    
    # Save config files to config_dir_path
    config_dir_path = os.path.join(env.config_dir, "experiment_configs")
    os.makedirs(config_dir_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config_dir_path, Experiment.CONFIG_FILE))
    OmegaConf.save(base_conf, os.path.join(config_dir_path, Experiment.BASE_CONFIG))
    OmegaConf.save(trials_conf, os.path.join(config_dir_path, Experiment.TRIALS_CONFIG))
    
    return config


# ============= Directory Structure Tests =============

class TestTrialDirectories:
    def test_basic_trial_structure(self, env, trial_config):
        """Test basic trial directory structure creation"""
        trial = Trial.from_config(trial_config, env)
        
        # Check main directories exist
        trial_dir = os.path.join(env.workspace, trial_config.name)
        assert os.path.exists(trial_dir), "Trial directory not created"
        
        # Check subdirectories
        subdirs = ["logs", "configs", "artifacts"]
        for subdir in subdirs:
            path = os.path.join(trial_dir, subdir)
            assert os.path.exists(path), f"{subdir} directory not created"
            assert os.path.isdir(path), f"{subdir} is not a directory"
            assert os.access(path, os.R_OK | os.W_OK), f"{subdir} not accessible"
    
    def test_nested_trial_structure(self, env, trial_config):
        """Test trial directory structure in nested environment"""
        nested_env = env.copy()
        nested_env.set_workspace("nested", inner=True)
        nested_env.setup_environment()
        
        trial = Trial.from_config(trial_config, nested_env)
        
        # Check paths
        nested_dir = os.path.join(env.workspace, "nested")
        trial_dir = os.path.join(nested_dir, trial_config.name)
        assert os.path.exists(nested_dir), "Nested directory not created"
        assert os.path.exists(trial_dir), "Trial directory not created in nested environment"


# ============= Configuration Tests =============

class TestTrialConfiguration:
    def test_config_saving(self, env, trial_config):
        """Test trial configuration is saved correctly"""
        trial = Trial.from_config(trial_config, env)
        
        # Check config file
        config_file = os.path.join(trial.env.config_dir, Trial.CONFIG_FILE)
        assert os.path.exists(config_file), "Config file not created"
        
        # Verify config contents
        saved_config = OmegaConf.load(config_file)
        assert saved_config.name == trial_config.name
        assert saved_config.id == trial_config.id
        assert saved_config.repeat == trial_config.repeat
        assert saved_config.settings.trial_param == trial_config.settings.trial_param
        assert saved_config.settings.nested.param == trial_config.settings.nested.param
    
    def test_experiment_trial_config_inheritance(self, env, experiment_config):
        """Test trials inherit settings from experiment correctly"""
        experiment = Experiment.from_config(experiment_config, env)
        
        # Run experiment to create trials
        experiment.run()
        
        # Check each trial's config
        trials_dir = os.path.join(experiment.env.workspace, "trials")
        for trial_conf in experiment_config.trials:
            trial_dir = os.path.join(trials_dir, trial_conf.name)
            config_file = os.path.join(trial_dir, "configs", Trial.CONFIG_FILE)
            
            assert os.path.exists(config_file), f"Config file not found for trial {trial_conf.name}"
            
            # Load and verify config
            saved_config = OmegaConf.load(config_file)
            assert saved_config.name == trial_conf.name
            assert saved_config.id == trial_conf.id
            assert saved_config.settings.base_param == experiment_config.settings.base_param
            assert saved_config.settings.trial_specific == trial_conf.settings.trial_specific


# ============= Logging Tests =============

class TestTrialLogging:
    def test_basic_trial_logging(self, env, trial_config):
        """Test basic trial logging"""
        trial = Trial.from_config(trial_config, env)
        trial.run()
        
        # Find log file
        log_files = [f for f in os.listdir(trial.env.log_dir) if f.endswith('.log')]
        assert len(log_files) > 0, "No log files found"
        
        # Check log contents
        log_file = os.path.join(trial.env.log_dir, log_files[0])
        with open(log_file, 'r') as f:
            content = f.read()
            assert f"Trial '{trial_config.name}' (ID: {trial_config.id}) created" in content
            assert f"Running trial '{trial_config.name}' (ID: {trial_config.id})" in content
            
            # Check repeat logs
            for i in range(trial_config.repeat):
                assert f"repeat {i}" in content
                assert f"repeat {i} completed" in content
    
    def test_experiment_trial_logging(self, env, experiment_config):
        """Test trial logging when created from experiment"""
        experiment = Experiment.from_config(experiment_config, env)
        experiment.run()
        
        # Check each trial's logs
        trials_dir = os.path.join(experiment.env.workspace, "trials")
        for trial_conf in experiment_config.trials:
            trial_dir = os.path.join(trials_dir, trial_conf.name)
            log_dir = os.path.join(trial_dir, "logs")
            
            # Find log file
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
            assert len(log_files) > 0, f"No log files for trial {trial_conf.name}"
            
            # Check log contents
            log_file = os.path.join(log_dir, log_files[0])
            with open(log_file, 'r') as f:
                content = f.read()
                assert f"Trial '{trial_conf.name}' (ID: {trial_conf.id}) created" in content
                assert f"Running trial '{trial_conf.name}' (ID: {trial_conf.id})" in content
