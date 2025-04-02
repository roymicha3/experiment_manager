import os
import pytest
from omegaconf import OmegaConf
from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment

@pytest.fixture
def env_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return OmegaConf.load(os.path.join(current_dir, "env.yaml"))

@pytest.fixture
def env(env_config, tmp_path):
    workspace = os.path.join(str(tmp_path), env_config.workspace)
    env = Environment(workspace=workspace, config=env_config)
    env.setup_environment()
    return env

@pytest.fixture
def experiment_configs(env):
    # Create and save test configurations
    base_config = OmegaConf.create({
        "model": {
            "type": "test_model",
            "params": {"num_classes": 2}
        }
    })
    
    trials_config = OmegaConf.create([
        {"learning_rate": 0.1, "settings": {}},
        {"learning_rate": 0.01, "settings": {}}
    ])
    
    exp_config = OmegaConf.create({
        "name": "test_exp",
        "description": "Test experiment",
        "settings": {}
    })
    
    # Save configurations
    os.makedirs(env.config_dir, exist_ok=True)
    OmegaConf.save(base_config, os.path.join(env.config_dir, Experiment.BASE_CONFIG))
    OmegaConf.save(trials_config, os.path.join(env.config_dir, Experiment.TRIALS_CONFIG))
    OmegaConf.save(exp_config, os.path.join(env.config_dir, Experiment.CONFIG_FILE))
    
    return base_config, trials_config, exp_config

@pytest.fixture
def experiment(env, experiment_configs):
    return Experiment(
        name="test_exp",
        id=1,
        desc="Test experiment",
        env=env,
        config_dir_path=env.config_dir
    )

def test_experiment_initialization(experiment):
    """Test that experiment is initialized correctly"""
    assert experiment.name == "test_exp"
    assert experiment.id == 1
    assert experiment.desc == "Test experiment"
    assert experiment.env is not None

def test_experiment_config_loading(experiment, experiment_configs):
    """Test that configurations are loaded correctly"""
    base_config, trials_config, exp_config = experiment_configs
    
    assert experiment.base_config == base_config
    assert experiment.trials_config == trials_config
    assert experiment.config == exp_config

def test_experiment_missing_configs(env):
    """Test that experiment raises error when configs are missing"""
    with pytest.raises(ValueError):
        Experiment(
            name="test_exp",
            id=1,
            desc="Test experiment",
            env=env
        )

def test_experiment_config_paths(experiment):
    """Test that experiment config paths are correct"""
    assert os.path.exists(os.path.join(experiment.config_dir_path, Experiment.CONFIG_FILE))
    assert os.path.exists(os.path.join(experiment.config_dir_path, Experiment.BASE_CONFIG))
    assert os.path.exists(os.path.join(experiment.config_dir_path, Experiment.TRIALS_CONFIG))
