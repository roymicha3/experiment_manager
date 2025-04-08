import os
import pytest
from pathlib import Path
import json
from omegaconf import OmegaConf

from experiment_manager.common.common import Level, Metric
from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment
from experiment_manager.trackers.log_tracker import LogTracker
from experiment_manager.trackers.db_tracker import DBTracker
from tests.pipelines.simple_classifier import SimpleClassifierPipeline
from tests.pipelines.dummy_pipeline_factory import DummyPipelineFactory

@pytest.fixture
def config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "simple_classifier.yaml")
    return OmegaConf.load(config_path)

@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path / "workspace")

def test_simple_classifier_single_run(workspace, config):
    """Test single run of simple classifier with both trackers."""
    # Initialize environment with config
    env_config = OmegaConf.create({
        "workspace": workspace,
        "verbose": True,
        "trackers": [
            {
                "type": "LogTracker",
                "name": "test_tracker.log",
                "verbose": True
            },
            {
                "type": "DBTracker",
                "name": "test_tracker.db",
                "recreate": True
            }
        ]
    })
    
    # Create experiment
    experiment_config = OmegaConf.create({
        "name": "test_simple_classifier",
        "id": 1,
        "desc": "Test Simple Classifier",
        "settings": config.pipeline,
        "config_dir_path": os.path.join(os.path.dirname(__file__), "configs", "test_simple_classifier"),
        "trials": [{
            "name": "single_run",
            "settings": {},
            "pipeline": "SimpleClassifierPipeline"
        }]
    })
    
    # Create and run experiment
    env = Environment(workspace=workspace, config=env_config, factory=DummyPipelineFactory)
    env.setup_environment()
    
    experiment = Experiment.from_config(experiment_config, env)
    experiment.run()
    
    # Verify log file contains experiment title and trial name
    log_path = os.path.join(workspace, next(t["name"] for t in env_config.trackers if t["type"] == "LogTracker"))
    assert os.path.exists(log_path)
    with open(log_path, 'r') as f:
        content = f.read()
        assert experiment_config.name in content
        assert experiment_config.trials[0].name in content
    
    # Verify metrics in database
    db_tracker = next(t for t in env.tracker_manager.trackers if isinstance(t, DBTracker))
    
    # Verify experiment was created
    experiments = db_tracker.db_manager._execute_query(
        "SELECT title FROM EXPERIMENT"
    ).fetchall()
    assert len(experiments) > 0
    assert any(e[0] == experiment_config.name for e in experiments)
    
    # Verify trial was created
    trials = db_tracker.db_manager._execute_query(
        "SELECT name FROM TRIAL"
    ).fetchall()
    assert len(trials) > 0
    assert any(t[0] == experiment_config.trials[0].name for t in trials)
    
    # Verify metrics were recorded
    metrics = db_tracker.db_manager._execute_query(
        "SELECT type FROM METRIC"
    ).fetchall()
    assert len(metrics) > 0
    assert any(m[0] == Metric.TEST_ACC.value for m in metrics)

def test_simple_classifier_experiment(workspace, config):
    """Test running multiple trials of the simple classifier experiment."""
    # Initialize environment with config
    env_config = OmegaConf.create({
        "workspace": workspace,
        "verbose": True,
        "trackers": [
            {
                "type": "LogTracker",
                "name": "test_tracker.log",
                "verbose": True
            },
            {
                "type": "DBTracker",
                "name": "test_tracker.db",
                "recreate": True
            }
        ]
    })
    
    # Create experiment
    experiment_config = OmegaConf.create({
        "name": "test_simple_classifier_multi",
        "id": 2,
        "desc": "Test Simple Classifier with Multiple Trials",
        "settings": config.pipeline,
        "config_dir_path": os.path.join(os.path.dirname(__file__), "configs", "test_simple_classifier_multi"),
        "trials": [
            {
                "name": "more_samples",
                "settings": {"n_samples": 2000},
                "pipeline": "SimpleClassifierPipeline"
            },
            {
                "name": "fewer_features",
                "settings": {"n_features": 10},
                "pipeline": "SimpleClassifierPipeline"
            }
        ]
    })
    
    # Create and run experiment
    env = Environment(workspace=workspace, config=env_config, factory=DummyPipelineFactory)
    env.setup_environment()
    
    experiment = Experiment.from_config(experiment_config, env)
    experiment.run()
    
    # Verify log file contains experiment title and trial names
    log_path = os.path.join(workspace, next(t["name"] for t in env_config.trackers if t["type"] == "LogTracker"))
    assert os.path.exists(log_path)
    with open(log_path, 'r') as f:
        content = f.read()
        assert experiment_config.name in content
        for trial in experiment_config.trials:
            assert trial.name in content
    
    # Verify metrics in database
    db_tracker = next(t for t in env.tracker_manager.trackers if isinstance(t, DBTracker))
    
    # Verify experiment was created
    experiments = db_tracker.db_manager._execute_query(
        "SELECT title FROM EXPERIMENT"
    ).fetchall()
    assert len(experiments) > 0
    assert any(e[0] == experiment_config.name for e in experiments)
    
    # Verify trials were created
    trials = db_tracker.db_manager._execute_query(
        "SELECT name FROM TRIAL"
    ).fetchall()
    assert len(trials) == len(experiment_config.trials)
    trial_names = [t[0] for t in trials]
    assert all(t.name in trial_names for t in experiment_config.trials)
    
    # Verify metrics were recorded
    metrics = db_tracker.db_manager._execute_query(
        "SELECT type FROM METRIC WHERE type = ?",
        (Metric.TEST_ACC.value,)
    ).fetchall()
    assert len(metrics) >= len(experiment_config.trials)
