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
from experiment_manager.pipelines.pipeline_factory import PipelineFactory

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
        "verbose": False,
        "trackers": [
            {
                "type": "LogTracker",
                "name": "test_tracker.log",
                "verbose": False
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
            "id": 1,
            "repeat": 1,
            "settings": config.pipeline,
            "pipeline": {
                "type": "SimpleClassifierPipeline"
            }
        }]
    })
    
    # Create and run experiment
    env = Environment(workspace=workspace, config=env_config, factory=PipelineFactory)
    env.setup_environment()
    
    experiment = Experiment.from_config(experiment_config, env)
    experiment.run()
    
    # Verify log file contains experiment title and trial name
    for tracker in experiment.env.tracker_manager.trackers:
        if isinstance(tracker, LogTracker):
            log_path = tracker.log_path
            assert os.path.exists(log_path), f"Log file {log_path} does not exist"
            with open(log_path, 'r') as f:
                content = f.read()
                assert experiment_config.name in content, f"Log file {log_path} does not contain experiment name: {experiment_config.name}"
    
    # Verify metrics in database
    db_tracker = next(t for t in env.tracker_manager.trackers if isinstance(t, DBTracker))
    
    # print the entire content of the DB
    print("\n\n\n")
    print("##################### Experiment Table: #####################")
    print(db_tracker.db_manager._execute_query("SELECT * FROM EXPERIMENT").fetchall())
    print("##################### Trial Table: #####################")
    print(db_tracker.db_manager._execute_query("SELECT * FROM TRIAL").fetchall())
    print("##################### Metric Table: #####################")
    print(db_tracker.db_manager._execute_query("SELECT * FROM METRIC").fetchall())
    print("##################### Trial Run Table: #####################")
    print(db_tracker.db_manager._execute_query("SELECT * FROM TRIAL_RUN").fetchall())
    print("##################### Epoch Table: #####################")
    print(db_tracker.db_manager._execute_query("SELECT * FROM EPOCH").fetchall())
    print("\n\n\n")

    # Verify experiment was created
    experiments = db_tracker.db_manager._execute_query(
        "SELECT title FROM EXPERIMENT"
    ).fetchall()
    assert len(experiments) > 0, f"Expected at least one experiment, found {len(experiments)}"
    assert any(e[0] == experiment_config.name for e in experiments), f"Expected experiment {experiment_config.name}, found {', '.join(e[0] for e in experiments)}"
    
    # Verify trial was created
    trials = db_tracker.db_manager._execute_query(
        "SELECT name FROM TRIAL"
    ).fetchall()
    assert len(trials) > 0, f"Expected at least one trial, found {len(trials)}"
    assert any(t[0] == experiment_config.trials[0].name for t in trials), f"Expected trial {experiment_config.trials[0].name}, found {', '.join(t[0] for t in trials)}"
    
    # Verify metrics were recorded
    metrics = db_tracker.db_manager._execute_query(
        "SELECT type FROM METRIC"
    ).fetchall()
    assert len(metrics) > 0, f"Expected at least one metric, found {len(metrics)}"
    assert any(m[0] == Metric.TEST_ACC.value for m in metrics), f"Expected metric {Metric.TEST_ACC.value}, found {', '.join(m[0] for m in metrics)}"

def test_simple_classifier_experiment(workspace, config):
    """Test running multiple trials of the simple classifier experiment."""
    # Initialize environment with config
    env_config = OmegaConf.create({
        "workspace": workspace,
        "verbose": False,
        "trackers": [
            {
                "type": "LogTracker",
                "name": "test_tracker.log",
                "verbose": False
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
                "id": 1,
                "repeat": 1,
                "settings": {
                    **config.pipeline,
                    "n_samples": 2000
                },
                "pipeline": {
                    "type": "SimpleClassifierPipeline"
                }
            },
            {
                "name": "fewer_features",
                "id": 2,
                "repeat": 1,
                "settings": {
                    **config.pipeline,
                    "n_features": 10
                },
                "pipeline": {
                    "type": "SimpleClassifierPipeline"
                }
            }
        ]
    })
    
    # Create and run experiment
    env = Environment(workspace=workspace, config=env_config, factory=PipelineFactory)
    env.setup_environment()
    
    experiment = Experiment.from_config(experiment_config, env)
    experiment.run()
    
    # Verify log file contains experiment title and trial names
    for tracker in experiment.env.tracker_manager.trackers:
        if isinstance(tracker, LogTracker):
            log_path = tracker.log_path
            assert os.path.exists(log_path)
            with open(log_path, 'r') as f:
                content = f.read()
                assert experiment_config.name in content, f"Log file does not contain experiment name: {experiment_config.name}"
                trial_names = [trial.name for trial in experiment_config.trials]
                assert all(name in content.strip("\"'()[]{}") for name in trial_names), \
                    f"Log file does not contain all trial names: {', '.join(trial_names)}"
    
    # Verify metrics in database
    db_tracker = next(t for t in env.tracker_manager.trackers if isinstance(t, DBTracker))
    
    # Verify experiment was created
    experiments = db_tracker.db_manager._execute_query(
        "SELECT title FROM EXPERIMENT"
    ).fetchall()
    assert len(experiments) > 0
    assert any(e[0] == experiment_config.name for e in experiments)

    # print the entire content of the DB
    print("\n\n\n")
    print("##################### Experiment Table: #####################")
    print(db_tracker.db_manager._execute_query("SELECT * FROM EXPERIMENT").fetchall())
    print("##################### Trial Table: #####################")
    print(db_tracker.db_manager._execute_query("SELECT * FROM TRIAL").fetchall())
    print("##################### Metric Table: #####################")
    print(db_tracker.db_manager._execute_query("SELECT * FROM METRIC").fetchall())
    print("##################### Trial Run Table: #####################")
    print(db_tracker.db_manager._execute_query("SELECT * FROM TRIAL_RUN").fetchall())
    print("##################### Epoch Table: #####################")
    print(db_tracker.db_manager._execute_query("SELECT * FROM EPOCH").fetchall())
    print("\n\n\n")

    # Verify trials were created
    trials = db_tracker.db_manager._execute_query(
        "SELECT name FROM TRIAL"
    ).fetchall()
    assert len(trials) == len(experiment_config.trials), f"Expected {len(experiment_config.trials)} trials, found {len(trials)}"
    trial_names = [t[0] for t in trials]
    assert all(t.name in trial_names for t in experiment_config.trials), f"Expected {experiment_config.trials} in trials, found {trial_names}"
    
    # Verify metrics were recorded
    metrics = db_tracker.db_manager._execute_query(
        "SELECT type FROM METRIC WHERE type = ?",
        (Metric.TEST_ACC.value,)
    ).fetchall()
    assert len(metrics) >= len(experiment_config.trials), f"Expected {len(experiment_config.trials)} metrics, found {len(metrics)}"
