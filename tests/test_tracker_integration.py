import os
import pytest
from omegaconf import OmegaConf, DictConfig

from experiment_manager.environment import Environment
from experiment_manager.common.common import Metric
from experiment_manager.trackers.log_tracker import LogTracker
from tests.pipelines.dummy_pipeline import DummyPipeline
from tests.pipelines.dummy_pipeline_factory import DummyPipelineFactory

@pytest.fixture
def env_config():
    return OmegaConf.create({
        "workspace": "test_workspace",
        "settings": {
            "debug": True,
            "verbose": True,
            "epochs": 3,
            "batch_size": 32
        },
        "trackers": [
            {
                "type": "LogTracker",
                "name": "test_tracker.log",
                "verbose": True
            }
        ]
    })

@pytest.fixture
def experiment_config():
    return OmegaConf.create({
        "name": "test_experiment",
        "id": 1,
        "desc": "Test experiment for tracking",
        "settings": {
            "param1": "value1"
        }
    })

@pytest.fixture
def trials_config():
    return OmegaConf.create([
        {
            "name": "trial_1",
            "id": 1,
            "repeat": 2,
            "settings": {
                "learning_rate": 0.001
            },
            "pipeline": {
                "type": "DummyPipeline",
                "settings": {
                    "epochs": 3
                }
            }
        },
        {
            "name": "trial_2",
            "id": 2,
            "repeat": 1,
            "settings": {
                "learning_rate": 0.01
            },
            "pipeline": {
                "type": "DummyPipeline",
                "settings": {
                    "epochs": 3
                }
            }
        }
    ])

@pytest.fixture
def env(env_config, tmp_path):
    workspace = os.path.join(str(tmp_path), env_config.workspace)
    env = Environment(
        workspace=workspace,
        config=env_config,
        factory=DummyPipelineFactory,
        verbose=True
    )
    env.setup_environment()
    return env

def test_basic_log_tracker(env):
    """Test that individual metrics are properly logged to file"""
    # Track some metrics
    metrics = [
        ("test_accuracy", 0.95, Metric.TEST_ACC),
        ("test_loss", 0.15, Metric.TEST_LOSS),
        ("val_accuracy", 0.92, Metric.VAL_ACC)
    ]
    
    for i, (name, value, metric_type) in enumerate(metrics):
        env.tracker_manager.track(
            metric=metric_type,
            value=value,
            step=i
        )
    
    # Verify log tracker setup
    tracker = env.tracker_manager.trackers[0]
    assert isinstance(tracker, LogTracker)
    assert tracker.name == "test_tracker.log"
    assert tracker.verbose == True
    
    # Verify log file exists and contains metrics
    log_path = os.path.join(tracker.workspace, tracker.name)
    assert os.path.exists(log_path)
    
    with open(log_path, 'r') as f:
        content = f.read()
        for name, value, metric_type in metrics:
            assert metric_type.name in content
            assert str(value) in content
            assert str(metric_type.value) in content

def test_dummy_pipeline(env):
    """Test tracking metrics through a dummy pipeline run"""
    # Create pipeline config
    pipeline_config = DictConfig({
        'type': 'DummyPipeline',
        'epochs': 5,
        'batch_size': 32,
        'device': 'cpu'
    })
    
    # Create and run pipeline
    pipeline = env.factory.create(
        name=pipeline_config.type,
        config=pipeline_config,
        env=env,
        id=1
    )
    status = pipeline.run(pipeline_config)
    assert status == "completed"
    
    # Verify log tracker
    tracker = env.tracker_manager.trackers[0]
    assert isinstance(tracker, LogTracker)
    
    # Verify log file exists and contains metrics
    log_path = os.path.join(tracker.workspace, tracker.name)
    assert os.path.exists(log_path)
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        # Check training metrics
        for epoch in range(pipeline_config.epochs):
            # Training metrics
            train_acc = 0.5 + (epoch * 0.05)
            train_loss = 1.0 - (epoch * 0.1)
            assert str(round(train_acc, 2)) in content
            assert str(round(train_loss, 2)) in content
            
            # Validation metrics
            val_acc = train_acc - 0.05
            val_loss = train_loss + 0.1
            assert str(round(val_acc, 2)) in content
            assert str(round(val_loss, 2)) in content
        
        # Check final test metrics
        final_train_acc = 0.5 + ((pipeline_config.epochs - 1) * 0.05)
        final_test_acc = final_train_acc + 0.02
        final_train_loss = 1.0 - ((pipeline_config.epochs - 1) * 0.1)
        final_test_loss = final_train_loss - 0.05
        
        assert str(round(final_test_acc, 2)) in content
        assert str(round(final_test_loss, 2)) in content
        
        # Check metric names
        assert Metric.TEST_ACC.name in content
        assert Metric.TEST_LOSS.name in content
        assert Metric.VAL_ACC.name in content
        assert Metric.VAL_LOSS.name in content
