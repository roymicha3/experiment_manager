import os
import pytest
import sqlite3
from omegaconf import OmegaConf

from experiment_manager.experiment import Experiment, ConfigPaths
from experiment_manager.environment import Environment
from tests.pipelines.dummy_pipeline_factory import DummyPipelineFactory


def get_db_connection(experiment_path):
    """Get connection to the experiment database"""
    db_path = os.path.join(experiment_path, "artifacts", "experiment.db")
    return sqlite3.connect(db_path)


@pytest.fixture
def config_dir():
    """Get the test experiment config directory"""
    return os.path.join("tests", "configs", "test_experiment")


@pytest.fixture
def prepare_env(tmp_path, config_dir):
    """Prepare environment configuration"""
    env_path = os.path.join(config_dir, ConfigPaths.ENV_CONFIG.value)
    env = OmegaConf.load(env_path)
    env.workspace = os.path.join(tmp_path, "test_outputs")
    OmegaConf.save(env, env_path)
    return env


def setup_and_run_test_experiment(tmp_path, config_dir):
    """Helper to set up and run a test experiment"""
    experiment = Experiment.create(config_dir, DummyPipelineFactory)
    experiment.run()
    return experiment


def test_experiment_structure_after_run(tmp_path, prepare_env, config_dir):
    """Test that experiment records are properly structured in DB after a run"""
    # Run experiment
    experiment = setup_and_run_test_experiment(tmp_path, config_dir)
    
    # Print workspace and check if database exists
    print(f"\nWorkspace: {experiment.env.workspace}")
    db_path = os.path.join(experiment.env.workspace, "artifacts", "experiment.db")
    print(f"DB path: {db_path}")
    print(f"DB exists: {os.path.exists(db_path)}")
    
    # Get DB connection
    db = get_db_connection(experiment.env.workspace)
    cursor = db.cursor()
    
    # Test experiment record exists
    cursor.execute("SELECT * FROM EXPERIMENT")
    exp_record = cursor.fetchone()
    assert exp_record is not None, "Experiment record not found"

    assert exp_record[1] == "test_experiment", "Experiment name does not match"
    
    # Test trials are linked correctly
    cursor.execute("SELECT * FROM TRIAL")
    trials = cursor.fetchall()
    assert len(trials) > 0
    
    # Test trial runs exist for each trial
    for trial in trials:
        trial_id = trial[0]
        cursor.execute("SELECT COUNT(*) FROM TRIAL_RUN WHERE trial_id = ?", (trial_id,))
        
        run_count = cursor.fetchone()[0]
        assert run_count in [2, 5]


def test_metric_hierarchy(tmp_path, prepare_env, config_dir):
    """Test that metrics are properly associated with correct trial runs"""
    experiment = setup_and_run_test_experiment(tmp_path, config_dir)
    db = get_db_connection(experiment.env.workspace)
    cursor = db.cursor()
    
    # Get all metrics with their hierarchy
    cursor.execute("""
        SELECT m.*, tr.trial_id
        FROM METRIC m
        JOIN TRIAL_RUN tr ON m.trial_run_id = tr.id
    """)
    metrics = cursor.fetchall()
    
    assert len(metrics) > 0
    
    # Check metric values follow expected patterns
    for metric in metrics:
        value = metric[3]  # value column
        name = metric[2]   # name column
        if name == 'test_acc':
            assert 0 <= value <= 1
        elif name == 'test_loss':
            assert value >= 0


def test_artifact_references(tmp_path, prepare_env, config_dir):
    """Test that artifacts are properly referenced in DB"""
    experiment = setup_and_run_test_experiment(tmp_path, config_dir)
    db = get_db_connection(experiment.env.workspace)
    cursor = db.cursor()
    
    # Check artifacts exist and are linked
    cursor.execute("""
        SELECT a.*, tr.trial_id
        FROM ARTIFACT a
        JOIN TRIAL_RUN tr ON a.trial_run_id = tr.id
    """)
    artifacts = cursor.fetchall()
    
    for artifact in artifacts:
        path = artifact[2]  # path column
        # Verify artifact belongs to correct hierarchy
        assert path.startswith(experiment.env.workspace)


def test_data_consistency(tmp_path, prepare_env, config_dir):
    """Test referential integrity and data consistency in DB"""
    experiment = setup_and_run_test_experiment(tmp_path, config_dir)
    db = get_db_connection(experiment.env.workspace)
    cursor = db.cursor()
    
    # Test no orphaned trials
    cursor.execute("""
        SELECT COUNT(*) FROM TRIAL t 
        LEFT JOIN experiments e ON t.experiment_id = e.id
        WHERE e.id IS NULL
    """)
    assert cursor.fetchone()[0] == 0
    
    # Test no orphaned runs
    cursor.execute("""
        SELECT COUNT(*) FROM trial_runs tr
        LEFT JOIN trials t ON tr.trial_id = t.id
        WHERE t.id IS NULL
    """)
    assert cursor.fetchone()[0] == 0
    
    # Test metric counts match expectations
    cursor.execute("SELECT * FROM trials")
    for trial in cursor.fetchall():
        trial_id = trial[0]
        repeat = trial[3]  # repeat count
        
        # Count metrics for this trial
        cursor.execute("""
            SELECT COUNT(*) FROM metrics m
            JOIN trial_runs tr ON m.trial_run_id = tr.id
            WHERE tr.trial_id = ?
        """, (trial_id,))
        metrics_count = cursor.fetchone()[0]
        
        # Each trial run should have metrics for each epoch
        # In DummyPipeline: 2 metrics (acc, loss) * 2 phases (train, val) * epochs
        epochs = 3  # from test experiment config
        expected_count = repeat * epochs * 4  # 4 metrics per epoch
        assert metrics_count >= expected_count  # >= because we also have final test metrics
