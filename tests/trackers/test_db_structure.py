import os
import pytest
import sqlite3
from omegaconf import OmegaConf

from experiment_manager.experiment import Experiment, ConfigPaths
from tests.pipelines.dummy_pipeline_factory import DummyPipelineFactory


def get_db_connection(experiment_path):
    """Get connection to the experiment database"""
    db_path = os.path.join(experiment_path, "artifacts", "experiment.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


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
    
    # Get all metrics with their hierarchy through RESULTS_METRIC
    cursor.execute("""
        SELECT m.*, tr.trial_id
        FROM METRIC m
        JOIN RESULTS_METRIC rm ON m.id = rm.metric_id
        JOIN RESULTS r ON rm.results_id = r.trial_run_id
        JOIN TRIAL_RUN tr ON r.trial_run_id = tr.id
    """)
    results_metrics = cursor.fetchall()
    
    # Get all metrics with their hierarchy through EPOCH_METRIC
    cursor.execute("""
        SELECT m.*, tr.trial_id, e.idx as epoch
        FROM METRIC m
        JOIN EPOCH_METRIC em ON m.id = em.metric_id
        JOIN EPOCH e ON em.epoch_idx = e.idx AND em.epoch_trial_run_id = e.trial_run_id
        JOIN TRIAL_RUN tr ON e.trial_run_id = tr.id
    """)
    epoch_metrics = cursor.fetchall()
    
    assert len(results_metrics) > 0 or len(epoch_metrics) > 0
    
    # Check metric values follow expected patterns
    for metric in results_metrics + epoch_metrics:
        value = float(metric['total_val'])
        type_name = metric['type']
        if type_name in ['TEST_ACC', 'TRAIN_ACC', 'VAL_ACC']:
            assert 0 <= value <= 1, f"Invalid {type_name} value: {value}"
        elif type_name in ['TEST_LOSS', 'TRAIN_LOSS', 'VAL_LOSS']:
            assert value >= 0, f"Invalid {type_name} value: {value}"


def test_artifact_references(tmp_path, prepare_env, config_dir):
    """Test that artifacts are properly referenced in DB"""
    experiment = setup_and_run_test_experiment(tmp_path, config_dir)
    db = get_db_connection(experiment.env.workspace)
    cursor = db.cursor()
    
    # Check artifacts exist and are linked through various tables
    cursor.execute("""
        SELECT DISTINCT a.*, tr.trial_id
        FROM ARTIFACT a
        LEFT JOIN EXPERIMENT_ARTIFACT ea ON a.id = ea.artifact_id
        LEFT JOIN TRIAL_ARTIFACT ta ON a.id = ta.artifact_id
        LEFT JOIN TRIAL_RUN_ARTIFACT tra ON a.id = tra.artifact_id
        LEFT JOIN TRIAL_RUN tr ON tra.trial_run_id = tr.id
        LEFT JOIN RESULTS_ARTIFACT ra ON a.id = ra.artifact_id
        LEFT JOIN RESULTS r ON ra.results_id = r.trial_run_id
        LEFT JOIN EPOCH_ARTIFACT epa ON a.id = epa.artifact_id
        LEFT JOIN EPOCH e ON epa.epoch_idx = e.idx AND epa.epoch_trial_run_id = e.trial_run_id
    """)
    artifacts = cursor.fetchall()
    
    # Verify any found artifacts belong to correct hierarchy
    for artifact in artifacts:
        path = artifact['loc']  # location column
        assert path.startswith(experiment.env.workspace)


@pytest.fixture
def test_db(tmp_path, prepare_env, config_dir):
    """Setup test database and return connection"""
    experiment = setup_and_run_test_experiment(tmp_path, config_dir)
    db = get_db_connection(experiment.env.workspace)
    yield db, experiment
    db.close()


def test_no_orphaned_trials(test_db):
    """Test that all trials are linked to an experiment"""
    db, _ = test_db
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as cnt FROM TRIAL t 
        LEFT JOIN EXPERIMENT e ON t.experiment_id = e.id
        WHERE e.id IS NULL
    """)
    assert cursor.fetchone()['cnt'] == 0


def test_no_orphaned_trial_runs(test_db):
    """Test that all trial runs are linked to a trial"""
    db, _ = test_db
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as cnt FROM TRIAL_RUN tr
        LEFT JOIN TRIAL t ON tr.trial_id = t.id
        WHERE t.id IS NULL
    """)
    assert cursor.fetchone()['cnt'] == 0


def test_no_orphaned_epochs(test_db):
    """Test that all epochs are linked to a trial run"""
    db, _ = test_db
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as cnt FROM EPOCH e
        LEFT JOIN TRIAL_RUN tr ON e.trial_run_id = tr.id
        WHERE tr.id IS NULL
    """)
    assert cursor.fetchone()['cnt'] == 0


def test_no_orphaned_results(test_db):
    """Test that all results are linked to a trial run"""
    db, _ = test_db
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as cnt FROM RESULTS r
        LEFT JOIN TRIAL_RUN tr ON r.trial_run_id = tr.id
        WHERE tr.id IS NULL
    """)
    assert cursor.fetchone()['cnt'] == 0


def test_no_orphaned_metrics(test_db):
    """Test that all metrics are linked to either an epoch or results"""
    db, _ = test_db
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as cnt FROM METRIC m
        LEFT JOIN EPOCH_METRIC em ON m.id = em.metric_id
        LEFT JOIN RESULTS_METRIC rm ON m.id = rm.metric_id
        WHERE em.metric_id IS NULL AND rm.metric_id IS NULL
    """)
    assert cursor.fetchone()['cnt'] == 0


def test_no_orphaned_artifacts(test_db):
    """Test that all artifacts are linked to an experiment, trial, trial run, or epoch"""
    db, _ = test_db
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as cnt FROM ARTIFACT a
        LEFT JOIN EXPERIMENT_ARTIFACT ea ON a.id = ea.artifact_id
        LEFT JOIN TRIAL_ARTIFACT ta ON a.id = ta.artifact_id
        LEFT JOIN TRIAL_RUN_ARTIFACT tra ON a.id = tra.artifact_id
        LEFT JOIN EPOCH_ARTIFACT epa ON a.id = epa.artifact_id
        WHERE ea.artifact_id IS NULL
            AND ta.artifact_id IS NULL
            AND tra.artifact_id IS NULL
            AND epa.artifact_id IS NULL
    """)
    assert cursor.fetchone()['cnt'] == 0


def test_artifact_paths(test_db):
    """Test that all artifact paths are within the experiment workspace"""
    db, experiment = test_db
    cursor = db.cursor()
    
    cursor.execute("SELECT loc FROM ARTIFACT")
    artifacts = cursor.fetchall()
    
    for artifact in artifacts:
        path = artifact['loc']  # location column
        assert path.startswith(experiment.env.workspace)


def test_checkpoint_artifact_count_matches_files(test_db):
    """Test that the number of checkpoint artifacts in the DB matches the number of checkpoint files on disk."""
    db, experiment = test_db
    cursor = db.cursor()
    # Find all checkpoint artifacts in the DB
    cursor.execute("SELECT loc FROM ARTIFACT WHERE type = 'checkpoint'")
    db_checkpoints = set([row['loc'] for row in cursor.fetchall()])
    # Find all checkpoint files on disk
    checkpoint_files = set()
    for root, dirs, files in os.walk(experiment.env.workspace):
        for f in files:
            if "checkpoint" in f:
                checkpoint_files.add(os.path.join(root, f))
    assert len(db_checkpoints) == len(checkpoint_files), (
        f"Checkpoint artifact count mismatch: DB has {len(db_checkpoints)}, files found {len(checkpoint_files)}."
    )
    # Optionally, check that all DB artifact paths exist as files
    for path in db_checkpoints:
        assert os.path.exists(path), f"Checkpoint artifact in DB does not exist as file: {path}"


def test_epoch_metrics(test_db):
    """Test that each trial run has the correct number of epoch metrics"""
    db, _ = test_db
    cursor = db.cursor()
    
    for trial_run in cursor.execute("SELECT id, trial_id FROM TRIAL_RUN ORDER BY trial_id, id").fetchall():
        run_id = trial_run['id']
        trial_id = trial_run['trial_id']
        
        # Get unique epoch indices
        cursor.execute("""
            SELECT DISTINCT e.idx as epoch_idx
            FROM EPOCH e
            WHERE e.trial_run_id = ?
            ORDER BY e.idx
        """, (run_id,))
        epochs = cursor.fetchall()
        print(f"\nTrial {trial_id} Run {run_id} epochs: {[e['epoch_idx'] for e in epochs]}")
        
        # Get epoch metrics
        cursor.execute("""
            SELECT e.idx as epoch_idx, m.type as metric_type, m.total_val
            FROM METRIC m
            JOIN EPOCH_METRIC em ON m.id = em.metric_id
            JOIN EPOCH e ON em.epoch_idx = e.idx AND em.epoch_trial_run_id = e.trial_run_id
            WHERE e.trial_run_id = ?
            ORDER BY e.idx, m.type
        """, (run_id,))
        epoch_metrics = cursor.fetchall()
        
        print(f"\nTrial {trial_id} Run {run_id}:")
        print("Epoch metrics:")
        for metric in epoch_metrics:
            print(f"  Epoch {metric['epoch_idx']}: {metric['metric_type']} = {metric['total_val']}")
        
        # Each trial run should have metrics for each epoch
        # In DummyPipeline: 2 metrics (acc, loss) * 2 phases (train, val) * epochs
        epochs = 3  # from test experiment config
        expected_epoch_count = epochs * 4  # 4 metrics per epoch (train_acc, train_loss, val_acc, val_loss)
        expected_results_count = 2  # final test metrics (acc, loss)
        
        # Assert we have the expected number of metrics for this trial run
        assert len(epoch_metrics) == expected_epoch_count, \
            f"Trial {trial_id} Run {run_id}: Expected {expected_epoch_count} epoch metrics but got {len(epoch_metrics)}"
        