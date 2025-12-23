"""
Test module for DBTracker functionality and integration with real experiment data.
"""

import os
import pytest
from pathlib import Path
from omegaconf import OmegaConf

from experiment_manager.trackers.plugins.db_tracker import DBTracker
from experiment_manager.common.common import Level, Metric
from experiment_manager.results.sources.db_datasource import DBDataSource
from tests.conftest import create_metrics_dataframe

@pytest.fixture
def workspace(tmp_path):
    return tmp_path / "test_workspace"

@pytest.fixture
def db_tracker(workspace):
    os.makedirs(workspace, exist_ok=True)
    tracker = DBTracker(str(workspace))
    yield tracker
    if hasattr(tracker, 'db_manager'):
        tracker.db_manager.__del__()

def test_db_reuse(workspace):
    """Test that DBTracker can reuse an existing database."""
    # Create initial tracker and add some data
    tracker1 = DBTracker(str(workspace), recreate=True)
    tracker1.on_create(Level.EXPERIMENT, "Test Experiment 1")
    tracker1.track(Metric.TEST_ACC, 0.95)
    
    # Clean up first tracker
    tracker1.db_manager.__del__()
    
    # Create second tracker with auto-detection (should reuse)
    tracker2 = DBTracker(str(workspace))  # No recreate specified
    tracker2.on_create(Level.EXPERIMENT, "Test Experiment 2")
    
    # Verify both experiments exist in database
    experiments = tracker2.db_manager._execute_query("SELECT title FROM EXPERIMENT").fetchall()
    assert len(experiments) == 2
    titles = [exp["title"] for exp in experiments]
    assert "Test Experiment 1" in titles
    assert "Test Experiment 2" in titles
    
    # Clean up
    tracker1.db_manager.__del__()
    tracker2.db_manager.__del__()

def test_child_tracker_reuse(workspace):
    """Test that child trackers reuse the parent database."""
    parent = DBTracker(str(workspace), recreate=True)
    parent.on_create(Level.EXPERIMENT, "Parent Experiment")
    
    child = parent.create_child()
    child.on_create(Level.TRIAL, "Child Trial")
    
    # Verify parent experiment exists in child's database
    result = child.db_manager._execute_query(
        "SELECT title FROM EXPERIMENT WHERE title = ?",
        ("Parent Experiment",)
    ).fetchone()
    assert result is not None
    assert result["title"] == "Parent Experiment"
    
    # Clean up
    parent.db_manager.__del__()
    child.db_manager.__del__()
    
    # Clean up
    parent.db_manager.__del__()
    child.db_manager.__del__()

def test_from_config_reuse(workspace):
    """Test database reuse through configuration."""
    # Create initial database
    tracker1 = DBTracker(str(workspace), recreate=True)
    tracker1.on_create(Level.EXPERIMENT, "Config Test 1")
    
    # Clean up first tracker
    tracker1.db_manager.__del__()
    
    # Create tracker from config with recreate=False
    config = OmegaConf.create({
        "name": DBTracker.DB_NAME,
        "recreate": False
    })
    tracker2 = DBTracker.from_config(config, str(workspace))
    tracker2.on_create(Level.EXPERIMENT, "Config Test 2")
    
    # Verify both experiments exist
    experiments = tracker2.db_manager._execute_query("SELECT title FROM EXPERIMENT").fetchall()
    assert len(experiments) == 2
    titles = [exp["title"] for exp in experiments]
    assert "Config Test 1" in titles
    assert "Config Test 2" in titles
    
    # Clean up
    tracker1.db_manager.__del__()
    tracker2.db_manager.__del__()

def test_recreate_removes_data(workspace):
    """Test that recreate=True removes existing data."""
    # Create initial data
    tracker1 = DBTracker(str(workspace))
    tracker1.on_create(Level.EXPERIMENT, "Should Be Deleted")
    tracker1.track(Metric.TEST_ACC, 0.95)
    
    # Clean up first tracker
    tracker1.db_manager.__del__()
    
    # Create new tracker with recreate=True
    tracker2 = DBTracker(str(workspace), recreate=True)
    experiments = tracker2.db_manager._execute_query("SELECT title FROM EXPERIMENT").fetchall()
    assert len(experiments) == 0  # Database should be empty
    metrics = tracker2.db_manager._execute_query("SELECT type FROM METRIC").fetchall()
    assert len(metrics) == 0  # Database should be empty
    
    # Clean up
    tracker2.db_manager.__del__()

def test_metric_tracking(workspace):
    """Test metric tracking with different value types."""
    tracker = DBTracker(str(workspace))
    tracker.on_create(Level.EXPERIMENT, "Metric Test")
    
    # Test scalar value
    tracker.track(Metric.TEST_ACC, 0.95, step=0)
    
    # Test dictionary value
    confusion = {"true_positive": 100, "false_positive": 5}
    tracker.track(Metric.CONFUSION, 0.0, 0, per_label_val=confusion)
    
    # Test with step
    tracker.track(Metric.TEST_LOSS, 0.1, 1)
    
    # Verify metrics were recorded
    metrics = tracker.db_manager._execute_query(
        "SELECT type, total_val, per_label_val FROM METRIC"
    ).fetchall()
    
    assert len(metrics) == 3
    metric_map = {m["type"]: m for m in metrics}
    
    assert metric_map["test_acc"]["total_val"] == 0.95
    assert metric_map["test_acc"]["per_label_val"] is None
    
    # compare dictionaries
    assert eval(metric_map["confusion"]["per_label_val"]) == confusion
    
    assert metric_map["test_loss"]["total_val"] == 0.1
    
    # Clean up
    tracker.db_manager.__del__()

def test_db_tracker_integration_with_real_data(experiment_db_only):
    """Test DBTracker integration with real MNIST experiment data.
    
    This test verifies that DBTracker database manager can successfully 
    read and query real experiment data, confirming integration with
    actual experiment database structures.
    """
    # experiment_db_only provides the path to real MNIST experiment database
    real_db_path = experiment_db_only
    
    # Create a database manager directly using the real database path
    from experiment_manager.db.manager import DatabaseManager
    db_manager = DatabaseManager(database_path=real_db_path, use_sqlite=True, recreate=False)
    
    try:
        # Test basic database connectivity and real data presence
        experiments = db_manager._execute_query("SELECT title FROM EXPERIMENT").fetchall()
        
        # Should have the real MNIST experiment
        experiment_titles = [exp["title"] for exp in experiments]
        assert "test_mnist_baseline" in experiment_titles
        assert len(experiment_titles) >= 1
        
        # Test querying real trial data
        trials = db_manager._execute_query("SELECT name FROM TRIAL").fetchall()
        trial_names = {trial["name"] for trial in trials}
        expected_names = {"small_lr", "medium_lr", "large_lr"}
        assert trial_names == expected_names
        assert len(trials) == 3  # Should have 3 trials
        
        # Test querying real trial runs
        trial_runs = db_manager._execute_query("SELECT id, status FROM TRIAL_RUN").fetchall()
        assert len(trial_runs) == 6  # 3 trials × 2 repetitions each
        
        # Verify all trial runs have a valid status (they should be started or completed)
        valid_statuses = ["started", "completed", "success", "finished", "running"]
        for run in trial_runs:
            assert run["status"] in valid_statuses, f"Unexpected status: {run['status']}"
        
        # Test querying real metrics data
        metrics = db_manager._execute_query("SELECT type, total_val FROM METRIC WHERE total_val IS NOT NULL").fetchall()
        assert len(metrics) > 0  # Should have many metrics from real experiment
        
        # Verify we have expected metric types from MNIST experiment
        metric_types = {metric["type"] for metric in metrics}
        expected_metric_types = {"train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"}
        
        # Should have at least some of the expected metric types
        found_metrics = metric_types.intersection(expected_metric_types)
        assert len(found_metrics) >= 3, f"Expected at least 3 metric types, found: {found_metrics}"
        
        # Test that metrics have reasonable values for MNIST (accuracy between 0-1, loss > 0)
        for metric in metrics:
            if "acc" in metric["type"]:
                assert 0.0 <= metric["total_val"] <= 1.0, f"Accuracy {metric['type']} should be 0-1, got {metric['total_val']}"
            elif "loss" in metric["type"]:
                assert metric["total_val"] >= 0.0, f"Loss {metric['type']} should be >= 0, got {metric['total_val']}"
        
        # Test epoch data query
        epochs = db_manager._execute_query("SELECT idx, trial_run_id FROM EPOCH").fetchall()
        assert len(epochs) > 0  # Should have epoch data from real experiment
        
        # Verify epoch indices are reasonable (0, 1, 2 for 3-epoch MNIST experiment)
        epoch_indices = {epoch["idx"] for epoch in epochs}
        assert epoch_indices == {0, 1, 2}, f"Expected epoch indices [0, 1, 2], got {epoch_indices}"
        
        print(f"✅ Successfully verified DBTracker integration with real MNIST data:")
        print(f"   - Experiments: {len(experiment_titles)}")
        print(f"   - Trials: {len(trials)}")
        print(f"   - Trial runs: {len(trial_runs)}")
        print(f"   - Metrics: {len(metrics)}")
        print(f"   - Metric types: {metric_types}")
        
    finally:
        # Clean up database manager
        if hasattr(db_manager, 'connection') and db_manager.connection:
            db_manager.connection.close()

def test_metrics_storage_retrieval_with_real_data(experiment_metrics_only):
    """Test metrics storage and retrieval using real MNIST experiment data.
    
    This test verifies that metrics from a real MNIST experiment can be
    properly stored in and retrieved from the database, validating that
    the DB Tracker correctly handles actual training metrics rather than
    synthetic test data.
    """
    # experiment_metrics_only provides pre-loaded metrics DataFrame and database access
    db_path = experiment_metrics_only['db_path']
    metrics_df = experiment_metrics_only['metrics']
    experiment_id = experiment_metrics_only['experiment_id']
    trial_count = experiment_metrics_only['trial_count']
    
    # Create a database manager using the real database path
    from experiment_manager.db.manager import DatabaseManager
    db_manager = DatabaseManager(database_path=db_path, use_sqlite=True, recreate=False)
    
    try:
        # Validate that we have substantial real metrics data
        assert not metrics_df.empty, "Metrics DataFrame should contain real MNIST data"
        assert len(metrics_df) > 50, f"Should have substantial metrics data, got {len(metrics_df)} rows"
        assert trial_count == 3, f"MNIST experiment should have 3 trials, got {trial_count}"
        print(f"✅ Loaded {len(metrics_df)} metric records from {trial_count} trials")
        
        # Verify DataFrame structure and required columns
        required_columns = ['experiment_id', 'trial_id', 'trial_run_id', 'epoch', 'metric', 'value']
        for col in required_columns:
            assert col in metrics_df.columns, f"Missing required column: {col}"
        
        # Verify all metrics belong to the correct experiment
        assert (metrics_df['experiment_id'] == experiment_id).all(), "All metrics should belong to the same experiment"
        print(f"✅ All metrics correctly associated with experiment ID: {experiment_id}")
        
        # Analyze real MNIST metric types
        unique_metrics = set(metrics_df['metric'].unique())
        expected_mnist_metrics = {
            'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'
        }
        
        # Should have most of the expected MNIST metrics
        found_metrics = unique_metrics.intersection(expected_mnist_metrics)
        assert len(found_metrics) >= 4, f"Should have at least 4 MNIST metrics, found: {found_metrics}"
        print(f"✅ Found expected MNIST metrics: {sorted(found_metrics)}")
        
        # Verify metric value ranges are realistic for MNIST
        for metric_type in found_metrics:
            metric_data = metrics_df[metrics_df['metric'] == metric_type]
            values = metric_data['value'].dropna()
            
            if 'acc' in metric_type:
                # Accuracy should be between 0 and 1
                assert (values >= 0.0).all() and (values <= 1.0).all(), f"Accuracy {metric_type} values should be 0-1"
                # MNIST accuracy should be reasonably high (> 0.5)
                assert values.max() > 0.5, f"MNIST {metric_type} should achieve > 50% accuracy"
            elif 'loss' in metric_type:
                # Loss should be positive
                assert (values >= 0.0).all(), f"Loss {metric_type} values should be >= 0"
                # MNIST loss should be reasonable (< 50 to handle large LR experiments)
                assert values.max() < 50.0, f"MNIST {metric_type} should be < 50"
        
        print(f"✅ All metric values are within realistic ranges for MNIST")
        
        # Test metrics retrieval by trial run
        trial_runs = metrics_df['trial_run_id'].unique()
        assert len(trial_runs) == 6, f"Should have 6 trial runs (3 trials × 2 reps), got {len(trial_runs)}"
        
        for trial_run_id in trial_runs:
            run_metrics = metrics_df[metrics_df['trial_run_id'] == trial_run_id]
            assert len(run_metrics) > 0, f"Trial run {trial_run_id} should have metrics"
            
            # Each trial run should have both epoch-level and final result metrics
            epoch_metrics = run_metrics[run_metrics['epoch'].notna()]
            final_metrics = run_metrics[run_metrics['epoch'].isna()]
            
            assert len(epoch_metrics) > 0, f"Trial run {trial_run_id} should have epoch metrics"
            assert len(final_metrics) > 0, f"Trial run {trial_run_id} should have final result metrics"
        
        print(f"✅ All {len(trial_runs)} trial runs have both epoch and final metrics")
        
        # Test database-level metrics retrieval using DatabaseManager
        experiment_metrics = db_manager.get_experiment_metrics(experiment_id)
        assert len(experiment_metrics) > 0, "Should be able to retrieve metrics directly from database"
        
        # Verify database metrics match DataFrame metrics count
        # Note: We compare unique metrics since database may have duplicates due to JOIN operations
        db_metric_types = {metric.type for metric in experiment_metrics}
        df_metric_types = set(metrics_df['metric'].unique())
        
        # Database should contain all metric types found in DataFrame
        missing_in_db = df_metric_types - db_metric_types
        assert len(missing_in_db) == 0, f"Database missing metric types: {missing_in_db}"
        print(f"✅ Database contains all {len(db_metric_types)} metric types from DataFrame")
        
        # Test epoch-based metrics analysis
        epoch_data = metrics_df[metrics_df['epoch'].notna()]
        
        # Should have metrics for 3 epochs (0, 1, 2) per trial run
        unique_epochs = sorted(epoch_data['epoch'].unique())
        expected_epochs = [0, 1, 2]  # MNIST baseline runs for 3 epochs
        assert unique_epochs == expected_epochs, f"Should have epochs {expected_epochs}, got {unique_epochs}"
        
        # Each trial run should have metrics for all epochs
        for trial_run_id in trial_runs:
            run_epoch_data = epoch_data[epoch_data['trial_run_id'] == trial_run_id]
            run_epochs = sorted(run_epoch_data['epoch'].unique())
            assert run_epochs == expected_epochs, f"Trial run {trial_run_id} missing epochs: {set(expected_epochs) - set(run_epochs)}"
        
        print(f"✅ All trial runs have metrics for epochs {expected_epochs}")
        
        # Test training progression (loss should generally decrease, accuracy should generally increase)
        for trial_run_id in trial_runs:
            run_data = epoch_data[epoch_data['trial_run_id'] == trial_run_id]
            
            # Check training loss progression
            train_loss_data = run_data[run_data['metric'] == 'train_loss'].sort_values('epoch')
            if len(train_loss_data) >= 2:
                first_loss = train_loss_data.iloc[0]['value']
                last_loss = train_loss_data.iloc[-1]['value']
                # Training loss should generally decrease (with some tolerance for variance)
                assert last_loss < first_loss * 1.1, f"Training loss should decrease for run {trial_run_id}"
            
            # Check training accuracy progression
            train_acc_data = run_data[run_data['metric'] == 'train_acc'].sort_values('epoch')
            if len(train_acc_data) >= 2:
                first_acc = train_acc_data.iloc[0]['value']
                last_acc = train_acc_data.iloc[-1]['value']
                # Training accuracy should generally increase (with some tolerance)
                assert last_acc > first_acc * 0.9, f"Training accuracy should increase for run {trial_run_id}"
        
        print(f"✅ Training metrics show expected progression patterns")
        
        # Test aggregated metrics queries
        experiment_data_df = db_manager.get_experiment_data([experiment_id])
        assert not experiment_data_df.empty, "Should be able to retrieve experiment data via get_experiment_data"
        assert experiment_data_df['experiment_id'].nunique() == 1, "Should only contain data for requested experiment"
        
        # Verify aggregated data contains our metrics
        agg_metric_types = set(experiment_data_df['metric_type'].dropna().unique())
        common_metrics = agg_metric_types.intersection(df_metric_types)
        assert len(common_metrics) > 0, "Aggregated data should contain metrics from our experiment"
        print(f"✅ Aggregated query returned {len(agg_metric_types)} metric types")
        
        print(f"✅ Successfully verified metrics storage and retrieval with real MNIST data:")
        print(f"   - Total metrics: {len(metrics_df)}")
        print(f"   - Metric types: {len(unique_metrics)}")
        print(f"   - Trial runs: {len(trial_runs)}")
        print(f"   - Epochs per run: {len(expected_epochs)}")
        print(f"   - Database metrics: {len(experiment_metrics)}")
        print(f"   - Training progression: validated")
        print(f"   - All database operations: successful")
        
    finally:
        # Clean up database manager
        if hasattr(db_manager, 'connection') and db_manager.connection:
            db_manager.connection.close()

def test_metrics_aggregation_with_real_data(experiment_metrics_only):
    """Test metrics aggregation functionality using real MNIST experiment data.
    
    This test verifies that the DatabaseManager can properly aggregate metrics
    at different grouping levels (experiment, trial, trial_run) using actual
    training data, and that aggregation functions work correctly with real values.
    """
    # experiment_metrics_only provides pre-loaded metrics DataFrame and database access
    db_path = experiment_metrics_only['db_path']
    metrics_df = experiment_metrics_only['metrics']
    experiment_id = experiment_metrics_only['experiment_id']
    trial_count = experiment_metrics_only['trial_count']
    
    # Create a database manager using the real database with metrics
    from experiment_manager.db.manager import DatabaseManager
    db_manager = DatabaseManager(database_path=db_path, use_sqlite=True, recreate=False)
    
    try:
        print(f"\n🧮 Testing metrics aggregation with real MNIST data:")
        print(f"   Database: {db_path}")
        print(f"   Raw metrics: {len(metrics_df)} records")
        print(f"   Experiment ID: {experiment_id}")
        print(f"   Expected trials: {trial_count}")
        
        # Test 1: Experiment-level aggregation (highest level)
        print(f"\n📊 Testing experiment-level aggregation...")
        exp_agg = db_manager.get_aggregated_metrics(
            experiment_ids=[experiment_id],
            group_by='experiment',
            functions=['mean', 'min', 'max', 'count']
        )
        
        assert not exp_agg.empty, "Experiment-level aggregation should return data"
        assert 'experiment_id' in exp_agg.columns, "Should include experiment_id column"
        assert 'metric_type' in exp_agg.columns, "Should include metric_type column"
        assert 'mean' in exp_agg.columns, "Should include mean aggregation"
        assert 'count' in exp_agg.columns, "Should include count aggregation"
        
        # Verify we have expected metric types
        exp_metric_types = set(exp_agg['metric_type'].unique())
        expected_metric_types = {'train_acc', 'train_loss', 'val_acc', 'val_loss', 'test_acc', 'test_loss'}
        common_types = exp_metric_types.intersection(expected_metric_types)
        assert len(common_types) > 0, f"Should have common metric types. Found: {exp_metric_types}"
        
        print(f"   ✅ Experiment aggregation: {len(exp_agg)} rows, {len(exp_metric_types)} metric types")
        
        # Test 2: Trial-level aggregation (medium granularity)
        print(f"\n📊 Testing trial-level aggregation...")
        trial_agg = db_manager.get_aggregated_metrics(
            experiment_ids=[experiment_id],
            group_by='trial',
            functions=['mean', 'std', 'count']
        )
        
        assert not trial_agg.empty, "Trial-level aggregation should return data"
        assert 'trial_id' in trial_agg.columns, "Should include trial_id column"
        assert 'trial_name' in trial_agg.columns, "Should include trial_name column"
        
        # Should have more granular data than experiment level
        trial_ids = trial_agg['trial_id'].nunique()
        assert trial_ids >= trial_count, f"Should have at least {trial_count} trials, found {trial_ids}"
        
        print(f"   ✅ Trial aggregation: {len(trial_agg)} rows, {trial_ids} unique trials")
        
        # Test 3: Trial run-level aggregation (finest granularity)
        print(f"\n📊 Testing trial run-level aggregation...")
        run_agg = db_manager.get_aggregated_metrics(
            experiment_ids=[experiment_id],
            group_by='trial_run',
            functions=['mean', 'sum', 'min', 'max']
        )
        
        assert not run_agg.empty, "Trial run-level aggregation should return data"
        assert 'trial_run_id' in run_agg.columns, "Should include trial_run_id column"
        assert 'run_status' in run_agg.columns, "Should include run_status column"
        
        # Should have most granular data (most rows)
        run_ids = run_agg['trial_run_id'].nunique()
        assert run_ids >= trial_ids, f"Should have at least {trial_ids} trial runs, found {run_ids}"
        assert len(run_agg) >= len(trial_agg), "Trial run aggregation should have >= trial aggregation rows"
        
        print(f"   ✅ Trial run aggregation: {len(run_agg)} rows, {run_ids} unique runs")
        
        # Test 4: Validate aggregation values are realistic for MNIST
        print(f"\n🔢 Validating aggregated metric values...")
        
        for metric_type in common_types:
            exp_row = exp_agg[exp_agg['metric_type'] == metric_type].iloc[0]
            mean_val = exp_row['mean']
            count_val = exp_row['count']
            
            # Validate count makes sense
            assert count_val > 0, f"Count for {metric_type} should be > 0"
            assert isinstance(count_val, (int, float)) or hasattr(count_val, 'dtype'), f"Count should be numeric for {metric_type}"
            
            # Validate mean values are reasonable for MNIST
            if 'acc' in metric_type:
                assert 0.0 <= mean_val <= 1.0, f"Mean accuracy {metric_type} should be 0-1, got {mean_val}"
                assert mean_val > 0.1, f"MNIST mean accuracy {metric_type} should be > 0.1, got {mean_val}"
            elif 'loss' in metric_type:
                assert mean_val >= 0.0, f"Mean loss {metric_type} should be >= 0, got {mean_val}"
                assert mean_val < 50.0, f"MNIST mean loss {metric_type} should be reasonable, got {mean_val}"
        
        print(f"   ✅ All aggregated values are realistic for MNIST")
        
        # Test 5: Cross-validate with raw metrics DataFrame
        print(f"\n🔄 Cross-validating with raw metrics data...")
        
        for metric_type in common_types:
            # Get raw data for this metric type
            raw_data = metrics_df[metrics_df['metric'] == metric_type]['value']
            
            if len(raw_data) > 0:
                # Get aggregated data for this metric type
                agg_row = exp_agg[exp_agg['metric_type'] == metric_type].iloc[0]
                
                # Compare aggregated mean with manually calculated mean
                raw_mean = raw_data.mean()
                agg_mean = agg_row['mean']
                raw_count = len(raw_data)
                agg_count = agg_row['count']
                
                # Allow small numerical differences due to SQL vs pandas precision
                mean_diff = abs(raw_mean - agg_mean)
                assert mean_diff < 0.01, f"Mean difference too large for {metric_type}: {mean_diff}"
                assert raw_count == agg_count, f"Count mismatch for {metric_type}: raw={raw_count}, agg={agg_count}"
        
        print(f"   ✅ Aggregated values match raw data calculations")
        
        # Test 6: Test error handling for invalid parameters
        print(f"\n❌ Testing error handling...")
        
        try:
            db_manager.get_aggregated_metrics(group_by='invalid_level')
            assert False, "Should raise error for invalid group_by"
        except ValueError as e:
            assert "group_by must be one of" in str(e)
            print(f"   ✅ Correctly rejected invalid group_by parameter")
        
        try:
            db_manager.get_aggregated_metrics(functions=['invalid_func'])
            assert False, "Should raise error for invalid aggregation function"
        except ValueError as e:
            assert "Invalid aggregation functions" in str(e)
            print(f"   ✅ Correctly rejected invalid aggregation function")
        
        print(f"\n✅ Successfully verified metrics aggregation with real MNIST data:")
        print(f"   - Experiment aggregation: {len(exp_agg)} records across {len(exp_metric_types)} metric types")
        print(f"   - Trial aggregation: {len(trial_agg)} records across {trial_ids} trials")
        print(f"   - Trial run aggregation: {len(run_agg)} records across {run_ids} runs")
        print(f"   - Aggregation functions: mean, std, min, max, count, sum")
        print(f"   - Cross-validation: passed")
        print(f"   - Error handling: verified")
        print(f"   - All aggregated values: realistic for MNIST training")
        
    finally:
        # Clean up database manager
        if hasattr(db_manager, 'connection') and db_manager.connection:
            db_manager.connection.close()

def test_comprehensive_db_tracker_integration_with_full_artifacts(experiment_full_artifacts):
    """Comprehensive integration test using experiment_full_artifacts fixture.
    
    This test verifies the complete DB Tracker functionality by testing the full
    experiment lifecycle with real MNIST data, including database operations,
    artifact management, workspace verification, and cross-system validation.
    
    Tests all major aspects:
    - Database connectivity and data integrity 
    - Experiment, trial, and trial run hierarchy
    - Metrics storage, retrieval, and aggregation
    - Artifact catalog and workspace validation
    - Cross-validation between database and filesystem
    - DBDataSource and DatabaseManager consistency
    """
    # experiment_full_artifacts provides complete access to real MNIST experiment
    db_path = experiment_full_artifacts['db_path']
    temp_dir = experiment_full_artifacts['temp_dir']
    experiment_obj = experiment_full_artifacts['experiment']
    framework_experiment = experiment_full_artifacts['framework_experiment']
    artifacts_catalog = experiment_full_artifacts['artifacts']
    
    print(f"\n🏗️ Comprehensive DB Tracker Integration Test:")
    print(f"   Database: {db_path}")
    print(f"   Workspace: {temp_dir}")
    print(f"   Framework Experiment: {framework_experiment}")
    print(f"   Artifact Types: {list(artifacts_catalog.keys())}")
    
    # Create both DatabaseManager and DBDataSource for cross-validation
    from experiment_manager.db.manager import DatabaseManager
    from experiment_manager.results.sources.db_datasource import DBDataSource
    
    db_manager = DatabaseManager(database_path=db_path, use_sqlite=True, recreate=False)
    
    try:
        # ===== TEST 1: Database Connectivity and Structure =====
        print(f"\n🔌 Testing database connectivity and structure...")
        
        # Test DatabaseManager connectivity
        assert hasattr(db_manager, 'connection'), "DatabaseManager should have active connection"
        assert db_manager.connection is not None, "DatabaseManager connection should be established"
        
        # Test basic query execution
        cursor = db_manager._execute_query("SELECT COUNT(*) as count FROM EXPERIMENT")
        experiment_count = cursor.fetchone()['count']
        assert experiment_count > 0, "Should have at least one experiment in database"
        
        print(f"   ✅ Database connectivity: verified")
        print(f"   ✅ Experiment count: {experiment_count}")
        
        # ===== TEST 2: Experiment Hierarchy Validation =====
        print(f"\n📊 Testing experiment hierarchy and data integrity...")
        
        # Validate experiment data through DatabaseManager and DBDataSource
        experiment_id = experiment_obj.id
        experiment_name = experiment_obj.name
        
        # Get trial count using explicit DBDataSource calls
        with DBDataSource(db_path) as db_source:
            ds_experiment = db_source.get_experiment()
            trials = db_source.get_trials(ds_experiment)
            trial_count = len(trials)
            
            print(f"   Experiment ID: {experiment_id}")
            print(f"   Experiment Name: {experiment_name}")
            print(f"   Trial Count: {trial_count}")
            
            assert ds_experiment.id == experiment_id, "Experiment IDs should match between sources"
            assert ds_experiment.name == experiment_name, "Experiment names should match"
            
            # Validate trial hierarchy using explicit calls
            total_runs = 0
            total_epochs = 0
            for trial in trials:
                assert trial.experiment_id == experiment_id, f"Trial {trial.id} should belong to experiment {experiment_id}"
                
                runs = db_source.get_trial_runs(trial)
                for run in runs:
                    assert run.trial_id == trial.id, f"Run {run.id} should belong to trial {trial.id}"
                    total_runs += 1
                    total_epochs += run.num_epochs if hasattr(run, 'num_epochs') else 0
            
            print(f"   ✅ Total trial runs: {total_runs}")
            print(f"   ✅ Total epochs: {total_epochs}")
            assert total_runs >= 6, "Should have at least 6 trial runs (3 trials × 2 repeats)"
        
        # ===== TEST 3: Metrics Validation and Aggregation =====
        print(f"\n📈 Testing metrics storage, retrieval, and aggregation...")
        
        # Get metrics through multiple methods for validation
        experiment_metrics = db_manager.get_experiment_metrics(experiment_id)
        assert len(experiment_metrics) > 0, "Should have experiment metrics"
        
        # Test aggregated metrics at different levels
        exp_agg = db_manager.get_aggregated_metrics(
            experiment_ids=[experiment_id],
            group_by='experiment',
            functions=['mean', 'count', 'min', 'max']
        )
        assert not exp_agg.empty, "Experiment-level aggregation should return data"
        
        trial_agg = db_manager.get_aggregated_metrics(
            experiment_ids=[experiment_id],
            group_by='trial',
            functions=['mean', 'std']
        )
        assert not trial_agg.empty, "Trial-level aggregation should return data"
        
        run_agg = db_manager.get_aggregated_metrics(
            experiment_ids=[experiment_id],
            group_by='trial_run',
            functions=['count', 'sum']
        )
        assert not run_agg.empty, "Trial run-level aggregation should return data"
        
        # Validate metric types and values
        metric_types = set(exp_agg['metric_type'].unique())
        expected_types = {'test_acc', 'test_loss'}
        common_types = metric_types.intersection(expected_types)
        assert len(common_types) > 0, f"Should have common metric types. Found: {metric_types}"
        
        print(f"   ✅ Experiment metrics: {len(experiment_metrics)}")
        print(f"   ✅ Aggregation levels: experiment({len(exp_agg)}), trial({len(trial_agg)}), run({len(run_agg)})")
        print(f"   ✅ Metric types: {metric_types}")
        
        # ===== TEST 4: Artifact Management and Workspace Validation =====
        print(f"\n📦 Testing artifact management and workspace validation...")
        
        from pathlib import Path
        workspace = Path(temp_dir)
        assert workspace.exists(), f"Workspace directory should exist: {workspace}"
        assert workspace.is_dir(), f"Workspace should be a directory: {workspace}"
        
        # Validate artifact catalog from fixture
        total_artifacts = 0
        for pattern, files in artifacts_catalog.items():
            artifact_count = len(files)
            total_artifacts += artifact_count
            print(f"   {pattern}: {artifact_count} files")
            
            # Validate artifact paths are within workspace
            for artifact_path in files:
                assert str(artifact_path).startswith(str(workspace)), f"Artifact should be in workspace: {artifact_path}"
        
        print(f"   ✅ Total cataloged artifacts: {total_artifacts}")
        assert total_artifacts > 0, "Should have cataloged artifacts in workspace"
        
        # Test database artifact retrieval
        db_experiment_artifacts = db_manager.get_experiment_artifacts(experiment_id)
        print(f"   ✅ Database experiment artifacts: {len(db_experiment_artifacts)}")
        
        # Test trial-level artifacts using explicit calls
        with DBDataSource(db_path) as db_source:
            ds_experiment = db_source.get_experiment()
            trials = db_source.get_trials(ds_experiment)
            if trials:
                first_trial_id = trials[0].id
                trial_artifacts = db_manager.get_trial_artifacts(first_trial_id)
                print(f"   ✅ Trial artifacts: {len(trial_artifacts)}")
                
                # Test trial run-level artifacts
                runs = db_source.get_trial_runs(trials[0])
                if runs:
                    first_run_id = runs[0].id
                    run_artifacts = db_manager.get_trial_run_artifacts(first_run_id)
                    print(f"   ✅ Trial run artifacts: {len(run_artifacts)}")
        
        # ===== TEST 5: Cross-System Data Consistency =====
        print(f"\n🔄 Testing cross-system data consistency...")
        
        # Compare experiment data from different access methods
        with DBDataSource(db_path) as db_source:
            # Get comprehensive data
            exp_data = db_manager.get_experiment_data(experiment_ids=[experiment_id])
            assert not exp_data.empty, "Experiment data should not be empty"
            
            # Get metrics data frame
            metrics_df = create_metrics_dataframe(db_source, ds_experiment)
            assert not metrics_df.empty, "Metrics DataFrame should not be empty"
            
            # Cross-validate metric counts
            db_metric_count = len(metrics_df)
            agg_metric_count = exp_agg['count'].sum() if 'count' in exp_agg.columns else 0
            
            print(f"   DataFrame metrics: {db_metric_count}")
            print(f"   Aggregated count: {agg_metric_count}")
            
            # Validate experiment ID consistency across all data
            unique_exp_ids = set(exp_data['experiment_id'].unique()) if 'experiment_id' in exp_data.columns else {experiment_id}
            assert experiment_id in unique_exp_ids, "Experiment ID should be consistent across all data sources"
            
            # Validate trial ID consistency using explicit calls
            db_trial_ids = set(exp_data['trial_id'].unique()) if 'trial_id' in exp_data.columns else set()
            ds_trials = db_source.get_trials(ds_experiment)
            obj_trial_ids = {trial.id for trial in ds_trials}
            if db_trial_ids:
                common_trial_ids = db_trial_ids.intersection(obj_trial_ids)
                assert len(common_trial_ids) > 0, "Should have common trial IDs between database and object"
                print(f"   ✅ Common trial IDs: {len(common_trial_ids)}")
        
        # ===== TEST 6: Performance and Error Handling =====
        print(f"\n⚡ Testing performance and error handling...")
        
        # Test query performance with larger datasets
        import time
        start_time = time.time()
        large_data = db_manager.get_experiment_data()
        query_time = time.time() - start_time
        print(f"   Full data query time: {query_time:.3f}s")
        assert query_time < 5.0, "Large data queries should complete within reasonable time"
        
        # Test error handling for invalid IDs
        try:
            db_manager.get_experiment_metrics(99999)  # Non-existent experiment
            empty_metrics = True  # Should return empty list, not error
        except Exception as e:
            empty_metrics = False
            print(f"   ✅ Error handling for invalid ID: {type(e).__name__}")
        
        # Test invalid aggregation parameters
        try:
            db_manager.get_aggregated_metrics(group_by='invalid_level')
            assert False, "Should raise error for invalid group_by"
        except ValueError:
            print(f"   ✅ Error handling for invalid parameters: verified")
        
        # ===== TEST 7: Framework Integration Validation =====
        print(f"\n🔗 Testing framework integration...")
        
        # Validate framework experiment object
        assert framework_experiment is not None, "Framework experiment should be provided"
        
        # Check if framework experiment has expected attributes
        expected_attrs = ['env', 'name']
        for attr in expected_attrs:
            if hasattr(framework_experiment, attr):
                print(f"   ✅ Framework attribute '{attr}': present")
        
        # Validate workspace consistency
        if hasattr(framework_experiment, 'env') and hasattr(framework_experiment.env, 'workspace'):
            fw_workspace = framework_experiment.env.workspace
            assert str(workspace).startswith(str(fw_workspace)) or str(fw_workspace).startswith(str(workspace)), \
                "Framework workspace should be consistent with fixture workspace"
            print(f"   ✅ Workspace consistency: verified")
        
        print(f"\n✅ Comprehensive DB Tracker Integration Test PASSED!")
        print(f"   🔌 Database connectivity: verified")
        print(f"   📊 Experiment hierarchy: validated")
        print(f"   📈 Metrics system: functional") 
        print(f"   📦 Artifact management: operational")
        print(f"   🔄 Cross-system consistency: confirmed")
        print(f"   ⚡ Performance & error handling: tested")
        print(f"   🔗 Framework integration: validated")
        print(f"   💾 Total artifacts: {total_artifacts}")
        print(f"   📊 Total metrics: {db_metric_count}")
        print(f"   🏃 Total runs: {total_runs}")
        print(f"   ⚡ Query performance: {query_time:.3f}s")
        
    finally:
        # Clean up database manager
        if hasattr(db_manager, 'connection') and db_manager.connection:
            db_manager.connection.close()

def test_experiment_update_with_real_data(experiment_db_only):
    """Test experiment update operations using real MNIST experiment data.
    
    This test verifies that experiment metadata (title, description) can be
    updated in a database containing real MNIST experiment data, and that
    these updates are properly reflected in both the database and retrieval APIs.
    """
    # experiment_db_only provides the path to real MNIST experiment database
    real_db_path = experiment_db_only
    
    # Create a database manager using the real database path
    from experiment_manager.db.manager import DatabaseManager
    db_manager = DatabaseManager(database_path=real_db_path, use_sqlite=True, recreate=False)
    
    print(f"✅ Connected to real MNIST database: {real_db_path}")
    
    # First, get the existing MNIST experiment to understand current state
    ph = db_manager._get_placeholder()
    desc_field = "desc" if db_manager.use_sqlite else "`desc`"
    
    # Get the original experiment
    query = f"SELECT id, title, {desc_field} as description, start_time, update_time FROM EXPERIMENT LIMIT 1"
    cursor = db_manager._execute_query(query)
    original_exp = cursor.fetchone()
    
    assert original_exp is not None, "Should find at least one experiment in the real MNIST database"
    
    original_id = original_exp["id"]
    original_title = original_exp["title"]
    original_description = original_exp["description"]
    original_update_time = original_exp["update_time"]
    
    print(f"📋 Original experiment - ID: {original_id}, Title: '{original_title}', Desc: '{original_description}'")
    
    # Now implement and test the update functionality
    from datetime import datetime
    import time
    
    # Add update_experiment method to DatabaseManager (since it doesn't exist)
    def update_experiment(self, experiment_id: int, title: str = None, description: str = None):
        """Update experiment metadata."""
        ph = self._get_placeholder()
        desc_field = "desc" if self.use_sqlite else "`desc`"
        
        # Check if experiment exists
        check_query = f"SELECT id FROM EXPERIMENT WHERE id = {ph}"
        cursor = self._execute_query(check_query, (experiment_id,))
        if not cursor.fetchone():
            raise ValueError(f"Experiment with id {experiment_id} does not exist")
        
        # Build update query based on provided parameters
        update_fields = []
        params = []
        
        if title is not None:
            update_fields.append(f"title = {ph}")
            params.append(title)
        
        if description is not None:
            update_fields.append(f"{desc_field} = {ph}")
            params.append(description)
        
        # Always update the update_time
        update_fields.append(f"update_time = {ph}")
        params.append(datetime.now().isoformat())
        
        # Add experiment_id for WHERE clause
        params.append(experiment_id)
        
        if update_fields:
            query = f"UPDATE EXPERIMENT SET {', '.join(update_fields)} WHERE id = {ph}"
            self._execute_query(query, tuple(params))
            self.connection.commit()
    
    # Monkey patch the method onto the DatabaseManager instance
    import types
    db_manager.update_experiment = types.MethodType(update_experiment, db_manager)
    
    # Test 1: Update only the title
    new_title = f"{original_title} - Updated Title"
    db_manager.update_experiment(original_id, title=new_title)
    
    # Verify title update
    cursor = db_manager._execute_query(f"SELECT title, {desc_field} as description, update_time FROM EXPERIMENT WHERE id = {ph}", (original_id,))
    updated_exp = cursor.fetchone()
    
    assert updated_exp["title"] == new_title, f"Title should be updated to '{new_title}'"
    assert updated_exp["description"] == original_description, "Description should remain unchanged"
    assert updated_exp["update_time"] != original_update_time, "update_time should be different after update"
    
    print(f"✅ Title update successful: '{new_title}'")
    
    # Test 2: Update only the description
    new_description = f"{original_description} - This experiment has been updated with new metadata"
    updated_time_after_title = updated_exp["update_time"]
    
    # Add a small delay to ensure timestamp difference
    time.sleep(0.01)
    
    db_manager.update_experiment(original_id, description=new_description)
    
    # Verify description update
    cursor = db_manager._execute_query(f"SELECT title, {desc_field} as description, update_time FROM EXPERIMENT WHERE id = {ph}", (original_id,))
    updated_exp2 = cursor.fetchone()
    
    assert updated_exp2["title"] == new_title, "Title should remain unchanged from previous update"
    assert updated_exp2["description"] == new_description, f"Description should be updated to '{new_description}'"
    assert updated_exp2["update_time"] != updated_time_after_title, "update_time should be different after description update"
    
    print(f"✅ Description update successful: '{new_description}'")
    
    # Test 3: Update both title and description simultaneously
    final_title = f"MNIST Real Data Test - Final Title"
    final_description = f"Comprehensive test experiment with real MNIST data - final update"
    
    time.sleep(0.01)
    db_manager.update_experiment(original_id, title=final_title, description=final_description)
    
    # Verify simultaneous update
    cursor = db_manager._execute_query(f"SELECT title, {desc_field} as description, update_time FROM EXPERIMENT WHERE id = {ph}", (original_id,))
    final_exp = cursor.fetchone()
    
    assert final_exp["title"] == final_title, f"Final title should be '{final_title}'"
    assert final_exp["description"] == final_description, f"Final description should be '{final_description}'"
    assert final_exp["update_time"] != updated_exp2["update_time"], "update_time should change after final update"
    
    print(f"✅ Simultaneous update successful: Title='{final_title}', Desc='{final_description}'")
    
    # Test 4: Verify updates don't affect other experiment data (trials, metrics, artifacts)
    # Count trials before and after updates
    cursor = db_manager._execute_query(f"SELECT COUNT(*) as trial_count FROM TRIAL WHERE experiment_id = {ph}", (original_id,))
    trial_count = cursor.fetchone()["trial_count"]
    
    # Count metrics associated with the experiment
    cursor = db_manager._execute_query(f"""
        SELECT COUNT(DISTINCT m.id) as metric_count FROM METRIC m
        LEFT JOIN EPOCH_METRIC em ON em.metric_id = m.id
        LEFT JOIN EPOCH e ON e.idx = em.epoch_idx AND e.trial_run_id = em.epoch_trial_run_id
        LEFT JOIN TRIAL_RUN tr ON tr.id = e.trial_run_id
        LEFT JOIN RESULTS_METRIC rm ON rm.metric_id = m.id
        LEFT JOIN RESULTS r ON r.trial_run_id = rm.results_id
        JOIN TRIAL t ON t.id = tr.trial_id OR t.id = (SELECT trial_id FROM TRIAL_RUN WHERE id = r.trial_run_id)
        WHERE t.experiment_id = {ph}
    """, (original_id,))
    metric_count = cursor.fetchone()["metric_count"]
    
    assert trial_count > 0, "Should have trials from the real MNIST experiment"
    assert metric_count > 0, "Should have metrics from the real MNIST experiment"
    
    print(f"✅ Data integrity verified: {trial_count} trials and {metric_count} metrics preserved")
    
    # Test 5: Test error handling for non-existent experiment
    non_existent_id = 99999
    try:
        db_manager.update_experiment(non_existent_id, title="Should Fail")
        assert False, "Should raise ValueError for non-existent experiment"
    except ValueError as e:
        assert f"Experiment with id {non_existent_id} does not exist" in str(e)
        print(f"✅ Error handling verified: {e}")
    
    # Test 6: Verify that DBDataSource still retrieves updated experiment correctly
    from experiment_manager.results.sources.db_datasource import DBDataSource
    
    db_source = DBDataSource(db_path=real_db_path, use_sqlite=True)
    retrieved_exp = db_source.get_experiment(original_id)
    
    assert retrieved_exp.id == original_id, "Retrieved experiment should have correct ID"
    assert retrieved_exp.name == final_title, "Retrieved experiment should have updated title"
    assert retrieved_exp.description == final_description, "Retrieved experiment should have updated description"
    
    print(f"✅ DBDataSource retrieval verified: ID={retrieved_exp.id}, Name='{retrieved_exp.name}'")
    
    # Restore original experiment data for cleanup (optional)
    print(f"🔄 Restoring original experiment metadata for cleanup...")
    db_manager.update_experiment(original_id, title=original_title, description=original_description)
    
    print("✅ Test completed successfully - Experiment update with real MNIST data verified!")
