"""Tests for data migration utilities."""
import json
import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, call

from experiment_manager.db.manager import DatabaseManager
from experiment_manager.db.data_migration import (
    DataMigrationManager, MigrationStrategy, MigrationProgress,
    MetricTransformer, HierarchyPreserver, DataValidator, SnapshotManager,
    DataMigrationError, DataSnapshot
)

@pytest.fixture
def db_manager():
    """Create a test database manager with sample data."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    manager = DatabaseManager(database_path=db_path, use_sqlite=True)
    
    # Create sample data for testing
    experiment = manager.create_experiment("Test Experiment", "Test description")
    trial = manager.create_trial(experiment.id, "Test Trial")
    trial_run = manager.create_trial_run(trial.id)
    
    # Create epoch
    manager.create_epoch(1, trial_run.id)
    
    # Create results entry
    ph = manager._get_placeholder()
    query = f"INSERT INTO RESULTS (trial_run_id, time) VALUES ({ph}, {ph})"
    manager._execute_query(query, (trial_run.id, datetime.now().isoformat()))
    manager.connection.commit()
    
    # Create metrics with JSON data
    metric1 = manager.record_metric(0.95, "accuracy", {"class_0": 0.92, "class_1": 0.98})
    metric2 = manager.record_metric(0.85, "precision", {"class_0": 0.80, "class_1": 0.90})
    
    # Link metrics
    manager.link_results_metric(trial_run.id, metric1.id)
    manager.add_epoch_metric(1, trial_run.id, metric2.id)
    
    yield manager
    
    # Close connections and cleanup
    try:
        if hasattr(manager, 'connection') and manager.connection:
            manager.connection.close()
    except:
        pass
    
    try:
        if Path(db_path).exists():
            Path(db_path).unlink()
    except:
        pass

@pytest.fixture
def migration_manager(db_manager):
    """Create a test migration manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        return DataMigrationManager(
            db_manager, 
            migration_dir=Path(temp_dir) / "migrations",
            snapshot_dir=Path(temp_dir) / "snapshots"
        )

class TestMetricTransformer:
    """Test the MetricTransformer utility class."""
    
    def test_validate_metric_json_valid_dict(self):
        """Test validation of valid metric JSON."""
        valid_data = {"class_0": 0.95, "class_1": 0.87}
        is_valid, error = MetricTransformer.validate_metric_json(valid_data)
        assert is_valid
        assert error is None
    
    def test_validate_metric_json_valid_string(self):
        """Test validation of valid metric JSON string."""
        valid_json = '{"class_0": 0.95, "class_1": 0.87}'
        is_valid, error = MetricTransformer.validate_metric_json(valid_json)
        assert is_valid
        assert error is None
    
    def test_validate_metric_json_invalid_structure(self):
        """Test validation of invalid metric structure."""
        invalid_data = ["not", "a", "dict"]
        is_valid, error = MetricTransformer.validate_metric_json(invalid_data)
        assert not is_valid
        assert "must be a dictionary" in error
    
    def test_validate_metric_json_invalid_keys(self):
        """Test validation with non-string keys."""
        invalid_data = {1: 0.95, "class_1": 0.87}
        is_valid, error = MetricTransformer.validate_metric_json(invalid_data)
        assert not is_valid
        assert "keys must be strings" in error
    
    def test_validate_metric_json_invalid_values(self):
        """Test validation with non-numeric values."""
        invalid_data = {"class_0": "not_a_number", "class_1": 0.87}
        is_valid, error = MetricTransformer.validate_metric_json(invalid_data)
        assert not is_valid
        assert "values must be numeric" in error
    
    def test_validate_metric_json_invalid_json(self):
        """Test validation of invalid JSON string."""
        invalid_json = '{"class_0": 0.95, "class_1":}'
        is_valid, error = MetricTransformer.validate_metric_json(invalid_json)
        assert not is_valid
        assert "Invalid JSON" in error
    
    def test_transform_metric_format(self):
        """Test metric format transformation."""
        original = {"accuracy": 95.5, "precision": 87.2}
        rules = {
            "accuracy": lambda x: x / 100.0,
            "precision": lambda x: round(x, 1)
        }
        
        transformed = MetricTransformer.transform_metric_format(original, rules)
        
        assert transformed["accuracy"] == 0.955
        assert transformed["precision"] == 87.2
    
    def test_transform_metric_format_missing_field(self):
        """Test transformation with missing field."""
        original = {"accuracy": 95.5}
        rules = {"precision": lambda x: x / 100.0}
        
        transformed = MetricTransformer.transform_metric_format(original, rules)
        
        # Should return original data unchanged
        assert transformed == original
    
    def test_transform_metric_format_error_handling(self):
        """Test transformation error handling."""
        original = {"accuracy": "not_a_number"}
        rules = {"accuracy": lambda x: x / 100.0}
        
        # Should not raise exception, but log warning
        transformed = MetricTransformer.transform_metric_format(original, rules)
        
        # Original value should be preserved on error
        assert transformed["accuracy"] == "not_a_number"
    
    def test_aggregate_per_label_metrics_mean(self):
        """Test mean aggregation of per-label metrics."""
        metrics = [
            {"class_0": 0.9, "class_1": 0.8},
            {"class_0": 0.8, "class_1": 0.9},
            {"class_0": 0.95, "class_1": 0.85}
        ]
        
        result = MetricTransformer.aggregate_per_label_metrics(metrics, "mean")
        
        assert result["class_0"] == pytest.approx(0.883, rel=1e-2)
        assert result["class_1"] == pytest.approx(0.850, rel=1e-2)
    
    def test_aggregate_per_label_metrics_sum(self):
        """Test sum aggregation of per-label metrics."""
        metrics = [
            {"class_0": 0.9, "class_1": 0.8},
            {"class_0": 0.1, "class_1": 0.2}
        ]
        
        result = MetricTransformer.aggregate_per_label_metrics(metrics, "sum")
        
        assert result["class_0"] == 1.0
        assert result["class_1"] == 1.0
    
    def test_aggregate_per_label_metrics_max(self):
        """Test max aggregation of per-label metrics."""
        metrics = [
            {"class_0": 0.9, "class_1": 0.8},
            {"class_0": 0.7, "class_1": 0.95}
        ]
        
        result = MetricTransformer.aggregate_per_label_metrics(metrics, "max")
        
        assert result["class_0"] == 0.9
        assert result["class_1"] == 0.95
    
    def test_aggregate_per_label_metrics_min(self):
        """Test min aggregation of per-label metrics."""
        metrics = [
            {"class_0": 0.9, "class_1": 0.8},
            {"class_0": 0.7, "class_1": 0.95}
        ]
        
        result = MetricTransformer.aggregate_per_label_metrics(metrics, "min")
        
        assert result["class_0"] == 0.7
        assert result["class_1"] == 0.8
    
    def test_aggregate_per_label_metrics_empty(self):
        """Test aggregation with empty metrics list."""
        result = MetricTransformer.aggregate_per_label_metrics([], "mean")
        assert result == {}
    
    def test_aggregate_per_label_metrics_invalid_function(self):
        """Test aggregation with invalid function."""
        metrics = [{"class_0": 0.9}]
        
        with pytest.raises(ValueError, match="Unsupported aggregation function"):
            MetricTransformer.aggregate_per_label_metrics(metrics, "invalid")

class TestHierarchyPreserver:
    """Test the HierarchyPreserver utility class."""
    
    def test_get_experiment_hierarchy(self, db_manager):
        """Test getting complete experiment hierarchy."""
        preserver = HierarchyPreserver(db_manager)
        
        # Get first experiment ID
        cursor = db_manager._execute_query("SELECT id FROM EXPERIMENT LIMIT 1")
        experiment_id = cursor.fetchone()["id"]
        
        hierarchy = preserver.get_experiment_hierarchy(experiment_id)
        
        assert "experiment" in hierarchy
        assert "trials" in hierarchy
        assert hierarchy["experiment"]["id"] == experiment_id
        assert len(hierarchy["trials"]) > 0
        
        # Check trial structure
        trial = hierarchy["trials"][0]
        assert "trial" in trial
        assert "runs" in trial
        assert len(trial["runs"]) > 0
        
        # Check run structure
        run = trial["runs"][0]
        assert "run" in run
        assert "epochs" in run
        assert "results" in run
    
    def test_get_experiment_hierarchy_not_found(self, db_manager):
        """Test getting hierarchy for non-existent experiment."""
        preserver = HierarchyPreserver(db_manager)
        
        with pytest.raises(DataMigrationError, match="Experiment 999 not found"):
            preserver.get_experiment_hierarchy(999)
    
    def test_validate_hierarchy_integrity_valid(self, db_manager):
        """Test validation of valid hierarchy."""
        preserver = HierarchyPreserver(db_manager)
        
        # Get first experiment ID
        cursor = db_manager._execute_query("SELECT id FROM EXPERIMENT LIMIT 1")
        experiment_id = cursor.fetchone()["id"]
        
        is_valid, issues = preserver.validate_hierarchy_integrity(experiment_id)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_hierarchy_integrity_invalid_experiment(self, db_manager):
        """Test validation with non-existent experiment."""
        preserver = HierarchyPreserver(db_manager)
        
        is_valid, issues = preserver.validate_hierarchy_integrity(999)
        
        assert not is_valid
        assert len(issues) > 0
        assert "does not exist" in issues[0]

class TestDataValidator:
    """Test the DataValidator utility class."""
    
    def test_validate_foreign_keys_valid(self, db_manager):
        """Test foreign key validation with valid data."""
        validator = DataValidator(db_manager)
        
        violations = validator.validate_foreign_keys()
        
        # Should have no violations with our test data
        assert len(violations) == 0
    
    def test_validate_json_metrics_valid(self, db_manager):
        """Test JSON metric validation with valid data."""
        validator = DataValidator(db_manager)
        
        issues = validator.validate_json_metrics()
        
        # Our test data has valid JSON metrics
        assert len(issues) == 0
    
    def test_validate_json_metrics_invalid(self, db_manager):
        """Test JSON metric validation with invalid data."""
        validator = DataValidator(db_manager)
        
        # Insert invalid JSON metric
        ph = db_manager._get_placeholder()
        query = f"INSERT INTO METRIC (type, total_val, per_label_val) VALUES ({ph}, {ph}, {ph})"
        db_manager._execute_query(query, ("test_metric", 0.5, "invalid_json"))
        db_manager.connection.commit()
        
        issues = validator.validate_json_metrics()
        
        assert len(issues) > 0
        assert any("Invalid JSON" in issue["error"] for issue in issues)
    
    def test_validate_data_consistency_pass(self, db_manager):
        """Test data consistency validation that passes."""
        validator = DataValidator(db_manager)
        
        results = validator.validate_data_consistency()
        
        assert "timestamp" in results
        assert "foreign_key_violations" in results
        assert "json_metric_issues" in results
        assert "summary" in results
        assert results["summary"]["overall_status"] == "PASS"
    
    def test_validate_data_consistency_fail(self, db_manager):
        """Test data consistency validation that fails."""
        validator = DataValidator(db_manager)
        
        # Insert invalid data
        ph = db_manager._get_placeholder()
        query = f"INSERT INTO METRIC (type, total_val, per_label_val) VALUES ({ph}, {ph}, {ph})"
        db_manager._execute_query(query, ("test_metric", 0.5, "invalid_json"))
        db_manager.connection.commit()
        
        results = validator.validate_data_consistency()
        
        assert results["summary"]["overall_status"] == "FAIL"
        assert results["summary"]["total_json_metric_issues"] > 0

class TestSnapshotManager:
    """Test the SnapshotManager utility class."""
    
    def test_create_snapshot(self, db_manager):
        """Test creating a database snapshot."""
        # Skip on Windows due to file locking issues in test environment
        import sys
        if sys.platform == "win32":
            pytest.skip("Skipping snapshot test on Windows due to file locking issues")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_manager = SnapshotManager(db_manager, temp_dir)
            
            snapshot = snapshot_manager.create_snapshot("Test snapshot")
            
            assert isinstance(snapshot, DataSnapshot)
            assert snapshot.snapshot_id.startswith("snapshot_")
            assert snapshot.description == "Test snapshot"
            assert snapshot.file_path.exists()
            assert snapshot.size_bytes > 0
            assert isinstance(snapshot.metadata, dict)
    
    def test_list_snapshots(self, db_manager):
        """Test listing snapshots."""
        # Skip on Windows due to file locking issues in test environment
        import sys
        if sys.platform == "win32":
            pytest.skip("Skipping snapshot test on Windows due to file locking issues")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_manager = SnapshotManager(db_manager, temp_dir)
            
            # Create a few snapshots
            snapshot1 = snapshot_manager.create_snapshot("First snapshot")
            snapshot2 = snapshot_manager.create_snapshot("Second snapshot")
            
            snapshots = snapshot_manager.list_snapshots()
            
            assert len(snapshots) == 2
            # Should be sorted by creation time (newest first)
            assert snapshots[0].created_at >= snapshots[1].created_at
    
    def test_restore_snapshot_sqlite(self, db_manager):
        """Test restoring a SQLite snapshot."""
        # Skip on Windows due to file locking issues in test environment
        import sys
        if sys.platform == "win32":
            pytest.skip("Skipping snapshot test on Windows due to file locking issues")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_manager = SnapshotManager(db_manager, temp_dir)
            
            # Create snapshot
            snapshot = snapshot_manager.create_snapshot("Test snapshot")
            
            # Modify database
            db_manager.create_experiment("New Experiment", "After snapshot")
            
            # Restore snapshot
            snapshot_manager.restore_snapshot(snapshot.snapshot_id)
            
            # Verify restoration (new experiment should be gone)
            cursor = db_manager._execute_query("SELECT COUNT(*) as count FROM EXPERIMENT WHERE title = 'New Experiment'")
            count = cursor.fetchone()["count"]
            assert count == 0
    
    def test_restore_snapshot_not_found(self, db_manager):
        """Test restoring non-existent snapshot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_manager = SnapshotManager(db_manager, temp_dir)
            
            with pytest.raises(DataMigrationError, match="Snapshot nonexistent not found"):
                snapshot_manager.restore_snapshot("nonexistent")

class TestMigrationProgress:
    """Test the MigrationProgress data class."""
    
    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        progress = MigrationProgress(
            total_items=100,
            processed_items=25,
            failed_items=5,
            start_time=datetime.now()
        )
        
        assert progress.completion_percentage == 25.0
    
    def test_completion_percentage_zero_total(self):
        """Test completion percentage with zero total items."""
        progress = MigrationProgress(
            total_items=0,
            processed_items=0,
            failed_items=0,
            start_time=datetime.now()
        )
        
        assert progress.completion_percentage == 100.0
    
    def test_success_rate(self):
        """Test success rate calculation."""
        progress = MigrationProgress(
            total_items=100,
            processed_items=20,
            failed_items=5,
            start_time=datetime.now()
        )
        
        assert progress.success_rate == 75.0  # (20-5)/20 * 100
    
    def test_success_rate_zero_processed(self):
        """Test success rate with zero processed items."""
        progress = MigrationProgress(
            total_items=100,
            processed_items=0,
            failed_items=0,
            start_time=datetime.now()
        )
        
        assert progress.success_rate == 100.0
    
    def test_update_eta(self):
        """Test ETA calculation."""
        import time
        
        start_time = datetime.now()
        progress = MigrationProgress(
            total_items=100,
            processed_items=25,
            failed_items=0,
            start_time=start_time
        )
        
        # Simulate some processing time
        time.sleep(0.1)
        progress.update_eta()
        
        assert progress.estimated_completion is not None
        assert progress.estimated_completion > datetime.now()

class TestDataMigrationManager:
    """Test the main DataMigrationManager class."""
    
    def test_initialization(self, db_manager):
        """Test migration manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataMigrationManager(
                db_manager,
                migration_dir=Path(temp_dir) / "migrations",
                snapshot_dir=Path(temp_dir) / "snapshots"
            )
            
            assert manager.db_manager == db_manager
            assert isinstance(manager.migration_manager, object)
            assert isinstance(manager.snapshot_manager, SnapshotManager)
            assert isinstance(manager.validator, DataValidator)
            assert isinstance(manager.hierarchy_preserver, HierarchyPreserver)
    
    def test_add_progress_callback(self, migration_manager):
        """Test adding progress callbacks."""
        callback = Mock()
        
        migration_manager.add_progress_callback(callback)
        
        assert callback in migration_manager._progress_callbacks
    
    def test_batch_transform_metrics(self, migration_manager):
        """Test batch metric transformation."""
        # Setup transformation rules
        transformation_rules = {
            "class_0": lambda x: x * 2,
            "class_1": lambda x: x * 2
        }
        
        # Mock progress callback
        callback = Mock()
        migration_manager.add_progress_callback(callback)
        
        # Perform transformation
        progress = migration_manager.batch_transform_metrics(
            transformation_rules=transformation_rules,
            create_snapshot=False  # Skip snapshot for test speed
        )
        
        assert isinstance(progress, MigrationProgress)
        assert progress.processed_items >= 0
        assert callback.called
    
    def test_migrate_experiment_data_new_target(self, migration_manager):
        """Test experiment data migration to new experiment."""
        # Get source experiment ID
        cursor = migration_manager.db_manager._execute_query("SELECT id FROM EXPERIMENT LIMIT 1")
        source_id = cursor.fetchone()["id"]
        
        # Perform migration
        progress = migration_manager.migrate_experiment_data(
            source_experiment_id=source_id,
            target_experiment_id=None,  # Create new
            strategy=MigrationStrategy.BALANCED,
            create_snapshot=False  # Skip for test speed
        )
        
        assert isinstance(progress, MigrationProgress)
        assert progress.processed_items > 0
        
        # Verify target experiment was created
        cursor = migration_manager.db_manager._execute_query("SELECT COUNT(*) as count FROM EXPERIMENT")
        experiment_count = cursor.fetchone()["count"]
        assert experiment_count >= 2  # Original + migrated
    
    def test_migrate_experiment_data_conservative_strategy(self, migration_manager):
        """Test migration with conservative strategy."""
        # Get source experiment ID
        cursor = migration_manager.db_manager._execute_query("SELECT id FROM EXPERIMENT LIMIT 1")
        source_id = cursor.fetchone()["id"]
        
        # Perform migration with conservative strategy
        progress = migration_manager.migrate_experiment_data(
            source_experiment_id=source_id,
            strategy=MigrationStrategy.CONSERVATIVE,
            create_snapshot=False
        )
        
        assert isinstance(progress, MigrationProgress)
        # Conservative strategy should succeed with valid data
        assert progress.failed_items == 0
    
    def test_migrate_experiment_data_not_found(self, migration_manager):
        """Test migration with non-existent source experiment."""
        with pytest.raises(DataMigrationError):
            migration_manager.migrate_experiment_data(
                source_experiment_id=999,
                create_snapshot=False
            )
    
    def test_get_migration_status_none(self, migration_manager):
        """Test getting migration status when none in progress."""
        status = migration_manager.get_migration_status()
        assert status is None
    
    def test_migration_with_transformation_rules(self, migration_manager):
        """Test migration with transformation rules."""
        # Get source experiment ID
        cursor = migration_manager.db_manager._execute_query("SELECT id FROM EXPERIMENT LIMIT 1")
        source_id = cursor.fetchone()["id"]
        
        # Define transformation rules
        transformation_rules = {
            "accuracy": lambda x: x / 100.0  # Convert percentage to decimal
        }
        
        # Perform migration
        progress = migration_manager.migrate_experiment_data(
            source_experiment_id=source_id,
            transformation_rules=transformation_rules,
            create_snapshot=False
        )
        
        assert isinstance(progress, MigrationProgress)
        assert progress.processed_items > 0

class TestDataMigrationCLI:
    """Test the CLI interface (unit tests for logic)."""
    
    def test_create_db_manager_sqlite(self):
        """Test creating SQLite database manager."""
        from experiment_manager.db.data_migration_cli import create_db_manager
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            manager = create_db_manager(db_path, use_sqlite=True)
            assert manager.use_sqlite
            assert manager.connection is not None
            
            # Clean up
            manager.connection.close()
        finally:
            try:
                Path(db_path).unlink()
            except:
                pass
    
    def test_create_db_manager_mysql(self):
        """Test creating MySQL database manager."""
        from experiment_manager.db.data_migration_cli import create_db_manager
        from experiment_manager.db.manager import ConnectionError
        
        # Expect connection failure since no MySQL server is running
        with pytest.raises(ConnectionError):
            create_db_manager(
                "test_db", 
                use_sqlite=False, 
                host="localhost", 
                user="test_user", 
                password="test_pass"
            )

class TestIntegrationScenarios:
    """Integration tests for complete migration scenarios."""
    
    def test_complete_migration_workflow(self, db_manager):
        """Test a complete migration workflow."""
        # Skip on Windows due to file locking issues in test environment
        import sys
        if sys.platform == "win32":
            pytest.skip("Skipping integration test on Windows due to file locking issues")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataMigrationManager(
                db_manager,
                snapshot_dir=temp_dir
            )
            
            # 1. Create snapshot
            snapshot = manager.snapshot_manager.create_snapshot("Pre-migration")
            assert snapshot.file_path.exists()
            
            # 2. Validate data
            validation_results = manager.validator.validate_data_consistency()
            assert validation_results["summary"]["overall_status"] == "PASS"
            
            # 3. Get experiment hierarchy
            cursor = db_manager._execute_query("SELECT id FROM EXPERIMENT LIMIT 1")
            exp_id = cursor.fetchone()["id"]
            hierarchy = manager.hierarchy_preserver.get_experiment_hierarchy(exp_id)
            assert "experiment" in hierarchy
            
            # 4. Perform migration
            progress = manager.migrate_experiment_data(
                source_experiment_id=exp_id,
                create_snapshot=False  # Already created one
            )
            assert progress.processed_items > 0
            
            # 5. Validate after migration
            post_validation = manager.validator.validate_data_consistency()
            assert post_validation["summary"]["overall_status"] == "PASS"
    
    def test_metric_transformation_workflow(self, db_manager):
        """Test metric transformation workflow."""
        # Skip on Windows due to file locking issues in test environment
        import sys
        if sys.platform == "win32":
            pytest.skip("Skipping integration test on Windows due to file locking issues")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataMigrationManager(db_manager, snapshot_dir=temp_dir)
            
            # 1. Create snapshot
            snapshot = manager.snapshot_manager.create_snapshot("Pre-transformation")
            
            # 2. Define transformation rules
            rules = {
                "class_0": lambda x: round(x, 2),
                "class_1": lambda x: round(x, 2)
            }
            
            # 3. Transform metrics
            progress = manager.batch_transform_metrics(
                transformation_rules=rules,
                create_snapshot=False
            )
            
            assert progress.processed_items >= 0
            
            # 4. Validate JSON metrics still valid
            issues = manager.validator.validate_json_metrics()
            assert len(issues) == 0
    
    def test_error_recovery_workflow(self, db_manager):
        """Test error recovery using snapshots."""
        # Skip on Windows due to file locking issues in test environment
        import sys
        if sys.platform == "win32":
            pytest.skip("Skipping integration test on Windows due to file locking issues")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataMigrationManager(db_manager, snapshot_dir=temp_dir)
            
            # 1. Create snapshot
            snapshot = manager.snapshot_manager.create_snapshot("Before error")
            
            # 2. Introduce some error (corrupt data)
            ph = db_manager._get_placeholder()
            query = f"INSERT INTO METRIC (type, total_val, per_label_val) VALUES ({ph}, {ph}, {ph})"
            db_manager._execute_query(query, ("corrupt", 0.0, "invalid_json"))
            db_manager.connection.commit()
            
            # 3. Verify data is corrupted
            validation = manager.validator.validate_data_consistency()
            assert validation["summary"]["overall_status"] == "FAIL"
            
            # 4. Restore from snapshot
            manager.snapshot_manager.restore_snapshot(snapshot.snapshot_id)
            
            # 5. Verify data is recovered
            validation = manager.validator.validate_data_consistency()
            assert validation["summary"]["overall_status"] == "PASS" 