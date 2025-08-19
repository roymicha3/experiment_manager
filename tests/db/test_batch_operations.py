"""
Basic database operations tests for batch functionality.

These tests focus on the core database operations for batch tracking,
separate from the full pipeline integration tests.
"""

import pytest
import tempfile
from datetime import datetime

from experiment_manager.db.manager import DatabaseManager
from experiment_manager.db.tables import Batch


class TestBatchDatabaseOperations:
    """Test suite for core batch database operations."""
    
    @pytest.fixture
    def db_manager(self):
        """Create a temporary database manager for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        db = DatabaseManager(database_path=db_path, use_sqlite=True, recreate=True)
        yield db
        
        try:
            db.connection.close()
        except:
            pass
    
    def test_create_batch(self, db_manager):
        """Test basic batch creation."""
        # Set up experiment hierarchy
        experiment = db_manager.create_experiment("Test Experiment", "Testing batch creation")
        trial = db_manager.create_trial(experiment.id, "Test Trial")
        trial_run = db_manager.create_trial_run(trial_id=trial.id)
        
        # Create epoch first
        db_manager.create_epoch(epoch_idx=0, trial_run_id=trial_run.id)
        
        # Create batch
        db_manager.create_batch(batch_idx=0, epoch_idx=0, trial_run_id=trial_run.id)
        
        # Verify batch was created
        cursor = db_manager._execute_query(
            "SELECT * FROM BATCH WHERE idx = ? AND epoch_idx = ? AND trial_run_id = ?",
            (0, 0, trial_run.id)
        )
        batch_record = cursor.fetchone()
        
        assert batch_record is not None, "Batch should be created"
        assert batch_record["idx"] == 0
        assert batch_record["epoch_idx"] == 0
        assert batch_record["trial_run_id"] == trial_run.id
    
    def test_multiple_batches_per_epoch(self, db_manager):
        """Test creating multiple batches within the same epoch."""
        # Set up experiment hierarchy
        experiment = db_manager.create_experiment("Test Experiment", "Testing multiple batches")
        trial = db_manager.create_trial(experiment.id, "Test Trial")
        trial_run = db_manager.create_trial_run(trial_id=trial.id)
        
        # Create epoch
        db_manager.create_epoch(epoch_idx=0, trial_run_id=trial_run.id)
        
        # Create multiple batches
        batch_count = 5
        for batch_idx in range(batch_count):
            db_manager.create_batch(batch_idx=batch_idx, epoch_idx=0, trial_run_id=trial_run.id)
        
        # Verify all batches were created
        cursor = db_manager._execute_query(
            "SELECT COUNT(*) as count FROM BATCH WHERE epoch_idx = ? AND trial_run_id = ?",
            (0, trial_run.id)
        )
        result = cursor.fetchone()
        assert result["count"] == batch_count, f"Expected {batch_count} batches, got {result['count']}"
    
    def test_batch_metric_linking(self, db_manager):
        """Test linking metrics to batches."""
        # Set up experiment hierarchy
        experiment = db_manager.create_experiment("Test Experiment", "Testing batch metrics")
        trial = db_manager.create_trial(experiment.id, "Test Trial")
        trial_run = db_manager.create_trial_run(trial_id=trial.id)
        
        # Create epoch and batch
        db_manager.create_epoch(epoch_idx=0, trial_run_id=trial_run.id)
        db_manager.create_batch(batch_idx=0, epoch_idx=0, trial_run_id=trial_run.id)
        
        # Create and link metric to batch
        metric = db_manager.record_metric(0.95, "batch_accuracy", {"details": "test"})
        db_manager.add_batch_metric(
            batch_idx=0, 
            epoch_idx=0, 
            trial_run_id=trial_run.id, 
            metric_id=metric.id
        )
        
        # Verify metric is linked to batch
        cursor = db_manager._execute_query(
            "SELECT COUNT(*) as count FROM BATCH_METRIC WHERE batch_idx = ? AND epoch_idx = ? AND trial_run_id = ?",
            (0, 0, trial_run.id)
        )
        result = cursor.fetchone()
        assert result["count"] == 1, "Metric should be linked to batch"
    
    def test_batch_artifact_linking(self, db_manager):
        """Test linking artifacts to batches."""
        # Set up experiment hierarchy
        experiment = db_manager.create_experiment("Test Experiment", "Testing batch artifacts")
        trial = db_manager.create_trial(experiment.id, "Test Trial")
        trial_run = db_manager.create_trial_run(trial_id=trial.id)
        
        # Create epoch and batch
        db_manager.create_epoch(epoch_idx=0, trial_run_id=trial_run.id)
        db_manager.create_batch(batch_idx=0, epoch_idx=0, trial_run_id=trial_run.id)
        
        # Create and link artifact to batch
        artifact = db_manager.record_artifact("checkpoint", "batch_checkpoint.pth")
        db_manager.link_batch_artifact(
            batch_idx=0,
            epoch_idx=0, 
            trial_run_id=trial_run.id,
            artifact_id=artifact.id
        )
        
        # Verify artifact is linked to batch
        cursor = db_manager._execute_query(
            "SELECT COUNT(*) as count FROM BATCH_ARTIFACT WHERE batch_idx = ? AND epoch_idx = ? AND trial_run_id = ?",
            (0, 0, trial_run.id)
        )
        result = cursor.fetchone()
        assert result["count"] == 1, "Artifact should be linked to batch"
        
        # Test getting batch artifacts
        artifacts = db_manager.get_batch_artifacts(
            batch_idx=0,
            epoch_idx=0,
            trial_run_id=trial_run.id
        )
        assert len(artifacts) == 1, "Should retrieve one artifact"
        assert artifacts[0].location == "batch_checkpoint.pth"
    
    def test_foreign_key_constraints(self, db_manager):
        """Test that foreign key constraints work properly for batches."""
        # Set up experiment hierarchy
        experiment = db_manager.create_experiment("Test Experiment", "Testing constraints")
        trial = db_manager.create_trial(experiment.id, "Test Trial")
        trial_run = db_manager.create_trial_run(trial_id=trial.id)
        
        # Create epoch
        db_manager.create_epoch(epoch_idx=0, trial_run_id=trial_run.id)
        
        # Try to create batch with invalid epoch reference
        with pytest.raises(Exception):
            db_manager.create_batch(batch_idx=0, epoch_idx=999, trial_run_id=trial_run.id)
        
        # Try to create batch with invalid trial_run reference  
        with pytest.raises(Exception):
            db_manager.create_batch(batch_idx=0, epoch_idx=0, trial_run_id=999)
    
    def test_batch_ordering(self, db_manager):
        """Test that batches maintain proper ordering within epochs."""
        # Set up experiment hierarchy
        experiment = db_manager.create_experiment("Test Experiment", "Testing batch ordering")
        trial = db_manager.create_trial(experiment.id, "Test Trial")
        trial_run = db_manager.create_trial_run(trial_id=trial.id)
        
        # Create epoch
        db_manager.create_epoch(epoch_idx=0, trial_run_id=trial_run.id)
        
        # Create batches in specific order
        batch_indices = [0, 1, 2, 3, 4]
        for batch_idx in batch_indices:
            db_manager.create_batch(batch_idx=batch_idx, epoch_idx=0, trial_run_id=trial_run.id)
        
        # Verify batches maintain order
        cursor = db_manager._execute_query(
            "SELECT idx FROM BATCH WHERE epoch_idx = ? AND trial_run_id = ? ORDER BY idx",
            (0, trial_run.id)
        )
        retrieved_indices = [row["idx"] for row in cursor.fetchall()]
        assert retrieved_indices == batch_indices, "Batches should maintain ordering"


if __name__ == "__main__":
    # Run the test directly for debugging
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
