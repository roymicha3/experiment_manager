# Migration System API Reference

## Overview

This document provides a comprehensive reference for the programmatic API of the Experiment Manager Migration System. Use these APIs when you need to integrate migration functionality into your own scripts or applications.

## Core Classes

### DataMigrationManager

The main interface for all migration operations.

```python
from experiment_manager.db.data_migration import DataMigrationManager
from experiment_manager.db.manager import DatabaseManager

# Initialize
db_manager = DatabaseManager(database_path="experiment.db", use_sqlite=True)
migration_manager = DataMigrationManager(db_manager, snapshot_dir="./snapshots")
```

#### Constructor Parameters

- `db_manager`: `DatabaseManager` instance
- `snapshot_dir`: Directory for storing snapshots (default: "snapshots")
- `enable_logging`: Enable detailed logging (default: True)

#### Key Methods

##### Validation Methods

```python
# Validate data consistency
validation_results = migration_manager.validator.validate_data_consistency()
# Returns: dict with validation results and error details

# Check hierarchy integrity for specific experiment
is_valid, issues = migration_manager.hierarchy_preserver.validate_hierarchy_integrity(experiment_id)
# Returns: (bool, List[str]) - validity status and list of issues

# Validate JSON metrics
json_issues = migration_manager.validator.validate_json_metrics()
# Returns: List[dict] with metric validation issues
```

##### Snapshot Methods

```python
# Create snapshot
snapshot_id = migration_manager.snapshot_manager.create_snapshot("Description")
# Returns: str - unique snapshot ID

# List all snapshots
snapshots = migration_manager.snapshot_manager.list_snapshots()
# Returns: List[dict] with snapshot information

# Restore snapshot
migration_manager.snapshot_manager.restore_snapshot("20231201_143022")
# Returns: bool - success status

# Delete old snapshots
migration_manager.snapshot_manager.cleanup_old_snapshots(keep_count=10)
# Returns: int - number of snapshots deleted
```

##### Migration Methods

```python
# Migrate entire experiment
def progress_callback(progress: MigrationProgress):
    print(f"Progress: {progress.completion_percentage:.1f}%")

result = migration_manager.migrate_experiment(
    source_experiment_id=1,
    target_experiment_id=None,  # Creates new experiment
    strategy=MigrationStrategy.BALANCED,
    progress_callback=progress_callback,
    transformation_rules=None
)
# Returns: MigrationResult with details

# Transform metrics with custom rules
transformation_rules = {
    "metric_transformations": [
        {
            "metric_type": "accuracy",
            "transformation": "legacy_to_per_label"
        }
    ]
}
result = migration_manager.transform_metrics(
    experiment_ids=[1, 2, 3],
    transformation_rules=transformation_rules,
    batch_size=1000
)
# Returns: TransformationResult with statistics
```

##### Export/Import Methods

```python
# Export experiment data
experiment_data = migration_manager.export_experiment(
    experiment_id=1,
    include_metrics=True,
    include_artifacts=True,
    format='json'
)
# Returns: dict with complete experiment data

# Import experiment data
new_experiment_id = migration_manager.import_experiment_data(
    experiment_data,
    preserve_ids=False,
    validate_before_import=True
)
# Returns: int - new experiment ID
```

### MigrationProgress

Tracks and provides estimates for migration operations.

```python
class MigrationProgress:
    processed_items: int          # Items processed so far
    total_items: int              # Total items to process
    completion_percentage: float  # Completion percentage (0-100)
    success_rate: float          # Success rate (0-100)
    current_operation: str       # Current operation description
    estimated_completion: datetime # Estimated completion time
    errors: List[str]           # List of errors encountered
    start_time: datetime        # Operation start time
    
    def get_elapsed_time(self) -> timedelta:
        """Get elapsed time since start."""
        
    def get_remaining_time(self) -> timedelta:
        """Get estimated remaining time."""
        
    def add_error(self, error: str) -> None:
        """Add error to the progress tracker."""
```

### MigrationStrategy

Enumeration of available migration strategies.

```python
from experiment_manager.db.data_migration import MigrationStrategy

# Available strategies
MigrationStrategy.CONSERVATIVE  # Safest, slowest
MigrationStrategy.BALANCED     # Default, good balance
MigrationStrategy.AGGRESSIVE   # Fastest, least validation
```

### DataValidator

Handles data validation operations.

```python
# Access through migration manager
validator = migration_manager.validator

# Validate foreign key integrity
fk_violations = validator.validate_foreign_keys()
# Returns: dict mapping table names to violation lists

# Validate JSON metric structure
json_issues = validator.validate_json_metrics()
# Returns: List[dict] with validation issues

# Validate experiment hierarchy
hierarchy_issues = validator.validate_experiment_hierarchy(experiment_id)
# Returns: List[str] with hierarchy problems

# Custom validation
class CustomValidator(DataValidator):
    def validate_business_rules(self):
        # Add your custom validation logic
        pass
```

### SnapshotManager

Manages database snapshots for backup and rollback.

```python
# Access through migration manager
snapshot_manager = migration_manager.snapshot_manager

# Create snapshot with metadata
snapshot_id = snapshot_manager.create_snapshot(
    description="Pre-migration backup",
    metadata={"experiment_count": 10, "migration_type": "metric_transform"}
)

# List snapshots with filtering
snapshots = snapshot_manager.list_snapshots(
    since=datetime(2023, 12, 1),
    pattern="migration_*"
)

# Get snapshot details
details = snapshot_manager.get_snapshot_details("20231201_143022")
# Returns: dict with snapshot metadata and statistics
```

### MetricTransformer

Handles metric data transformations.

```python
from experiment_manager.db.data_migration import MetricTransformer

transformer = MetricTransformer(db_manager)

# Transform specific metrics
result = transformer.transform_metrics(
    metric_ids=[1, 2, 3],
    transformation_type="legacy_to_per_label",
    transformation_params={"default_classes": ["class_0", "class_1"]}
)

# Batch transform by experiment
result = transformer.batch_transform_experiment(
    experiment_id=1,
    transformations=[
        {
            "metric_type": "accuracy",
            "transformation": "normalize_json_keys",
            "params": {"key_mapping": {"cls_0": "class_0"}}
        }
    ]
)
```

## Data Structures

### Transformation Rules Format

```python
transformation_rules = {
    "metric_transformations": [
        {
            "metric_type": "accuracy",           # Target metric type
            "transformation": "legacy_to_per_label",  # Transformation type
            "rules": {                          # Transformation parameters
                "split_by": "class_",
                "default_classes": ["class_0", "class_1", "class_2"]
            }
        },
        {
            "metric_type": "f1_score",
            "transformation": "normalize_json_keys",
            "rules": {
                "key_mapping": {
                    "macro": "macro_avg",
                    "micro": "micro_avg"
                }
            }
        },
        {
            "metric_type": "loss",
            "transformation": "add_metadata",
            "rules": {
                "metadata": {
                    "aggregation_method": "mean",
                    "computed_at": "epoch_end"
                }
            }
        }
    ],
    "global_settings": {
        "backup_original": True,
        "validate_after_transform": True,
        "rollback_on_error": True
    }
}
```

### Export Data Format

```python
exported_data = {
    "experiment": {
        "id": 1,
        "title": "MNIST Training",
        "description": "Training on MNIST dataset",
        "start_time": "2023-12-01T10:00:00",
        "update_time": "2023-12-01T15:30:00"
    },
    "trials": [
        {
            "id": 1,
            "name": "SGD Optimizer",
            "experiment_id": 1,
            "start_time": "2023-12-01T10:05:00",
            "trial_runs": [
                {
                    "id": 1,
                    "trial_id": 1,
                    "status": "completed",
                    "start_time": "2023-12-01T10:05:00",
                    "epochs": [...],      # If include_metrics=True
                    "results": {...},     # If include_metrics=True
                    "artifacts": [...]    # If include_artifacts=True
                }
            ]
        }
    ],
    "artifacts": [...],  # Experiment-level artifacts
    "metadata": {
        "export_timestamp": "2023-12-01T16:00:00",
        "source_database": "experiment.db",
        "include_metrics": True,
        "include_artifacts": True
    }
}
```

## Error Handling

### Exception Classes

```python
from experiment_manager.db.data_migration import (
    DataMigrationError,      # Base migration error
    ValidationError,         # Data validation failed
    SnapshotError,          # Snapshot operation failed
    TransformationError,    # Metric transformation failed
    ImportExportError       # Data import/export failed
)

try:
    result = migration_manager.migrate_experiment(source_id=1)
except ValidationError as e:
    print(f"Validation failed: {e.validation_errors}")
except SnapshotError as e:
    print(f"Snapshot operation failed: {e}")
except DataMigrationError as e:
    print(f"Migration failed: {e}")
```

### Error Recovery

```python
# Example error recovery workflow
def safe_migration_with_recovery(migration_manager, source_id):
    # Create safety snapshot
    snapshot_id = migration_manager.snapshot_manager.create_snapshot(
        f"Pre-migration safety backup for experiment {source_id}"
    )
    
    try:
        # Attempt migration
        result = migration_manager.migrate_experiment(source_id)
        return result
        
    except DataMigrationError as e:
        # Log error
        logging.error(f"Migration failed: {e}")
        
        # Restore snapshot
        migration_manager.snapshot_manager.restore_snapshot(snapshot_id)
        
        # Re-raise with context
        raise DataMigrationError(f"Migration failed and restored to snapshot {snapshot_id}") from e
```

## Usage Examples

### Example 1: Complete Migration Workflow

```python
from experiment_manager.db.data_migration import DataMigrationManager
from experiment_manager.db.manager import DatabaseManager
import logging

# Setup
logging.basicConfig(level=logging.INFO)
db = DatabaseManager(database_path="production.db", use_sqlite=True)
migrator = DataMigrationManager(db)

# 1. Pre-migration validation
print("Validating database...")
validation = migrator.validator.validate_data_consistency()
if validation["summary"]["overall_status"] != "PASS":
    print("⚠️  Validation issues found, please review")
    
# 2. Create safety snapshot
print("Creating safety snapshot...")
snapshot_id = migrator.snapshot_manager.create_snapshot("Pre-migration backup")

# 3. Perform migration with progress tracking
def show_progress(progress):
    print(f"\r{progress.current_operation}: {progress.completion_percentage:.1f}%", end="")

try:
    print("Starting migration...")
    result = migrator.migrate_experiment(
        source_experiment_id=1,
        strategy=MigrationStrategy.BALANCED,
        progress_callback=show_progress
    )
    print(f"\n✅ Migration completed successfully!")
    print(f"   New experiment ID: {result.new_experiment_id}")
    print(f"   Processed items: {result.items_processed}")
    
except Exception as e:
    print(f"\n❌ Migration failed: {e}")
    print("Restoring from snapshot...")
    migrator.snapshot_manager.restore_snapshot(snapshot_id)
    print("Database restored to pre-migration state")
```

### Example 2: Metric Transformation

```python
# Define transformation rules
transform_rules = {
    "metric_transformations": [
        {
            "metric_type": "accuracy",
            "transformation": "legacy_to_per_label",
            "rules": {
                "split_by": "class_",
                "default_classes": ["class_0", "class_1", "class_2"]
            }
        }
    ]
}

# Apply transformations
result = migrator.transform_metrics(
    experiment_ids=[1, 2, 3],
    transformation_rules=transform_rules,
    batch_size=500
)

print(f"Transformed {result.metrics_updated} metrics")
print(f"Success rate: {result.success_rate:.1f}%")
```

### Example 3: Cross-Database Migration

```python
# Source database (SQLite)
source_db = DatabaseManager(database_path="research.db", use_sqlite=True)
source_migrator = DataMigrationManager(source_db)

# Target database (MySQL)
target_db = DatabaseManager(
    database_path="production",
    host="prod-db.company.com",
    user="migrator",
    password="secure_password",
    use_sqlite=False
)
target_migrator = DataMigrationManager(target_db)

# Export from source
experiment_data = source_migrator.export_experiment(
    experiment_id=42,
    include_metrics=True,
    include_artifacts=True
)

# Import to target
new_id = target_migrator.import_experiment_data(
    experiment_data,
    validate_before_import=True
)

print(f"Migrated experiment 42 to production as experiment {new_id}")
```

## Best Practices

### 1. Always Use Progress Callbacks for Long Operations

```python
def detailed_progress_callback(progress: MigrationProgress):
    elapsed = progress.get_elapsed_time()
    remaining = progress.get_remaining_time()
    
    print(f"{progress.current_operation}")
    print(f"Progress: {progress.completion_percentage:.1f}% "
          f"({progress.processed_items}/{progress.total_items})")
    print(f"Success Rate: {progress.success_rate:.1f}%")
    print(f"Elapsed: {elapsed} | Remaining: {remaining}")
    
    if progress.errors:
        print(f"Recent errors: {len(progress.errors)}")
```

### 2. Implement Robust Error Handling

```python
def robust_migration(migrator, source_id, max_retries=3):
    for attempt in range(max_retries):
        snapshot_id = migrator.snapshot_manager.create_snapshot(
            f"Attempt {attempt + 1} backup"
        )
        
        try:
            return migrator.migrate_experiment(source_id)
        except DataMigrationError as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
                migrator.snapshot_manager.restore_snapshot(snapshot_id)
            else:
                raise
```

### 3. Use Validation Before and After Operations

```python
def validated_operation(migrator, operation_func):
    # Pre-operation validation
    pre_validation = migrator.validator.validate_data_consistency()
    assert pre_validation["summary"]["overall_status"] == "PASS"
    
    # Perform operation
    result = operation_func()
    
    # Post-operation validation
    post_validation = migrator.validator.validate_data_consistency()
    assert post_validation["summary"]["overall_status"] == "PASS"
    
    return result
```

This API reference provides comprehensive coverage of the migration system's programmatic interface, enabling analysts and developers to integrate migration functionality into their workflows effectively. 