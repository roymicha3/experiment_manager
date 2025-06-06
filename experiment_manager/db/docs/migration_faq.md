# Migration System FAQ

## General Questions

### Q: How often should I run database validation?
**A:** For production databases, run validation weekly. For development databases, run before any major changes or monthly. Use automated scripts for regular validation:

```bash
# Weekly validation script
#!/bin/bash
DATE=$(date +%Y%m%d)
python -m experiment_manager.db.data_migration_cli validate \
  -d prod_experiments --mysql --host prod-db.company.com --user analyst \
  -o "weekly_validation_${DATE}.json"
```

### Q: How long do snapshots take to create?
**A:** Snapshot creation time depends on database size:
- Small databases (<100MB): 1-5 seconds
- Medium databases (1-5GB): 30 seconds - 2 minutes  
- Large databases (>10GB): 5-15 minutes

### Q: Can I run migrations on a live database?
**A:** Yes, but it's recommended to:
- Run during low-usage periods
- Use conservative migration strategy
- Monitor system resources
- Notify users of potential slowdowns

### Q: What happens if I interrupt a migration?
**A:** Migrations are designed to be interruptible. The system will:
- Complete the current batch
- Save progress state
- Allow resuming from the last completed batch

### Q: How do I migrate between different database types (SQLite ↔ MySQL)?
**A:** Use the export/import approach:

```bash
# 1. Export from source database
python -m experiment_manager.db.data_migration_cli export-experiment \
  -d research.db -e 1 -o experiment_export.json --include-metrics --include-artifacts

# 2. Import to target database using programmatic API
```

```python
from experiment_manager.db.data_migration import DataMigrationManager
from experiment_manager.db.manager import DatabaseManager
import json

# Load exported data
with open('experiment_export.json', 'r') as f:
    experiment_data = json.load(f)

# Import to target database
target_db = DatabaseManager(database_path="production", 
                           host="prod-db.company.com", 
                           user="analyst", use_sqlite=False)
migration_manager = DataMigrationManager(target_db)
new_exp_id = migration_manager.import_experiment_data(experiment_data)
```

## Technical Questions

### Q: Can I customize the validation checks?
**A:** Yes, you can extend the validation system:

```python
from experiment_manager.db.data_migration import DataValidator

class CustomValidator(DataValidator):
    def validate_custom_business_rules(self):
        """Add your custom validation logic."""
        issues = []
        
        # Example: Check for experiments without trials
        cursor = self.db_manager._execute_query("""
            SELECT e.id, e.title 
            FROM EXPERIMENT e 
            LEFT JOIN TRIAL t ON e.id = t.experiment_id 
            WHERE t.id IS NULL
        """)
        
        orphaned_experiments = cursor.fetchall()
        for exp in orphaned_experiments:
            issues.append(f"Experiment {exp['id']} has no trials")
        
        return issues

# Use custom validator
migration_manager.validator = CustomValidator(db_manager)
validation_results = migration_manager.validator.validate_custom_business_rules()
```

### Q: How do I handle very large experiments (millions of records)?
**A:** For large experiments:

```python
# Use aggressive migration strategy
result = migrator.migrate_experiment(
    source_experiment_id=1,
    strategy=MigrationStrategy.AGGRESSIVE,
    batch_size=10000,  # Large batch size
    progress_callback=None  # Disable progress callbacks
)

# Or use chunked processing
def process_large_experiment(migrator, exp_id, chunk_size=50000):
    cursor = migrator.db_manager._execute_query(
        "SELECT COUNT(*) FROM TRIAL_RUN WHERE trial_id IN (SELECT id FROM TRIAL WHERE experiment_id = ?)",
        (exp_id,)
    )
    total_runs = cursor.fetchone()[0]
    
    for offset in range(0, total_runs, chunk_size):
        print(f"Processing chunk {offset//chunk_size + 1}/{(total_runs//chunk_size) + 1}")
        process_chunk(migrator, exp_id, offset, chunk_size)
```

### Q: What file formats are supported for transformation rules?
**A:** The system supports JSON format for transformation rules:

```json
{
  "metric_transformations": [
    {
      "metric_type": "accuracy",
      "transformation": "legacy_to_per_label",
      "rules": {
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
    }
  ],
  "global_settings": {
    "backup_original": true,
    "validate_after_transform": true
  }
}
```

## Troubleshooting Questions

### Q: Why is my migration running slowly?
**A:** Common causes and solutions:

| Problem | Solution |
|---------|----------|
| Small batch size | Increase `--batch-size` to 2000-5000 |
| Conservative strategy | Use `--strategy balanced` or `aggressive` |
| Database locks | Run during low-usage periods |
| Insufficient resources | Monitor CPU/memory, close other applications |
| Progress callbacks | Disable callbacks for better performance |

### Q: How do I recover from a failed migration?
**A:** Follow the error recovery procedure:

```bash
# 1. Check error logs
grep "ERROR" ~/.experiment_manager/logs/migration_*.log | tail -20

# 2. Restore from pre-migration snapshot
python -m experiment_manager.db.data_migration_cli restore-snapshot \
  -d experiment.db --snapshot-id 20231201_143022

# 3. Validate restored state
python -m experiment_manager.db.data_migration_cli validate -d experiment.db

# 4. Investigate and fix root cause before retrying
```

### Q: What if I accidentally delete important data?
**A:** Recovery options in order of preference:

1. **Restore from snapshot** (if available):
   ```bash
   python -m experiment_manager.db.data_migration_cli restore-snapshot \
     -d experiment.db --snapshot-id <most_recent_snapshot>
   ```

2. **Restore from exported JSON** (if available):
   ```python
   migration_manager.import_experiment_data(exported_data)
   ```

3. **Contact database administrator** for backup restoration

4. **Partial recovery from logs** (if detailed logging was enabled)

### Q: How do I fix "foreign key violation" errors?
**A:** Use the automated cleanup approach:

```python
from experiment_manager.db.manager import DatabaseManager
from experiment_manager.db.data_migration import DataMigrationManager

db = DatabaseManager(database_path="experiment.db", use_sqlite=True)
migrator = DataMigrationManager(db)

# Create safety snapshot
snapshot_id = migrator.snapshot_manager.create_snapshot("Before FK cleanup")

# Run validation to identify issues
validation = migrator.validator.validate_data_consistency()

# Clean up orphaned records
cleanup_foreign_key_violations(db, validation['foreign_key_violations'])

def cleanup_foreign_key_violations(db, violations):
    """Clean up foreign key violations systematically."""
    
    cleanup_queries = {
        'TRIAL_RUN': "DELETE FROM TRIAL_RUN WHERE trial_id NOT IN (SELECT id FROM TRIAL)",
        'EPOCH': "DELETE FROM EPOCH WHERE trial_run_id NOT IN (SELECT id FROM TRIAL_RUN)",
        'RESULTS': "DELETE FROM RESULTS WHERE trial_run_id NOT IN (SELECT id FROM TRIAL_RUN)",
        'TRIAL_ARTIFACT': "DELETE FROM TRIAL_ARTIFACT WHERE trial_id NOT IN (SELECT id FROM TRIAL)",
        'RESULTS_METRIC': "DELETE FROM RESULTS_METRIC WHERE results_id NOT IN (SELECT trial_run_id FROM RESULTS)"
    }
    
    for table, query in cleanup_queries.items():
        if table in violations and violations[table]:
            cursor = db._execute_query(query)
            print(f"Cleaned up {cursor.rowcount} orphaned records from {table}")
```

## Best Practices Questions

### Q: Should I use SQLite or MySQL for production?
**A:** Use MySQL for production environments because:

**MySQL Advantages:**
- Better concurrent access support
- Superior performance for large datasets  
- Built-in backup and replication features
- Better security and access control
- Professional database administration tools

**SQLite Advantages:**
- Zero configuration
- Perfect for development and testing
- Single file database
- No separate server process

**Recommendation:**
- **Development/Testing:** SQLite
- **Production:** MySQL
- **Single-user scenarios:** SQLite
- **Multi-user/team environments:** MySQL

### Q: How do I set up automated backups?
**A:** Create scheduled scripts for different backup frequencies:

```bash
#!/bin/bash
# daily_backup.sh - Daily snapshot creation

DATE=$(date +%Y%m%d)
TIMESTAMP=$(date +%H%M%S)

python -m experiment_manager.db.data_migration_cli create-snapshot \
  -d production_experiments --mysql --host prod-db.company.com --user backup_user \
  --description "Daily automated backup ${DATE}_${TIMESTAMP}"

# Log the operation
echo "$(date): Daily backup completed" >> /var/log/experiment_backups.log
```

```bash
#!/bin/bash
# weekly_validation.sh - Weekly validation and cleanup

DATE=$(date +%Y%m%d)

# Run validation
python -m experiment_manager.db.data_migration_cli validate \
  -d production_experiments --mysql --host prod-db.company.com --user backup_user \
  -o "/backups/validation_reports/weekly_${DATE}.json"

# Check validation results and alert if issues found
python check_validation_results.py "/backups/validation_reports/weekly_${DATE}.json"
```

**Crontab setup:**
```bash
# Add to crontab (crontab -e)
# Daily backup at 2 AM
0 2 * * * /scripts/daily_backup.sh

# Weekly validation on Sundays at 3 AM  
0 3 * * 0 /scripts/weekly_validation.sh

# Monthly deep backup (first day of month at 1 AM)
0 1 1 * * /scripts/monthly_backup.sh
```

### Q: What's the recommended retention policy for snapshots?
**A:** Suggested retention policy:

| Frequency | Retention Period | Purpose |
|-----------|------------------|---------|
| Daily snapshots | 30 days | Recent recovery |
| Weekly snapshots | 12 weeks | Medium-term recovery |
| Monthly snapshots | 12 months | Long-term recovery |
| Major milestone snapshots | Indefinitely | Project milestones |

**Implementation:**
```python
def cleanup_old_snapshots(migrator, retention_policy):
    """Clean up snapshots according to retention policy."""
    from datetime import datetime, timedelta
    
    snapshots = migrator.snapshot_manager.list_snapshots()
    now = datetime.now()
    
    for snapshot in snapshots:
        snapshot_date = datetime.fromisoformat(snapshot['created_at'])
        age = now - snapshot_date
        
        # Apply retention rules
        should_delete = False
        
        if 'daily' in snapshot['description']:
            should_delete = age > timedelta(days=30)
        elif 'weekly' in snapshot['description']:
            should_delete = age > timedelta(weeks=12)
        elif 'monthly' in snapshot['description']:
            should_delete = age > timedelta(days=365)
        elif 'milestone' in snapshot['description']:
            should_delete = False  # Keep forever
            
        if should_delete:
            migrator.snapshot_manager.delete_snapshot(snapshot['id'])
            print(f"Deleted old snapshot: {snapshot['id']}")
```

### Q: How do I handle schema changes?
**A:** For schema evolution:

```python
def handle_schema_migration(migrator, new_schema_version):
    """Handle database schema migration safely."""
    
    # 1. Create comprehensive backup
    backup_id = migrator.snapshot_manager.create_snapshot(
        f"Pre-schema-migration to v{new_schema_version}"
    )
    
    # 2. Validate current state
    validation = migrator.validator.validate_data_consistency()
    if validation['summary']['overall_status'] != 'PASS':
        raise Exception("Database validation failed before schema migration")
    
    # 3. Test schema changes on copy
    test_db_path = f"test_schema_{new_schema_version}.db"
    create_test_database_copy(migrator.db_manager, test_db_path)
    
    # 4. Apply schema changes to test database
    apply_schema_changes(test_db_path, new_schema_version)
    
    # 5. Validate test database
    test_db = DatabaseManager(database_path=test_db_path, use_sqlite=True)
    test_migrator = DataMigrationManager(test_db)
    test_validation = test_migrator.validator.validate_data_consistency()
    
    if test_validation['summary']['overall_status'] != 'PASS':
        raise Exception("Schema migration failed validation on test database")
    
    # 6. Apply to production during maintenance window
    print("Schema changes validated. Ready for production deployment.")
    return backup_id

def apply_schema_changes(db_path, version):
    """Apply schema changes for specific version."""
    schema_changes = {
        '2.0': [
            "ALTER TABLE METRIC ADD COLUMN metadata TEXT",
            "CREATE INDEX idx_metric_type ON METRIC(type)",
            "UPDATE METRIC SET metadata = '{}' WHERE metadata IS NULL"
        ],
        '2.1': [
            "ALTER TABLE EXPERIMENT ADD COLUMN tags TEXT",
            "CREATE TABLE EXPERIMENT_TAG (experiment_id INTEGER, tag_name TEXT)"
        ]
    }
    
    if version in schema_changes:
        db = DatabaseManager(database_path=db_path, use_sqlite=True)
        for sql in schema_changes[version]:
            db._execute_query(sql)
```

### Q: How do I monitor migration performance?
**A:** Implement comprehensive monitoring:

```python
class MigrationMonitor:
    """Monitor migration performance and health."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics = {
            'items_processed': 0,
            'errors': 0,
            'memory_peak': 0,
            'batch_times': []
        }
    
    def log_batch_completion(self, batch_size, duration, memory_usage):
        """Log metrics for completed batch."""
        self.metrics['items_processed'] += batch_size
        self.metrics['batch_times'].append(duration)
        self.metrics['memory_peak'] = max(self.metrics['memory_peak'], memory_usage)
        
        # Calculate performance metrics
        avg_batch_time = sum(self.metrics['batch_times']) / len(self.metrics['batch_times'])
        items_per_second = batch_size / duration if duration > 0 else 0
        
        print(f"Batch completed: {batch_size} items in {duration:.2f}s "
              f"({items_per_second:.1f} items/s)")
    
    def generate_report(self):
        """Generate performance report."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        overall_rate = self.metrics['items_processed'] / total_time if total_time > 0 else 0
        
        return {
            'total_items': self.metrics['items_processed'],
            'total_time': total_time,
            'overall_rate': overall_rate,
            'peak_memory_mb': self.metrics['memory_peak'] / 1024 / 1024,
            'error_count': self.metrics['errors'],
            'average_batch_time': sum(self.metrics['batch_times']) / len(self.metrics['batch_times'])
        }

# Use monitoring in migrations
monitor = MigrationMonitor()

def monitored_migration(migrator, source_id):
    """Run migration with performance monitoring."""
    import psutil
    
    def progress_callback(progress):
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        monitor.log_batch_completion(
            batch_size=100,  # Your batch size
            duration=1.5,    # Time for this batch
            memory_usage=memory_mb
        )
    
    result = migrator.migrate_experiment(
        source_experiment_id=source_id,
        progress_callback=progress_callback
    )
    
    # Generate performance report
    report = monitor.generate_report()
    print(f"Migration completed: {report}")
    
    return result, report
```

This FAQ covers the most common questions analysts have when working with the migration system, providing practical answers and code examples for immediate use. 