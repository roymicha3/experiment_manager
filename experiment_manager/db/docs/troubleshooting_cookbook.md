# Migration Troubleshooting Cookbook

## Overview

This cookbook provides step-by-step solutions for common issues encountered when using the Experiment Manager migration system. Each scenario includes symptoms, root causes, and detailed resolution steps.

## Common Issues Index

1. [Foreign Key Violations](#foreign-key-violations)
2. [JSON Metric Parsing Errors](#json-metric-parsing-errors)
3. [Migration Timeouts](#migration-timeouts)
4. [Snapshot Corruption](#snapshot-corruption)
5. [Memory Issues with Large Datasets](#memory-issues-with-large-datasets)
6. [Connection Problems](#connection-problems)
7. [Incomplete Migrations](#incomplete-migrations)
8. [Performance Degradation](#performance-degradation)

---

## Foreign Key Violations

### Symptoms
```
❌ Validation Results:
  Overall Status: FAIL
  Foreign Key Violations: 15
  
TRIAL_RUN table:
  - Record 245: trial_id=99 references non-existent TRIAL
  - Record 246: trial_id=99 references non-existent TRIAL
```

### Root Causes
- Incomplete data deletion (parent records removed but children remain)
- Data corruption during previous operations
- Manual database modifications outside the system
- Failed rollback operations

### Solution Steps

#### Step 1: Identify All Violations
```bash
python -m experiment_manager.db.data_migration_cli validate \
  -d experiment.db -o violation_report.json
```

#### Step 2: Analyze the Report
```python
import json
with open('violation_report.json', 'r') as f:
    report = json.load(f)

# Check foreign key violations by table
for table, violations in report['foreign_key_violations'].items():
    if violations:
        print(f"\n{table}: {len(violations)} violations")
        for violation in violations[:3]:  # Show first 3
            print(f"  - {violation}")
```

#### Step 3: Create Cleanup Script
```python
from experiment_manager.db.manager import DatabaseManager

db = DatabaseManager(database_path="experiment.db", use_sqlite=True)

# Create snapshot before cleanup
from experiment_manager.db.data_migration import DataMigrationManager
migrator = DataMigrationManager(db)
snapshot_id = migrator.snapshot_manager.create_snapshot("Before FK cleanup")

# Remove orphaned trial runs
cursor = db._execute_query("""
    DELETE FROM TRIAL_RUN 
    WHERE trial_id NOT IN (SELECT id FROM TRIAL)
""")
print(f"Removed {cursor.rowcount} orphaned trial runs")

# Remove orphaned epochs
cursor = db._execute_query("""
    DELETE FROM EPOCH 
    WHERE trial_run_id NOT IN (SELECT id FROM TRIAL_RUN)
""")
print(f"Removed {cursor.rowcount} orphaned epochs")

# Remove orphaned results
cursor = db._execute_query("""
    DELETE FROM RESULTS 
    WHERE trial_run_id NOT IN (SELECT id FROM TRIAL_RUN)
""")
print(f"Removed {cursor.rowcount} orphaned results")

# Clean up junction tables
cursor = db._execute_query("""
    DELETE FROM TRIAL_ARTIFACT 
    WHERE trial_id NOT IN (SELECT id FROM TRIAL)
""")
print(f"Removed {cursor.rowcount} orphaned trial-artifact links")
```

#### Step 4: Verify Fix
```bash
python -m experiment_manager.db.data_migration_cli validate -d experiment.db
```

### Prevention
- Always use the migration tools instead of manual database changes
- Create snapshots before any major operations
- Run regular validation checks

---

## JSON Metric Parsing Errors

### Symptoms
```
JSON Metric Issues:
  - Metric 1245 (accuracy): Invalid JSON structure
  - Metric 1247 (f1_score): Unexpected token in JSON
```

### Root Causes
- Single quotes instead of double quotes in JSON
- Unescaped special characters
- Malformed JSON structure from legacy imports
- Encoding issues

### Solution Steps

#### Step 1: Examine Problematic Metrics
```python
from experiment_manager.db.manager import DatabaseManager
import json

db = DatabaseManager(database_path="experiment.db", use_sqlite=True)

# Get problematic metrics
cursor = db._execute_query("""
    SELECT id, type, per_label_val 
    FROM METRIC 
    WHERE id IN (1245, 1247)
""")

for row in cursor.fetchall():
    print(f"Metric {row['id']} ({row['type']}):")
    print(f"  Raw value: {repr(row['per_label_val'])}")
    
    try:
        parsed = json.loads(row['per_label_val'])
        print(f"  ✅ Valid JSON")
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON Error: {e}")
```

#### Step 2: Identify Common JSON Issues
```python
def diagnose_json_issues(raw_json):
    """Diagnose common JSON formatting issues."""
    issues = []
    
    if isinstance(raw_json, str):
        if "'" in raw_json and '"' not in raw_json:
            issues.append("Uses single quotes instead of double quotes")
        
        if raw_json.count('{') != raw_json.count('}'):
            issues.append("Mismatched braces")
            
        if raw_json.count('[') != raw_json.count(']'):
            issues.append("Mismatched brackets")
            
        if '\\' in raw_json and '\\\\' not in raw_json:
            issues.append("Unescaped backslashes")
    
    return issues

# Check all metrics with JSON issues
cursor = db._execute_query("SELECT id, per_label_val FROM METRIC")
problem_metrics = []

for row in cursor.fetchall():
    if row['per_label_val']:
        try:
            json.loads(row['per_label_val'])
        except json.JSONDecodeError:
            issues = diagnose_json_issues(row['per_label_val'])
            problem_metrics.append({
                'id': row['id'],
                'issues': issues,
                'raw': row['per_label_val']
            })

print(f"Found {len(problem_metrics)} metrics with JSON issues")
```

#### Step 3: Fix JSON Issues Systematically
```python
def fix_json_value(raw_json):
    """Attempt to fix common JSON issues."""
    if not raw_json or raw_json.strip() == '':
        return '{}'
    
    # Make a copy to work with
    fixed = str(raw_json)
    
    # Fix single quotes to double quotes
    if "'" in fixed and '"' not in fixed:
        fixed = fixed.replace("'", '"')
    
    # Fix Python-style True/False/None
    fixed = fixed.replace(' True', ' true')
    fixed = fixed.replace(' False', ' false')
    fixed = fixed.replace(' None', ' null')
    
    # Fix trailing commas
    import re
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    
    return fixed

# Create snapshot before fixing
snapshot_id = migrator.snapshot_manager.create_snapshot("Before JSON fixes")

fixed_count = 0
for metric in problem_metrics:
    try:
        fixed_json = fix_json_value(metric['raw'])
        # Validate the fix
        json.loads(fixed_json)
        
        # Update the database
        db._execute_query(
            "UPDATE METRIC SET per_label_val = ? WHERE id = ?",
            (fixed_json, metric['id'])
        )
        fixed_count += 1
        
    except json.JSONDecodeError:
        print(f"Could not fix metric {metric['id']}: {metric['raw']}")

print(f"Fixed {fixed_count} metrics")
```

#### Step 4: Handle Unfixable Metrics
```python
# For metrics that couldn't be auto-fixed, set to empty JSON
cursor = db._execute_query("SELECT id, per_label_val FROM METRIC")

for row in cursor.fetchall():
    if row['per_label_val']:
        try:
            json.loads(row['per_label_val'])
        except json.JSONDecodeError:
            # Set to empty JSON object
            db._execute_query(
                "UPDATE METRIC SET per_label_val = '{}' WHERE id = ?",
                (row['id'],)
            )
            print(f"Reset metric {row['id']} to empty JSON")
```

### Prevention
- Use proper JSON serialization when storing metrics
- Validate JSON before database insertion
- Implement schema validation for metric structures

---

## Migration Timeouts

### Symptoms
```
Migration failed: Operation timed out after 1800 seconds
Progress: 45.2% (2260/5000) - Success: 99.8% - ETA: Unknown
```

### Root Causes
- Very large datasets with small batch sizes
- Database locks or contention
- Insufficient system resources
- Conservative migration strategy with excessive validation

### Solution Steps

#### Step 1: Analyze System Resources
```bash
# Check system resources during migration
# Memory usage
free -h

# Disk I/O
iostat -x 1

# Database size
du -h experiment.db
```

#### Step 2: Optimize Migration Settings
```python
from experiment_manager.db.data_migration import MigrationStrategy

# Use aggressive strategy for better performance
result = migrator.migrate_experiment(
    source_experiment_id=1,
    strategy=MigrationStrategy.AGGRESSIVE,  # Reduced validation
    batch_size=5000,  # Larger batches
    progress_callback=None  # Disable progress callbacks for speed
)
```

#### Step 3: Implement Chunked Migration
```python
def chunked_migration(migrator, source_id, chunk_size=1000):
    """Break large migration into smaller chunks."""
    
    # Get total item count
    cursor = migrator.db_manager._execute_query("""
        SELECT COUNT(*) as total FROM TRIAL_RUN 
        WHERE trial_id IN (
            SELECT id FROM TRIAL WHERE experiment_id = ?
        )
    """, (source_id,))
    total_items = cursor.fetchone()['total']
    
    # Process in chunks
    for offset in range(0, total_items, chunk_size):
        print(f"Processing chunk {offset//chunk_size + 1}")
        
        # Export chunk
        chunk_data = migrator.export_experiment_chunk(
            experiment_id=source_id,
            offset=offset,
            limit=chunk_size
        )
        
        # Import chunk
        migrator.import_experiment_chunk(chunk_data)
        
        # Small delay to prevent resource exhaustion
        import time
        time.sleep(1)
```

#### Step 4: Alternative: Export/Import Approach
```bash
# For very large migrations, use export/import
python -m experiment_manager.db.data_migration_cli export-experiment \
  -d experiment.db -e 1 -o large_experiment.json \
  --include-metrics --include-artifacts

# Process the JSON file in smaller batches
python process_large_export.py large_experiment.json
```

### Prevention
- Monitor system resources before large migrations
- Use appropriate batch sizes for your system
- Schedule migrations during low-usage periods
- Consider database optimization before migration

---

## Snapshot Corruption

### Symptoms
```
❌ Snapshot restoration failed: Corrupted snapshot file
Snapshot file appears to be truncated or damaged
```

### Root Causes
- Disk space exhaustion during snapshot creation
- System crash during snapshot operation
- File system corruption
- Permission issues

### Solution Steps

#### Step 1: Examine Snapshot Integrity
```bash
# List all snapshots with details
python -m experiment_manager.db.data_migration_cli list-snapshots -d experiment.db
```

```python
import os
from pathlib import Path

# Check snapshot files manually
snapshot_dir = Path("snapshots")
for snapshot_file in snapshot_dir.glob("*.db"):
    size = snapshot_file.stat().st_size
    print(f"{snapshot_file.name}: {size} bytes")
    
    # Check if file is readable
    try:
        with open(snapshot_file, 'rb') as f:
            header = f.read(100)
            if b'SQLite' in header:
                print(f"  ✅ Valid SQLite file")
            else:
                print(f"  ❌ Not a valid SQLite file")
    except Exception as e:
        print(f"  ❌ Cannot read file: {e}")
```

#### Step 2: Find Valid Snapshots
```python
def test_snapshot_integrity(snapshot_path):
    """Test if a snapshot can be opened and queried."""
    try:
        import sqlite3
        conn = sqlite3.connect(snapshot_path)
        cursor = conn.cursor()
        
        # Try a simple query
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        conn.close()
        return True, len(tables)
    except Exception as e:
        return False, str(e)

# Test all snapshots
snapshot_dir = Path("snapshots")
valid_snapshots = []

for snapshot_file in snapshot_dir.glob("*.db"):
    is_valid, info = test_snapshot_integrity(snapshot_file)
    if is_valid:
        valid_snapshots.append((snapshot_file, info))
        print(f"✅ {snapshot_file.name}: {info} tables")
    else:
        print(f"❌ {snapshot_file.name}: {info}")

print(f"\nFound {len(valid_snapshots)} valid snapshots")
```

#### Step 3: Restore from Valid Snapshot
```bash
# Use the most recent valid snapshot
python -m experiment_manager.db.data_migration_cli restore-snapshot \
  -d experiment.db --snapshot-id <valid_snapshot_id>
```

#### Step 4: Prevent Future Corruption
```python
def create_verified_snapshot(migrator, description):
    """Create snapshot with integrity verification."""
    import shutil
    
    # Create snapshot
    snapshot_id = migrator.snapshot_manager.create_snapshot(description)
    
    # Verify immediately after creation
    snapshot_path = Path(migrator.snapshot_dir) / f"{snapshot_id}.db"
    is_valid, info = test_snapshot_integrity(snapshot_path)
    
    if not is_valid:
        # Remove corrupted snapshot
        snapshot_path.unlink()
        raise Exception(f"Snapshot creation failed verification: {info}")
    
    # Create backup copy
    backup_path = Path(migrator.snapshot_dir) / f"{snapshot_id}_backup.db"
    shutil.copy2(snapshot_path, backup_path)
    
    return snapshot_id

# Use verified snapshot creation
snapshot_id = create_verified_snapshot(migrator, "Verified pre-migration backup")
```

### Prevention
- Monitor disk space before creating snapshots
- Implement snapshot verification after creation
- Keep multiple copies of critical snapshots
- Regular filesystem health checks

---

## Memory Issues with Large Datasets

### Symptoms
```
MemoryError: Cannot allocate memory for migration operation
Process killed (OOM killer)
Migration slows down significantly over time
```

### Root Causes
- Loading entire datasets into memory
- Memory leaks in long-running operations
- Insufficient system RAM for dataset size
- Large JSON objects in metrics

### Solution Steps

#### Step 1: Monitor Memory Usage
```python
import psutil
import gc

def monitor_memory():
    """Monitor current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
    print(f"Available: {psutil.virtual_memory().available / 1024 / 1024:.1f} MB")

# Monitor before migration
monitor_memory()
```

#### Step 2: Implement Streaming Processing
```python
def stream_migration(migrator, source_id, batch_size=100):
    """Process migration in small batches to control memory."""
    
    # Get trial runs in batches
    offset = 0
    while True:
        cursor = migrator.db_manager._execute_query("""
            SELECT id FROM TRIAL_RUN 
            WHERE trial_id IN (
                SELECT id FROM TRIAL WHERE experiment_id = ?
            )
            LIMIT ? OFFSET ?
        """, (source_id, batch_size, offset))
        
        trial_runs = cursor.fetchall()
        if not trial_runs:
            break
            
        # Process this batch
        for run in trial_runs:
            process_trial_run(migrator, run['id'])
            
        # Force garbage collection
        gc.collect()
        
        # Monitor memory
        if offset % 1000 == 0:  # Every 10 batches
            monitor_memory()
            
        offset += batch_size

def process_trial_run(migrator, trial_run_id):
    """Process a single trial run efficiently."""
    # Load minimal data
    cursor = migrator.db_manager._execute_query(
        "SELECT * FROM TRIAL_RUN WHERE id = ?", (trial_run_id,)
    )
    trial_run = cursor.fetchone()
    
    # Process without loading large objects
    # ... migration logic here ...
    
    # Clear local variables
    del trial_run
```

#### Step 3: Optimize Database Queries
```python
def get_large_metrics_info(db_manager):
    """Find metrics with large JSON objects."""
    cursor = db_manager._execute_query("""
        SELECT id, type, LENGTH(per_label_val) as json_size
        FROM METRIC 
        WHERE LENGTH(per_label_val) > 10000
        ORDER BY json_size DESC
        LIMIT 20
    """)
    
    large_metrics = cursor.fetchall()
    for metric in large_metrics:
        size_kb = metric['json_size'] / 1024
        print(f"Metric {metric['id']} ({metric['type']}): {size_kb:.1f} KB")
    
    return large_metrics

# Identify problematic metrics
large_metrics = get_large_metrics_info(migrator.db_manager)

# Process large metrics separately
for metric in large_metrics:
    # Process one at a time
    process_single_metric(migrator, metric['id'])
    gc.collect()
```

#### Step 4: Implement Memory-Efficient Export
```python
def memory_efficient_export(migrator, experiment_id, output_file):
    """Export experiment data without loading everything into memory."""
    import json
    
    with open(output_file, 'w') as f:
        f.write('{\n')
        
        # Export experiment metadata
        f.write('"experiment": ')
        exp_data = migrator.get_experiment_metadata(experiment_id)
        json.dump(exp_data, f)
        f.write(',\n')
        
        # Export trials one by one
        f.write('"trials": [\n')
        
        cursor = migrator.db_manager._execute_query(
            "SELECT id FROM TRIAL WHERE experiment_id = ?", (experiment_id,)
        )
        trial_ids = [row['id'] for row in cursor.fetchall()]
        
        for i, trial_id in enumerate(trial_ids):
            if i > 0:
                f.write(',\n')
            
            trial_data = migrator.get_trial_data(trial_id)
            json.dump(trial_data, f)
            
            # Clear from memory
            del trial_data
            gc.collect()
        
        f.write('\n]\n}')

# Use memory-efficient export
memory_efficient_export(migrator, 1, "large_experiment_export.json")
```

### Prevention
- Use appropriate batch sizes for available memory
- Implement streaming for large datasets
- Monitor memory usage during development
- Consider database pagination for large result sets

---

## Connection Problems

### Symptoms
```
ConnectionError: Cannot connect to MySQL database
sqlite3.OperationalError: database is locked
Connection timeout after 30 seconds
```

### Root Causes
- Database server is down or unreachable
- Incorrect connection parameters
- Network connectivity issues
- Database locks from other processes

### Solution Steps

#### Step 1: Test Basic Connectivity
```python
def test_database_connection(db_config):
    """Test database connection with detailed diagnostics."""
    
    if db_config.get('use_sqlite', True):
        # SQLite connection test
        import sqlite3
        try:
            conn = sqlite3.connect(db_config['database_path'], timeout=10)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True, "SQLite connection successful"
        except Exception as e:
            return False, f"SQLite error: {e}"
    else:
        # MySQL connection test
        import mysql.connector
        try:
            conn = mysql.connector.connect(
                host=db_config['host'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database_path'],
                connection_timeout=10
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True, "MySQL connection successful"
        except Exception as e:
            return False, f"MySQL error: {e}"

# Test connection
db_config = {
    'database_path': 'experiment.db',
    'use_sqlite': True
}

success, message = test_database_connection(db_config)
print(f"Connection test: {message}")
```

#### Step 2: Diagnose SQLite Lock Issues
```python
def diagnose_sqlite_locks(db_path):
    """Diagnose SQLite database lock issues."""
    import sqlite3
    import os
    
    # Check if database file exists
    if not os.path.exists(db_path):
        return f"Database file does not exist: {db_path}"
    
    # Check file permissions
    if not os.access(db_path, os.R_OK | os.W_OK):
        return f"Insufficient permissions for: {db_path}"
    
    # Check for WAL files (indicate active connections)
    wal_file = f"{db_path}-wal"
    shm_file = f"{db_path}-shm"
    
    if os.path.exists(wal_file):
        print(f"WAL file exists: {wal_file}")
    if os.path.exists(shm_file):
        print(f"SHM file exists: {shm_file}")
    
    # Try to open with immediate mode
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.close()
        return "Database is accessible in read-only mode"
    except Exception as e:
        return f"Cannot access database: {e}"

# Diagnose SQLite issues
diagnosis = diagnose_sqlite_locks("experiment.db")
print(diagnosis)
```

#### Step 3: Implement Connection Retry Logic
```python
import time
import random

def connect_with_retry(db_config, max_retries=5):
    """Connect to database with exponential backoff retry."""
    
    for attempt in range(max_retries):
        try:
            if db_config.get('use_sqlite', True):
                # SQLite with timeout
                import sqlite3
                conn = sqlite3.connect(
                    db_config['database_path'], 
                    timeout=30
                )
                # Test the connection
                conn.execute("SELECT 1")
                return conn
            else:
                # MySQL with retry
                import mysql.connector
                conn = mysql.connector.connect(
                    host=db_config['host'],
                    user=db_config['user'],
                    password=db_config['password'],
                    database=db_config['database_path'],
                    connection_timeout=30,
                    autocommit=True
                )
                return conn
                
        except Exception as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Connection attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                raise e

# Use retry logic
db_config = {
    'database_path': 'production',
    'host': 'db.company.com',
    'user': 'analyst',
    'password': 'password',
    'use_sqlite': False
}

conn = connect_with_retry(db_config)
```

#### Step 4: Handle Connection Pooling
```python
class DatabaseConnectionManager:
    """Manage database connections with pooling."""
    
    def __init__(self, db_config, pool_size=5):
        self.db_config = db_config
        self.pool_size = pool_size
        self.connections = []
        self.in_use = set()
    
    def get_connection(self):
        """Get a connection from the pool."""
        # Try to reuse existing connection
        for conn in self.connections:
            if conn not in self.in_use:
                self.in_use.add(conn)
                return conn
        
        # Create new connection if pool not full
        if len(self.connections) < self.pool_size:
            conn = connect_with_retry(self.db_config)
            self.connections.append(conn)
            self.in_use.add(conn)
            return conn
        
        # Wait for available connection
        raise Exception("Connection pool exhausted")
    
    def release_connection(self, conn):
        """Return connection to pool."""
        self.in_use.discard(conn)
    
    def close_all(self):
        """Close all connections."""
        for conn in self.connections:
            try:
                conn.close()
            except:
                pass
        self.connections.clear()
        self.in_use.clear()

# Use connection manager
conn_manager = DatabaseConnectionManager(db_config)
```

### Prevention
- Implement connection pooling for high-load scenarios
- Use appropriate timeouts for your network environment
- Monitor database server health
- Implement proper connection cleanup

This troubleshooting cookbook provides practical solutions for the most common migration issues. Each scenario includes both diagnostic steps and proven solutions that analysts can apply immediately when problems arise. 