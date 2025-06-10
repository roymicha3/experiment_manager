# Database Migration System - Comprehensive Guide for Analysts

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start Guide](#quick-start-guide)
3. [Migration CLI Reference](#migration-cli-reference)
4. [Common Migration Scenarios](#common-migration-scenarios)
5. [Best Practices for Production](#best-practices-for-production)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [API Reference](#api-reference)
8. [FAQ](#faq)

## Introduction

The Experiment Manager Database Migration System provides comprehensive tools for analysts to manage database schema evolution, data migration, and system maintenance. This system is designed with analysts in mind, offering both command-line tools and programmatic APIs for various skill levels.

### Key Features
- **Schema Evolution**: Version-controlled schema changes with rollback capabilities
- **Data Migration**: Safe transformation of existing experiment data
- **Validation Tools**: Comprehensive data integrity checking
- **Backup & Recovery**: Automated snapshot creation and restoration
- **Cross-Database Support**: Works with both SQLite (development) and MySQL (production)

### Who Should Use This Guide
- **Data Analysts**: Working with experiment results and need to understand data structure changes
- **ML Engineers**: Managing production experiment tracking databases
- **Database Administrators**: Maintaining experiment databases in production environments
- **Research Teams**: Collaborating on experiments with shared databases

## Quick Start Guide

### Prerequisites
- Python 3.7+ with experiment_manager installed
- Access to experiment tracking database (SQLite or MySQL)
- Basic familiarity with command-line tools

### Your First Migration

1. **Check current database status**:
   ```bash
   # For SQLite database
   python -m experiment_manager.db.data_migration_cli validate -d /path/to/experiment.db

   # For MySQL database
   python -m experiment_manager.db.data_migration_cli validate -d experiment_db --mysql --host localhost --user analyst
   ```

2. **Create a safety snapshot**:
   ```bash
   python -m experiment_manager.db.data_migration_cli create-snapshot -d /path/to/experiment.db --description "Before first migration"
   ```

3. **Validate data integrity**:
   ```bash
   python -m experiment_manager.db.data_migration_cli check-hierarchy -d /path/to/experiment.db
   ```

4. **Export experiment data** (optional, for external analysis):
   ```bash
   python -m experiment_manager.db.data_migration_cli export-experiment -d /path/to/experiment.db -e 1 -o experiment_1_backup.json --include-metrics --include-artifacts
   ```

### Understanding Your Database

Before performing any migrations, understand your current database structure:

```bash
# Check all available snapshots
python -m experiment_manager.db.data_migration_cli list-snapshots -d /path/to/experiment.db

# Validate data consistency
python -m experiment_manager.db.data_migration_cli validate -d /path/to/experiment.db -o validation_report.json
```

## Migration CLI Reference

### Database Connection Options

All commands support both SQLite and MySQL databases:

**SQLite (Development)**:
```bash
-d /path/to/database.db --sqlite
```

**MySQL (Production)**:
```bash
-d database_name --mysql --host hostname --user username --password
```

### Core Commands

#### 1. Data Validation

**`validate`** - Comprehensive data integrity checking

```bash
python -m experiment_manager.db.data_migration_cli validate [OPTIONS]

Options:
  -d, --database TEXT     Database path or name [required]
  --sqlite / --mysql      Database type (default: SQLite)
  --host TEXT            MySQL host (if using MySQL)
  --user TEXT            MySQL user (if using MySQL)
  --password TEXT        MySQL password (if using MySQL)
  -o, --output TEXT      Output file for validation report

Examples:
  # Basic validation with report
  python -m experiment_manager.db.data_migration_cli validate -d experiment.db -o report.json
  
  # MySQL validation
  python -m experiment_manager.db.data_migration_cli validate -d prod_experiments --mysql --host db.company.com --user analyst
```

**What it checks**:
- Foreign key integrity across all tables
- JSON metric structure validation
- Hierarchy consistency (experiments → trials → runs → epochs)
- Orphaned records detection

#### 2. Hierarchy Validation

**`check-hierarchy`** - Validate experiment hierarchy integrity

```bash
python -m experiment_manager.db.data_migration_cli check-hierarchy [OPTIONS]

Options:
  -e, --experiment-id INTEGER  Specific experiment ID to check

Examples:
  # Check specific experiment
  python -m experiment_manager.db.data_migration_cli check-hierarchy -d experiment.db -e 5
  
  # Check all experiments
  python -m experiment_manager.db.data_migration_cli check-hierarchy -d experiment.db
```

#### 3. Snapshot Management

**`create-snapshot`** - Create database backup

```bash
python -m experiment_manager.db.data_migration_cli create-snapshot [OPTIONS]

Options:
  --description TEXT        Description for the snapshot
  --snapshot-dir TEXT       Directory to store snapshots (default: snapshots)

Examples:
  # Create snapshot with description
  python -m experiment_manager.db.data_migration_cli create-snapshot -d experiment.db --description "Before metric transformation"
```

**`list-snapshots`** - List available snapshots

```bash
python -m experiment_manager.db.data_migration_cli list-snapshots -d experiment.db
```

**`restore-snapshot`** - Restore from snapshot

```bash
python -m experiment_manager.db.data_migration_cli restore-snapshot [OPTIONS]

Options:
  --snapshot-id TEXT        Snapshot ID to restore [required]
  --snapshot-dir TEXT       Directory containing snapshots

Examples:
  # Restore specific snapshot (with confirmation prompt)
  python -m experiment_manager.db.data_migration_cli restore-snapshot -d experiment.db --snapshot-id 20231201_143022
```

#### 4. Experiment Migration

**`migrate-experiment`** - Migrate experiment data

```bash
python -m experiment_manager.db.data_migration_cli migrate-experiment [OPTIONS]

Options:
  -s, --source-experiment INTEGER    Source experiment ID [required]
  -t, --target-experiment INTEGER    Target experiment ID (creates new if not provided)
  --strategy [conservative|balanced|aggressive]  Migration strategy (default: balanced)
  --batch-size INTEGER              Batch size for processing (default: 1000)
  --no-snapshot                     Skip creating pre-migration snapshot
  --transformation-file TEXT        JSON file containing transformation rules

Examples:
  # Basic experiment migration
  python -m experiment_manager.db.data_migration_cli migrate-experiment -d experiment.db -s 1 -t 2
  
  # Migration with custom transformation rules
  python -m experiment_manager.db.data_migration_cli migrate-experiment -d experiment.db -s 1 --transformation-file transforms.json
```

#### 5. Metric Transformation

**`transform-metrics`** - Transform metric data

```bash
python -m experiment_manager.db.data_migration_cli transform-metrics [OPTIONS]

Options:
  --experiment-ids TEXT             Comma-separated experiment IDs
  --batch-size INTEGER              Batch size for processing (default: 1000)
  --transformation-file TEXT        JSON file containing transformation rules

Examples:
  # Transform metrics for specific experiments
  python -m experiment_manager.db.data_migration_cli transform-metrics -d experiment.db --experiment-ids "1,2,3"
  
  # Transform all experiments with custom rules
  python -m experiment_manager.db.data_migration_cli transform-metrics -d experiment.db --transformation-file metric_transforms.json
```

#### 6. Data Export

**`export-experiment`** - Export experiment data

```bash
python -m experiment_manager.db.data_migration_cli export-experiment [OPTIONS]

Options:
  -e, --experiment-id INTEGER       Experiment ID to export [required]
  -o, --output TEXT                 Output JSON file [required]
  --include-metrics                 Include metric data in export
  --include-artifacts               Include artifact information in export

Examples:
  # Export complete experiment data
  python -m experiment_manager.db.data_migration_cli export-experiment -d experiment.db -e 1 -o exp1_complete.json --include-metrics --include-artifacts
  
  # Export basic experiment structure only
  python -m experiment_manager.db.data_migration_cli export-experiment -d experiment.db -e 1 -o exp1_basic.json
```

#### 7. Transformation Templates

**`create-transformation-template`** - Generate transformation template

```bash
python -m experiment_manager.db.data_migration_cli create-transformation-template
```

This creates a template JSON file showing the structure for custom transformations.
