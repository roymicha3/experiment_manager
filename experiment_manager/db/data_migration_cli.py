"""Command-line interface for data migration utilities."""
import json
import click
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
import sys

from experiment_manager.db.manager import DatabaseManager
from experiment_manager.db.data_migration import (
    DataMigrationManager, MigrationStrategy, MigrationProgress,
    MetricTransformer, DataMigrationError
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_db_manager(database_path: str, use_sqlite: bool = True, 
                     host: str = None, user: str = None, password: str = None) -> DatabaseManager:
    """Create a database manager instance."""
    if use_sqlite:
        return DatabaseManager(database_path=database_path, use_sqlite=True)
    else:
        return DatabaseManager(
            database_path=database_path,
            host=host or "localhost",
            user=user or "root",
            password=password or "",
            use_sqlite=False
        )

def print_progress(progress: MigrationProgress):
    """Print migration progress to console."""
    completion = progress.completion_percentage
    success_rate = progress.success_rate
    eta_str = progress.estimated_completion.strftime("%H:%M:%S") if progress.estimated_completion else "Unknown"
    
    click.echo(f"\r{progress.current_operation} - "
               f"Progress: {completion:.1f}% "
               f"({progress.processed_items}/{progress.total_items}) - "
               f"Success: {success_rate:.1f}% - "
               f"ETA: {eta_str}", nl=False)
    
    if progress.completion_percentage >= 100:
        click.echo()  # New line when complete

@click.group()
def cli():
    """Data Migration Utilities for Experiment Manager.
    
    This tool provides comprehensive data migration capabilities for analysts
    working with experiment tracking data.
    """
    pass

@cli.command()
@click.option('--database', '-d', required=True, help='Database path or name')
@click.option('--sqlite/--mysql', default=True, help='Database type (default: SQLite)')
@click.option('--host', help='MySQL host (if using MySQL)')
@click.option('--user', help='MySQL user (if using MySQL)')
@click.option('--password', help='MySQL password (if using MySQL)')
@click.option('--output', '-o', help='Output file for validation report')
def validate(database, sqlite, host, user, password, output):
    """Validate data consistency and integrity.
    
    Performs comprehensive validation checks on the database including:
    - Foreign key integrity
    - JSON metric structure validation
    - Hierarchy consistency checks
    """
    try:
        db_manager = create_db_manager(database, sqlite, host, user, password)
        migration_manager = DataMigrationManager(db_manager)
        
        click.echo("🔍 Starting data validation...")
        
        # Perform validation
        validation_results = migration_manager.validator.validate_data_consistency()
        
        # Display summary
        summary = validation_results["summary"]
        status_emoji = "✅" if summary["overall_status"] == "PASS" else "❌"
        
        click.echo(f"\n{status_emoji} Validation Results:")
        click.echo(f"  Overall Status: {summary['overall_status']}")
        click.echo(f"  Foreign Key Violations: {summary['total_foreign_key_violations']}")
        click.echo(f"  JSON Metric Issues: {summary['total_json_metric_issues']}")
        
        # Show details if there are issues
        if summary["overall_status"] == "FAIL":
            click.echo("\n📋 Detailed Issues:")
            
            # Foreign key violations
            for table, violations in validation_results["foreign_key_violations"].items():
                if violations:
                    click.echo(f"\n  {table}:")
                    for violation in violations[:5]:  # Show first 5
                        click.echo(f"    - {violation}")
                    if len(violations) > 5:
                        click.echo(f"    ... and {len(violations) - 5} more")
            
            # JSON metric issues
            json_issues = validation_results["json_metric_issues"]
            if json_issues:
                click.echo(f"\n  JSON Metric Issues:")
                for issue in json_issues[:5]:  # Show first 5
                    click.echo(f"    - Metric {issue['metric_id']} ({issue['metric_type']}): {issue['error']}")
                if len(json_issues) > 5:
                    click.echo(f"    ... and {len(json_issues) - 5} more")
        
        # Save report if requested
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            click.echo(f"\n💾 Validation report saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--database', '-d', required=True, help='Database path or name')
@click.option('--sqlite/--mysql', default=True, help='Database type (default: SQLite)')
@click.option('--host', help='MySQL host (if using MySQL)')
@click.option('--user', help='MySQL user (if using MySQL)')
@click.option('--password', help='MySQL password (if using MySQL)')
@click.option('--experiment-id', '-e', type=int, help='Specific experiment ID to check')
def check_hierarchy(database, sqlite, host, user, password, experiment_id):
    """Check experiment hierarchy integrity.
    
    Validates the hierarchical structure of experiments, trials, trial runs, 
    epochs, and results to ensure referential integrity.
    """
    try:
        db_manager = create_db_manager(database, sqlite, host, user, password)
        migration_manager = DataMigrationManager(db_manager)
        
        if experiment_id:
            click.echo(f"🔍 Checking hierarchy for experiment {experiment_id}...")
            
            is_valid, issues = migration_manager.hierarchy_preserver.validate_hierarchy_integrity(experiment_id)
            
            if is_valid:
                click.echo("✅ Hierarchy is valid!")
            else:
                click.echo("❌ Hierarchy issues found:")
                for issue in issues:
                    click.echo(f"  - {issue}")
        else:
            # Check all experiments
            click.echo("🔍 Checking hierarchy for all experiments...")
            
            # Get all experiment IDs
            cursor = db_manager._execute_query("SELECT id FROM EXPERIMENT")
            experiments = cursor.fetchall()
            
            total_issues = 0
            for exp in experiments:
                exp_id = exp["id"]
                is_valid, issues = migration_manager.hierarchy_preserver.validate_hierarchy_integrity(exp_id)
                
                if not is_valid:
                    click.echo(f"\n❌ Experiment {exp_id} has issues:")
                    for issue in issues:
                        click.echo(f"  - {issue}")
                    total_issues += len(issues)
            
            if total_issues == 0:
                click.echo("✅ All experiment hierarchies are valid!")
            else:
                click.echo(f"\n📊 Summary: {total_issues} total hierarchy issues found")
        
    except Exception as e:
        click.echo(f"❌ Hierarchy check failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--database', '-d', required=True, help='Database path or name')
@click.option('--sqlite/--mysql', default=True, help='Database type (default: SQLite)')
@click.option('--host', help='MySQL host (if using MySQL)')
@click.option('--user', help='MySQL user (if using MySQL)')
@click.option('--password', help='MySQL password (if using MySQL)')
@click.option('--description', help='Description for the snapshot')
@click.option('--snapshot-dir', default='snapshots', help='Directory to store snapshots')
def create_snapshot(database, sqlite, host, user, password, description, snapshot_dir):
    """Create a database snapshot for backup/rollback purposes.
    
    Creates a complete backup of the current database state that can be
    used for rollback in case of migration issues.
    """
    try:
        db_manager = create_db_manager(database, sqlite, host, user, password)
        migration_manager = DataMigrationManager(db_manager, snapshot_dir=snapshot_dir)
        
        desc = description or f"Manual snapshot created at {datetime.now().isoformat()}"
        
        click.echo("📸 Creating database snapshot...")
        snapshot = migration_manager.snapshot_manager.create_snapshot(desc)
        
        click.echo(f"✅ Snapshot created successfully!")
        click.echo(f"  ID: {snapshot.snapshot_id}")
        click.echo(f"  Size: {snapshot.size_bytes / 1024 / 1024:.2f} MB")
        click.echo(f"  Path: {snapshot.file_path}")
        click.echo(f"  Description: {snapshot.description}")
        
        # Show metadata
        if snapshot.metadata:
            click.echo(f"\n📊 Database Statistics:")
            for key, value in snapshot.metadata.items():
                if key.endswith('_count'):
                    table_name = key.replace('_count', '').upper()
                    click.echo(f"  {table_name}: {value} records")
                elif key == 'schema_version':
                    click.echo(f"  Schema Version: {value}")
        
    except Exception as e:
        click.echo(f"❌ Snapshot creation failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--database', '-d', required=True, help='Database path or name')
@click.option('--sqlite/--mysql', default=True, help='Database type (default: SQLite)')
@click.option('--host', help='MySQL host (if using MySQL)')
@click.option('--user', help='MySQL user (if using MySQL)')
@click.option('--password', help='MySQL password (if using MySQL)')
@click.option('--snapshot-dir', default='snapshots', help='Directory containing snapshots')
def list_snapshots(database, sqlite, host, user, password, snapshot_dir):
    """List all available database snapshots.
    
    Shows all snapshots with their metadata, creation time, and size.
    """
    try:
        db_manager = create_db_manager(database, sqlite, host, user, password)
        migration_manager = DataMigrationManager(db_manager, snapshot_dir=snapshot_dir)
        
        snapshots = migration_manager.snapshot_manager.list_snapshots()
        
        if not snapshots:
            click.echo("📭 No snapshots found.")
            return
        
        click.echo(f"📋 Found {len(snapshots)} snapshot(s):\n")
        
        for snapshot in snapshots:
            click.echo(f"🔹 {snapshot.snapshot_id}")
            click.echo(f"   Created: {snapshot.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"   Size: {snapshot.size_bytes / 1024 / 1024:.2f} MB")
            click.echo(f"   Description: {snapshot.description}")
            click.echo(f"   Path: {snapshot.file_path}")
            
            if snapshot.metadata and 'schema_version' in snapshot.metadata:
                click.echo(f"   Schema Version: {snapshot.metadata['schema_version']}")
            
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Failed to list snapshots: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--database', '-d', required=True, help='Database path or name')
@click.option('--sqlite/--mysql', default=True, help='Database type (default: SQLite)')
@click.option('--host', help='MySQL host (if using MySQL)')
@click.option('--user', help='MySQL user (if using MySQL)')
@click.option('--password', help='MySQL password (if using MySQL)')
@click.option('--snapshot-id', required=True, help='Snapshot ID to restore')
@click.option('--snapshot-dir', default='snapshots', help='Directory containing snapshots')
@click.confirmation_option(prompt='Are you sure you want to restore this snapshot? This will overwrite the current database.')
def restore_snapshot(database, sqlite, host, user, password, snapshot_id, snapshot_dir):
    """Restore database from a snapshot.
    
    WARNING: This will completely replace the current database with the snapshot data.
    """
    try:
        db_manager = create_db_manager(database, sqlite, host, user, password)
        migration_manager = DataMigrationManager(db_manager, snapshot_dir=snapshot_dir)
        
        click.echo(f"🔄 Restoring snapshot {snapshot_id}...")
        migration_manager.snapshot_manager.restore_snapshot(snapshot_id)
        
        click.echo(f"✅ Snapshot {snapshot_id} restored successfully!")
        click.echo("⚠️  Please restart any active connections to pick up the restored data.")
        
    except Exception as e:
        click.echo(f"❌ Snapshot restoration failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--database', '-d', required=True, help='Database path or name')
@click.option('--sqlite/--mysql', default=True, help='Database type (default: SQLite)')
@click.option('--host', help='MySQL host (if using MySQL)')
@click.option('--user', help='MySQL user (if using MySQL)')
@click.option('--password', help='MySQL password (if using MySQL)')
@click.option('--source-experiment', '-s', required=True, type=int, help='Source experiment ID to migrate')
@click.option('--target-experiment', '-t', type=int, help='Target experiment ID (creates new if not provided)')
@click.option('--strategy', type=click.Choice(['conservative', 'balanced', 'aggressive']), 
              default='balanced', help='Migration strategy')
@click.option('--batch-size', default=1000, help='Batch size for processing')
@click.option('--no-snapshot', is_flag=True, help='Skip creating pre-migration snapshot')
@click.option('--transformation-file', help='JSON file containing transformation rules')
def migrate_experiment(database, sqlite, host, user, password, source_experiment, 
                      target_experiment, strategy, batch_size, no_snapshot, transformation_file):
    """Migrate experiment data with hierarchy preservation.
    
    Migrates all data from a source experiment to a target experiment,
    preserving the full hierarchy structure.
    """
    try:
        db_manager = create_db_manager(database, sqlite, host, user, password)
        migration_manager = DataMigrationManager(db_manager)
        
        # Load transformation rules if provided
        transformation_rules = None
        if transformation_file:
            with open(transformation_file, 'r') as f:
                rules_config = json.load(f)
            # Convert string function names to actual functions (simplified)
            # In production, you'd want a more sophisticated transformation rule system
            transformation_rules = {}
            for field, rule in rules_config.items():
                if rule == "normalize":
                    transformation_rules[field] = lambda x: x / 100.0  # Example normalization
                # Add more transformation functions as needed
        
        # Set up progress callback
        migration_manager.add_progress_callback(print_progress)
        
        # Convert strategy string to enum
        strategy_map = {
            'conservative': MigrationStrategy.CONSERVATIVE,
            'balanced': MigrationStrategy.BALANCED,
            'aggressive': MigrationStrategy.AGGRESSIVE
        }
        migration_strategy = strategy_map[strategy]
        
        click.echo(f"🚀 Starting migration of experiment {source_experiment}...")
        click.echo(f"   Strategy: {strategy}")
        click.echo(f"   Batch size: {batch_size}")
        click.echo(f"   Create snapshot: {not no_snapshot}")
        
        # Perform migration
        progress = migration_manager.migrate_experiment_data(
            source_experiment_id=source_experiment,
            target_experiment_id=target_experiment,
            strategy=migration_strategy,
            create_snapshot=not no_snapshot,
            batch_size=batch_size,
            transformation_rules=transformation_rules
        )
        
        # Show final results
        click.echo(f"\n✅ Migration completed!")
        click.echo(f"   Total items processed: {progress.processed_items}")
        click.echo(f"   Failed items: {progress.failed_items}")
        click.echo(f"   Success rate: {progress.success_rate:.1f}%")
        click.echo(f"   Duration: {datetime.now() - progress.start_time}")
        
        if progress.errors:
            click.echo(f"\n⚠️  Errors encountered ({len(progress.errors)}):")
            for error in progress.errors[:5]:  # Show first 5 errors
                click.echo(f"   - {error}")
            if len(progress.errors) > 5:
                click.echo(f"   ... and {len(progress.errors) - 5} more errors")
        
    except Exception as e:
        click.echo(f"❌ Migration failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--database', '-d', required=True, help='Database path or name')
@click.option('--sqlite/--mysql', default=True, help='Database type (default: SQLite)')
@click.option('--host', help='MySQL host (if using MySQL)')
@click.option('--user', help='MySQL user (if using MySQL)')
@click.option('--password', help='MySQL password (if using MySQL)')
@click.option('--experiment-ids', help='Comma-separated experiment IDs (all if not provided)')
@click.option('--batch-size', default=1000, help='Batch size for processing')
@click.option('--no-snapshot', is_flag=True, help='Skip creating pre-transformation snapshot')
@click.option('--transformation-file', help='JSON file containing transformation rules')
def transform_metrics(database, sqlite, host, user, password, experiment_ids, 
                     batch_size, no_snapshot, transformation_file):
    """Batch transform JSON metrics across experiments.
    
    Applies transformation rules to per-label metrics in the database.
    """
    try:
        db_manager = create_db_manager(database, sqlite, host, user, password)
        migration_manager = DataMigrationManager(db_manager)
        
        # Parse experiment IDs
        exp_ids = None
        if experiment_ids:
            exp_ids = [int(x.strip()) for x in experiment_ids.split(',')]
        
        # Load transformation rules
        transformation_rules = {}
        if transformation_file:
            with open(transformation_file, 'r') as f:
                rules_config = json.load(f)
            
            # Convert to actual functions (simplified example)
            for field, rule in rules_config.items():
                if rule == "normalize":
                    transformation_rules[field] = lambda x: x / 100.0
                elif rule == "round":
                    transformation_rules[field] = lambda x: round(x, 2)
                # Add more transformation functions as needed
        
        if not transformation_rules:
            click.echo("⚠️  No transformation rules provided. Use --transformation-file to specify rules.")
            return
        
        # Set up progress callback
        migration_manager.add_progress_callback(print_progress)
        
        click.echo("🔄 Starting metric transformation...")
        click.echo(f"   Experiments: {exp_ids if exp_ids else 'All'}")
        click.echo(f"   Batch size: {batch_size}")
        click.echo(f"   Rules: {list(transformation_rules.keys())}")
        
        # Perform transformation
        progress = migration_manager.batch_transform_metrics(
            experiment_ids=exp_ids,
            transformation_rules=transformation_rules,
            batch_size=batch_size,
            create_snapshot=not no_snapshot
        )
        
        # Show final results
        click.echo(f"\n✅ Transformation completed!")
        click.echo(f"   Metrics processed: {progress.processed_items}")
        click.echo(f"   Failed transformations: {progress.failed_items}")
        click.echo(f"   Success rate: {progress.success_rate:.1f}%")
        click.echo(f"   Duration: {datetime.now() - progress.start_time}")
        
        if progress.errors:
            click.echo(f"\n⚠️  Errors encountered ({len(progress.errors)}):")
            for error in progress.errors[:3]:  # Show first 3 errors
                click.echo(f"   - {error}")
            if len(progress.errors) > 3:
                click.echo(f"   ... and {len(progress.errors) - 3} more errors")
        
    except Exception as e:
        click.echo(f"❌ Transformation failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--database', '-d', required=True, help='Database path or name')
@click.option('--sqlite/--mysql', default=True, help='Database type (default: SQLite)')
@click.option('--host', help='MySQL host (if using MySQL)')
@click.option('--user', help='MySQL user (if using MySQL)')
@click.option('--password', help='MySQL password (if using MySQL)')
@click.option('--experiment-id', '-e', type=int, required=True, help='Experiment ID to export')
@click.option('--output', '-o', required=True, help='Output JSON file')
@click.option('--include-metrics', is_flag=True, help='Include metric data in export')
@click.option('--include-artifacts', is_flag=True, help='Include artifact information in export')
def export_experiment(database, sqlite, host, user, password, experiment_id, 
                     output, include_metrics, include_artifacts):
    """Export experiment data to JSON file.
    
    Exports the complete hierarchy and associated data for an experiment
    to a JSON file for analysis or archival.
    """
    try:
        db_manager = create_db_manager(database, sqlite, host, user, password)
        migration_manager = DataMigrationManager(db_manager)
        
        click.echo(f"📤 Exporting experiment {experiment_id}...")
        
        # Get experiment hierarchy
        hierarchy = migration_manager.hierarchy_preserver.get_experiment_hierarchy(experiment_id)
        
        # Add metrics if requested
        if include_metrics:
            click.echo("   Including metrics data...")
            metrics = db_manager.get_experiment_metrics(experiment_id)
            hierarchy["metrics"] = [
                {
                    "id": m.id,
                    "type": m.type,
                    "total_val": m.total_val,
                    "per_label_val": m.per_label_val
                }
                for m in metrics
            ]
        
        # Add artifacts if requested
        if include_artifacts:
            click.echo("   Including artifacts data...")
            # Get experiment artifacts
            ph = db_manager._get_placeholder()
            artifacts_cursor = db_manager._execute_query(f"""
                SELECT DISTINCT a.* FROM ARTIFACT a
                JOIN EXPERIMENT_ARTIFACT ea ON ea.artifact_id = a.id
                WHERE ea.experiment_id = {ph}
            """, (experiment_id,))
            
            artifacts = artifacts_cursor.fetchall()
            hierarchy["artifacts"] = [dict(artifact) for artifact in artifacts]
        
        # Add export metadata
        hierarchy["export_metadata"] = {
            "exported_at": datetime.now().isoformat(),
            "experiment_id": experiment_id,
            "include_metrics": include_metrics,
            "include_artifacts": include_artifacts,
            "database_type": "SQLite" if sqlite else "MySQL"
        }
        
        # Write to file
        output_path = Path(output)
        with open(output_path, 'w') as f:
            json.dump(hierarchy, f, indent=2, default=str)
        
        click.echo(f"✅ Export completed!")
        click.echo(f"   Output file: {output_path}")
        click.echo(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Show summary
        stats = hierarchy["export_metadata"]
        click.echo(f"\n📊 Export Summary:")
        click.echo(f"   Trials: {len(hierarchy['trials'])}")
        total_runs = sum(len(trial['runs']) for trial in hierarchy['trials'])
        click.echo(f"   Trial Runs: {total_runs}")
        
        if include_metrics:
            click.echo(f"   Metrics: {len(hierarchy.get('metrics', []))}")
        if include_artifacts:
            click.echo(f"   Artifacts: {len(hierarchy.get('artifacts', []))}")
        
    except Exception as e:
        click.echo(f"❌ Export failed: {e}", err=True)
        sys.exit(1)

@cli.command()
def create_transformation_template():
    """Create a template file for metric transformation rules.
    
    Creates a sample JSON file showing how to define transformation rules
    for metric data migration.
    """
    template = {
        "_comment": "Transformation rules for metric data migration",
        "_available_functions": [
            "normalize: divide by 100",
            "round: round to 2 decimal places",
            "custom: define your own transformation function"
        ],
        "accuracy": "normalize",
        "precision": "round",
        "recall": "round",
        "f1_score": "normalize"
    }
    
    output_file = "transformation_rules_template.json"
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    click.echo(f"✅ Template created: {output_file}")
    click.echo("📝 Edit this file to define your transformation rules, then use it with:")
    click.echo("   data-migration transform-metrics --transformation-file transformation_rules_template.json")

if __name__ == '__main__':
    cli() 