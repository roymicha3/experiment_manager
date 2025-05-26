"""Command-line interface for schema comparison and diff generation tools."""
import click
import logging
import sys
from pathlib import Path
from typing import Optional
import webbrowser

from experiment_manager.db.manager import DatabaseManager, ConnectionError
from experiment_manager.db.schema_inspector import SchemaInspector
from experiment_manager.db.schema_comparator import SchemaComparator

logger = logging.getLogger(__name__)

def create_db_manager(database_path: str, use_sqlite: bool = True, 
                     host: str = "localhost", user: str = "root", 
                     password: str = "", port: int = 3306) -> DatabaseManager:
    """Create and return a database manager instance.
    
    Args:
        database_path: Database path (file for SQLite, name for MySQL)
        use_sqlite: Whether to use SQLite or MySQL
        host: MySQL host (ignored for SQLite)
        user: MySQL username (ignored for SQLite)
        password: MySQL password (ignored for SQLite)
        port: MySQL port (ignored for SQLite, not supported by DatabaseManager)
        
    Returns:
        DatabaseManager: Configured database manager
        
    Raises:
        ConnectionError: If database connection fails
    """
    if use_sqlite:
        return DatabaseManager(database_path=database_path, use_sqlite=True)
    else:
        # Note: DatabaseManager doesn't support port parameter
        return DatabaseManager(
            database_path=database_path,
            use_sqlite=False,
            host=host,
            user=user,
            password=password
        )

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """🔍 Schema Comparison and Diff Tools for Database Analysis
    
    Tools for comparing database schemas, generating visual diffs, and analyzing
    the impact of database changes for safe migrations.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

@cli.command()
@click.option('--database', '-d', required=True, help='Database path (SQLite file) or name (MySQL)')
@click.option('--mysql', is_flag=True, help='Use MySQL instead of SQLite')
@click.option('--host', default='localhost', help='MySQL host (default: localhost)')
@click.option('--user', default='root', help='MySQL username (default: root)')
@click.option('--password', prompt=True, hide_input=True, help='MySQL password')
@click.option('--port', default=3306, help='MySQL port (default: 3306)')
@click.option('--output', '-o', help='Output file path for schema (default: schema_<db_name>.json)')
@click.option('--include-stats', is_flag=True, default=True, help='Include row counts and data sizes')
@click.option('--format', 'output_format', type=click.Choice(['json']), default='json', help='Output format')
def extract_schema(database, mysql, host, user, password, port, output, include_stats, output_format):
    """📊 Extract complete database schema information
    
    Extracts detailed schema information including tables, columns, indexes,
    foreign keys, constraints, and optionally data statistics.
    
    Example:
        schema-diff extract-schema --database experiment.db
        schema-diff extract-schema --database mydb --mysql --host localhost --user admin
    """
    try:
        click.echo("🔍 Extracting database schema...")
        
        # Create database manager
        if mysql and not password:
            password = click.prompt('MySQL password', hide_input=True)
        
        db_manager = create_db_manager(
            database_path=database,
            use_sqlite=not mysql,
            host=host,
            user=user,
            password=password,
            port=port
        )
        
        # Extract schema
        inspector = SchemaInspector(db_manager)
        schema = inspector.extract_full_schema(include_data_stats=include_stats)
        
        # Determine output path
        if not output:
            db_name = Path(database).stem if not mysql else database
            output = f"schema_{db_name}.json"
        
        # Save schema
        inspector.save_schema_to_file(schema, output)
        
        # Display summary
        click.echo(f"\n✅ Schema extracted successfully!")
        click.echo(f"📄 Output: {output}")
        click.echo(f"🗃️  Database: {schema.database_name} ({schema.database_type})")
        click.echo(f"📊 Tables: {len(schema.tables)}")
        
        if include_stats:
            total_rows = sum(table.row_count or 0 for table in schema.tables)
            click.echo(f"📈 Total rows: {total_rows:,}")
        
        click.echo(f"🏷️  Schema version: {schema.schema_version or 'Not versioned'}")
        click.echo(f"⏰ Extracted at: {schema.extracted_at}")
        
    except ConnectionError as e:
        click.echo(f"❌ Database connection failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Schema extraction failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--source', '-s', required=True, help='Source schema file (JSON)')
@click.option('--target', '-t', required=True, help='Target schema file (JSON)')
@click.option('--output', '-o', help='Output directory for diff reports (default: schema_diff_reports/)')
@click.option('--format', 'output_formats', multiple=True, 
              type=click.Choice(['json', 'html']), default=['json', 'html'],
              help='Output formats (default: json html)')
@click.option('--open-html', is_flag=True, help='Open HTML report in browser after generation')
def compare_schemas(source, target, output, output_formats, open_html):
    """🔄 Compare two database schemas and generate diff reports
    
    Compares two schema files and generates comprehensive diff reports showing
    changes, impact analysis, migration strategy, and risk assessment.
    
    Example:
        schema-diff compare-schemas --source old_schema.json --target new_schema.json
        schema-diff compare-schemas -s v1.json -t v2.json --output reports/ --open-html
    """
    try:
        click.echo("🔄 Comparing database schemas...")
        
        # Validate input files
        source_path = Path(source)
        target_path = Path(target)
        
        if not source_path.exists():
            click.echo(f"❌ Source schema file not found: {source}", err=True)
            sys.exit(1)
        
        if not target_path.exists():
            click.echo(f"❌ Target schema file not found: {target}", err=True)
            sys.exit(1)
        
        # Load schemas
        inspector = SchemaInspector(None)  # Don't need DB connection for file loading
        source_schema = inspector.load_schema_from_file(source)
        target_schema = inspector.load_schema_from_file(target)
        
        # Compare schemas
        comparator = SchemaComparator()
        schema_diff = comparator.compare_schemas(source_schema, target_schema)
        
        # Prepare output directory
        if not output:
            output = "schema_diff_reports"
        
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate reports
        report_files = []
        
        if 'json' in output_formats:
            json_path = output_dir / "schema_diff.json"
            comparator.save_diff_to_json(schema_diff, str(json_path))
            report_files.append(str(json_path))
            click.echo(f"📄 JSON report: {json_path}")
        
        if 'html' in output_formats:
            html_path = output_dir / "schema_diff_report.html"
            comparator.generate_html_diff_report(schema_diff, str(html_path))
            report_files.append(str(html_path))
            click.echo(f"🌐 HTML report: {html_path}")
            
            if open_html:
                try:
                    webbrowser.open(f"file://{html_path.absolute()}")
                    click.echo("🚀 Opening HTML report in browser...")
                except Exception as e:
                    click.echo(f"⚠️  Could not open browser: {e}")
        
        # Display summary
        click.echo(f"\n✅ Schema comparison completed!")
        click.echo(f"📊 Summary:")
        click.echo(f"   • Overall Impact: {schema_diff.overall_impact.value}")
        click.echo(f"   • Migration Strategy: {schema_diff.migration_strategy}")
        click.echo(f"   • Estimated Downtime: {schema_diff.estimated_downtime}")
        click.echo(f"   • Rollback Complexity: {schema_diff.rollback_complexity}")
        
        summary = schema_diff.summary
        click.echo(f"📈 Changes:")
        click.echo(f"   • Tables: {summary.get('added_tables', 0)} added, {summary.get('removed_tables', 0)} removed, {summary.get('modified_tables', 0)} modified")
        click.echo(f"   • Columns: {summary.get('added_columns', 0)} added, {summary.get('removed_columns', 0)} removed, {summary.get('modified_columns', 0)} modified")
        click.echo(f"   • Indexes: {summary.get('added_indexes', 0)} added, {summary.get('removed_indexes', 0)} removed, {summary.get('modified_indexes', 0)} modified")
        
        # Risk assessment
        if schema_diff.overall_impact.value in ['BREAKING', 'MAJOR']:
            click.echo(f"\n⚠️  Risk Assessment: {schema_diff.risk_assessment}")
        
        if summary.get('breaking_changes', 0) > 0:
            click.echo(f"🚨 Breaking changes detected: {summary.get('breaking_changes', 0)}")
            click.echo("   Careful migration planning required!")
        
    except Exception as e:
        click.echo(f"❌ Schema comparison failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--source-db', required=True, help='Source database path/name')
@click.option('--target-db', required=True, help='Target database path/name')
@click.option('--source-mysql', is_flag=True, help='Source is MySQL database')
@click.option('--target-mysql', is_flag=True, help='Target is MySQL database')
@click.option('--source-host', default='localhost', help='Source MySQL host')
@click.option('--source-user', default='root', help='Source MySQL username')
@click.option('--source-password', help='Source MySQL password')
@click.option('--target-host', default='localhost', help='Target MySQL host')
@click.option('--target-user', default='root', help='Target MySQL username')
@click.option('--target-password', help='Target MySQL password')
@click.option('--output', '-o', help='Output directory for reports (default: live_schema_diff/)')
@click.option('--format', 'output_formats', multiple=True, 
              type=click.Choice(['json', 'html']), default=['html'],
              help='Output formats (default: html)')
@click.option('--open-html', is_flag=True, default=True, help='Open HTML report in browser')
def compare_live_schemas(source_db, target_db, source_mysql, target_mysql,
                        source_host, source_user, source_password,
                        target_host, target_user, target_password,
                        output, output_formats, open_html):
    """🔗 Compare schemas directly from live databases
    
    Connects to two live databases, extracts their schemas, and generates
    comparison reports without needing intermediate schema files.
    
    Example:
        schema-diff compare-live-schemas --source-db old.db --target-db new.db
        schema-diff compare-live-schemas --source-db olddb --source-mysql --target-db newdb --target-mysql
    """
    try:
        click.echo("🔗 Connecting to databases and extracting schemas...")
        
        # Prompt for passwords if not provided
        if source_mysql and not source_password:
            source_password = click.prompt('Source MySQL password', hide_input=True)
        
        if target_mysql and not target_password:
            target_password = click.prompt('Target MySQL password', hide_input=True)
        
        # Create database managers
        source_manager = create_db_manager(
            database_path=source_db,
            use_sqlite=not source_mysql,
            host=source_host,
            user=source_user,
            password=source_password or ""
        )
        
        target_manager = create_db_manager(
            database_path=target_db,
            use_sqlite=not target_mysql,
            host=target_host,
            user=target_user,
            password=target_password or ""
        )
        
        # Extract schemas
        click.echo("📊 Extracting source schema...")
        source_inspector = SchemaInspector(source_manager)
        source_schema = source_inspector.extract_full_schema()
        
        click.echo("📊 Extracting target schema...")
        target_inspector = SchemaInspector(target_manager)
        target_schema = target_inspector.extract_full_schema()
        
        # Compare schemas
        click.echo("🔄 Comparing schemas...")
        comparator = SchemaComparator()
        schema_diff = comparator.compare_schemas(source_schema, target_schema)
        
        # Prepare output directory
        if not output:
            output = "live_schema_diff"
        
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save intermediate schema files for reference
        source_inspector.save_schema_to_file(source_schema, str(output_dir / "source_schema.json"))
        target_inspector.save_schema_to_file(target_schema, str(output_dir / "target_schema.json"))
        
        # Generate reports
        if 'json' in output_formats:
            json_path = output_dir / "schema_diff.json"
            comparator.save_diff_to_json(schema_diff, str(json_path))
            click.echo(f"📄 JSON report: {json_path}")
        
        if 'html' in output_formats:
            html_path = output_dir / "schema_diff_report.html"
            comparator.generate_html_diff_report(schema_diff, str(html_path))
            click.echo(f"🌐 HTML report: {html_path}")
            
            if open_html:
                try:
                    webbrowser.open(f"file://{html_path.absolute()}")
                    click.echo("🚀 Opening HTML report in browser...")
                except Exception as e:
                    click.echo(f"⚠️  Could not open browser: {e}")
        
        # Display summary
        click.echo(f"\n✅ Live schema comparison completed!")
        _display_comparison_summary(schema_diff)
        
    except ConnectionError as e:
        click.echo(f"❌ Database connection failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Live schema comparison failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--schema-file', '-s', required=True, help='Schema file to analyze (JSON)')
@click.option('--output', '-o', help='Output file for analysis report (default: schema_analysis.json)')
def analyze_schema(schema_file, output):
    """🔬 Analyze a single schema for complexity and potential issues
    
    Analyzes a schema file to identify potential issues, complexity metrics,
    and recommendations for optimization or migration planning.
    
    Example:
        schema-diff analyze-schema --schema-file current_schema.json
    """
    try:
        click.echo("🔬 Analyzing database schema...")
        
        # Validate input file
        schema_path = Path(schema_file)
        if not schema_path.exists():
            click.echo(f"❌ Schema file not found: {schema_file}", err=True)
            sys.exit(1)
        
        # Load schema
        inspector = SchemaInspector(None)
        schema = inspector.load_schema_from_file(schema_file)
        
        # Analyze schema
        analysis = _analyze_schema_complexity(schema)
        
        # Save analysis
        if not output:
            output = "schema_analysis.json"
        
        with open(output, 'w') as f:
            import json
            json.dump(analysis, f, indent=2, default=str)
        
        # Display results
        click.echo(f"\n✅ Schema analysis completed!")
        click.echo(f"📄 Analysis report: {output}")
        click.echo(f"\n📊 Analysis Summary:")
        click.echo(f"   • Total Tables: {analysis['table_count']}")
        click.echo(f"   • Total Columns: {analysis['column_count']}")
        click.echo(f"   • Total Indexes: {analysis['index_count']}")
        click.echo(f"   • Foreign Keys: {analysis['foreign_key_count']}")
        click.echo(f"   • Complexity Score: {analysis['complexity_score']}/10")
        
        if analysis['recommendations']:
            click.echo(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                click.echo(f"   • {rec}")
        
        if analysis['potential_issues']:
            click.echo(f"\n⚠️  Potential Issues:")
            for issue in analysis['potential_issues']:
                click.echo(f"   • {issue}")
        
    except Exception as e:
        click.echo(f"❌ Schema analysis failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--diff-file', '-d', required=True, help='Schema diff file (JSON)')
@click.option('--output', '-o', help='Output file for migration script (default: migration_plan.sql)')
@click.option('--strategy', type=click.Choice(['safe', 'fast', 'minimal-downtime']), 
              default='safe', help='Migration strategy (default: safe)')
def generate_migration_plan(diff_file, output, strategy):
    """🛠️ Generate migration plan from schema diff
    
    Analyzes a schema diff and generates a detailed migration plan with
    SQL scripts, timing recommendations, and rollback procedures.
    
    Example:
        schema-diff generate-migration-plan --diff-file schema_diff.json
    """
    try:
        click.echo("🛠️ Generating migration plan...")
        
        # Load diff file
        diff_path = Path(diff_file)
        if not diff_path.exists():
            click.echo(f"❌ Diff file not found: {diff_file}", err=True)
            sys.exit(1)
        
        with open(diff_file, 'r') as f:
            import json
            diff_data = json.load(f)
        
        # Generate migration plan
        migration_plan = _generate_migration_plan_from_diff(diff_data, strategy)
        
        # Save migration plan
        if not output:
            output = "migration_plan.sql"
        
        with open(output, 'w') as f:
            f.write(migration_plan)
        
        click.echo(f"\n✅ Migration plan generated!")
        click.echo(f"📄 Migration script: {output}")
        click.echo(f"🔧 Strategy: {strategy}")
        
        # Display key information from diff
        overall_impact = diff_data.get('overall_impact', 'UNKNOWN')
        estimated_downtime = diff_data.get('estimated_downtime', 'Unknown')
        rollback_complexity = diff_data.get('rollback_complexity', 'Unknown')
        
        click.echo(f"\n📊 Migration Overview:")
        click.echo(f"   • Impact Level: {overall_impact}")
        click.echo(f"   • Estimated Downtime: {estimated_downtime}")
        click.echo(f"   • Rollback Complexity: {rollback_complexity}")
        
        if overall_impact in ['BREAKING', 'MAJOR']:
            click.echo(f"\n⚠️  Warning: This migration has significant impact!")
            click.echo("   • Test thoroughly in staging environment")
            click.echo("   • Plan for rollback procedures")
            click.echo("   • Consider maintenance window")
        
    except Exception as e:
        click.echo(f"❌ Migration plan generation failed: {e}", err=True)
        sys.exit(1)

def _display_comparison_summary(schema_diff):
    """Display a formatted summary of schema comparison results."""
    click.echo(f"📊 Summary:")
    click.echo(f"   • Overall Impact: {schema_diff.overall_impact.value}")
    click.echo(f"   • Migration Strategy: {schema_diff.migration_strategy}")
    click.echo(f"   • Estimated Downtime: {schema_diff.estimated_downtime}")
    click.echo(f"   • Rollback Complexity: {schema_diff.rollback_complexity}")
    
    summary = schema_diff.summary
    click.echo(f"📈 Changes:")
    click.echo(f"   • Tables: {summary.get('added_tables', 0)} added, {summary.get('removed_tables', 0)} removed, {summary.get('modified_tables', 0)} modified")
    click.echo(f"   • Columns: {summary.get('added_columns', 0)} added, {summary.get('removed_columns', 0)} removed, {summary.get('modified_columns', 0)} modified")
    click.echo(f"   • Indexes: {summary.get('added_indexes', 0)} added, {summary.get('removed_indexes', 0)} removed, {summary.get('modified_indexes', 0)} modified")
    
    # Risk assessment
    if schema_diff.overall_impact.value in ['BREAKING', 'MAJOR']:
        click.echo(f"\n⚠️  Risk Assessment: {schema_diff.risk_assessment}")
    
    if summary.get('breaking_changes', 0) > 0:
        click.echo(f"🚨 Breaking changes detected: {summary.get('breaking_changes', 0)}")
        click.echo("   Careful migration planning required!")

def _analyze_schema_complexity(schema):
    """Analyze schema complexity and generate recommendations."""
    analysis = {
        'table_count': len(schema.tables),
        'column_count': sum(len(table.columns) for table in schema.tables),
        'index_count': sum(len(table.indexes) for table in schema.tables),
        'foreign_key_count': sum(len(table.foreign_keys) for table in schema.tables),
        'complexity_score': 0,
        'recommendations': [],
        'potential_issues': [],
        'table_analysis': []
    }
    
    # Calculate complexity score (0-10)
    complexity_factors = []
    
    # Table count factor
    table_factor = min(analysis['table_count'] / 50, 1.0) * 2  # 0-2 points
    complexity_factors.append(table_factor)
    
    # Column density factor
    avg_columns = analysis['column_count'] / max(analysis['table_count'], 1)
    column_factor = min(avg_columns / 20, 1.0) * 2  # 0-2 points
    complexity_factors.append(column_factor)
    
    # Relationship complexity
    fk_density = analysis['foreign_key_count'] / max(analysis['table_count'], 1)
    fk_factor = min(fk_density / 5, 1.0) * 3  # 0-3 points
    complexity_factors.append(fk_factor)
    
    # Index density
    idx_density = analysis['index_count'] / max(analysis['column_count'], 1)
    idx_factor = min(idx_density / 0.3, 1.0) * 3  # 0-3 points
    complexity_factors.append(idx_factor)
    
    analysis['complexity_score'] = round(sum(complexity_factors), 1)
    
    # Generate recommendations
    if analysis['table_count'] > 100:
        analysis['recommendations'].append("Consider schema partitioning for large table count")
    
    if avg_columns > 30:
        analysis['recommendations'].append("Review tables with many columns for normalization opportunities")
    
    if fk_density < 0.1:
        analysis['recommendations'].append("Consider adding foreign key constraints for better data integrity")
    
    if idx_density < 0.1:
        analysis['recommendations'].append("Review indexing strategy - may need more indexes for performance")
    elif idx_density > 0.5:
        analysis['recommendations'].append("Review indexing strategy - may have too many indexes")
    
    # Identify potential issues
    for table in schema.tables:
        if len(table.columns) > 50:
            analysis['potential_issues'].append(f"Table '{table.name}' has {len(table.columns)} columns - consider normalization")
        
        if not table.indexes and len(table.columns) > 5:
            analysis['potential_issues'].append(f"Table '{table.name}' has no indexes - may impact performance")
        
        pk_columns = [col for col in table.columns if col.is_primary_key]
        if not pk_columns:
            analysis['potential_issues'].append(f"Table '{table.name}' has no primary key")
    
    return analysis

def _generate_migration_plan_from_diff(diff_data, strategy):
    """Generate SQL migration plan from diff data."""
    plan = f"""-- Migration Plan Generated from Schema Diff
-- Strategy: {strategy}
-- Generated at: {diff_data.get('generated_at', 'Unknown')}
-- Overall Impact: {diff_data.get('overall_impact', 'Unknown')}

-- MIGRATION OVERVIEW
-- Estimated Downtime: {diff_data.get('estimated_downtime', 'Unknown')}
-- Rollback Complexity: {diff_data.get('rollback_complexity', 'Unknown')}
-- Risk Assessment: {diff_data.get('risk_assessment', 'Unknown')}

-- WARNING: This is a generated template. Review and test thoroughly before execution!

"""
    
    if strategy == 'safe':
        plan += """-- SAFE MIGRATION STRATEGY
-- 1. Create backup
-- 2. Apply changes with rollback points
-- 3. Validate after each step
-- 4. Monitor performance

"""
    elif strategy == 'fast':
        plan += """-- FAST MIGRATION STRATEGY
-- 1. Minimal validation
-- 2. Batch operations
-- 3. Reduced safety checks
-- WARNING: Higher risk approach

"""
    else:  # minimal-downtime
        plan += """-- MINIMAL DOWNTIME STRATEGY
-- 1. Phased migration
-- 2. Hot swapping where possible
-- 3. Online schema changes
-- 4. Blue-green deployment considerations

"""
    
    plan += """-- BEGIN MIGRATION
BEGIN TRANSACTION;

-- Add your migration steps here based on the schema diff
-- This template should be customized for your specific changes

-- Example steps:
-- 1. ADD new tables
-- 2. ADD new columns with defaults
-- 3. MODIFY existing columns (careful!)
-- 4. ADD indexes
-- 5. ADD foreign keys
-- 6. DROP old constraints
-- 7. DROP old columns (data loss!)
-- 8. DROP old tables (data loss!)

-- TODO: Replace this section with actual migration steps
-- based on the changes identified in the schema diff

COMMIT;
-- END MIGRATION

-- ROLLBACK PROCEDURE
-- In case of issues, execute the following:
-- BEGIN TRANSACTION;
-- -- Add rollback steps here (reverse of migration)
-- COMMIT;
"""
    
    return plan

if __name__ == '__main__':
    cli() 