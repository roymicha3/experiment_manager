"""
CLI Interface for Migration Validation

This module provides a command-line interface for database migration validation
with support for test generation, validation execution, and report management.
"""
import click
import sys
import json
import logging
from pathlib import Path
from typing import Optional
import webbrowser

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .database_adapter import DatabaseManager
from .migration_validator import MigrationValidator, ValidationReport
from .test_generator import TestGenerator, TestSuite
from .performance_profiler import PerformanceProfiler

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--db-type', type=click.Choice(['sqlite', 'mysql']), default='sqlite', 
              help='Database type (default: sqlite)')
@click.option('--db-path', type=str, help='Database path for SQLite')
@click.option('--db-host', type=str, help='Database host for MySQL')
@click.option('--db-port', type=int, help='Database port for MySQL')
@click.option('--db-user', type=str, help='Database user for MySQL')
@click.option('--db-password', type=str, help='Database password for MySQL')
@click.option('--db-name', type=str, help='Database name for MySQL')
@click.pass_context
def cli(ctx, verbose, db_type, db_path, db_host, db_port, db_user, db_password, db_name):
    """🔍 Migration Validation and Testing Framework
    
    Comprehensive tools for validating database migrations with automated testing,
    performance benchmarking, and safety analysis.
    """
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Store database configuration in context
    ctx.ensure_object(dict)
    ctx.obj['db_config'] = {
        'db_type': db_type,
        'db_path': db_path,
        'db_host': db_host,
        'db_port': db_port,
        'db_user': db_user,
        'db_password': db_password,
        'db_name': db_name
    }


@cli.command()
@click.argument('migration_script', type=click.Path(exists=True))
@click.option('--test-data', type=click.Path(exists=True), 
              help='Optional test data script')
@click.option('--test-suite', type=click.Path(exists=True), 
              help='Custom test suite JSON file')
@click.option('--output-dir', type=click.Path(), default='validation_reports',
              help='Directory to save validation reports')
@click.option('--no-rollback', is_flag=True, 
              help='Skip rollback testing')
@click.option('--no-performance', is_flag=True, 
              help='Skip performance benchmarking')
@click.option('--open-report', is_flag=True, 
              help='Open HTML report in browser after validation')
@click.pass_context
def validate(ctx, migration_script, test_data, test_suite, output_dir, 
            no_rollback, no_performance, open_report):
    """🔍 Validate a migration script with comprehensive testing
    
    Performs dry-run execution, schema validation, integrity checking,
    performance benchmarking, and rollback testing.
    
    Example:
        migration-validator validate migration.sql --test-data data.sql --open-report
    """
    try:
        # Create database manager
        db_manager = _create_db_manager(ctx.obj['db_config'])
        
        # Read migration script
        with open(migration_script, 'r') as f:
            migration_sql = f.read()
        
        # Read test data script if provided
        test_data_sql = None
        if test_data:
            with open(test_data, 'r') as f:
                test_data_sql = f.read()
        
        # Load custom test suite if provided
        integrity_checks = None
        if test_suite:
            test_generator = TestGenerator(db_manager)
            loaded_suite = test_generator.load_test_suite(Path(test_suite))
            integrity_checks = loaded_suite.integrity_checks
            console.print(f"📋 Loaded custom test suite: {loaded_suite.name}")
        
        console.print(Panel.fit("🚀 Starting Migration Validation", style="bold blue"))
        
        # Create validator and run validation
        validator = MigrationValidator(db_manager)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating migration...", total=None)
            
            report = validator.validate_migration(
                migration_script=migration_sql,
                test_data_script=test_data_sql,
                integrity_checks=integrity_checks,
                perform_rollback_test=not no_rollback,
                benchmark_performance=not no_performance
            )
            
            progress.update(task, description="Validation completed!")
        
        # Display summary
        _display_validation_summary(report)
        
        # Save reports
        output_path = Path(output_dir)
        json_path, html_path = validator.save_validation_report(report, output_path)
        
        console.print(f"\n📄 Reports saved:")
        console.print(f"   JSON: {json_path}")
        console.print(f"   HTML: {html_path}")
        
        # Open HTML report if requested
        if open_report:
            try:
                webbrowser.open(f"file://{html_path.absolute()}")
                console.print("🌐 Opening HTML report in browser...")
            except Exception as e:
                console.print(f"⚠️ Could not open browser: {e}")
        
        # Exit with appropriate code
        if not report.is_safe_to_deploy:
            console.print("\n❌ Migration validation failed - not safe for deployment!")
            sys.exit(1)
        else:
            console.print("\n✅ Migration validation passed - safe for deployment!")
            
    except Exception as e:
        console.print(f"❌ Validation failed: {e}", style="red")
        sys.exit(1)


@cli.command()
@click.argument('migration_script', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), default='test_suite.json',
              help='Output path for generated test suite')
@click.option('--name', type=str, default='auto_generated_tests',
              help='Name for the test suite')
@click.pass_context
def generate_tests(ctx, migration_script, output, name):
    """🧪 Generate automated test suite for migration validation
    
    Analyzes a migration script and generates comprehensive test cases
    including integrity checks, rollback tests, and performance benchmarks.
    
    Example:
        migration-validator generate-tests migration.sql --output tests.json
    """
    try:
        # Create database manager and test generator
        db_manager = _create_db_manager(ctx.obj['db_config'])
        test_generator = TestGenerator(db_manager)
        
        # Read migration script
        with open(migration_script, 'r') as f:
            migration_sql = f.read()
        
        console.print("🧪 Generating test suite...")
        
        # Generate test suite
        test_suite = test_generator.generate_migration_tests(
            migration_script=migration_sql,
            test_suite_name=name
        )
        
        # Save test suite
        output_path = test_generator.export_test_suite(test_suite, Path(output))
        
        # Display summary
        console.print(Panel.fit("✅ Test Suite Generated Successfully", style="bold green"))
        
        table = Table(title="Test Suite Summary")
        table.add_column("Component", style="cyan")
        table.add_column("Count", style="magenta")
        
        table.add_row("Integrity Checks", str(len(test_suite.integrity_checks)))
        table.add_row("Rollback Tests", str(len(test_suite.rollback_tests)))
        table.add_row("Performance Benchmarks", str(len(test_suite.performance_benchmarks)))
        
        console.print(table)
        console.print(f"\n📄 Test suite saved to: {output_path}")
        
        # Show some sample integrity checks
        if test_suite.integrity_checks:
            console.print("\n📝 Sample Integrity Checks:")
            for i, check in enumerate(test_suite.integrity_checks[:3]):
                console.print(f"   {i+1}. {check.description}")
        
    except Exception as e:
        console.print(f"❌ Test generation failed: {e}", style="red")
        sys.exit(1)


@cli.command()
@click.argument('report_path', type=click.Path(exists=True))
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@click.option('--open-html', is_flag=True, 
              help='Open corresponding HTML report in browser')
def show_report(report_path, output_format, open_html):
    """📊 Display validation report details
    
    Shows detailed information from a validation report including test results,
    performance metrics, and recommendations.
    
    Example:
        migration-validator show-report validation_report_123456.json
    """
    try:
        # Load report
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        if output_format == 'json':
            console.print_json(data=report_data)
            return
        
        # Display formatted report
        console.print(Panel.fit(f"📊 Validation Report: {report_data['validation_id']}", style="bold blue"))
        
        # Basic info
        info_table = Table(title="Report Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Validation ID", report_data['validation_id'])
        info_table.add_row("Timestamp", report_data['timestamp'])
        info_table.add_row("Database Type", report_data['database_type'])
        info_table.add_row("Execution Time", f"{report_data['execution_time']:.2f}s")
        info_table.add_row("Total Tests", str(report_data['total_tests']))
        info_table.add_row("Passed Tests", str(report_data['passed_tests']))
        info_table.add_row("Failed Tests", str(report_data['failed_tests']))
        
        success_rate = (report_data['passed_tests'] / max(report_data['total_tests'], 1)) * 100
        info_table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        # Determine safety status
        critical_failures = sum(1 for r in report_data.get('results', []) 
                              if not r.get('passed', True) and r.get('severity') == 'CRITICAL')
        is_safe = critical_failures == 0 and success_rate >= 90.0
        
        safety_status = "✅ SAFE" if is_safe else "⚠️ RISKY"
        info_table.add_row("Deployment Status", safety_status)
        
        console.print(info_table)
        
        # Test results
        if report_data.get('results'):
            console.print("\n📋 Test Results:")
            
            results_table = Table()
            results_table.add_column("Test", style="cyan")
            results_table.add_column("Status", style="white")
            results_table.add_column("Message", style="white")
            results_table.add_column("Severity", style="white")
            results_table.add_column("Time", style="magenta")
            
            for result in report_data['results']:
                status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
                severity = result.get('severity', 'INFO')
                
                # Color code severity
                severity_color = {
                    'INFO': 'green',
                    'WARNING': 'yellow', 
                    'ERROR': 'red',
                    'CRITICAL': 'bold red'
                }.get(severity, 'white')
                
                results_table.add_row(
                    result.get('test_name', 'Unknown'),
                    status,
                    result.get('message', ''),
                    Text(severity, style=severity_color),
                    f"{result.get('execution_time', 0):.3f}s"
                )
            
            console.print(results_table)
        
        # Performance metrics
        if report_data.get('performance_metrics'):
            console.print("\n⚡ Performance Metrics:")
            perf_data = report_data['performance_metrics']
            
            perf_table = Table()
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="white")
            
            for key, value in perf_data.items():
                if isinstance(value, (int, float)):
                    if 'time' in key.lower():
                        perf_table.add_row(key.replace('_', ' ').title(), f"{value:.3f}s")
                    else:
                        perf_table.add_row(key.replace('_', ' ').title(), str(value))
                else:
                    perf_table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(perf_table)
        
        # Recommendations
        if report_data.get('recommendations'):
            console.print("\n💡 Recommendations:")
            for i, rec in enumerate(report_data['recommendations'], 1):
                console.print(f"   {i}. {rec}")
        
        # Try to open HTML report if requested
        if open_html:
            report_dir = Path(report_path).parent
            report_id = report_data['validation_id']
            html_path = report_dir / f"validation_report_{report_id}.html"
            
            if html_path.exists():
                try:
                    webbrowser.open(f"file://{html_path.absolute()}")
                    console.print("\n🌐 Opening HTML report in browser...")
                except Exception as e:
                    console.print(f"⚠️ Could not open browser: {e}")
            else:
                console.print(f"⚠️ HTML report not found: {html_path}")
        
    except Exception as e:
        console.print(f"❌ Failed to display report: {e}", style="red")
        sys.exit(1)


@cli.command()
@click.argument('test_suite_path', type=click.Path(exists=True))
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
def show_test_suite(test_suite_path, output_format):
    """🧪 Display test suite details
    
    Shows the contents of a generated test suite including integrity checks,
    rollback tests, and performance benchmarks.
    
    Example:
        migration-validator show-test-suite test_suite.json
    """
    try:
        # Load test suite
        with open(test_suite_path, 'r') as f:
            suite_data = json.load(f)
        
        if output_format == 'json':
            console.print_json(data=suite_data)
            return
        
        # Display formatted test suite
        console.print(Panel.fit(f"🧪 Test Suite: {suite_data['name']}", style="bold green"))
        
        console.print(f"📝 Description: {suite_data['description']}\n")
        
        # Integrity checks
        if suite_data.get('integrity_checks'):
            console.print("🔍 Integrity Checks:")
            
            checks_table = Table()
            checks_table.add_column("Check Name", style="cyan")
            checks_table.add_column("Description", style="white")
            checks_table.add_column("Table", style="magenta")
            checks_table.add_column("Expected Result", style="yellow")
            
            for check in suite_data['integrity_checks']:
                checks_table.add_row(
                    check.get('check_name', ''),
                    check.get('description', ''),
                    check.get('table', 'N/A'),
                    str(check.get('expected_result', ''))
                )
            
            console.print(checks_table)
        
        # Expected schema changes
        if suite_data.get('expected_schema_changes'):
            console.print("\n📊 Expected Schema Changes:")
            changes = suite_data['expected_schema_changes']
            
            changes_table = Table()
            changes_table.add_column("Change Type", style="cyan")
            changes_table.add_column("Count", style="white")
            changes_table.add_column("Items", style="magenta")
            
            for change_type, items in changes.items():
                if isinstance(items, list):
                    count = len(items)
                    items_str = ', '.join(items[:3])
                    if len(items) > 3:
                        items_str += f" ... (+{len(items) - 3} more)"
                else:
                    count = str(items)
                    items_str = "N/A"
                
                changes_table.add_row(
                    change_type.replace('_', ' ').title(),
                    str(count),
                    items_str
                )
            
            console.print(changes_table)
        
        # Rollback tests
        if suite_data.get('rollback_tests'):
            console.print("\n🔄 Rollback Tests:")
            for i, test in enumerate(suite_data['rollback_tests'], 1):
                console.print(f"   {i}. {test}")
        
        # Performance benchmarks
        if suite_data.get('performance_benchmarks'):
            console.print("\n⚡ Performance Benchmarks:")
            
            bench_table = Table()
            bench_table.add_column("Benchmark", style="cyan")
            bench_table.add_column("Description", style="white")
            bench_table.add_column("Max Time", style="yellow")
            bench_table.add_column("Operation Type", style="magenta")
            
            for benchmark in suite_data['performance_benchmarks']:
                bench_table.add_row(
                    benchmark.get('name', ''),
                    benchmark.get('description', ''),
                    f"{benchmark.get('expected_max_time', 0)}s",
                    benchmark.get('operation_type', '')
                )
            
            console.print(bench_table)
        
        # Test data script preview
        if suite_data.get('test_data_script'):
            test_data = suite_data['test_data_script']
            if test_data.strip() and not test_data.startswith('-- No test data'):
                console.print("\n📄 Test Data Script Preview:")
                lines = test_data.split('\n')[:5]  # Show first 5 lines
                for line in lines:
                    console.print(f"   {line}")
                if len(test_data.split('\n')) > 5:
                    console.print("   ...")
        
    except Exception as e:
        console.print(f"❌ Failed to display test suite: {e}", style="red")
        sys.exit(1)


def _create_db_manager(db_config: dict) -> DatabaseManager:
    """Create database manager from configuration."""
    if db_config['db_type'] == 'sqlite':
        if not db_config['db_path']:
            raise click.ClickException("SQLite database path is required")
        return DatabaseManager(db_type='sqlite', db_path=db_config['db_path'])
    
    elif db_config['db_type'] == 'mysql':
        # Prompt for missing MySQL parameters
        host = db_config['db_host'] or click.prompt('MySQL host', default='localhost')
        port = db_config['db_port'] or click.prompt('MySQL port', default=3306, type=int)
        user = db_config['db_user'] or click.prompt('MySQL user', default='root')
        password = db_config['db_password'] or click.prompt('MySQL password', hide_input=True)
        database = db_config['db_name'] or click.prompt('MySQL database name')
        
        return DatabaseManager(
            db_type='mysql',
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
    
    else:
        raise click.ClickException(f"Unsupported database type: {db_config['db_type']}")


def _display_validation_summary(report):
    """Display a formatted validation summary."""
    # Create summary panel
    summary_text = f"""
🆔 Validation ID: {report.validation_id}
⏱️  Execution Time: {report.execution_time:.2f} seconds
🧪 Total Tests: {report.total_tests}
✅ Passed: {report.passed_tests}
❌ Failed: {report.failed_tests}
⚠️  Warnings: {report.warnings}
📊 Success Rate: {report.success_rate:.1f}%
🚀 Safe to Deploy: {'✅ YES' if report.is_safe_to_deploy else '❌ NO'}
"""
    
    console.print(Panel(summary_text.strip(), title="Validation Summary", style="bold"))
    
    # Show failed tests if any
    failed_tests = [r for r in report.results if not r.passed]
    if failed_tests:
        console.print("\n❌ Failed Tests:")
        for test in failed_tests:
            console.print(f"   • {test.test_name}: {test.message}")
    
    # Show critical warnings
    critical_results = [r for r in report.results if r.severity == 'CRITICAL']
    if critical_results:
        console.print("\n🚨 Critical Issues:")
        for result in critical_results:
            console.print(f"   • {result.test_name}: {result.message}")


if __name__ == '__main__':
    cli() 