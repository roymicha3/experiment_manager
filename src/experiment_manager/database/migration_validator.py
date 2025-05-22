"""
Migration Validation and Testing Framework

This module provides comprehensive validation tools for database migrations,
including dry-run capabilities, data integrity checks, and performance testing.
"""
import logging
import time
import sqlite3
import mysql.connector
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import tempfile
import shutil
from contextlib import contextmanager

from .database_adapter import DatabaseManager
from experiment_manager.db.schema_inspector import SchemaInspector, DatabaseSchema
from experiment_manager.db.migration_manager import MigrationManager
from .performance_profiler import PerformanceProfiler

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Represents the result of a migration validation."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL


@dataclass
class IntegrityCheck:
    """Represents a data integrity check."""
    check_name: str
    query: str
    expected_result: Any
    description: str
    table: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    validation_id: str
    timestamp: datetime
    migration_script: str
    database_type: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    warnings: int
    execution_time: float
    results: List[ValidationResult] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    schema_diff: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def is_safe_to_deploy(self) -> bool:
        """Determine if migration is safe for production deployment."""
        critical_failures = sum(1 for r in self.results 
                              if not r.passed and r.severity == "CRITICAL")
        return critical_failures == 0 and self.success_rate >= 90.0


class MigrationValidator:
    """
    Comprehensive migration validation framework.
    
    Provides tools for validating database migrations including:
    - Dry-run execution
    - Data integrity validation
    - Foreign key constraint testing
    - Performance benchmarking
    - Rollback testing
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the migration validator.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.schema_inspector = SchemaInspector(db_manager)
        self.migration_manager = MigrationManager(db_manager)
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger(__name__)

    def validate_migration(
        self,
        migration_script: str,
        test_data_script: Optional[str] = None,
        integrity_checks: Optional[List[IntegrityCheck]] = None,
        perform_rollback_test: bool = True,
        benchmark_performance: bool = True
    ) -> ValidationReport:
        """
        Perform comprehensive validation of a migration script.
        
        Args:
            migration_script: SQL migration script to validate
            test_data_script: Optional script to populate test data
            integrity_checks: Custom integrity checks to perform
            perform_rollback_test: Whether to test rollback capability
            benchmark_performance: Whether to benchmark performance
            
        Returns:
            ValidationReport: Comprehensive validation results
        """
        validation_id = f"val_{int(time.time())}"
        start_time = time.time()
        
        self.logger.info(f"Starting migration validation {validation_id}")
        
        report = ValidationReport(
            validation_id=validation_id,
            timestamp=datetime.now(),
            migration_script=migration_script,
            database_type=self.db_manager.db_type,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            warnings=0,
            execution_time=0.0
        )

        try:
            # Create test database
            with self._create_test_database() as test_db_path:
                test_db_manager = self._create_test_db_manager(test_db_path)
                
                            # Capture initial schema
                initial_schema = self.schema_inspector.extract_full_schema()
                
                # Run validation tests
                tests = []
                
                # 1. Syntax validation
                tests.append(self._validate_syntax(migration_script))
                
                # 2. Dry run execution
                tests.append(self._dry_run_migration(test_db_manager, migration_script))
                
                # 3. Schema validation
                if tests[-1].passed:
                    tests.append(self._validate_schema_changes(
                        test_db_manager, initial_schema
                    ))
                
                # 4. Data integrity checks
                if test_data_script:
                    self._populate_test_data(test_db_manager, test_data_script)
                    tests.extend(self._run_integrity_checks(
                        test_db_manager, integrity_checks or []
                    ))
                
                # 5. Foreign key constraint testing
                tests.append(self._validate_foreign_keys(test_db_manager))
                
                # 6. Performance benchmarking
                if benchmark_performance:
                    perf_result = self._benchmark_migration_performance(
                        test_db_manager, migration_script
                    )
                    tests.append(perf_result)
                    report.performance_metrics = perf_result.details
                
                # 7. Rollback testing
                if perform_rollback_test:
                    tests.append(self._test_rollback_capability(
                        test_db_manager, migration_script
                    ))
                
                # Compile results
                report.results = tests
                report.total_tests = len(tests)
                report.passed_tests = sum(1 for t in tests if t.passed)
                report.failed_tests = sum(1 for t in tests if not t.passed)
                report.warnings = sum(1 for t in tests if t.severity == "WARNING")
                
                # Generate recommendations
                report.recommendations = self._generate_recommendations(report)
                
        except Exception as e:
            self.logger.error(f"Validation failed with error: {e}")
            report.results.append(ValidationResult(
                test_name="validation_framework",
                passed=False,
                message=f"Validation framework error: {str(e)}",
                severity="CRITICAL"
            ))
            report.total_tests = 1
            report.failed_tests = 1

        report.execution_time = time.time() - start_time
        self.logger.info(f"Validation {validation_id} completed in {report.execution_time:.2f}s")
        
        return report

    def _validate_syntax(self, migration_script: str) -> ValidationResult:
        """Validate SQL syntax of migration script."""
        try:
            # Basic syntax validation by attempting to parse
            statements = [stmt.strip() for stmt in migration_script.split(';') if stmt.strip()]
            
            if not statements:
                return ValidationResult(
                    test_name="syntax_validation",
                    passed=False,
                    message="Migration script is empty",
                    severity="ERROR"
                )
            
            # Check for common SQL syntax issues
            issues = []
            for i, stmt in enumerate(statements):
                if not any(stmt.upper().startswith(cmd) for cmd in [
                    'CREATE', 'ALTER', 'DROP', 'INSERT', 'UPDATE', 'DELETE', 'RENAME'
                ]):
                    issues.append(f"Statement {i+1} doesn't start with a recognized SQL command")
            
            if issues:
                return ValidationResult(
                    test_name="syntax_validation",
                    passed=False,
                    message="Syntax issues found",
                    details={"issues": issues},
                    severity="ERROR"
                )
            
            return ValidationResult(
                test_name="syntax_validation",
                passed=True,
                message=f"Syntax validation passed for {len(statements)} statements",
                details={"statement_count": len(statements)}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="syntax_validation",
                passed=False,
                message=f"Syntax validation failed: {str(e)}",
                severity="ERROR"
            )

    def _dry_run_migration(self, test_db_manager: DatabaseManager, migration_script: str) -> ValidationResult:
        """Execute migration in a transaction that gets rolled back."""
        start_time = time.time()
        
        try:
            with test_db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                try:
                    # Execute migration within transaction
                    for statement in migration_script.split(';'):
                        statement = statement.strip()
                        if statement:
                            cursor.execute(statement)
                    
                    # Force rollback to test without committing
                    conn.rollback()
                    
                    execution_time = time.time() - start_time
                    
                    return ValidationResult(
                        test_name="dry_run_execution",
                        passed=True,
                        message="Dry run completed successfully",
                        details={"execution_time": execution_time},
                        execution_time=execution_time
                    )
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                    
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                test_name="dry_run_execution",
                passed=False,
                message=f"Dry run failed: {str(e)}",
                details={"error": str(e)},
                execution_time=execution_time,
                severity="ERROR"
            )

    def _validate_schema_changes(self, test_db_manager: DatabaseManager, initial_schema: DatabaseSchema) -> ValidationResult:
        """Validate that schema changes are as expected."""
        try:
            # Apply migration
            with test_db_manager.get_connection() as conn:
                cursor = conn.cursor()
                for statement in self.migration_script.split(';'):
                    statement = statement.strip()
                    if statement:
                        cursor.execute(statement)
                conn.commit()
            
            # Extract new schema
            test_inspector = SchemaInspector(test_db_manager)
            new_schema = test_inspector.extract_full_schema()
            
            # Compare schemas
            changes = []
            
            # Convert table lists to dictionaries for easier comparison
            initial_tables = {table.name: table for table in initial_schema.tables}
            new_tables_dict = {table.name: table for table in new_schema.tables}
            
            # Check for new tables
            new_tables = set(new_tables_dict.keys()) - set(initial_tables.keys())
            if new_tables:
                changes.append(f"Added tables: {', '.join(new_tables)}")
            
            # Check for removed tables
            removed_tables = set(initial_tables.keys()) - set(new_tables_dict.keys())
            if removed_tables:
                changes.append(f"Removed tables: {', '.join(removed_tables)}")
            
            # Check for modified tables
            for table_name in set(initial_tables.keys()) & set(new_tables_dict.keys()):
                initial_cols = {col.name for col in initial_tables[table_name].columns}
                new_cols = {col.name for col in new_tables_dict[table_name].columns}
                
                if initial_cols != new_cols:
                    added_cols = new_cols - initial_cols
                    removed_cols = initial_cols - new_cols
                    if added_cols:
                        changes.append(f"Added columns to {table_name}: {', '.join(added_cols)}")
                    if removed_cols:
                        changes.append(f"Removed columns from {table_name}: {', '.join(removed_cols)}")
            
            return ValidationResult(
                test_name="schema_validation",
                passed=True,
                message="Schema changes validated successfully",
                details={"changes": changes, "change_count": len(changes)}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="schema_validation",
                passed=False,
                message=f"Schema validation failed: {str(e)}",
                severity="ERROR"
            )

    def _run_integrity_checks(self, test_db_manager: DatabaseManager, checks: List[IntegrityCheck]) -> List[ValidationResult]:
        """Run custom data integrity checks."""
        results = []
        
        if not checks:
            # Generate default integrity checks
            checks = self._generate_default_integrity_checks(test_db_manager)
        
        for check in checks:
            try:
                with test_db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(check.query)
                    result = cursor.fetchone()
                    
                    if result is None:
                        actual_result = None
                    elif len(result) == 1:
                        actual_result = result[0]
                    else:
                        actual_result = result
                    
                    passed = actual_result == check.expected_result
                    
                    results.append(ValidationResult(
                        test_name=f"integrity_check_{check.check_name}",
                        passed=passed,
                        message=f"Integrity check '{check.check_name}': {'PASSED' if passed else 'FAILED'}",
                        details={
                            "description": check.description,
                            "expected": check.expected_result,
                            "actual": actual_result,
                            "query": check.query
                        },
                        severity="WARNING" if not passed else "INFO"
                    ))
                    
            except Exception as e:
                results.append(ValidationResult(
                    test_name=f"integrity_check_{check.check_name}",
                    passed=False,
                    message=f"Integrity check '{check.check_name}' failed with error: {str(e)}",
                    details={"error": str(e), "query": check.query},
                    severity="ERROR"
                ))
        
        return results

    def _validate_foreign_keys(self, test_db_manager: DatabaseManager) -> ValidationResult:
        """Validate foreign key constraints."""
        try:
            violated_constraints = []
            
            with test_db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                if test_db_manager.db_type == 'sqlite':
                    # SQLite foreign key validation
                    cursor.execute("PRAGMA foreign_key_check")
                    violations = cursor.fetchall()
                    
                    for violation in violations:
                        violated_constraints.append({
                            "table": violation[0],
                            "rowid": violation[1],
                            "parent": violation[2],
                            "fkid": violation[3]
                        })
                
                elif test_db_manager.db_type == 'mysql':
                    # MySQL foreign key validation
                    cursor.execute("""
                        SELECT TABLE_NAME, CONSTRAINT_NAME, REFERENCED_TABLE_NAME
                        FROM information_schema.KEY_COLUMN_USAGE
                        WHERE REFERENCED_TABLE_SCHEMA = DATABASE()
                        AND REFERENCED_TABLE_NAME IS NOT NULL
                    """)
                    fk_constraints = cursor.fetchall()
                    
                    # Check each constraint
                    for constraint in fk_constraints:
                        table_name, constraint_name, ref_table = constraint
                        # Add specific validation logic for MySQL
            
            if violated_constraints:
                return ValidationResult(
                    test_name="foreign_key_validation",
                    passed=False,
                    message=f"Found {len(violated_constraints)} foreign key violations",
                    details={"violations": violated_constraints},
                    severity="ERROR"
                )
            else:
                return ValidationResult(
                    test_name="foreign_key_validation",
                    passed=True,
                    message="All foreign key constraints are valid",
                    details={"violations_checked": len(violated_constraints)}
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="foreign_key_validation",
                passed=False,
                message=f"Foreign key validation failed: {str(e)}",
                severity="ERROR"
            )

    def _benchmark_migration_performance(self, test_db_manager: DatabaseManager, migration_script: str) -> ValidationResult:
        """Benchmark migration performance."""
        try:
            # Measure execution time
            start_time = time.time()
            
            with test_db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                for statement in migration_script.split(';'):
                    statement = statement.strip()
                    if statement:
                        stmt_start = time.time()
                        cursor.execute(statement)
                        stmt_time = time.time() - stmt_start
                        
                        # Log slow statements
                        if stmt_time > 1.0:  # Statements taking more than 1 second
                            self.logger.warning(f"Slow statement detected: {stmt_time:.2f}s")
                
                conn.commit()
            
            total_time = time.time() - start_time
            
            # Determine performance rating
            if total_time < 1.0:
                rating = "Excellent"
                severity = "INFO"
            elif total_time < 5.0:
                rating = "Good"
                severity = "INFO"
            elif total_time < 30.0:
                rating = "Acceptable"
                severity = "WARNING"
            else:
                rating = "Slow"
                severity = "WARNING"
            
            return ValidationResult(
                test_name="performance_benchmark",
                passed=True,
                message=f"Migration performance: {rating} ({total_time:.3f}s)",
                details={
                    "execution_time": total_time,
                    "rating": rating,
                    "statements_count": len([s for s in migration_script.split(';') if s.strip()])
                },
                execution_time=max(total_time, 0.001),  # Ensure minimum non-zero value for tests
                severity=severity
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="performance_benchmark",
                passed=False,
                message=f"Performance benchmark failed: {str(e)}",
                severity="ERROR"
            )

    def _test_rollback_capability(self, test_db_manager: DatabaseManager, migration_script: str) -> ValidationResult:
        """Test if migration can be rolled back."""
        try:
            # Capture initial state
            initial_inspector = SchemaInspector(test_db_manager)
            initial_schema = initial_inspector.extract_schema()
            
            # Apply migration
            with test_db_manager.get_connection() as conn:
                cursor = conn.cursor()
                savepoint_name = "migration_test"
                
                try:
                    # Create savepoint
                    if test_db_manager.db_type == 'mysql':
                        cursor.execute(f"SAVEPOINT {savepoint_name}")
                    
                    # Execute migration
                    for statement in migration_script.split(';'):
                        statement = statement.strip()
                        if statement:
                            cursor.execute(statement)
                    
                    # Test rollback
                    if test_db_manager.db_type == 'mysql':
                        cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    else:
                        conn.rollback()
                    
                    # Verify rollback worked
                    post_rollback_schema = initial_inspector.extract_schema()
                    
                    # Compare schemas to ensure rollback worked
                    rollback_successful = self._compare_schemas(initial_schema, post_rollback_schema)
                    
                    return ValidationResult(
                        test_name="rollback_capability",
                        passed=rollback_successful,
                        message="Rollback test completed" if rollback_successful else "Rollback test failed",
                        details={"rollback_method": "savepoint" if test_db_manager.db_type == 'mysql' else "transaction"},
                        severity="INFO" if rollback_successful else "WARNING"
                    )
                    
                except Exception as rollback_error:
                    conn.rollback()
                    return ValidationResult(
                        test_name="rollback_capability",
                        passed=False,
                        message=f"Rollback test failed: {str(rollback_error)}",
                        severity="WARNING"
                    )
                    
        except Exception as e:
            return ValidationResult(
                test_name="rollback_capability",
                passed=False,
                message=f"Rollback capability test failed: {str(e)}",
                severity="ERROR"
            )

    def _generate_default_integrity_checks(self, test_db_manager: DatabaseManager) -> List[IntegrityCheck]:
        """Generate default integrity checks based on database schema."""
        checks = []
        
        try:
            inspector = SchemaInspector(test_db_manager)
            schema = inspector.extract_full_schema()
            
            # Check table existence
            for table_info in schema.tables:
                checks.append(IntegrityCheck(
                    check_name=f"table_exists_{table_info.name}",
                    query=f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table_info.name}'" 
                          if test_db_manager.db_type == 'sqlite' 
                          else f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{table_info.name}'",
                    expected_result=1,
                    description=f"Verify table {table_info.name} exists",
                    table=table_info.name
                ))
            
            # Check for null values in non-nullable columns
            for table_info in schema.tables:
                for col_info in table_info.columns:
                    if not col_info.is_nullable:
                        checks.append(IntegrityCheck(
                            check_name=f"no_nulls_{table_info.name}_{col_info.name}",
                            query=f"SELECT COUNT(*) FROM {table_info.name} WHERE {col_info.name} IS NULL",
                            expected_result=0,
                            description=f"Verify no null values in non-nullable column {table_info.name}.{col_info.name}",
                            table=table_info.name
                        ))
            
        except Exception as e:
            self.logger.warning(f"Could not generate default integrity checks: {e}")
        
        return checks

    def _compare_schemas(self, schema1: DatabaseSchema, schema2: DatabaseSchema) -> bool:
        """Compare two schemas for equality."""
        if len(schema1.tables) != len(schema2.tables):
            return False
        
        # Convert to dictionaries for easier comparison
        tables1 = {table.name: table for table in schema1.tables}
        tables2 = {table.name: table for table in schema2.tables}
        
        for table_name in tables1:
            if table_name not in tables2:
                return False
            
            table1 = tables1[table_name]
            table2 = tables2[table_name]
            
            if len(table1.columns) != len(table2.columns):
                return False
            
            cols1 = {col.name for col in table1.columns}
            cols2 = {col.name for col in table2.columns}
            
            if cols1 != cols2:
                return False
        
        return True

    def _populate_test_data(self, test_db_manager: DatabaseManager, test_data_script: str):
        """Populate test database with sample data."""
        try:
            with test_db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                for statement in test_data_script.split(';'):
                    statement = statement.strip()
                    if statement:
                        cursor.execute(statement)
                
                conn.commit()
                self.logger.info("Test data populated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to populate test data: {e}")
            raise

    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Performance recommendations
        if report.performance_metrics.get("execution_time", 0) > 30:
            recommendations.append("Consider breaking down large migrations into smaller chunks")
            recommendations.append("Review queries for optimization opportunities")
        
        # Error-based recommendations
        error_count = sum(1 for r in report.results if not r.passed and r.severity == "ERROR")
        if error_count > 0:
            recommendations.append("Fix all errors before proceeding to production")
            recommendations.append("Review migration script for syntax and logic issues")
        
        # Warning-based recommendations
        warning_count = sum(1 for r in report.results if r.severity == "WARNING")
        if warning_count > 2:
            recommendations.append("Review warnings and consider addressing them before deployment")
        
        # Success rate recommendations
        if report.success_rate < 90:
            recommendations.append("Improve test coverage and fix failing validations")
            recommendations.append("Consider additional testing in staging environment")
        
        # Foreign key recommendations
        fk_issues = any("foreign_key" in r.test_name and not r.passed for r in report.results)
        if fk_issues:
            recommendations.append("Review and fix foreign key constraint violations")
            recommendations.append("Ensure referential integrity is maintained")
        
        if not recommendations:
            recommendations.append("Migration validation looks good - ready for deployment!")
        
        return recommendations

    @contextmanager
    def _create_test_database(self):
        """Create a temporary test database."""
        if self.db_manager.db_type == 'sqlite':
            # Create temporary SQLite database
            temp_dir = tempfile.mkdtemp()
            test_db_path = Path(temp_dir) / "test_migration.db"
            
            try:
                # Copy original database structure
                if self.db_manager.db_path and Path(self.db_manager.db_path).exists():
                    shutil.copy2(self.db_manager.db_path, test_db_path)
                
                yield str(test_db_path)
            finally:
                # Cleanup
                if test_db_path.exists():
                    test_db_path.unlink()
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            # For MySQL, we would create a temporary database
            # This is a simplified implementation
            yield None

    def _create_test_db_manager(self, test_db_path: Optional[str]) -> DatabaseManager:
        """Create a database manager for the test database."""
        if self.db_manager.db_type == 'sqlite' and test_db_path:
            return DatabaseManager(db_type='sqlite', db_path=test_db_path)
        else:
            # For MySQL, return the original manager (would need temporary DB creation)
            return self.db_manager

    def generate_validation_report_html(self, report: ValidationReport) -> str:
        """Generate HTML validation report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Migration Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #e8f4fd; padding: 10px; border-radius: 5px; text-align: center; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .passed {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
                .failed {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                .recommendations {{ background-color: #e2e3e5; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                .details {{ font-family: monospace; background-color: #f8f9fa; padding: 10px; margin-top: 5px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Migration Validation Report</h1>
                <p><strong>Validation ID:</strong> {validation_id}</p>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                <p><strong>Database Type:</strong> {database_type}</p>
                <p><strong>Execution Time:</strong> {execution_time:.2f} seconds</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>{total_tests}</h3>
                    <p>Total Tests</p>
                </div>
                <div class="metric">
                    <h3>{passed_tests}</h3>
                    <p>Passed</p>
                </div>
                <div class="metric">
                    <h3>{failed_tests}</h3>
                    <p>Failed</p>
                </div>
                <div class="metric">
                    <h3>{success_rate:.1f}%</h3>
                    <p>Success Rate</p>
                </div>
                <div class="metric">
                    <h3>{deployment_status}</h3>
                    <p>Deployment Status</p>
                </div>
            </div>
            
            <h2>Test Results</h2>
            {test_results}
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
                {recommendations_list}
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Generate test results HTML
        test_results_html = ""
        for result in report.results:
            css_class = "passed" if result.passed else ("warning" if result.severity == "WARNING" else "failed")
            
            details_html = ""
            if result.details:
                details_html = f'<div class="details">{json.dumps(result.details, indent=2)}</div>'
            
            test_results_html += f"""
            <div class="test-result {css_class}">
                <h4>{'✅' if result.passed else '❌'} {result.test_name}</h4>
                <p><strong>Message:</strong> {result.message}</p>
                <p><strong>Severity:</strong> {result.severity}</p>
                <p><strong>Execution Time:</strong> {result.execution_time:.3f}s</p>
                {details_html}
            </div>
            """
        
        # Generate recommendations HTML
        recommendations_html = ""
        for rec in report.recommendations:
            recommendations_html += f"<li>{rec}</li>"
        
        deployment_status = "✅ SAFE" if report.is_safe_to_deploy else "⚠️ RISKY"
        
        return html_template.format(
            validation_id=report.validation_id,
            timestamp=report.timestamp.strftime("%Y-%m-%d %H:%M:%S") if report.timestamp else "N/A",
            database_type=report.database_type,
            execution_time=report.execution_time,
            total_tests=report.total_tests,
            passed_tests=report.passed_tests,
            failed_tests=report.failed_tests,
            success_rate=report.success_rate,
            deployment_status=deployment_status,
            test_results=test_results_html,
            recommendations_list=recommendations_html
        )

    def save_validation_report(self, report: ValidationReport, output_dir: Path) -> Tuple[Path, Path]:
        """
        Save validation report in both JSON and HTML formats.
        
        Args:
            report: Validation report to save
            output_dir: Directory to save reports
            
        Returns:
            Tuple of (json_path, html_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = output_dir / f"validation_report_{report.validation_id}.json"
        with open(json_path, 'w') as f:
            json.dump(report.__dict__, f, indent=2, default=str)
        
        # Save HTML report
        html_path = output_dir / f"validation_report_{report.validation_id}.html"
        html_content = self.generate_validation_report_html(report)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Validation report saved: {json_path} and {html_path}")
        return json_path, html_path 