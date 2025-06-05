"""
Tests for Migration Validation Framework

This module contains comprehensive tests for the database migration validation
framework including test generation, validation execution, and reporting.
"""
import pytest
import tempfile
import sqlite3
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.experiment_manager.database.database_adapter import DatabaseManager
from src.experiment_manager.database.migration_validator import (
    MigrationValidator, ValidationResult, IntegrityCheck, ValidationReport
)
from src.experiment_manager.database.performance_profiler import PerformanceProfiler
from experiment_manager.db.schema_inspector import SchemaInspector


class TestMigrationValidator:
    """Test the MigrationValidator class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary SQLite database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_migration.db"
        
        # Create basic test schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create test tables
        cursor.execute("""
            CREATE TABLE experiments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                metric_name TEXT NOT NULL,
                value REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)
        
        # Insert sample data
        cursor.execute("INSERT INTO experiments (name, status) VALUES (?, ?)", ("test_exp", "running"))
        cursor.execute("INSERT INTO metrics (experiment_id, metric_name, value) VALUES (?, ?, ?)", 
                      (1, "accuracy", 0.95))
        
        conn.commit()
        conn.close()
        
        yield str(db_path)
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def db_manager(self, temp_db):
        """Create a DatabaseManager for testing."""
        return DatabaseManager(db_type='sqlite', db_path=temp_db)

    @pytest.fixture
    def validator(self, db_manager):
        """Create a MigrationValidator for testing."""
        return MigrationValidator(db_manager)

    @pytest.fixture
    def test_migration_script(self):
        """Sample migration script for testing."""
        return """
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT
        );
        
        CREATE INDEX idx_test_name ON test_table(name);
        
        ALTER TABLE experiments ADD COLUMN metadata TEXT;
        """

    def test_initialization(self, validator, db_manager):
        """Test validator initialization."""
        assert validator.db_manager == db_manager
        assert isinstance(validator.schema_inspector, SchemaInspector)
        assert isinstance(validator.profiler, PerformanceProfiler)

    def test_validate_syntax_valid_script(self, validator):
        """Test syntax validation with valid SQL."""
        valid_script = "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);"
        result = validator._validate_syntax(valid_script)
        
        assert result.passed
        assert result.test_name == "syntax_validation"
        assert "1 statements" in result.message

    def test_validate_syntax_empty_script(self, validator):
        """Test syntax validation with empty script."""
        empty_script = ""
        result = validator._validate_syntax(empty_script)
        
        assert not result.passed
        assert result.severity == "ERROR"
        assert "empty" in result.message

    def test_validate_syntax_invalid_script(self, validator):
        """Test syntax validation with invalid SQL."""
        invalid_script = "INVALID SQL STATEMENT"
        result = validator._validate_syntax(invalid_script)
        
        assert not result.passed
        assert result.severity == "ERROR"

    def test_dry_run_migration_success(self, validator, db_manager, test_migration_script):
        """Test successful dry run execution."""
        result = validator._dry_run_migration(db_manager, test_migration_script)
        
        assert result.passed
        assert result.test_name == "dry_run_execution"
        assert result.execution_time >= 0  # Allow for very fast execution

    def test_dry_run_migration_failure(self, validator, db_manager):
        """Test dry run with failing SQL."""
        failing_script = "CREATE TABLE invalid syntax here"
        result = validator._dry_run_migration(db_manager, failing_script)
        
        assert not result.passed
        assert result.severity == "ERROR"
        assert "Dry run failed" in result.message

    def test_validate_foreign_keys_sqlite(self, validator, db_manager):
        """Test foreign key validation for SQLite."""
        result = validator._validate_foreign_keys(db_manager)
        
        assert result.test_name == "foreign_key_validation"
        # Should pass as our test data maintains referential integrity
        assert result.passed

    def test_benchmark_migration_performance(self, validator, db_manager):
        """Test performance benchmarking."""
        simple_script = "CREATE TABLE perf_test (id INTEGER);"
        result = validator._benchmark_migration_performance(db_manager, simple_script)
        
        assert result.test_name == "performance_benchmark"
        assert result.passed
        assert "execution_time" in result.details
        assert result.execution_time >= 0  # Allow for very fast execution

    def test_generate_default_integrity_checks(self, validator, db_manager):
        """Test generation of default integrity checks."""
        checks = validator._generate_default_integrity_checks(db_manager)
        
        assert len(checks) > 0
        # Should have checks for existing tables
        table_check_names = [check.check_name for check in checks]
        assert any("experiments" in name for name in table_check_names)
        assert any("metrics" in name for name in table_check_names)

    def test_run_integrity_checks(self, validator, db_manager):
        """Test running integrity checks."""
        custom_checks = [
            IntegrityCheck(
                check_name="test_table_count",
                query="SELECT COUNT(*) FROM experiments",
                expected_result=1,
                description="Check experiment count"
            ),
            IntegrityCheck(
                check_name="test_metric_exists",
                query="SELECT COUNT(*) FROM metrics WHERE metric_name = 'accuracy'",
                expected_result=1,
                description="Check accuracy metric exists"
            )
        ]
        
        results = validator._run_integrity_checks(db_manager, custom_checks)
        
        assert len(results) == 2
        assert all(isinstance(r, ValidationResult) for r in results)
        assert all(r.passed for r in results)  # Our test data should pass these checks

    def test_run_integrity_checks_failure(self, validator, db_manager):
        """Test integrity checks with expected failures."""
        failing_check = IntegrityCheck(
            check_name="failing_check",
            query="SELECT COUNT(*) FROM experiments WHERE status = 'completed'",
            expected_result=1,  # We know there are no completed experiments
            description="This should fail"
        )
        
        results = validator._run_integrity_checks(db_manager, [failing_check])
        
        assert len(results) == 1
        assert not results[0].passed
        assert results[0].severity == "WARNING"

    @patch('src.experiment_manager.database.migration_validator.MigrationValidator._create_test_database')
    def test_full_validation_workflow(self, mock_create_test_db, validator, db_manager, test_migration_script):
        """Test the complete validation workflow."""
        # Mock the test database creation to use the main database
        mock_create_test_db.return_value.__enter__.return_value = db_manager.db_path
        
        # Mock the migration script access
        validator.migration_script = test_migration_script
        
        report = validator.validate_migration(
            migration_script=test_migration_script,
            test_data_script=None,
            integrity_checks=None,
            perform_rollback_test=False,  # Skip rollback for this test
            benchmark_performance=True
        )
        
        assert isinstance(report, ValidationReport)
        assert report.total_tests > 0
        assert report.execution_time >= 0  # Allow for very fast execution
        assert len(report.results) > 0
        assert len(report.recommendations) > 0

    def test_generate_recommendations(self, validator):
        """Test recommendation generation."""
        # Create a mock report with various scenarios
        mock_report = ValidationReport(
            validation_id="test",
            timestamp=None,
            migration_script="test",
            database_type="sqlite",
            total_tests=10,
            passed_tests=7,
            failed_tests=3,
            warnings=2,
            execution_time=35.0  # Slow execution
        )
        
        # Add mock results
        mock_report.results = [
            ValidationResult("test1", False, "Failed", severity="ERROR"),
            ValidationResult("test2", False, "Failed", severity="WARNING"),
            ValidationResult("foreign_key_test", False, "FK violation", severity="ERROR")
        ]
        
        mock_report.performance_metrics = {"execution_time": 35.0}
        
        recommendations = validator._generate_recommendations(mock_report)
        
        assert len(recommendations) > 0
        # Should recommend breaking down migration due to slow execution
        assert any("breaking down" in rec.lower() for rec in recommendations)
        # Should recommend fixing errors
        assert any("fix all errors" in rec.lower() for rec in recommendations)
        # Should mention foreign key issues
        assert any("foreign key" in rec.lower() for rec in recommendations)

    def test_save_validation_report(self, validator, temp_db):
        """Test saving validation reports."""
        report = ValidationReport(
            validation_id="test_save",
            timestamp=None,
            migration_script="CREATE TABLE test (id INTEGER);",
            database_type="sqlite",
            total_tests=3,
            passed_tests=3,
            failed_tests=0,
            warnings=0,
            execution_time=1.5
        )
        
        output_dir = Path(temp_db).parent / "reports"
        json_path, html_path = validator.save_validation_report(report, output_dir)
        
        assert json_path.exists()
        assert html_path.exists()
        assert json_path.suffix == '.json'
        assert html_path.suffix == '.html'
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            data = json.load(f)
        assert data["validation_id"] == "test_save"
        assert data["total_tests"] == 3
        
        # Verify HTML content
        with open(html_path, 'r') as f:
            html_content = f.read()
        assert "Migration Validation Report" in html_content
        assert "test_save" in html_content


class TestPerformanceProfiler:
    """Test the PerformanceProfiler class."""

    @pytest.fixture
    def profiler(self):
        """Create a PerformanceProfiler for testing."""
        return PerformanceProfiler(slow_query_threshold=0.1)

    def test_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.slow_query_threshold == 0.1
        assert len(profiler.metrics_history) == 0

    def test_profile_operation_context_manager(self, profiler):
        """Test the profile_operation context manager."""
        import time
        
        with profiler.profile_operation("test_operation") as metrics:
            time.sleep(0.01)  # Small delay to measure
            metrics.query_count = 5
        
        # Check that metrics were recorded
        assert len(profiler.metrics_history) == 1
        recorded_metrics = profiler.metrics_history[0]
        
        assert recorded_metrics.operation_name == "test_operation"
        assert recorded_metrics.duration > 0
        assert recorded_metrics.query_count == 5

    def test_profile_query_normal(self, profiler):
        """Test profiling a normal (fast) query."""
        is_slow = profiler.profile_query("SELECT 1", 0.05)
        assert not is_slow

    def test_profile_query_slow(self, profiler):
        """Test profiling a slow query."""
        is_slow = profiler.profile_query("SELECT * FROM large_table", 0.5)
        assert is_slow

    def test_get_summary_statistics(self, profiler):
        """Test summary statistics generation."""
        # Add some mock metrics
        import time
        
        with profiler.profile_operation("op1"):
            time.sleep(0.01)
        
        with profiler.profile_operation("op2"):
            time.sleep(0.01)
        
        stats = profiler.get_summary_statistics()
        
        assert stats["total_operations"] == 2
        assert stats["total_duration"] > 0
        assert stats["average_duration"] > 0
        assert len(stats["operations"]) == 2

    def test_clear_history(self, profiler):
        """Test clearing metrics history."""
        with profiler.profile_operation("test"):
            pass
        
        assert len(profiler.metrics_history) == 1
        
        profiler.clear_history()
        assert len(profiler.metrics_history) == 0

    def test_export_metrics(self, profiler):
        """Test exporting metrics."""
        with profiler.profile_operation("export_test") as metrics:
            metrics.query_count = 3
            metrics.slow_queries = ["slow query"]
        
        exported = profiler.export_metrics()
        
        assert len(exported) == 1
        assert exported[0]["operation_name"] == "export_test"
        assert exported[0]["query_count"] == 3
        assert exported[0]["slow_queries"] == ["slow query"]


class TestValidationResult:
    """Test the ValidationResult class."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult instances."""
        result = ValidationResult(
            test_name="test_validation",
            passed=True,
            message="Test passed successfully",
            details={"query": "SELECT 1", "result": 1},
            execution_time=0.05,
            severity="INFO"
        )
        
        assert result.test_name == "test_validation"
        assert result.passed
        assert result.message == "Test passed successfully"
        assert result.details["query"] == "SELECT 1"
        assert result.execution_time == 0.05
        assert result.severity == "INFO"


class TestValidationReport:
    """Test the ValidationReport class."""

    def test_validation_report_properties(self):
        """Test ValidationReport calculated properties."""
        report = ValidationReport(
            validation_id="test_report",
            timestamp=None,
            migration_script="CREATE TABLE test (id INTEGER);",
            database_type="sqlite",
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            warnings=1,
            execution_time=5.0
        )
        
        # Add mock results for safety calculation
        report.results = [
            ValidationResult("test1", True, "Passed"),
            ValidationResult("test2", False, "Failed", severity="WARNING"),
            ValidationResult("test3", False, "Critical failure", severity="CRITICAL")
        ]
        
        assert report.success_rate == 80.0
        assert not report.is_safe_to_deploy  # Has critical failure

    def test_validation_report_safe_deployment(self):
        """Test safe deployment determination."""
        report = ValidationReport(
            validation_id="safe_report",
            timestamp=None,
            migration_script="CREATE TABLE test (id INTEGER);",
            database_type="sqlite",
            total_tests=10,
            passed_tests=10,
            failed_tests=0,
            warnings=0,
            execution_time=2.0
        )
        
        # Add only passing results
        report.results = [
            ValidationResult("test1", True, "Passed"),
            ValidationResult("test2", True, "Passed"),
        ]
        
        assert report.success_rate == 100.0
        assert report.is_safe_to_deploy


if __name__ == '__main__':
    pytest.main([__file__]) 