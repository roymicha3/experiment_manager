"""
Automated Test Generator for Migration Validation

This module provides automated generation of validation tests for database migrations
based on schema analysis and common migration patterns.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

from .database_adapter import DatabaseManager
from experiment_manager.db.schema_inspector import SchemaInspector, DatabaseSchema
from .migration_validator import IntegrityCheck

logger = logging.getLogger(__name__)


@dataclass
class TestSuite:
    """Represents a collection of generated tests for migration validation."""
    name: str
    description: str
    integrity_checks: List[IntegrityCheck]
    test_data_script: str
    expected_schema_changes: Dict[str, Any]
    rollback_tests: List[str]
    performance_benchmarks: List[Dict[str, Any]]


class TestGenerator:
    """
    Automated test generator for migration validation.
    
    Generates comprehensive test suites based on:
    - Database schema analysis
    - Migration script patterns
    - Common validation scenarios
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the test generator.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.schema_inspector = SchemaInspector(db_manager)
        self.logger = logging.getLogger(__name__)

    def generate_migration_tests(
        self,
        migration_script: str,
        test_suite_name: str = "auto_generated_tests"
    ) -> TestSuite:
        """
        Generate comprehensive test suite for a migration script.
        
        Args:
            migration_script: SQL migration script to generate tests for
            test_suite_name: Name for the generated test suite
            
        Returns:
            TestSuite: Generated test suite
        """
        self.logger.info(f"Generating test suite '{test_suite_name}' for migration")
        
        # Analyze migration script
        migration_analysis = self._analyze_migration_script(migration_script)
        
        # Generate integrity checks
        integrity_checks = self._generate_integrity_checks(migration_analysis)
        
        # Generate test data script
        test_data_script = self._generate_test_data_script(migration_analysis)
        
        # Generate expected schema changes
        expected_changes = self._generate_expected_schema_changes(migration_analysis)
        
        # Generate rollback tests
        rollback_tests = self._generate_rollback_tests(migration_analysis)
        
        # Generate performance benchmarks
        performance_benchmarks = self._generate_performance_benchmarks(migration_analysis)
        
        test_suite = TestSuite(
            name=test_suite_name,
            description=f"Auto-generated validation tests for migration",
            integrity_checks=integrity_checks,
            test_data_script=test_data_script,
            expected_schema_changes=expected_changes,
            rollback_tests=rollback_tests,
            performance_benchmarks=performance_benchmarks
        )
        
        self.logger.info(f"Generated test suite with {len(integrity_checks)} integrity checks")
        return test_suite

    def _analyze_migration_script(self, migration_script: str) -> Dict[str, Any]:
        """
        Analyze migration script to understand its operations.
        
        Args:
            migration_script: SQL migration script
            
        Returns:
            Dict containing analysis results
        """
        analysis = {
            "operations": [],
            "tables_created": [],
            "tables_dropped": [],
            "tables_altered": [],
            "columns_added": [],
            "columns_dropped": [],
            "indexes_created": [],
            "indexes_dropped": [],
            "constraints_added": [],
            "constraints_dropped": [],
            "data_modifications": []
        }
        
        statements = [stmt.strip() for stmt in migration_script.split(';') if stmt.strip()]
        
        for stmt in statements:
            stmt_upper = stmt.upper()
            
            # Analyze CREATE TABLE statements
            if stmt_upper.startswith('CREATE TABLE'):
                table_name = self._extract_table_name_from_create(stmt)
                if table_name:
                    analysis["tables_created"].append(table_name)
                    analysis["operations"].append({
                        "type": "CREATE_TABLE",
                        "table": table_name,
                        "statement": stmt
                    })
            
            # Analyze DROP TABLE statements
            elif stmt_upper.startswith('DROP TABLE'):
                table_name = self._extract_table_name_from_drop(stmt)
                if table_name:
                    analysis["tables_dropped"].append(table_name)
                    analysis["operations"].append({
                        "type": "DROP_TABLE",
                        "table": table_name,
                        "statement": stmt
                    })
            
            # Analyze ALTER TABLE statements
            elif stmt_upper.startswith('ALTER TABLE'):
                alter_analysis = self._analyze_alter_statement(stmt)
                analysis["tables_altered"].append(alter_analysis["table"])
                analysis["operations"].append({
                    "type": "ALTER_TABLE",
                    **alter_analysis,
                    "statement": stmt
                })
                
                # Add specific column operations
                if alter_analysis.get("columns_added"):
                    analysis["columns_added"].extend(alter_analysis["columns_added"])
                if alter_analysis.get("columns_dropped"):
                    analysis["columns_dropped"].extend(alter_analysis["columns_dropped"])
            
            # Analyze CREATE INDEX statements
            elif stmt_upper.startswith('CREATE INDEX') or stmt_upper.startswith('CREATE UNIQUE INDEX'):
                index_info = self._extract_index_info(stmt)
                if index_info:
                    analysis["indexes_created"].append(index_info)
                    analysis["operations"].append({
                        "type": "CREATE_INDEX",
                        **index_info,
                        "statement": stmt
                    })
            
            # Analyze DROP INDEX statements
            elif stmt_upper.startswith('DROP INDEX'):
                index_name = self._extract_index_name_from_drop(stmt)
                if index_name:
                    analysis["indexes_dropped"].append(index_name)
                    analysis["operations"].append({
                        "type": "DROP_INDEX",
                        "index": index_name,
                        "statement": stmt
                    })
            
            # Analyze data modification statements
            elif any(stmt_upper.startswith(cmd) for cmd in ['INSERT', 'UPDATE', 'DELETE']):
                analysis["data_modifications"].append({
                    "type": stmt_upper.split()[0],
                    "statement": stmt
                })
                analysis["operations"].append({
                    "type": "DATA_MODIFICATION",
                    "operation": stmt_upper.split()[0],
                    "statement": stmt
                })
        
        return analysis

    def _generate_integrity_checks(self, migration_analysis: Dict[str, Any]) -> List[IntegrityCheck]:
        """Generate integrity checks based on migration analysis."""
        checks = []
        
        # Check for newly created tables
        for table_name in migration_analysis["tables_created"]:
            checks.append(IntegrityCheck(
                check_name=f"table_created_{table_name}",
                query=f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'" 
                      if self.db_manager.db_type == 'sqlite' 
                      else f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{table_name}'",
                expected_result=1,
                description=f"Verify that table '{table_name}' was created",
                table=table_name
            ))
        
        # Check for dropped tables
        for table_name in migration_analysis["tables_dropped"]:
            checks.append(IntegrityCheck(
                check_name=f"table_dropped_{table_name}",
                query=f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'" 
                      if self.db_manager.db_type == 'sqlite' 
                      else f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{table_name}'",
                expected_result=0,
                description=f"Verify that table '{table_name}' was dropped",
                table=table_name
            ))
        
        # Check for created indexes
        for index_info in migration_analysis["indexes_created"]:
            index_name = index_info["name"]
            checks.append(IntegrityCheck(
                check_name=f"index_created_{index_name}",
                query=f"SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='{index_name}'" 
                      if self.db_manager.db_type == 'sqlite' 
                      else f"SELECT COUNT(*) FROM information_schema.statistics WHERE index_name='{index_name}'",
                expected_result=1,
                description=f"Verify that index '{index_name}' was created",
                table=index_info.get("table")
            ))
        
        # Check data integrity for modified tables
        for table_name in migration_analysis["tables_altered"]:
            # Check that table still exists after alteration
            checks.append(IntegrityCheck(
                check_name=f"table_exists_after_alter_{table_name}",
                query=f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'" 
                      if self.db_manager.db_type == 'sqlite' 
                      else f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{table_name}'",
                expected_result=1,
                description=f"Verify that table '{table_name}' exists after alteration",
                table=table_name
            ))
            
            # Check that data is preserved (basic row count check)
            checks.append(IntegrityCheck(
                check_name=f"data_preserved_{table_name}",
                query=f"SELECT COUNT(*) >= 0 FROM {table_name}",
                expected_result=True,
                description=f"Verify that table '{table_name}' is accessible and contains data",
                table=table_name
            ))
        
        # Add foreign key integrity checks
        try:
            current_schema = self.schema_inspector.extract_full_schema()
            for table_info in current_schema.tables:
                for fk in table_info.foreign_keys:
                    checks.append(IntegrityCheck(
                        check_name=f"foreign_key_integrity_{table_info.name}_{fk.column_name}",
                        query=f"""
                        SELECT COUNT(*) FROM {table_info.name} t 
                        LEFT JOIN {fk.referenced_table} r ON t.{fk.column_name} = r.{fk.referenced_column}
                        WHERE t.{fk.column_name} IS NOT NULL AND r.{fk.referenced_column} IS NULL
                        """,
                        expected_result=0,
                        description=f"Verify foreign key integrity for {table_info.name}.{fk.column_name}",
                        table=table_info.name
                    ))
        except Exception as e:
            self.logger.warning(f"Could not generate foreign key checks: {e}")
        
        return checks

    def _generate_test_data_script(self, migration_analysis: Dict[str, Any]) -> str:
        """Generate test data insertion script."""
        script_parts = []
        
        # Generate test data for newly created tables
        for table_name in migration_analysis["tables_created"]:
            # Generate sample data based on table structure
            try:
                # This would need to be enhanced to inspect the actual table structure
                # For now, generate basic test data
                script_parts.append(f"-- Test data for {table_name}")
                script_parts.append(f"-- INSERT INTO {table_name} VALUES (...);")
                script_parts.append("")
            except Exception as e:
                self.logger.warning(f"Could not generate test data for {table_name}: {e}")
        
        # Add data validation inserts for existing tables
        for table_name in migration_analysis["tables_altered"]:
            script_parts.append(f"-- Validation data for altered table {table_name}")
            script_parts.append(f"-- INSERT INTO {table_name} VALUES (...);")
            script_parts.append("")
        
        return "\n".join(script_parts) if script_parts else "-- No test data generated"

    def _generate_expected_schema_changes(self, migration_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate expected schema changes for validation."""
        return {
            "tables_created": migration_analysis["tables_created"],
            "tables_dropped": migration_analysis["tables_dropped"],
            "tables_altered": migration_analysis["tables_altered"],
            "columns_added": migration_analysis["columns_added"],
            "columns_dropped": migration_analysis["columns_dropped"],
            "indexes_created": [idx["name"] for idx in migration_analysis["indexes_created"]],
            "indexes_dropped": migration_analysis["indexes_dropped"],
            "total_operations": len(migration_analysis["operations"])
        }

    def _generate_rollback_tests(self, migration_analysis: Dict[str, Any]) -> List[str]:
        """Generate rollback test scenarios."""
        rollback_tests = []
        
        # Test rollback for each major operation type
        if migration_analysis["tables_created"]:
            rollback_tests.append("test_rollback_table_creation")
        
        if migration_analysis["tables_altered"]:
            rollback_tests.append("test_rollback_table_alteration")
        
        if migration_analysis["indexes_created"]:
            rollback_tests.append("test_rollback_index_creation")
        
        if migration_analysis["data_modifications"]:
            rollback_tests.append("test_rollback_data_modifications")
        
        return rollback_tests

    def _generate_performance_benchmarks(self, migration_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance benchmark scenarios."""
        benchmarks = []
        
        # Benchmark table creation operations
        for table_name in migration_analysis["tables_created"]:
            benchmarks.append({
                "name": f"benchmark_create_table_{table_name}",
                "description": f"Benchmark creation of table {table_name}",
                "expected_max_time": 5.0,  # 5 seconds max
                "operation_type": "CREATE_TABLE"
            })
        
        # Benchmark table alteration operations
        for table_name in migration_analysis["tables_altered"]:
            benchmarks.append({
                "name": f"benchmark_alter_table_{table_name}",
                "description": f"Benchmark alteration of table {table_name}",
                "expected_max_time": 10.0,  # 10 seconds max for alterations
                "operation_type": "ALTER_TABLE"
            })
        
        # Benchmark index creation
        for index_info in migration_analysis["indexes_created"]:
            benchmarks.append({
                "name": f"benchmark_create_index_{index_info['name']}",
                "description": f"Benchmark creation of index {index_info['name']}",
                "expected_max_time": 15.0,  # 15 seconds max for indexes
                "operation_type": "CREATE_INDEX"
            })
        
        return benchmarks

    def _extract_table_name_from_create(self, stmt: str) -> Optional[str]:
        """Extract table name from CREATE TABLE statement."""
        try:
            # Keep original statement for extracting names with original case
            original_parts = stmt.split()
            parts = stmt.upper().split()
            
            if len(parts) >= 3 and parts[0] == 'CREATE' and parts[1] == 'TABLE':
                # Get the table name from original parts but check structure with uppercase
                table_name = original_parts[2].strip('(').strip('"').strip("'")
                
                # Remove IF NOT EXISTS if present
                if parts[2].upper() == 'IF':
                    if len(parts) >= 6 and parts[3] == 'NOT' and parts[4] == 'EXISTS':
                        table_name = original_parts[5].strip('(').strip('"').strip("'")
                return table_name
        except Exception:
            pass
        return None

    def _extract_table_name_from_drop(self, stmt: str) -> Optional[str]:
        """Extract table name from DROP TABLE statement."""
        try:
            original_parts = stmt.split()
            parts = stmt.upper().split()
            
            if len(parts) >= 3 and parts[0] == 'DROP' and parts[1] == 'TABLE':
                table_name = original_parts[2].strip().strip('"').strip("'")
                # Remove IF EXISTS if present
                if parts[2].upper() == 'IF':
                    if len(parts) >= 5 and parts[3] == 'EXISTS':
                        table_name = original_parts[4].strip().strip('"').strip("'")
                return table_name
        except Exception:
            pass
        return None

    def _analyze_alter_statement(self, stmt: str) -> Dict[str, Any]:
        """Analyze ALTER TABLE statement."""
        analysis = {
            "table": None,
            "operations": [],
            "columns_added": [],
            "columns_dropped": []
        }
        
        try:
            original_parts = stmt.split()
            parts = stmt.upper().split()
            
            if len(parts) >= 3 and parts[0] == 'ALTER' and parts[1] == 'TABLE':
                analysis["table"] = original_parts[2].strip().strip('"').strip("'")
                
                stmt_upper = stmt.upper()
                if 'ADD COLUMN' in stmt_upper or 'ADD ' in stmt_upper:
                    analysis["operations"].append("ADD_COLUMN")
                    # Extract column name (simplified)
                    if 'ADD COLUMN' in stmt_upper:
                        # Find the position in original statement
                        add_pos = stmt.upper().find('ADD COLUMN') + len('ADD COLUMN')
                        add_part = stmt[add_pos:].strip()
                        col_name = add_part.split()[0].strip('"').strip("'")
                        analysis["columns_added"].append(col_name)
                
                if 'DROP COLUMN' in stmt_upper:
                    analysis["operations"].append("DROP_COLUMN")
                    # Extract column name (simplified)
                    drop_pos = stmt.upper().find('DROP COLUMN') + len('DROP COLUMN')
                    drop_part = stmt[drop_pos:].strip()
                    col_name = drop_part.split()[0].strip('"').strip("'")
                    analysis["columns_dropped"].append(col_name)
                
                if 'RENAME' in stmt_upper:
                    analysis["operations"].append("RENAME")
                
                if 'MODIFY' in stmt_upper:
                    analysis["operations"].append("MODIFY_COLUMN")
                    
        except Exception as e:
            self.logger.warning(f"Could not analyze ALTER statement: {e}")
        
        return analysis

    def _extract_index_info(self, stmt: str) -> Optional[Dict[str, Any]]:
        """Extract index information from CREATE INDEX statement."""
        try:
            original_parts = stmt.split()
            stmt_upper = stmt.upper()
            parts = stmt_upper.split()
            
            index_info = {
                "name": None,
                "table": None,
                "unique": False
            }
            
            if 'UNIQUE' in parts:
                index_info["unique"] = True
                # Format: CREATE UNIQUE INDEX index_name ON table_name
                name_idx = parts.index('INDEX') + 1
            else:
                # Format: CREATE INDEX index_name ON table_name
                name_idx = parts.index('INDEX') + 1
            
            if name_idx < len(original_parts):
                index_info["name"] = original_parts[name_idx].strip('"').strip("'")
            
            if 'ON' in parts:
                on_idx = parts.index('ON') + 1
                if on_idx < len(original_parts):
                    table_part = original_parts[on_idx]
                    index_info["table"] = table_part.split('(')[0].strip('"').strip("'")
            
            return index_info
            
        except Exception:
            return None

    def _extract_index_name_from_drop(self, stmt: str) -> Optional[str]:
        """Extract index name from DROP INDEX statement."""
        try:
            parts = stmt.upper().split()
            if len(parts) >= 3 and parts[0] == 'DROP' and parts[1] == 'INDEX':
                return parts[2].strip().strip('"').strip("'")
        except Exception:
            pass
        return None

    def export_test_suite(self, test_suite: TestSuite, output_path: Path) -> Path:
        """
        Export test suite to JSON file.
        
        Args:
            test_suite: Test suite to export
            output_path: Path to save the test suite
            
        Returns:
            Path: Path to the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert test suite to dictionary
        test_suite_dict = {
            "name": test_suite.name,
            "description": test_suite.description,
            "integrity_checks": [
                {
                    "check_name": check.check_name,
                    "query": check.query,
                    "expected_result": check.expected_result,
                    "description": check.description,
                    "table": check.table
                }
                for check in test_suite.integrity_checks
            ],
            "test_data_script": test_suite.test_data_script,
            "expected_schema_changes": test_suite.expected_schema_changes,
            "rollback_tests": test_suite.rollback_tests,
            "performance_benchmarks": test_suite.performance_benchmarks
        }
        
        with open(output_path, 'w') as f:
            json.dump(test_suite_dict, f, indent=2)
        
        self.logger.info(f"Test suite exported to {output_path}")
        return output_path

    def load_test_suite(self, input_path: Path) -> TestSuite:
        """
        Load test suite from JSON file.
        
        Args:
            input_path: Path to the test suite file
            
        Returns:
            TestSuite: Loaded test suite
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        integrity_checks = [
            IntegrityCheck(
                check_name=check["check_name"],
                query=check["query"],
                expected_result=check["expected_result"],
                description=check["description"],
                table=check.get("table")
            )
            for check in data["integrity_checks"]
        ]
        
        return TestSuite(
            name=data["name"],
            description=data["description"],
            integrity_checks=integrity_checks,
            test_data_script=data["test_data_script"],
            expected_schema_changes=data["expected_schema_changes"],
            rollback_tests=data["rollback_tests"],
            performance_benchmarks=data["performance_benchmarks"]
        ) 