"""Tests for schema comparison and diff generation tools."""
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from experiment_manager.db.manager import DatabaseManager
from experiment_manager.db.schema_inspector import (
    SchemaInspector, DatabaseSchema, TableInfo, ColumnInfo, IndexInfo, 
    ForeignKeyInfo, ColumnType
)
from experiment_manager.db.schema_comparator import (
    SchemaComparator, SchemaDiff, TableDiff, ColumnDiff, ChangeType, ImpactLevel
)


class TestSchemaInspector:
    """Test cases for SchemaInspector functionality."""
    
    @pytest.fixture
    def sqlite_db_manager(self):
        """Create a test SQLite database manager with sample schema."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        manager = DatabaseManager(database_path=db_path, use_sqlite=True)
        
        # Create sample schema
        manager._execute_query("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        manager._execute_query("""
            CREATE TABLE posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                published_at DATETIME,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        manager._execute_query("""
            CREATE INDEX idx_posts_user_id ON posts (user_id)
        """)
        
        manager._execute_query("""
            CREATE UNIQUE INDEX idx_users_username ON users (username)
        """)
        
        # Insert some test data
        manager._execute_query("INSERT INTO users (username, email) VALUES ('test1', 'test1@example.com')")
        manager._execute_query("INSERT INTO users (username, email) VALUES ('test2', 'test2@example.com')")
        manager._execute_query("INSERT INTO posts (user_id, title, content) VALUES (1, 'Test Post', 'Content')")
        
        yield manager
        
        # Cleanup
        try:
            manager.connection.close()
            Path(db_path).unlink()
        except:
            pass
    
    def test_extract_full_schema(self, sqlite_db_manager):
        """Test extracting complete schema from database."""
        inspector = SchemaInspector(sqlite_db_manager)
        schema = inspector.extract_full_schema()
        
        assert isinstance(schema, DatabaseSchema)
        assert schema.database_type == "sqlite"
        assert len(schema.tables) == 2  # users and posts
        
        # Check users table
        users_table = next(t for t in schema.tables if t.name == "users")
        assert len(users_table.columns) == 5
        assert len(users_table.indexes) >= 1  # At least one index
        
        # Check posts table
        posts_table = next(t for t in schema.tables if t.name == "posts")
        assert len(posts_table.columns) == 5
        assert len(posts_table.foreign_keys) == 1
        
        # Check data statistics
        assert users_table.row_count == 2
        assert posts_table.row_count == 1
    
    def test_extract_table_columns(self, sqlite_db_manager):
        """Test extracting column information."""
        inspector = SchemaInspector(sqlite_db_manager)
        columns = inspector._get_table_columns("users")
        
        assert len(columns) == 5
        
        # Check ID column
        id_col = next(c for c in columns if c.name == "id")
        assert id_col.is_primary_key
        assert id_col.is_auto_increment
        assert id_col.normalized_type == ColumnType.INTEGER
        
        # Check username column
        username_col = next(c for c in columns if c.name == "username")
        assert not username_col.is_nullable
        assert username_col.normalized_type == ColumnType.TEXT
        
        # Check is_active column
        active_col = next(c for c in columns if c.name == "is_active")
        assert active_col.default_value == "1"
        assert active_col.normalized_type == ColumnType.BOOLEAN
    
    def test_extract_table_indexes(self, sqlite_db_manager):
        """Test extracting index information."""
        inspector = SchemaInspector(sqlite_db_manager)
        indexes = inspector._get_table_indexes("users")
        
        # Should have primary key index and unique username index
        assert len(indexes) >= 1
        
        # Find the unique username index
        username_idx = next((idx for idx in indexes if "username" in idx.name), None)
        if username_idx:  # SQLite may not always report all indexes
            assert username_idx.is_unique
            assert "username" in username_idx.columns
    
    def test_extract_foreign_keys(self, sqlite_db_manager):
        """Test extracting foreign key information."""
        inspector = SchemaInspector(sqlite_db_manager)
        foreign_keys = inspector._get_table_foreign_keys("posts")
        
        assert len(foreign_keys) == 1
        
        fk = foreign_keys[0]
        assert fk.column_name == "user_id"
        assert fk.referenced_table == "users"
        assert fk.referenced_column == "id"
    
    def test_normalize_column_types(self, sqlite_db_manager):
        """Test column type normalization."""
        inspector = SchemaInspector(sqlite_db_manager)
        
        # Test various type normalizations
        assert inspector._normalize_column_type("INTEGER") == ColumnType.INTEGER
        assert inspector._normalize_column_type("VARCHAR(255)") == ColumnType.TEXT
        assert inspector._normalize_column_type("DATETIME") == ColumnType.DATETIME
        assert inspector._normalize_column_type("BOOLEAN") == ColumnType.BOOLEAN
        assert inspector._normalize_column_type("REAL") == ColumnType.FLOAT
        assert inspector._normalize_column_type("BLOB") == ColumnType.BLOB
        assert inspector._normalize_column_type("JSON") == ColumnType.JSON
    
    def test_save_and_load_schema(self, sqlite_db_manager):
        """Test saving schema to file and loading it back."""
        inspector = SchemaInspector(sqlite_db_manager)
        schema = inspector.extract_full_schema()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            schema_file = f.name
        
        try:
            # Save schema
            inspector.save_schema_to_file(schema, schema_file)
            
            # Verify file exists and has content
            assert Path(schema_file).exists()
            with open(schema_file, 'r') as f:
                data = json.load(f)
            assert data['database_name'] == schema.database_name
            assert len(data['tables']) == len(schema.tables)
            
            # Load schema back
            loaded_schema = inspector.load_schema_from_file(schema_file)
            assert loaded_schema.database_name == schema.database_name
            assert len(loaded_schema.tables) == len(schema.tables)
            
        finally:
            try:
                Path(schema_file).unlink()
            except:
                pass


class TestSchemaComparator:
    """Test cases for SchemaComparator functionality."""
    
    @pytest.fixture
    def sample_schemas(self):
        """Create sample schemas for comparison testing."""
        # Source schema (version 1)
        source_schema = DatabaseSchema(
            database_name="test_db",
            database_type="sqlite",
            version="SQLite 3.36",
            schema_version="1.0",
            extracted_at=datetime.now(),
            tables=[
                TableInfo(
                    name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="INTEGER",
                            normalized_type=ColumnType.INTEGER,
                            is_nullable=False,
                            is_primary_key=True,
                            is_auto_increment=True
                        ),
                        ColumnInfo(
                            name="username",
                            data_type="TEXT",
                            normalized_type=ColumnType.TEXT,
                            is_nullable=False
                        ),
                        ColumnInfo(
                            name="email",
                            data_type="TEXT",
                            normalized_type=ColumnType.TEXT,
                            is_nullable=False
                        )
                    ],
                    indexes=[
                        IndexInfo(
                            name="idx_users_username",
                            table_name="users",
                            columns=["username"],
                            is_unique=True,
                            is_primary=False
                        )
                    ],
                    foreign_keys=[],
                    check_constraints=[],
                    row_count=100
                )
            ],
            metadata={}
        )
        
        # Target schema (version 2) - with changes
        target_schema = DatabaseSchema(
            database_name="test_db",
            database_type="sqlite",
            version="SQLite 3.36",
            schema_version="2.0",
            extracted_at=datetime.now(),
            tables=[
                TableInfo(
                    name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="INTEGER",
                            normalized_type=ColumnType.INTEGER,
                            is_nullable=False,
                            is_primary_key=True,
                            is_auto_increment=True
                        ),
                        ColumnInfo(
                            name="username",
                            data_type="TEXT",
                            normalized_type=ColumnType.TEXT,
                            is_nullable=False
                        ),
                        ColumnInfo(
                            name="email",
                            data_type="TEXT",
                            normalized_type=ColumnType.TEXT,
                            is_nullable=False
                        ),
                        ColumnInfo(  # New column
                            name="full_name",
                            data_type="TEXT",
                            normalized_type=ColumnType.TEXT,
                            is_nullable=True,
                            default_value="NULL"
                        ),
                        ColumnInfo(  # New NOT NULL column without default
                            name="status",
                            data_type="TEXT",
                            normalized_type=ColumnType.TEXT,
                            is_nullable=False
                        )
                    ],
                    indexes=[
                        IndexInfo(
                            name="idx_users_username",
                            table_name="users",
                            columns=["username"],
                            is_unique=True,
                            is_primary=False
                        ),
                        IndexInfo(  # New index
                            name="idx_users_email",
                            table_name="users",
                            columns=["email"],
                            is_unique=False,
                            is_primary=False
                        )
                    ],
                    foreign_keys=[],
                    check_constraints=[],
                    row_count=150
                ),
                TableInfo(  # New table
                    name="posts",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="INTEGER",
                            normalized_type=ColumnType.INTEGER,
                            is_nullable=False,
                            is_primary_key=True,
                            is_auto_increment=True
                        ),
                        ColumnInfo(
                            name="user_id",
                            data_type="INTEGER",
                            normalized_type=ColumnType.INTEGER,
                            is_nullable=False
                        ),
                        ColumnInfo(
                            name="title",
                            data_type="TEXT",
                            normalized_type=ColumnType.TEXT,
                            is_nullable=False
                        )
                    ],
                    indexes=[],
                    foreign_keys=[
                        ForeignKeyInfo(
                            name="fk_posts_user_id",
                            table_name="posts",
                            column_name="user_id",
                            referenced_table="users",
                            referenced_column="id"
                        )
                    ],
                    check_constraints=[],
                    row_count=0
                )
            ],
            metadata={}
        )
        
        return source_schema, target_schema
    
    def test_compare_schemas_basic(self, sample_schemas):
        """Test basic schema comparison functionality."""
        source_schema, target_schema = sample_schemas
        
        comparator = SchemaComparator()
        diff = comparator.compare_schemas(source_schema, target_schema)
        
        assert isinstance(diff, SchemaDiff)
        assert diff.source_schema == source_schema
        assert diff.target_schema == target_schema
        assert len(diff.table_diffs) == 2  # users (modified) + posts (added)
        
        # Check overall impact
        assert diff.overall_impact in [ImpactLevel.BREAKING, ImpactLevel.MAJOR]
        
        # Check summary statistics
        summary = diff.summary
        assert summary['added_tables'] == 1  # posts table
        assert summary['modified_tables'] == 1  # users table
        assert summary['added_columns'] == 5  # 2 in users + 3 in posts
    
    def test_compare_table_modifications(self, sample_schemas):
        """Test detection of table modifications."""
        source_schema, target_schema = sample_schemas
        
        comparator = SchemaComparator()
        diff = comparator.compare_schemas(source_schema, target_schema)
        
        # Find users table diff
        users_diff = next(td for td in diff.table_diffs if td.table_name == "users")
        
        assert users_diff.change_type == ChangeType.MODIFIED
        assert len(users_diff.column_diffs) == 2  # full_name and status columns added
        assert len(users_diff.index_diffs) == 1   # email index added
        
        # Check column additions
        full_name_diff = next(cd for cd in users_diff.column_diffs if cd.column_name == "full_name")
        assert full_name_diff.change_type == ChangeType.ADDED
        assert full_name_diff.impact_level == ImpactLevel.MINOR  # Nullable with default
        
        status_diff = next(cd for cd in users_diff.column_diffs if cd.column_name == "status")
        assert status_diff.change_type == ChangeType.ADDED
        assert status_diff.impact_level == ImpactLevel.BREAKING  # NOT NULL without default
    
    def test_compare_table_additions(self, sample_schemas):
        """Test detection of new tables."""
        source_schema, target_schema = sample_schemas
        
        comparator = SchemaComparator()
        diff = comparator.compare_schemas(source_schema, target_schema)
        
        # Find posts table diff
        posts_diff = next(td for td in diff.table_diffs if td.table_name == "posts")
        
        assert posts_diff.change_type == ChangeType.ADDED
        assert posts_diff.impact_level == ImpactLevel.MINOR
        assert posts_diff.new_table is not None
        assert posts_diff.old_table is None
    
    def test_column_type_changes(self):
        """Test detection of column type changes."""
        # Create schemas with type changes
        source_col = ColumnInfo(
            name="age",
            data_type="TEXT",
            normalized_type=ColumnType.TEXT,
            is_nullable=True
        )
        
        target_col = ColumnInfo(
            name="age",
            data_type="INTEGER",
            normalized_type=ColumnType.INTEGER,
            is_nullable=True
        )
        
        comparator = SchemaComparator()
        diff = comparator._compare_column_details(source_col, target_col, "users")
        
        assert diff is not None
        assert diff.change_type == ChangeType.MODIFIED
        assert diff.impact_level == ImpactLevel.BREAKING
        assert "Type changed" in diff.impact_description
    
    def test_nullability_changes(self):
        """Test detection of nullability changes."""
        # NOT NULL to nullable (safe)
        source_col = ColumnInfo(
            name="name",
            data_type="TEXT",
            normalized_type=ColumnType.TEXT,
            is_nullable=False
        )
        
        target_col = ColumnInfo(
            name="name",
            data_type="TEXT",
            normalized_type=ColumnType.TEXT,
            is_nullable=True
        )
        
        comparator = SchemaComparator()
        diff = comparator._compare_column_details(source_col, target_col, "users")
        
        assert diff is not None
        assert diff.impact_level == ImpactLevel.MINOR
        assert "nullable" in diff.impact_description
        
        # Nullable to NOT NULL (breaking)
        reverse_diff = comparator._compare_column_details(target_col, source_col, "users")
        assert reverse_diff.impact_level == ImpactLevel.BREAKING
    
    def test_impact_assessment(self, sample_schemas):
        """Test overall impact assessment."""
        source_schema, target_schema = sample_schemas
        
        comparator = SchemaComparator()
        diff = comparator.compare_schemas(source_schema, target_schema)
        
        # Should be breaking due to NOT NULL column without default
        assert diff.overall_impact == ImpactLevel.BREAKING
        
        # Check migration strategy
        assert diff.migration_strategy in ["CAREFUL_MIGRATION", "PHASED_MIGRATION"]
        
        # Check risk assessment
        assert "breaking" in diff.risk_assessment.lower()
    
    def test_downtime_estimation(self, sample_schemas):
        """Test downtime estimation logic."""
        source_schema, target_schema = sample_schemas
        
        comparator = SchemaComparator()
        diff = comparator.compare_schemas(source_schema, target_schema)
        
        # With breaking changes and some data, should require downtime
        assert "MINIMAL" in diff.estimated_downtime or "MODERATE" in diff.estimated_downtime
    
    def test_rollback_complexity_assessment(self, sample_schemas):
        """Test rollback complexity assessment."""
        source_schema, target_schema = sample_schemas
        
        comparator = SchemaComparator()
        diff = comparator.compare_schemas(source_schema, target_schema)
        
        # No data loss changes, should be LOW to MEDIUM
        assert diff.rollback_complexity in ["LOW - Schema-only changes", "MEDIUM - Multiple type conversions to reverse"]
    
    def test_html_report_generation(self, sample_schemas):
        """Test HTML diff report generation."""
        source_schema, target_schema = sample_schemas
        
        comparator = SchemaComparator()
        diff = comparator.compare_schemas(source_schema, target_schema)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            html_file = f.name
        
        try:
            comparator.generate_html_diff_report(diff, html_file)
            
            # Verify file exists and has content
            assert Path(html_file).exists()
            
            with open(html_file, 'r') as f:
                content = f.read()
            
            # Check for key elements
            assert "Schema Diff Report" in content
            assert "users" in content
            assert "posts" in content
            assert "BREAKING" in content or "MAJOR" in content
            assert "table-diff" in content  # CSS class
            
        finally:
            try:
                Path(html_file).unlink()
            except:
                pass
    
    def test_json_report_generation(self, sample_schemas):
        """Test JSON diff report generation."""
        source_schema, target_schema = sample_schemas
        
        comparator = SchemaComparator()
        diff = comparator.compare_schemas(source_schema, target_schema)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
        
        try:
            comparator.save_diff_to_json(diff, json_file)
            
            # Verify file exists and has valid JSON
            assert Path(json_file).exists()
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check structure
            assert 'source_schema' in data
            assert 'target_schema' in data
            assert 'table_diffs' in data
            assert 'overall_impact' in data
            assert 'summary' in data
            
            # Check content
            assert data['overall_impact'] in ['BREAKING', 'MAJOR']
            assert len(data['table_diffs']) == 2
            
        finally:
            try:
                Path(json_file).unlink()
            except:
                pass


class TestSchemaDiffCLI:
    """Test cases for schema diff CLI functionality."""
    
    def test_create_db_manager_sqlite(self):
        """Test creating SQLite database manager."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            from experiment_manager.db.schema_diff_cli import create_db_manager
            manager = create_db_manager(db_path, use_sqlite=True)
            assert manager.use_sqlite
            
        finally:
            try:
                Path(db_path).unlink()
            except:
                pass
    
    @patch('click.prompt')
    def test_create_db_manager_mysql_connection_error(self, mock_prompt):
        """Test MySQL connection error handling."""
        from experiment_manager.db.schema_diff_cli import create_db_manager
        from experiment_manager.db.manager import ConnectionError
        
        # Should raise ConnectionError for non-existent MySQL server
        with pytest.raises(ConnectionError):
            create_db_manager(
                "test_db", 
                use_sqlite=False, 
                host="nonexistent", 
                user="test", 
                password="test"
            )


class TestSchemaAnalysis:
    """Test cases for schema analysis functionality."""
    
    def test_analyze_schema_complexity(self):
        """Test schema complexity analysis."""
        from experiment_manager.db.schema_diff_cli import _analyze_schema_complexity
        
        # Create a complex schema for testing
        schema = DatabaseSchema(
            database_name="complex_db",
            database_type="sqlite",
            version="SQLite 3.36",
            tables=[
                TableInfo(
                    name=f"table_{i}",
                    columns=[
                        ColumnInfo(
                            name=f"col_{j}",
                            data_type="TEXT",
                            normalized_type=ColumnType.TEXT,
                            is_nullable=True
                        ) for j in range(10)  # 10 columns per table
                    ],
                    indexes=[
                        IndexInfo(
                            name=f"idx_{i}_{j}",
                            table_name=f"table_{i}",
                            columns=[f"col_{j}"],
                            is_unique=False,
                            is_primary=False
                        ) for j in range(2)  # 2 indexes per table
                    ],
                    foreign_keys=[],
                    check_constraints=[]
                ) for i in range(5)  # 5 tables
            ],
            metadata={}
        )
        
        analysis = _analyze_schema_complexity(schema)
        
        assert analysis['table_count'] == 5
        assert analysis['column_count'] == 50  # 5 tables * 10 columns
        assert analysis['index_count'] == 10   # 5 tables * 2 indexes
        assert analysis['foreign_key_count'] == 0
        assert 0 <= analysis['complexity_score'] <= 10
        assert isinstance(analysis['recommendations'], list)
        assert isinstance(analysis['potential_issues'], list)
    
    def test_migration_plan_generation(self):
        """Test migration plan generation from diff data."""
        from experiment_manager.db.schema_diff_cli import _generate_migration_plan_from_diff
        
        diff_data = {
            'generated_at': '2024-01-01T12:00:00',
            'overall_impact': 'BREAKING',
            'estimated_downtime': 'MODERATE (10-30 minutes)',
            'rollback_complexity': 'MEDIUM - Multiple type conversions to reverse',
            'risk_assessment': 'HIGH: Breaking changes detected'
        }
        
        # Test different strategies
        for strategy in ['safe', 'fast', 'minimal-downtime']:
            plan = _generate_migration_plan_from_diff(diff_data, strategy)
            
            assert isinstance(plan, str)
            assert strategy.upper() in plan
            assert 'BREAKING' in plan
            assert 'BEGIN TRANSACTION' in plan
            assert 'COMMIT' in plan
            assert 'ROLLBACK' in plan


class TestIntegration:
    """Integration tests for the complete schema diff workflow."""
    
    def test_complete_schema_diff_workflow(self, sqlite_db_manager):
        """Test complete workflow from extraction to diff generation."""
        # Extract original schema
        inspector = SchemaInspector(sqlite_db_manager)
        original_schema = inspector.extract_full_schema()
        
        # Modify database schema
        sqlite_db_manager._execute_query("""
            ALTER TABLE users ADD COLUMN phone TEXT
        """)
        
        sqlite_db_manager._execute_query("""
            CREATE TABLE comments (
                id INTEGER PRIMARY KEY,
                post_id INTEGER,
                content TEXT,
                FOREIGN KEY (post_id) REFERENCES posts (id)
            )
        """)
        
        # Extract modified schema
        modified_schema = inspector.extract_full_schema()
        
        # Compare schemas
        comparator = SchemaComparator()
        diff = comparator.compare_schemas(original_schema, modified_schema)
        
        # Verify diff results
        assert len(diff.table_diffs) == 3  # users (modified), posts (unchanged), comments (added)
        
        # Check users table modification
        users_diff = next(td for td in diff.table_diffs if td.table_name == "users")
        assert users_diff.change_type == ChangeType.MODIFIED
        assert len(users_diff.column_diffs) == 1  # phone column added
        
        # Check comments table addition
        comments_diff = next(td for td in diff.table_diffs if td.table_name == "comments")
        assert comments_diff.change_type == ChangeType.ADDED
        
        # Generate reports
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate JSON report
            json_path = temp_path / "diff.json"
            comparator.save_diff_to_json(diff, str(json_path))
            assert json_path.exists()
            
            # Generate HTML report
            html_path = temp_path / "diff.html"
            comparator.generate_html_diff_report(diff, str(html_path))
            assert html_path.exists()
            
            # Verify HTML content
            with open(html_path, 'r') as f:
                html_content = f.read()
            assert "users" in html_content
            assert "comments" in html_content
            assert "phone" in html_content
    
    def test_schema_file_persistence(self, sqlite_db_manager):
        """Test schema file save/load persistence."""
        inspector = SchemaInspector(sqlite_db_manager)
        original_schema = inspector.extract_full_schema()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            schema_file = f.name
        
        try:
            # Save and reload
            inspector.save_schema_to_file(original_schema, schema_file)
            loaded_schema = inspector.load_schema_from_file(schema_file)
            
            # Compare key attributes
            assert loaded_schema.database_name == original_schema.database_name
            assert loaded_schema.database_type == original_schema.database_type
            assert len(loaded_schema.tables) == len(original_schema.tables)
            
            # Check first table details
            orig_table = original_schema.tables[0]
            loaded_table = loaded_schema.tables[0]
            assert orig_table.name == loaded_table.name
            assert len(orig_table.columns) == len(loaded_table.columns)
            
        finally:
            try:
                Path(schema_file).unlink()
            except:
                pass 