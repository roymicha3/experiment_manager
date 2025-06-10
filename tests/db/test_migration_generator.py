"""Tests for migration generation tools."""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from experiment_manager.db.migration_generator import (
    MigrationGenerator, DatabaseType, MigrationType, MigrationScript,
    MigrationOperation, MigrationError, MigrationTemplates
)
from experiment_manager.db.migration_manager import MigrationManager
from experiment_manager.db.manager import DatabaseManager

@pytest.fixture
def temp_migration_dir():
    """Create a temporary directory for migration files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sqlite_generator(temp_migration_dir):
    """Create a SQLite migration generator."""
    return MigrationGenerator(temp_migration_dir, DatabaseType.SQLITE)

@pytest.fixture
def mysql_generator(temp_migration_dir):
    """Create a MySQL migration generator."""
    return MigrationGenerator(temp_migration_dir, DatabaseType.MYSQL)

@pytest.fixture
def db_manager(tmp_path):
    """Create a test database manager."""
    db_path = tmp_path / "test_migrations.db"
    return DatabaseManager(database_path=str(db_path), use_sqlite=True, recreate=True)

@pytest.fixture
def migration_manager(db_manager, temp_migration_dir):
    """Create a migration manager."""
    return MigrationManager(db_manager, temp_migration_dir)

class TestMigrationGenerator:
    """Test the MigrationGenerator class."""
    
    def test_initialization_sqlite(self, temp_migration_dir):
        """Test SQLite generator initialization."""
        generator = MigrationGenerator(temp_migration_dir, DatabaseType.SQLITE)
        assert generator.database_type == DatabaseType.SQLITE
        assert generator.migration_dir == temp_migration_dir
        assert generator.templates == MigrationTemplates.SQLITE_TEMPLATES
        assert temp_migration_dir.exists()
    
    def test_initialization_mysql(self, temp_migration_dir):
        """Test MySQL generator initialization."""
        generator = MigrationGenerator(temp_migration_dir, DatabaseType.MYSQL)
        assert generator.database_type == DatabaseType.MYSQL
        assert generator.templates == MigrationTemplates.MYSQL_TEMPLATES
    
    def test_generate_migration_name(self, sqlite_generator):
        """Test migration name generation."""
        name = sqlite_generator.generate_migration_name("Create user table")
        assert "create_user_table" in name
        assert len(name.split('_')[0]) == 8  # Date part YYYYMMDD
        assert len(name.split('_')[1]) == 6  # Time part HHMMSS
        
        # Test special characters
        name = sqlite_generator.generate_migration_name("Add user's email column!")
        assert "add_users_email_column" in name
    
    def test_validate_sql_syntax(self, sqlite_generator):
        """Test SQL syntax validation."""
        # Valid SQL
        valid, error = sqlite_generator.validate_sql_syntax("CREATE TABLE test (id INT);")
        assert valid is True
        assert error is None
        
        # Empty SQL
        valid, error = sqlite_generator.validate_sql_syntax("")
        assert valid is False
        assert "Empty SQL statement" in error
        
        # Invalid start
        valid, error = sqlite_generator.validate_sql_syntax("INVALID TABLE test;")
        assert valid is False
        assert "valid keyword" in error
        
        # Unbalanced parentheses
        valid, error = sqlite_generator.validate_sql_syntax("CREATE TABLE test (id INT;")
        assert valid is False
        assert "Unbalanced parentheses" in error
        
        # No semicolon
        valid, error = sqlite_generator.validate_sql_syntax("CREATE TABLE test (id INT)")
        assert valid is False
        assert "semicolon" in error
    
    def test_create_table_migration_sqlite(self, sqlite_generator):
        """Test CREATE TABLE migration for SQLite."""
        columns = [
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "name", "type": "TEXT", "not_null": True},
            {"name": "email", "type": "TEXT", "unique": True},
            {"name": "created_at", "type": "TEXT", "default": "CURRENT_TIMESTAMP"}
        ]
        constraints = ["UNIQUE(name, email)"]
        
        migration = sqlite_generator.create_table_migration(
            "users", columns, constraints, "Create users table"
        )
        
        assert isinstance(migration, MigrationScript)
        assert migration.description == "Create users table"
        assert "CREATE TABLE IF NOT EXISTS users" in migration.up_script
        assert "INTEGER PRIMARY KEY AUTOINCREMENT" in migration.up_script
        assert "TEXT NOT NULL" in migration.up_script
        assert "TEXT UNIQUE" in migration.up_script
        assert "DEFAULT CURRENT_TIMESTAMP" in migration.up_script
        assert "DROP TABLE IF EXISTS users" in migration.down_script
    
    def test_create_table_migration_mysql(self, mysql_generator):
        """Test CREATE TABLE migration for MySQL."""
        columns = [
            {"name": "id", "type": "INT", "primary_key": True},
            {"name": "name", "type": "VARCHAR(255)", "not_null": True}
        ]
        
        migration = mysql_generator.create_table_migration("users", columns)
        
        assert "INT PRIMARY KEY AUTO_INCREMENT" in migration.up_script
        assert "VARCHAR(255) NOT NULL" in migration.up_script
    
    def test_add_column_migration_sqlite(self, sqlite_generator):
        """Test ADD COLUMN migration for SQLite."""
        migration = sqlite_generator.add_column_migration(
            "users", "age", "INTEGER", not_null=True, default="0"
        )
        
        assert migration.description == "Add age column to users"
        assert "ALTER TABLE users ADD COLUMN age INTEGER NOT NULL DEFAULT 0" in migration.up_script
        assert "SQLite doesn't support DROP COLUMN easily" in migration.down_script
    
    def test_add_column_migration_mysql(self, mysql_generator):
        """Test ADD COLUMN migration for MySQL."""
        migration = mysql_generator.add_column_migration(
            "users", "age", "INT", not_null=True, default="0"
        )
        
        assert "ALTER TABLE users ADD COLUMN age INT NOT NULL DEFAULT 0" in migration.up_script
        assert "ALTER TABLE users DROP COLUMN age" in migration.down_script
    
    def test_add_index_migration(self, sqlite_generator):
        """Test ADD INDEX migration."""
        migration = sqlite_generator.add_index_migration(
            "users", ["email", "name"], unique=True
        )
        
        assert "index" in migration.description.lower()
        assert "unique" in migration.description.lower()
        assert "CREATE UNIQUE INDEX" in migration.up_script
        assert "users" in migration.up_script
        assert "email, name" in migration.up_script
        assert "DROP INDEX" in migration.down_script
    
    def test_custom_migration(self, sqlite_generator):
        """Test custom migration creation."""
        up_sql = "ALTER TABLE users ADD COLUMN status VARCHAR(20) DEFAULT 'active';"
        down_sql = "ALTER TABLE users DROP COLUMN status;"
        
        migration = sqlite_generator.custom_migration(
            up_sql, down_sql, "Add user status column"
        )
        
        assert migration.description == "Add user status column"
        assert migration.up_script == up_sql.strip()
        assert migration.down_script == down_sql.strip()
        assert migration.operations[0].operation_type == MigrationType.CUSTOM
    
    def test_custom_migration_invalid_sql(self, sqlite_generator):
        """Test custom migration with invalid SQL."""
        with pytest.raises(MigrationError, match="Invalid up SQL"):
            sqlite_generator.custom_migration(
                "INVALID SQL",
                "DROP TABLE test;",
                "Invalid migration"
            )
    
    def test_save_migration(self, sqlite_generator, temp_migration_dir):
        """Test saving migration to file."""
        migration = sqlite_generator.custom_migration(
            "CREATE TABLE test (id INT);",
            "DROP TABLE test;",
            "Test migration"
        )
        
        filepath = sqlite_generator.save_migration(migration, "1.0.0")
        
        assert filepath.exists()
        assert filepath.suffix == ".sql"
        
        # Check file content
        content = filepath.read_text()
        assert "-- Migration:" in content
        assert "-- Version: 1.0.0" in content
        assert "-- Description: Test migration" in content
        assert "-- Database: sqlite" in content
        assert "-- UP" in content
        assert "CREATE TABLE test (id INT);" in content
        assert "-- DOWN" in content
        assert "DROP TABLE test;" in content
    
    def test_save_migration_invalid_version(self, sqlite_generator):
        """Test saving migration with invalid version."""
        migration = sqlite_generator.custom_migration(
            "CREATE TABLE test (id INT);",
            "DROP TABLE test;",
            "Test migration"
        )
        
        with pytest.raises(MigrationError, match="Invalid version format"):
            sqlite_generator.save_migration(migration, "invalid.version")
    
    def test_list_migrations(self, sqlite_generator, temp_migration_dir):
        """Test listing migration files."""
        # Create some migration files
        migration1 = sqlite_generator.custom_migration(
            "CREATE TABLE test1 (id INT);", "DROP TABLE test1;", "First migration"
        )
        migration2 = sqlite_generator.custom_migration(
            "CREATE TABLE test2 (id INT);", "DROP TABLE test2;", "Second migration"
        )
        
        sqlite_generator.save_migration(migration1, "1.0.0")
        sqlite_generator.save_migration(migration2, "1.1.0")
        
        migrations = sqlite_generator.list_migrations()
        
        assert len(migrations) == 2
        assert all(m['version'] in ['1.0.0', '1.1.0'] for m in migrations)
        assert all('filename' in m for m in migrations)
        assert all('filepath' in m for m in migrations)

class TestMigrationManager:
    """Test the MigrationManager class."""
    
    def test_initialization(self, migration_manager, db_manager, temp_migration_dir):
        """Test migration manager initialization."""
        assert migration_manager.db_manager == db_manager
        assert migration_manager.migration_dir == temp_migration_dir
        assert migration_manager.generator.database_type == DatabaseType.SQLITE
    
    def test_initialize_migrations(self, migration_manager):
        """Test migration system initialization."""
        migration_manager.initialize_migrations("1.0.0")
        
        current_version = migration_manager.db_manager.get_current_schema_version()
        assert current_version is not None
        assert current_version.version == "1.0.0"
        assert current_version.migration_name == "initial_schema"
    
    def test_initialize_migrations_already_initialized(self, migration_manager):
        """Test initializing when already initialized."""
        migration_manager.initialize_migrations("1.0.0")
        # Should not raise error on second initialization
        migration_manager.initialize_migrations("1.1.0")
        
        # Should still be at original version
        current_version = migration_manager.db_manager.get_current_schema_version()
        assert current_version.version == "1.0.0"
    
    def test_create_migration(self, migration_manager):
        """Test creating a new migration file."""
        migration_manager.initialize_migrations("1.0.0")
        
        filepath = migration_manager.create_migration("Add user table")
        
        assert Path(filepath).exists()
        content = Path(filepath).read_text()
        assert "-- Version: 1.1.0" in content
        assert "-- Description: Add user table" in content
        assert "-- Add your migration SQL here" in content
    
    def test_generate_table_migration(self, migration_manager):
        """Test generating a table migration."""
        migration_manager.initialize_migrations("1.0.0")
        
        columns = [
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "name", "type": "TEXT", "not_null": True}
        ]
        
        filepath = migration_manager.generate_table_migration(
            "users", columns, description="Create users table"
        )
        
        assert Path(filepath).exists()
        content = Path(filepath).read_text()
        assert "CREATE TABLE IF NOT EXISTS users" in content
        assert "INTEGER PRIMARY KEY AUTOINCREMENT" in content
        assert "DROP TABLE IF EXISTS users" in content
    
    def test_generate_column_migration(self, migration_manager):
        """Test generating a column migration."""
        migration_manager.initialize_migrations("1.0.0")
        
        filepath = migration_manager.generate_column_migration(
            "users", "email", "TEXT", unique=True
        )
        
        assert Path(filepath).exists()
        content = Path(filepath).read_text()
        assert "ALTER TABLE users ADD COLUMN email TEXT UNIQUE" in content
    
    def test_generate_index_migration(self, migration_manager):
        """Test generating an index migration."""
        migration_manager.initialize_migrations("1.0.0")
        
        filepath = migration_manager.generate_index_migration(
            "users", ["email"], unique=True
        )
        
        assert Path(filepath).exists()
        content = Path(filepath).read_text()
        assert "CREATE UNIQUE INDEX" in content
        assert "users" in content
        assert "email" in content
    
    def test_apply_migration(self, migration_manager):
        """Test applying a migration."""
        migration_manager.initialize_migrations("1.0.0")
        
        # Create a simple migration
        migration_file = migration_manager.generate_table_migration(
            "test_table",
            [{"name": "id", "type": "INTEGER", "primary_key": True}],
            description="Create test table"
        )
        
        # Apply the migration
        migration_manager.apply_migration(migration_file)
        
        # Check that table was created
        cursor = migration_manager.db_manager._execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
        )
        tables = cursor.fetchall()
        assert len(tables) == 1
        assert tables[0]['name'] == 'test_table'
        
        # Check that migration was recorded
        current_version = migration_manager.db_manager.get_current_schema_version()
        assert current_version.version == "1.1.0"
    
    def test_apply_migration_already_applied(self, migration_manager, capsys):
        """Test applying a migration that's already applied."""
        migration_manager.initialize_migrations("1.0.0")
        
        migration_file = migration_manager.generate_table_migration(
            "test_table",
            [{"name": "id", "type": "INTEGER", "primary_key": True}],
            description="Create test table"
        )
        
        # Apply twice
        migration_manager.apply_migration(migration_file)
        migration_manager.apply_migration(migration_file)
        
        captured = capsys.readouterr()
        assert "already applied" in captured.out
    
    def test_list_migrations(self, migration_manager):
        """Test listing migrations with status."""
        migration_manager.initialize_migrations("1.0.0")
        
        # Create and apply one migration
        file1 = migration_manager.generate_table_migration(
            "table1", [{"name": "id", "type": "INTEGER", "primary_key": True}]
        )
        migration_manager.apply_migration(file1)
        
        # Create another migration without applying
        migration_manager.generate_table_migration(
            "table2", [{"name": "id", "type": "INTEGER", "primary_key": True}]
        )
        
        migrations = migration_manager.list_migrations()
        
        assert len(migrations) == 2
        applied_count = sum(1 for m in migrations if m['applied'])
        pending_count = sum(1 for m in migrations if not m['applied'])
        assert applied_count == 1
        assert pending_count == 1
    
    def test_status(self, migration_manager):
        """Test getting migration system status."""
        migration_manager.initialize_migrations("1.0.0")
        
        # Create some migrations
        migration_manager.generate_table_migration(
            "table1", [{"name": "id", "type": "INTEGER", "primary_key": True}]
        )
        migration_manager.generate_table_migration(
            "table2", [{"name": "id", "type": "INTEGER", "primary_key": True}]
        )
        
        status = migration_manager.status()
        
        assert status['current_version'] == "1.0.0"
        assert status['total_applied'] == 1  # Just the initial schema
        assert status['total_pending'] == 2  # The two generated migrations
        assert status['migration_files'] == 2
        assert status['database_type'] == 'SQLite'
        assert 'migration_dir' in status
    
    def test_parse_migration_file(self, migration_manager):
        """Test parsing migration file."""
        # Create a migration file
        migration_file = migration_manager.create_migration("Test migration", "1.5.0")
        
        # Parse it
        parsed = migration_manager._parse_migration_file(Path(migration_file))
        
        assert parsed['version'] == "1.5.0"
        assert parsed['description'] == "Test migration"
        assert 'up_sql' in parsed
        assert 'down_sql' in parsed
        assert 'name' in parsed
    
    def test_split_sql_statements(self, migration_manager):
        """Test SQL statement splitting."""
        sql = """
        CREATE TABLE test1 (id INT);
        -- This is a comment
        CREATE TABLE test2 (name TEXT);
        
        INSERT INTO test1 VALUES (1);
        """
        
        statements = migration_manager._split_sql_statements(sql)
        
        assert len(statements) == 3
        assert "CREATE TABLE test1" in statements[0]
        assert "CREATE TABLE test2" in statements[1]
        assert "INSERT INTO test1" in statements[2]

class TestDatabaseTypeCompatibility:
    """Test database type specific behavior."""
    
    def test_sqlite_templates(self):
        """Test SQLite specific templates."""
        templates = MigrationTemplates.SQLITE_TEMPLATES
        
        # SQLite should have special handling for DROP COLUMN
        drop_column_template = templates[MigrationType.DROP_COLUMN]
        assert "PRAGMA foreign_keys = OFF" in drop_column_template
        assert "table_name}_new" in drop_column_template
        
        # SQLite should use INTEGER PRIMARY KEY AUTOINCREMENT
        create_table_template = templates[MigrationType.CREATE_TABLE]
        assert "CREATE TABLE IF NOT EXISTS" in create_table_template
    
    def test_mysql_templates(self):
        """Test MySQL specific templates."""
        templates = MigrationTemplates.MYSQL_TEMPLATES
        
        # MySQL should support direct DROP COLUMN
        drop_column_template = templates[MigrationType.DROP_COLUMN]
        assert "ALTER TABLE" in drop_column_template
        assert "DROP COLUMN" in drop_column_template
        assert "PRAGMA" not in drop_column_template  # No SQLite pragmas
    
    def test_generator_template_selection(self, temp_migration_dir):
        """Test that generators select correct templates."""
        sqlite_gen = MigrationGenerator(temp_migration_dir, DatabaseType.SQLITE)
        mysql_gen = MigrationGenerator(temp_migration_dir, DatabaseType.MYSQL)
        
        assert sqlite_gen.templates == MigrationTemplates.SQLITE_TEMPLATES
        assert mysql_gen.templates == MigrationTemplates.MYSQL_TEMPLATES
        
        # Test actual template usage
        sqlite_migration = sqlite_gen.add_column_migration("test", "col", "TEXT")
        mysql_migration = mysql_gen.add_column_migration("test", "col", "TEXT")
        
        # SQLite rollback should mention manual intervention
        assert "SQLite doesn't support DROP COLUMN easily" in sqlite_migration.down_script
        
        # MySQL rollback should be proper DROP COLUMN
        assert "ALTER TABLE test DROP COLUMN col" in mysql_migration.down_script 