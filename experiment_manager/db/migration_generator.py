"""Migration script generation tools for database schema evolution."""
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from experiment_manager.db.version_utils import (
    validate_version_string, get_next_version, VersionError
)

class DatabaseType(Enum):
    """Supported database types for migration generation."""
    SQLITE = "sqlite"
    MYSQL = "mysql"

class MigrationType(Enum):
    """Types of migration operations."""
    CREATE_TABLE = "create_table"
    DROP_TABLE = "drop_table"
    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    MODIFY_COLUMN = "modify_column"
    RENAME_COLUMN = "rename_column"
    ADD_INDEX = "add_index"
    DROP_INDEX = "drop_index"
    ADD_FOREIGN_KEY = "add_foreign_key"
    DROP_FOREIGN_KEY = "drop_foreign_key"
    CUSTOM = "custom"

@dataclass
class MigrationOperation:
    """Represents a single migration operation."""
    operation_type: MigrationType
    table_name: str
    details: Dict
    rollback_details: Optional[Dict] = None

@dataclass
class MigrationScript:
    """Represents a complete migration script."""
    version: str
    name: str
    description: str
    operations: List[MigrationOperation]
    up_script: str
    down_script: str
    created_at: datetime

class MigrationError(Exception):
    """Error in migration script generation."""
    pass

class MigrationTemplates:
    """Templates for generating SQL migration scripts."""
    
    MYSQL_TEMPLATES = {
        MigrationType.CREATE_TABLE: """
CREATE TABLE IF NOT EXISTS {table_name} (
{columns}
{constraints}
);""",
        
        MigrationType.DROP_TABLE: """
DROP TABLE IF EXISTS {table_name};""",
        
        MigrationType.ADD_COLUMN: """
ALTER TABLE {table_name} ADD COLUMN {column_definition};""",
        
        MigrationType.DROP_COLUMN: """
ALTER TABLE {table_name} DROP COLUMN {column_name};""",
        
        MigrationType.MODIFY_COLUMN: """
ALTER TABLE {table_name} MODIFY COLUMN {column_definition};""",
        
        MigrationType.RENAME_COLUMN: """
ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name};""",
        
        MigrationType.ADD_INDEX: """
CREATE INDEX {index_name} ON {table_name} ({columns});""",
        
        MigrationType.DROP_INDEX: """
DROP INDEX {index_name};""",
        
        MigrationType.ADD_FOREIGN_KEY: """
ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} 
FOREIGN KEY ({column_name}) REFERENCES {ref_table}({ref_column});""",
        
        MigrationType.DROP_FOREIGN_KEY: """
ALTER TABLE {table_name} DROP FOREIGN KEY {constraint_name};"""
    }
    
    SQLITE_TEMPLATES = {
        MigrationType.CREATE_TABLE: """
CREATE TABLE IF NOT EXISTS {table_name} (
{columns}
{constraints}
);""",
        
        MigrationType.DROP_TABLE: """
DROP TABLE IF EXISTS {table_name};""",
        
        MigrationType.ADD_COLUMN: """
ALTER TABLE {table_name} ADD COLUMN {column_definition};""",
        
        # SQLite doesn't support DROP COLUMN directly
        MigrationType.DROP_COLUMN: """
-- SQLite doesn't support DROP COLUMN directly
-- This requires creating a new table and copying data
-- See: https://www.sqlite.org/lang_altertable.html
PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

-- Create new table without the column
CREATE TABLE {table_name}_new (
{new_columns}
);

-- Copy data from old table (excluding dropped column)
INSERT INTO {table_name}_new ({copy_columns})
SELECT {copy_columns} FROM {table_name};

-- Drop old table
DROP TABLE {table_name};

-- Rename new table
ALTER TABLE {table_name}_new RENAME TO {table_name};

COMMIT;
PRAGMA foreign_keys = ON;""",
        
        # SQLite doesn't support MODIFY COLUMN directly
        MigrationType.MODIFY_COLUMN: """
-- SQLite doesn't support MODIFY COLUMN directly
-- This requires recreating the table with the new column definition
PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

-- Create new table with modified column
CREATE TABLE {table_name}_new (
{new_columns}
);

-- Copy data with type conversion if needed
INSERT INTO {table_name}_new ({columns})
SELECT {columns} FROM {table_name};

-- Drop old table
DROP TABLE {table_name};

-- Rename new table
ALTER TABLE {table_name}_new RENAME TO {table_name};

COMMIT;
PRAGMA foreign_keys = ON;""",
        
        MigrationType.RENAME_COLUMN: """
ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name};""",
        
        MigrationType.ADD_INDEX: """
CREATE INDEX {index_name} ON {table_name} ({columns});""",
        
        MigrationType.DROP_INDEX: """
DROP INDEX IF EXISTS {index_name};""",
        
        # SQLite handles foreign keys differently
        MigrationType.ADD_FOREIGN_KEY: """
-- SQLite foreign keys are defined during table creation
-- To add a foreign key, table must be recreated
PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

-- Create new table with foreign key
CREATE TABLE {table_name}_new (
{new_columns_with_fk}
);

-- Copy existing data
INSERT INTO {table_name}_new ({columns})
SELECT {columns} FROM {table_name};

-- Drop old table
DROP TABLE {table_name};

-- Rename new table
ALTER TABLE {table_name}_new RENAME TO {table_name};

COMMIT;
PRAGMA foreign_keys = ON;""",
        
        MigrationType.DROP_FOREIGN_KEY: """
-- SQLite foreign keys are defined during table creation
-- To remove a foreign key, table must be recreated
PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

-- Create new table without foreign key
CREATE TABLE {table_name}_new (
{new_columns_without_fk}
);

-- Copy existing data
INSERT INTO {table_name}_new ({columns})
SELECT {columns} FROM {table_name};

-- Drop old table
DROP TABLE {table_name};

-- Rename new table
ALTER TABLE {table_name}_new RENAME TO {table_name};

COMMIT;
PRAGMA foreign_keys = ON;"""
    }

class MigrationGenerator:
    """Generates database migration scripts for schema evolution."""
    
    def __init__(self, migration_dir: Union[str, Path] = "migrations", 
                 database_type: DatabaseType = DatabaseType.SQLITE):
        """Initialize migration generator.
        
        Args:
            migration_dir: Directory to store migration files
            database_type: Target database type for migrations
        """
        self.migration_dir = Path(migration_dir)
        self.database_type = database_type
        self.templates = (MigrationTemplates.SQLITE_TEMPLATES 
                         if database_type == DatabaseType.SQLITE 
                         else MigrationTemplates.MYSQL_TEMPLATES)
        
        # Ensure migration directory exists
        self.migration_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_migration_name(self, description: str) -> str:
        """Generate a timestamped migration name.
        
        Args:
            description: Human-readable description of the migration
            
        Returns:
            str: Formatted migration name with timestamp
        """
        # Clean description for filename
        clean_desc = re.sub(r'[^a-zA-Z0-9_\s]', '', description)
        clean_desc = re.sub(r'\s+', '_', clean_desc.strip()).lower()
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{timestamp}_{clean_desc}"
    
    def validate_sql_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Basic SQL syntax validation.
        
        Args:
            sql: SQL statement to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic syntax checks
        sql = sql.strip()
        
        if not sql:
            return False, "Empty SQL statement"
        
        # Skip validation for template/placeholder content
        if sql.startswith("-- Add your migration SQL here") or sql.startswith("-- Add rollback SQL here"):
            return True, None
        
        # Remove comments and empty lines for validation
        lines = []
        for line in sql.split('\n'):
            line = line.strip()
            if line and not line.startswith('--'):
                lines.append(line)
        
        if not lines:
            return True, None  # Only comments, that's valid
        
        # Check first non-comment line
        first_statement = lines[0]
        
        # Check for basic SQL keywords
        valid_start_keywords = [
            'CREATE', 'DROP', 'ALTER', 'INSERT', 'UPDATE', 'DELETE', 
            'SELECT', 'PRAGMA', 'BEGIN', 'COMMIT', 'ROLLBACK'
        ]
        
        first_word = first_statement.split()[0].upper()
        if first_word not in valid_start_keywords:
            return False, f"SQL statement should start with a valid keyword, got: {first_word}"
        
        # Check for balanced parentheses in the entire SQL
        paren_count = sql.count('(') - sql.count(')')
        if paren_count != 0:
            return False, "Unbalanced parentheses in SQL statement"
        
        # Check for ending semicolon (at least one statement should end with semicolon)
        has_semicolon = any(line.rstrip().endswith(';') for line in lines)
        if not has_semicolon:
            return False, "SQL should contain at least one statement ending with semicolon"
        
        return True, None
    
    def create_table_migration(self, table_name: str, columns: List[Dict], 
                             constraints: List[str] = None, 
                             description: str = None) -> MigrationScript:
        """Generate a CREATE TABLE migration.
        
        Args:
            table_name: Name of the table to create
            columns: List of column definitions
            constraints: List of table constraints
            description: Description of the migration
            
        Returns:
            MigrationScript: Generated migration script
        """
        if not description:
            description = f"Create {table_name} table"
        
        # Format columns
        column_defs = []
        for col in columns:
            col_def = f"    {col['name']} {col['type']}"
            if col.get('primary_key'):
                if self.database_type == DatabaseType.MYSQL:
                    col_def += " PRIMARY KEY AUTO_INCREMENT"
                else:
                    col_def += " PRIMARY KEY AUTOINCREMENT"
            elif col.get('not_null'):
                col_def += " NOT NULL"
            if col.get('unique'):
                col_def += " UNIQUE"
            if col.get('default'):
                col_def += f" DEFAULT {col['default']}"
            column_defs.append(col_def)
        
        # Format constraints
        constraint_defs = []
        if constraints:
            for constraint in constraints:
                constraint_defs.append(f"    {constraint}")
        
        # Combine columns and constraints
        all_defs = column_defs
        if constraint_defs:
            all_defs.extend(constraint_defs)
        
        columns_str = ",\n".join(all_defs)
        
        # Generate up script
        up_script = self.templates[MigrationType.CREATE_TABLE].format(
            table_name=table_name,
            columns=columns_str,
            constraints=""
        ).strip()
        
        # Generate down script
        down_script = self.templates[MigrationType.DROP_TABLE].format(
            table_name=table_name
        ).strip()
        
        operation = MigrationOperation(
            operation_type=MigrationType.CREATE_TABLE,
            table_name=table_name,
            details={"columns": columns, "constraints": constraints or []}
        )
        
        return MigrationScript(
            version="",  # Will be set when script is saved
            name=self.generate_migration_name(description),
            description=description,
            operations=[operation],
            up_script=up_script,
            down_script=down_script,
            created_at=datetime.now()
        )
    
    def add_column_migration(self, table_name: str, column_name: str, 
                           column_type: str, **options) -> MigrationScript:
        """Generate an ADD COLUMN migration.
        
        Args:
            table_name: Name of the table
            column_name: Name of the new column
            column_type: Data type of the new column
            **options: Additional column options (not_null, default, etc.)
            
        Returns:
            MigrationScript: Generated migration script
        """
        description = f"Add {column_name} column to {table_name}"
        
        # Build column definition
        col_def = f"{column_name} {column_type}"
        if options.get('not_null'):
            col_def += " NOT NULL"
        if options.get('unique'):
            col_def += " UNIQUE"
        if options.get('default'):
            col_def += f" DEFAULT {options['default']}"
        
        # Generate up script
        up_script = self.templates[MigrationType.ADD_COLUMN].format(
            table_name=table_name,
            column_definition=col_def
        ).strip()
        
        # Generate down script (drop column)
        if self.database_type == DatabaseType.SQLITE:
            # SQLite requires special handling for dropping columns
            down_script = f"-- SQLite doesn't support DROP COLUMN easily\n-- Manual intervention required to drop {column_name} from {table_name}"
        else:
            down_script = self.templates[MigrationType.DROP_COLUMN].format(
                table_name=table_name,
                column_name=column_name
            ).strip()
        
        operation = MigrationOperation(
            operation_type=MigrationType.ADD_COLUMN,
            table_name=table_name,
            details={
                "column_name": column_name,
                "column_type": column_type,
                "options": options
            }
        )
        
        return MigrationScript(
            version="",
            name=self.generate_migration_name(description),
            description=description,
            operations=[operation],
            up_script=up_script,
            down_script=down_script,
            created_at=datetime.now()
        )
    
    def add_index_migration(self, table_name: str, columns: List[str], 
                          index_name: str = None, unique: bool = False) -> MigrationScript:
        """Generate an ADD INDEX migration.
        
        Args:
            table_name: Name of the table
            columns: List of column names for the index
            index_name: Name of the index (auto-generated if None)
            unique: Whether to create a unique index
            
        Returns:
            MigrationScript: Generated migration script
        """
        if not index_name:
            index_name = f"idx_{table_name}_{'_'.join(columns)}"
        
        description = f"Add {'unique ' if unique else ''}index {index_name} on {table_name}"
        
        # Generate up script
        index_type = "UNIQUE INDEX" if unique else "INDEX"
        if unique and self.database_type == DatabaseType.MYSQL:
            up_script = f"CREATE {index_type} {index_name} ON {table_name} ({', '.join(columns)});"
        else:
            up_script = self.templates[MigrationType.ADD_INDEX].format(
                index_name=index_name,
                table_name=table_name,
                columns=", ".join(columns)
            ).strip()
            if unique:
                up_script = up_script.replace("CREATE INDEX", "CREATE UNIQUE INDEX")
        
        # Generate down script
        down_script = self.templates[MigrationType.DROP_INDEX].format(
            index_name=index_name
        ).strip()
        
        operation = MigrationOperation(
            operation_type=MigrationType.ADD_INDEX,
            table_name=table_name,
            details={
                "index_name": index_name,
                "columns": columns,
                "unique": unique
            }
        )
        
        return MigrationScript(
            version="",
            name=self.generate_migration_name(description),
            description=description,
            operations=[operation],
            up_script=up_script,
            down_script=down_script,
            created_at=datetime.now()
        )
    
    def custom_migration(self, up_sql: str, down_sql: str, 
                        description: str) -> MigrationScript:
        """Generate a custom migration with provided SQL.
        
        Args:
            up_sql: SQL for the forward migration
            down_sql: SQL for the rollback migration
            description: Description of the migration
            
        Returns:
            MigrationScript: Generated migration script
            
        Raises:
            MigrationError: If SQL validation fails
        """
        # Validate SQL syntax
        up_valid, up_error = self.validate_sql_syntax(up_sql)
        if not up_valid:
            raise MigrationError(f"Invalid up SQL: {up_error}")
        
        down_valid, down_error = self.validate_sql_syntax(down_sql)
        if not down_valid:
            raise MigrationError(f"Invalid down SQL: {down_error}")
        
        operation = MigrationOperation(
            operation_type=MigrationType.CUSTOM,
            table_name="",
            details={"up_sql": up_sql, "down_sql": down_sql}
        )
        
        return MigrationScript(
            version="",
            name=self.generate_migration_name(description),
            description=description,
            operations=[operation],
            up_script=up_sql.strip(),
            down_script=down_sql.strip(),
            created_at=datetime.now()
        )
    
    def save_migration(self, migration: MigrationScript, 
                      version: str = None) -> Path:
        """Save migration script to file.
        
        Args:
            migration: Migration script to save
            version: Version number for the migration
            
        Returns:
            Path: Path to the saved migration file
            
        Raises:
            MigrationError: If version validation fails
        """
        if version:
            if not validate_version_string(version):
                raise MigrationError(f"Invalid version format: {version}")
            migration.version = version
        
        # Create migration file content
        content = f"""-- Migration: {migration.name}
-- Version: {migration.version}
-- Description: {migration.description}
-- Created: {migration.created_at.isoformat()}
-- Database: {self.database_type.value}

-- UP
{migration.up_script}

-- DOWN
{migration.down_script}
"""
        
        # Save to file
        filename = f"{migration.name}.sql"
        filepath = self.migration_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def list_migrations(self) -> List[Dict]:
        """List all migration files in the migration directory.
        
        Returns:
            List[Dict]: List of migration file information
        """
        migrations = []
        
        for file_path in self.migration_dir.glob("*.sql"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse metadata from file header
                lines = content.split('\n')
                metadata = {}
                
                for line in lines:
                    if line.startswith('-- Migration:'):
                        metadata['name'] = line.split(':', 1)[1].strip()
                    elif line.startswith('-- Version:'):
                        metadata['version'] = line.split(':', 1)[1].strip()
                    elif line.startswith('-- Description:'):
                        metadata['description'] = line.split(':', 1)[1].strip()
                    elif line.startswith('-- Created:'):
                        metadata['created'] = line.split(':', 1)[1].strip()
                    elif line.startswith('-- Database:'):
                        metadata['database'] = line.split(':', 1)[1].strip()
                
                metadata['filepath'] = str(file_path)
                metadata['filename'] = file_path.name
                migrations.append(metadata)
                
            except Exception as e:
                # Skip files that can't be parsed
                continue
        
        # Sort by creation timestamp in filename
        migrations.sort(key=lambda x: x.get('filename', ''))
        return migrations 