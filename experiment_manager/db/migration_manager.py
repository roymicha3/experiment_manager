"""Migration management for database schema evolution."""
import os
from pathlib import Path
from typing import List, Optional, Dict, Union
from datetime import datetime

from experiment_manager.db.manager import DatabaseManager, DatabaseError
from experiment_manager.db.migration_generator import (
    MigrationGenerator, MigrationScript, DatabaseType, MigrationError
)
from experiment_manager.db.version_utils import (
    get_next_version, compare_versions, is_compatible_upgrade, VersionError
)

class MigrationManager:
    """Manages database migrations and schema evolution."""
    
    def __init__(self, db_manager: DatabaseManager, 
                 migration_dir: Union[str, Path] = "migrations"):
        """Initialize migration manager.
        
        Args:
            db_manager: DatabaseManager instance
            migration_dir: Directory for migration files
        """
        self.db_manager = db_manager
        self.migration_dir = Path(migration_dir)
        
        # Determine database type
        db_type = DatabaseType.SQLITE if db_manager.use_sqlite else DatabaseType.MYSQL
        self.generator = MigrationGenerator(migration_dir, db_type)
        
        # Ensure migration directory exists
        self.migration_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize_migrations(self, initial_version: str = "1.0.0") -> None:
        """Initialize migration system for an existing database.
        
        Args:
            initial_version: Initial schema version to record
            
        Raises:
            MigrationError: If initialization fails
        """
        try:
            # Initialize schema versioning if not already done
            current_version = self.db_manager.get_current_schema_version()
            if current_version is None:
                self.db_manager.initialize_schema_versioning(initial_version)
                print(f"Initialized schema versioning at version {initial_version}")
            else:
                print(f"Schema versioning already initialized at version {current_version.version}")
        
        except Exception as e:
            raise MigrationError(f"Failed to initialize migrations: {e}") from e
    
    def create_migration(self, description: str, version: str = None) -> str:
        """Create a new migration file template.
        
        Args:
            description: Description of the migration
            version: Target version (auto-generated if None)
            
        Returns:
            str: Path to the created migration file
            
        Raises:
            MigrationError: If migration creation fails
        """
        try:
            # Determine version
            if not version:
                current_version = self.db_manager.get_current_schema_version()
                if current_version:
                    version = get_next_version(current_version.version, "minor")
                else:
                    version = "1.0.0"
            
            # Create template migration
            template_up = "-- Add your migration SQL here\n-- Example: ALTER TABLE experiment ADD COLUMN new_field VARCHAR(255);"
            template_down = "-- Add rollback SQL here\n-- Example: ALTER TABLE experiment DROP COLUMN new_field;"
            
            migration = self.generator.custom_migration(
                up_sql=template_up,
                down_sql=template_down,
                description=description
            )
            
            # Save migration file
            filepath = self.generator.save_migration(migration, version)
            
            print(f"Created migration file: {filepath}")
            print(f"Version: {version}")
            print(f"Edit the file to add your migration SQL before applying.")
            
            return str(filepath)
        
        except Exception as e:
            raise MigrationError(f"Failed to create migration: {e}") from e
    
    def apply_migration(self, migration_file: Union[str, Path]) -> None:
        """Apply a migration to the database.
        
        Args:
            migration_file: Path to migration file
            
        Raises:
            MigrationError: If migration application fails
        """
        migration_file = Path(migration_file)
        
        if not migration_file.exists():
            raise MigrationError(f"Migration file not found: {migration_file}")
        
        try:
            # Parse migration file
            migration_info = self._parse_migration_file(migration_file)
            
            # Check if migration already applied first
            if migration_info['version'] and self.db_manager.check_version_exists(migration_info['version']):
                print(f"Migration {migration_info['version']} already applied, skipping.")
                return
            
            # Check version compatibility
            current_version = self.db_manager.get_current_schema_version()
            if current_version and migration_info['version']:
                if not is_compatible_upgrade(current_version.version, migration_info['version']):
                    raise MigrationError(
                        f"Cannot upgrade from {current_version.version} to {migration_info['version']}"
                    )
            
            # Apply the migration
            print(f"Applying migration: {migration_info['name']}")
            print(f"Version: {migration_info['version']}")
            print(f"Description: {migration_info['description']}")
            
            # Execute UP script
            up_statements = self._split_sql_statements(migration_info['up_sql'])
            for statement in up_statements:
                if statement.strip():
                    try:
                        self.db_manager._execute_query(statement)
                    except Exception as e:
                        # Rollback and re-raise
                        self.db_manager.connection.rollback()
                        raise MigrationError(f"Migration failed at statement: {statement[:100]}...\nError: {e}")
            
            # Commit the migration
            self.db_manager.connection.commit()
            
            # Record migration in schema versions
            if migration_info['version']:
                self.db_manager.record_migration(
                    version=migration_info['version'],
                    migration_name=migration_info['name'],
                    description=migration_info['description'],
                    rollback_script=migration_info['down_sql']
                )
            
            print(f"Successfully applied migration: {migration_info['name']}")
        
        except Exception as e:
            raise MigrationError(f"Failed to apply migration: {e}") from e
    
    def rollback_migration(self, version: str = None) -> None:
        """Rollback the last migration or to a specific version.
        
        Args:
            version: Version to rollback to (rollback last migration if None)
            
        Raises:
            MigrationError: If rollback fails
        """
        try:
            current_version = self.db_manager.get_current_schema_version()
            if not current_version:
                raise MigrationError("No migrations to rollback")
            
            if version:
                # Rollback to specific version
                all_versions = self.db_manager.get_all_schema_versions()
                target_idx = None
                for i, v in enumerate(all_versions):
                    if v.version == version:
                        target_idx = i
                        break
                
                if target_idx is None:
                    raise MigrationError(f"Version {version} not found")
                
                # Rollback all versions after target
                versions_to_rollback = all_versions[target_idx + 1:]
            else:
                # Rollback just the last migration
                versions_to_rollback = [current_version]
            
            # Apply rollbacks in reverse order
            for version_record in reversed(versions_to_rollback):
                print(f"Rolling back migration: {version_record.migration_name} (v{version_record.version})")
                
                if version_record.rollback_script:
                    # Execute rollback script
                    rollback_statements = self._split_sql_statements(version_record.rollback_script)
                    for statement in rollback_statements:
                        if statement.strip():
                            self.db_manager._execute_query(statement)
                    
                    self.db_manager.connection.commit()
                else:
                    print(f"Warning: No rollback script for {version_record.migration_name}")
                
                # Remove from schema versions (if rolling back completely)
                if not version or version_record.version != version:
                    ph = self.db_manager._get_placeholder()
                    delete_query = f"DELETE FROM SCHEMA_VERSION WHERE version = {ph}"
                    self.db_manager._execute_query(delete_query, (version_record.version,))
                    self.db_manager.connection.commit()
                
                print(f"Successfully rolled back migration: {version_record.migration_name}")
        
        except Exception as e:
            self.db_manager.connection.rollback()
            raise MigrationError(f"Failed to rollback migration: {e}") from e
    
    def list_migrations(self, show_applied: bool = True) -> List[Dict]:
        """List migrations and their application status.
        
        Args:
            show_applied: Whether to include applied migrations
            
        Returns:
            List[Dict]: Migration information with status
        """
        # Get file-based migrations
        file_migrations = self.generator.list_migrations()
        
        # Get applied migrations
        applied_versions = set()
        if show_applied:
            applied_migrations = self.db_manager.get_all_schema_versions()
            applied_versions = {m.version for m in applied_migrations}
        
        # Combine information
        migration_list = []
        for migration in file_migrations:
            migration_info = {
                **migration,
                'applied': migration.get('version', '') in applied_versions,
                'status': 'applied' if migration.get('version', '') in applied_versions else 'pending'
            }
            migration_list.append(migration_info)
        
        return migration_list
    
    def status(self) -> Dict:
        """Get migration system status.
        
        Returns:
            Dict: System status information
        """
        current_version = self.db_manager.get_current_schema_version()
        all_versions = self.db_manager.get_all_schema_versions()
        file_migrations = self.generator.list_migrations()
        
        # Find pending migrations
        applied_versions = {v.version for v in all_versions}
        pending_migrations = [
            m for m in file_migrations 
            if m.get('version') and m['version'] not in applied_versions
        ]
        
        return {
            'current_version': current_version.version if current_version else None,
            'total_applied': len(all_versions),
            'total_pending': len(pending_migrations),
            'migration_files': len(file_migrations),
            'migration_dir': str(self.migration_dir),
            'database_type': 'SQLite' if self.db_manager.use_sqlite else 'MySQL'
        }
    
    def generate_table_migration(self, table_name: str, columns: List[Dict], 
                                constraints: List[str] = None, 
                                description: str = None, version: str = None) -> str:
        """Generate a CREATE TABLE migration.
        
        Args:
            table_name: Name of the table to create
            columns: List of column definitions
            constraints: List of table constraints
            description: Description of the migration
            version: Target version
            
        Returns:
            str: Path to the created migration file
        """
        try:
            if not version:
                current_version = self.db_manager.get_current_schema_version()
                version = get_next_version(current_version.version if current_version else "1.0.0", "minor")
            
            migration = self.generator.create_table_migration(
                table_name=table_name,
                columns=columns,
                constraints=constraints,
                description=description
            )
            
            filepath = self.generator.save_migration(migration, version)
            print(f"Generated table migration: {filepath}")
            
            return str(filepath)
        
        except Exception as e:
            raise MigrationError(f"Failed to generate table migration: {e}") from e
    
    def generate_column_migration(self, table_name: str, column_name: str, 
                                column_type: str, version: str = None, 
                                **options) -> str:
        """Generate an ADD COLUMN migration.
        
        Args:
            table_name: Name of the table
            column_name: Name of the new column
            column_type: Data type of the new column
            version: Target version
            **options: Additional column options
            
        Returns:
            str: Path to the created migration file
        """
        try:
            if not version:
                current_version = self.db_manager.get_current_schema_version()
                version = get_next_version(current_version.version if current_version else "1.0.0", "patch")
            
            migration = self.generator.add_column_migration(
                table_name=table_name,
                column_name=column_name,
                column_type=column_type,
                **options
            )
            
            filepath = self.generator.save_migration(migration, version)
            print(f"Generated column migration: {filepath}")
            
            return str(filepath)
        
        except Exception as e:
            raise MigrationError(f"Failed to generate column migration: {e}") from e
    
    def generate_index_migration(self, table_name: str, columns: List[str],
                               index_name: str = None, unique: bool = False,
                               version: str = None) -> str:
        """Generate an ADD INDEX migration.
        
        Args:
            table_name: Name of the table
            columns: List of column names for the index
            index_name: Name of the index
            unique: Whether to create a unique index
            version: Target version
            
        Returns:
            str: Path to the created migration file
        """
        try:
            if not version:
                current_version = self.db_manager.get_current_schema_version()
                version = get_next_version(current_version.version if current_version else "1.0.0", "patch")
            
            migration = self.generator.add_index_migration(
                table_name=table_name,
                columns=columns,
                index_name=index_name,
                unique=unique
            )
            
            filepath = self.generator.save_migration(migration, version)
            print(f"Generated index migration: {filepath}")
            
            return str(filepath)
        
        except Exception as e:
            raise MigrationError(f"Failed to generate index migration: {e}") from e
    
    def _parse_migration_file(self, filepath: Path) -> Dict:
        """Parse migration file and extract metadata and SQL.
        
        Args:
            filepath: Path to migration file
            
        Returns:
            Dict: Parsed migration information
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        metadata = {}
        up_sql_lines = []
        down_sql_lines = []
        
        current_section = None
        
        for line in lines:
            line = line.rstrip()
            
            # Parse metadata
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
            
            # Parse SQL sections
            elif line == '-- UP':
                current_section = 'up'
                continue
            elif line == '-- DOWN':
                current_section = 'down'
                continue
            elif line.startswith('--') or not line.strip():
                continue
            
            # Add SQL lines
            if current_section == 'up':
                up_sql_lines.append(line)
            elif current_section == 'down':
                down_sql_lines.append(line)
        
        metadata['up_sql'] = '\n'.join(up_sql_lines).strip()
        metadata['down_sql'] = '\n'.join(down_sql_lines).strip()
        
        return metadata
    
    def _split_sql_statements(self, sql: str) -> List[str]:
        """Split SQL into individual statements.
        
        Args:
            sql: SQL text to split
            
        Returns:
            List[str]: Individual SQL statements
        """
        # Simple statement splitting on semicolons
        # Note: This doesn't handle complex cases like semicolons in strings
        statements = []
        current_statement = []
        
        for line in sql.split('\n'):
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            
            current_statement.append(line)
            
            if line.endswith(';'):
                statement = ' '.join(current_statement)
                if statement.strip():
                    statements.append(statement)
                current_statement = []
        
        # Add any remaining statement
        if current_statement:
            statement = ' '.join(current_statement)
            if statement.strip():
                statements.append(statement)
        
        return statements 