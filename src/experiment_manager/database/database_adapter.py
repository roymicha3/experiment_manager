"""
Database Adapter for Migration Validation

This module provides an adapter to make the existing DatabaseManager compatible
with the migration validation framework's expected API.
"""
from typing import Optional
from contextlib import contextmanager
from experiment_manager.db.manager import DatabaseManager as OriginalDatabaseManager


class DatabaseManager:
    """
    Adapter for the original DatabaseManager to provide migration validation API.
    
    This adapter wraps the existing DatabaseManager and provides the interface
    expected by the migration validation framework.
    """
    
    def __init__(self, db_type: str, db_path: Optional[str] = None, 
                 host: str = "localhost", port: int = 3306, 
                 user: str = "root", password: str = "", database: str = ""):
        """
        Initialize the database manager adapter.
        
        Args:
            db_type: Database type ('sqlite' or 'mysql')
            db_path: Path to SQLite database file
            host: MySQL host (for MySQL connections)
            port: MySQL port (for MySQL connections)
            user: MySQL user (for MySQL connections)
            password: MySQL password (for MySQL connections)
            database: MySQL database name (for MySQL connections)
        """
        self.db_type = db_type
        self.db_path = db_path
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        
        # Add use_sqlite for compatibility with existing code
        self.use_sqlite = (db_type == 'sqlite')
        
        # Add database_path for compatibility with SchemaInspector
        self.database_path = db_path if db_type == 'sqlite' else database
        
        # Create the underlying database manager
        if db_type == 'sqlite':
            if not db_path:
                raise ValueError("db_path is required for SQLite connections")
            self._manager = OriginalDatabaseManager(
                database_path=db_path,
                use_sqlite=True,
                recreate=False
            )
        elif db_type == 'mysql':
            self._manager = OriginalDatabaseManager(
                database_path=database,  # database name
                use_sqlite=False,
                host=host,
                user=user,
                password=password,
                recreate=False
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection context manager.
        
        Yields:
            Database connection object
        """
        try:
            yield self._manager.connection
        except Exception as e:
            # Ensure connection is still valid
            if hasattr(self._manager, 'connection') and self._manager.connection:
                try:
                    self._manager.connection.rollback()
                except:
                    pass
            raise e
    
    def execute_query(self, query: str, params: Optional[tuple] = None):
        """
        Execute a database query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Query result cursor
        """
        return self._manager._execute_query(query, params)
    
    def _execute_query(self, query: str, params: Optional[tuple] = None):
        """
        Execute a database query (internal method expected by SchemaInspector).
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Query result cursor
        """
        return self._manager._execute_query(query, params)
    
    def get_cursor(self):
        """Get the database cursor."""
        return self._manager.cursor
    
    def commit(self):
        """Commit the current transaction."""
        self._manager.connection.commit()
    
    def rollback(self):
        """Rollback the current transaction."""
        self._manager.connection.rollback()
    
    def get_current_schema_version(self):
        """Get current schema version."""
        try:
            return self._manager.get_current_schema_version()
        except Exception:
            # Schema versioning might not be initialized
            return None
    
    def close(self):
        """Close the database connection."""
        if hasattr(self._manager, 'cursor') and self._manager.cursor:
            self._manager.cursor.close()
        if hasattr(self._manager, 'connection') and self._manager.connection:
            self._manager.connection.close()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except:
            pass 