import os
import sqlite3
import pytest
from experiment_manager.results.sources.db_datasource import DBDataSource
from experiment_manager.db.manager import DatabaseManager, ConnectionError, QueryError

@pytest.mark.datasource
class TestDBDataSourceReadonly:
    def test_readonly_mode_allows_reads_blocks_writes(self, tmp_path):
        """
        Create a temp SQLite DB, write some data, then reopen in readonly mode via DBDataSource and verify:
        - Reads succeed
        - Writes fail with an error
        """
        # Step 1: Create a writable SQLite DB and add a table/row
        db_path = tmp_path / "test_readonly.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO test (value) VALUES ('foo')")
        conn.commit()
        conn.close()

        # Step 2: Open in readonly mode using DBDataSource
        ds = DBDataSource(db_path=str(db_path), use_sqlite=True, readonly=True)
        # Read should succeed
        cursor = ds.db_manager.connection.execute("SELECT value FROM test")
        result = cursor.fetchone()
        assert result[0] == 'foo'

        # Write should fail
        with pytest.raises((sqlite3.OperationalError, QueryError)):
            ds.db_manager.connection.execute("INSERT INTO test (value) VALUES ('bar')")

    def test_db_manager_readonly_mode(self, tmp_path):
        """
        Directly test DatabaseManager readonly enforcement.
        """
        db_path = tmp_path / "test_db_manager_readonly.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test2 (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO test2 (value) VALUES ('baz')")
        conn.commit()
        conn.close()

        # Open in readonly mode
        dbm = DatabaseManager(database_path=str(db_path), use_sqlite=True, readonly=True)
        # Read should succeed
        cursor = dbm.connection.execute("SELECT value FROM test2")
        result = cursor.fetchone()
        assert result[0] == 'baz'

        # Write should fail
        with pytest.raises((sqlite3.OperationalError, QueryError)):
            dbm.connection.execute("INSERT INTO test2 (value) VALUES ('fail')")

    def test_open_nonexistent_db_readonly_fails(self, tmp_path):
        """
        Opening a nonexistent DB in readonly mode should fail with ConnectionError or sqlite3.OperationalError.
        """
        db_path = tmp_path / "does_not_exist.sqlite"
        with pytest.raises((sqlite3.OperationalError, ConnectionError)):
            DBDataSource(db_path=str(db_path), use_sqlite=True, readonly=True) 