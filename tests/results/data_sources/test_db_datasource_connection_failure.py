import os
import pytest
from experiment_manager.results.sources.db_datasource import DBDataSource
from experiment_manager.db.manager import ConnectionError

@pytest.mark.parametrize("db_path, use_sqlite, expect_error", [
    ("/nonexistent_dir/should_fail.db", True, True),  # Non-existent directory
    ("/", True, True),  # Path is a directory, not a file
    ("/root/forbidden.db", True, True),  # Likely not writable
    ("test_should_succeed.db", True, False),  # Should succeed (created in cwd)
])
def test_sqlite_connection_failures(tmp_path, db_path, use_sqlite, expect_error):
    # Adjust db_path for tmp_path if not absolute
    if not os.path.isabs(db_path):
        db_path = os.path.join(tmp_path, db_path)
    if expect_error:
        with pytest.raises(ConnectionError):
            DBDataSource(db_path, use_sqlite=use_sqlite)
    else:
        ds = DBDataSource(db_path, use_sqlite=use_sqlite)
        ds.close()

@pytest.mark.parametrize("host, user, password, expect_error", [
    ("invalid_host", "root", "", True),  # Unreachable host
    ("localhost", "invalid_user", "badpass", True),  # Invalid credentials (if MySQL running)
])
def test_mysql_connection_failures(host, user, password, expect_error):
    # Only run if mysql.connector is available
    try:
        import mysql.connector
    except ImportError:
        pytest.skip("mysql-connector not installed")
    if expect_error:
        with pytest.raises(ConnectionError):
            DBDataSource(
                db_path="test_db", use_sqlite=False, host=host, user=user, password=password
            ) 