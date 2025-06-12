import os
import pytest
from experiment_manager.results.sources.db_datasource import DBDataSource
from experiment_manager.db.manager import ConnectionError
import platform

def test_sqlite_connection_failures(tmp_path):
    paths = []
    # Path that is a directory (should always fail)
    paths.append((str(tmp_path), True, True))
    # Path to a non-existent drive (Windows) or /dev/null (Linux, as a file)
    if os.name == "nt":
        paths.append((r"Z:\\this_should_fail.db", True, True))
        paths.append((r"C:\\Windows\\System32\\forbidden.db", True, True))
    else:
        paths.append(("/dev/null/should_fail.db", True, True))
        paths.append(("/etc/forbidden.db", True, True))
    # Path that should succeed (created in cwd)
    paths.append((str(tmp_path / "test_should_succeed.db"), True, False))

    for db_path, use_sqlite, expect_error in paths:
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