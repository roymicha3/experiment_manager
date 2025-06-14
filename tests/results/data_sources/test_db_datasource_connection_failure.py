import os
import pytest
from experiment_manager.results.sources.db_datasource import DBDataSource
from experiment_manager.db.manager import ConnectionError, QueryError
import platform
import tempfile
import shutil

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

def test_invalid_file_format_raises_connection_error(tmp_path):
    # Create a text file, not a valid SQLite db
    text_file = tmp_path / "not_a_db.txt"
    text_file.write_text("this is not a database")
    with pytest.raises(ConnectionError):
        DBDataSource(str(text_file), use_sqlite=True)

def test_read_only_db_file_raises_on_write(tmp_path):
    # Create a valid db file
    db_path = tmp_path / "readonly.db"
    ds = DBDataSource(str(db_path), use_sqlite=True)
    ds.close()
    # Make it read-only
    os.chmod(db_path, 0o444)
    try:
        ds2 = DBDataSource(str(db_path), use_sqlite=True)
        # Attempt a write operation (should fail)
        with pytest.raises(QueryError):
            # Try to create a table (should fail on read-only DB)
            ds2.db_manager._execute_query("CREATE TABLE test_fail (id INTEGER)")
        ds2.close()
    finally:
        # Restore permissions so tmp_path can be cleaned up
        os.chmod(db_path, 0o666)

def test_special_characters_in_path(tmp_path):
    # Path with special characters
    special_path = tmp_path / "db_!@#$%^&*()[]{};,.db"
    # Try to create and open
    try:
        ds = DBDataSource(str(special_path), use_sqlite=True)
        ds.close()
    except ConnectionError:
        # Acceptable if OS/filesystem does not allow
        pass 