"""Tests for schema versioning functionality."""
import pytest
from datetime import datetime

from experiment_manager.db.manager import DatabaseManager, QueryError
from experiment_manager.db.tables import SchemaVersion
from experiment_manager.db.version_utils import (
    parse_version, validate_version_string, is_compatible_upgrade,
    is_backward_compatible, get_next_version, compare_versions,
    SemanticVersion, VersionError
)

@pytest.fixture
def db_manager(tmp_path):
    """Create a test database manager."""
    db_path = tmp_path / "test_experiment_manager.db"
    manager = DatabaseManager(database_path=str(db_path), use_sqlite=True, recreate=True)
    yield manager

class TestSchemaVersioning:
    """Test schema versioning functionality."""
    
    def test_schema_version_table_created(self, db_manager):
        """Test that SCHEMA_VERSION table is created."""
        cursor = db_manager._execute_query("SELECT name FROM sqlite_master WHERE type='table' AND name='SCHEMA_VERSION'")
        tables = cursor.fetchall()
        assert len(tables) == 1
        assert tables[0]['name'] == 'SCHEMA_VERSION'
    
    def test_get_current_schema_version_empty(self, db_manager):
        """Test getting current schema version when none exist."""
        version = db_manager.get_current_schema_version()
        assert version is None
    
    def test_record_migration(self, db_manager):
        """Test recording a migration."""
        version = db_manager.record_migration(
            version="1.0.0",
            migration_name="initial_migration",
            description="Initial schema setup",
            rollback_script="DROP DATABASE test;"
        )
        
        assert version.version == "1.0.0"
        assert version.migration_name == "initial_migration"
        assert version.description == "Initial schema setup"
        assert version.rollback_script == "DROP DATABASE test;"
        assert isinstance(version.applied_at, datetime)
        assert version.id is not None
    
    def test_get_current_schema_version_with_data(self, db_manager):
        """Test getting current schema version when data exists."""
        # Record multiple migrations
        db_manager.record_migration("1.0.0", "migration1", "First migration")
        db_manager.record_migration("1.1.0", "migration2", "Second migration")
        db_manager.record_migration("1.2.0", "migration3", "Third migration")
        
        current = db_manager.get_current_schema_version()
        assert current is not None
        assert current.version == "1.2.0"
        assert current.migration_name == "migration3"
    
    def test_get_all_schema_versions(self, db_manager):
        """Test getting all schema versions in order."""
        # Record migrations
        db_manager.record_migration("1.0.0", "migration1", "First migration")
        db_manager.record_migration("1.1.0", "migration2", "Second migration")
        db_manager.record_migration("1.2.0", "migration3", "Third migration")
        
        versions = db_manager.get_all_schema_versions()
        assert len(versions) == 3
        
        # Should be ordered by application time (ascending)
        assert versions[0].version == "1.0.0"
        assert versions[1].version == "1.1.0"
        assert versions[2].version == "1.2.0"
    
    def test_check_version_exists(self, db_manager):
        """Test checking if a version exists."""
        # Initially no versions
        assert not db_manager.check_version_exists("1.0.0")
        
        # Record a migration
        db_manager.record_migration("1.0.0", "migration1", "First migration")
        
        # Now it should exist
        assert db_manager.check_version_exists("1.0.0")
        assert not db_manager.check_version_exists("1.1.0")
    
    def test_initialize_schema_versioning(self, db_manager):
        """Test initializing schema versioning."""
        version = db_manager.initialize_schema_versioning("1.0.0")
        
        assert version.version == "1.0.0"
        assert version.migration_name == "initial_schema"
        assert version.description == "Initial database schema creation"
        assert version.rollback_script is None
        
        # Should be able to get it as current version
        current = db_manager.get_current_schema_version()
        assert current.version == "1.0.0"
    
    def test_initialize_schema_versioning_already_exists(self, db_manager):
        """Test that initializing schema versioning fails if already initialized."""
        db_manager.initialize_schema_versioning("1.0.0")
        
        with pytest.raises(QueryError, match="Schema versioning already initialized"):
            db_manager.initialize_schema_versioning("1.1.0")
    
    def test_initialize_schema_versioning_invalid_version(self, db_manager):
        """Test that initializing with invalid version fails."""
        with pytest.raises(VersionError, match="Invalid initial version format"):
            db_manager.initialize_schema_versioning("invalid.version")
    
    def test_get_version_compatibility_info_no_current(self, db_manager):
        """Test compatibility info when no current version exists."""
        info = db_manager.get_version_compatibility_info("1.0.0")
        
        assert info["current_version"] is None
        assert info["target_version"] == "1.0.0"
        assert info["can_upgrade"] is True
        assert info["is_compatible"] is True
        assert info["is_backward_compatible"] is True
        assert info["requires_initialization"] is True
    
    def test_get_version_compatibility_info_with_current(self, db_manager):
        """Test compatibility info with existing current version."""
        db_manager.initialize_schema_versioning("1.0.0")
        
        # Test compatible upgrade
        info = db_manager.get_version_compatibility_info("1.1.0")
        assert info["current_version"] == "1.0.0"
        assert info["target_version"] == "1.1.0"
        assert info["can_upgrade"] is True
        assert info["is_compatible"] is True
        assert info["is_backward_compatible"] is True
        assert info["requires_initialization"] is False
        
        # Test major version upgrade (not backward compatible)
        info = db_manager.get_version_compatibility_info("2.0.0")
        assert info["can_upgrade"] is True
        assert info["is_compatible"] is True
        assert info["is_backward_compatible"] is False
        
        # Test downgrade
        info = db_manager.get_version_compatibility_info("0.9.0")
        assert info["can_upgrade"] is False
        assert info["is_compatible"] is False


class TestVersionUtils:
    """Test version utility functions."""
    
    def test_semantic_version_creation(self):
        """Test creating SemanticVersion objects."""
        version = SemanticVersion(1, 2, 3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert str(version) == "1.2.3"
    
    def test_semantic_version_comparison(self):
        """Test comparing SemanticVersion objects."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)
        v3 = SemanticVersion(2, 0, 0)
        v4 = SemanticVersion(1, 0, 0)
        
        assert v1 < v2
        assert v2 < v3
        assert v1 == v4
        assert v3 > v2
        assert v2 >= v1
        assert v1 <= v4
    
    def test_parse_version_valid(self):
        """Test parsing valid version strings."""
        version = parse_version("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        
        # Test with 'v' prefix
        version = parse_version("v2.0.1")
        assert version.major == 2
        assert version.minor == 0
        assert version.patch == 1
    
    def test_parse_version_invalid(self):
        """Test parsing invalid version strings."""
        with pytest.raises(VersionError):
            parse_version("1.2")
        
        with pytest.raises(VersionError):
            parse_version("1.2.3.4")
        
        with pytest.raises(VersionError):
            parse_version("a.b.c")
        
        with pytest.raises(VersionError):
            parse_version("1.2.3-alpha")
    
    def test_validate_version_string(self):
        """Test version string validation."""
        assert validate_version_string("1.2.3") is True
        assert validate_version_string("v1.2.3") is True
        assert validate_version_string("0.0.0") is True
        assert validate_version_string("1.2") is False
        assert validate_version_string("invalid") is False
    
    def test_is_compatible_upgrade(self):
        """Test compatibility checking for upgrades."""
        assert is_compatible_upgrade("1.0.0", "1.1.0") is True
        assert is_compatible_upgrade("1.0.0", "2.0.0") is True
        assert is_compatible_upgrade("1.1.0", "1.0.0") is False
        assert is_compatible_upgrade("1.0.0", "1.0.0") is False
    
    def test_is_backward_compatible(self):
        """Test backward compatibility checking."""
        assert is_backward_compatible("1.0.0", "1.1.0") is True
        assert is_backward_compatible("1.0.0", "1.0.1") is True
        assert is_backward_compatible("1.0.0", "2.0.0") is False
        assert is_backward_compatible("1.1.0", "1.0.0") is False
    
    def test_get_next_version(self):
        """Test generating next version numbers."""
        assert get_next_version("1.0.0", "patch") == "1.0.1"
        assert get_next_version("1.0.0", "minor") == "1.1.0"
        assert get_next_version("1.0.0", "major") == "2.0.0"
        assert get_next_version("1.2.3", "patch") == "1.2.4"
        assert get_next_version("1.2.3", "minor") == "1.3.0"
        assert get_next_version("1.2.3", "major") == "2.0.0"
        
        with pytest.raises(VersionError):
            get_next_version("1.0.0", "invalid")
    
    def test_compare_versions(self):
        """Test version comparison function."""
        assert compare_versions("1.0.0", "1.1.0") == -1
        assert compare_versions("1.1.0", "1.0.0") == 1
        assert compare_versions("1.0.0", "1.0.0") == 0
        assert compare_versions("2.0.0", "1.9.9") == 1
        assert compare_versions("1.0.1", "1.1.0") == -1 