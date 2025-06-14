import os
import pytest
from omegaconf import OmegaConf

from experiment_manager.results.sources.db_datasource import DBDataSource


def _create_config(tmp_path):
    """Helper to build a minimal DictConfig for DBDataSource."""
    return OmegaConf.create(
        {
            "db_path": os.path.join(tmp_path, "test.db"),
            "use_sqlite": True,
            "host": "localhost",
            "user": "user",
            "password": "pass",
        }
    )


def test_db_datasource_yaml_round_trip(tmp_path):
    """Ensure that a DBDataSource configuration survives a YAML round-trip without loss."""

    # Prepare configuration and YAML file location
    cfg = _create_config(tmp_path)
    cfg.readonly = False  # Needs write access for test DB creation
    yaml_path = os.path.join(tmp_path, "db_config.yaml")

    # Serialize to YAML using OmegaConf (DBDataSource.save is not yet implemented)
    OmegaConf.save(cfg, yaml_path)

    # Reconstruct the data source from YAML
    datasource = DBDataSource.load(yaml_path)

    try:
        # Validate that critical attributes match the original config
        assert datasource.db_path == cfg.db_path
        # SQLite flag is accessible via the internal DatabaseManager
        assert datasource.db_manager.use_sqlite == cfg.use_sqlite
        # Host, user, and password are stored only in the DatabaseManager constructor parameters
        # They aren't exposed as attributes on DBDataSource, so verify via the stored config
        assert OmegaConf.to_container(datasource.config) == OmegaConf.to_container(cfg)

    finally:
        # Close the underlying DB connection to avoid locking the temp directory on Windows
        if hasattr(datasource, "db_manager") and hasattr(datasource.db_manager, "connection"):
            datasource.db_manager.connection.close() 