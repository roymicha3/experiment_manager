import os
import pytest
from omegaconf import OmegaConf

from experiment_manager.results.sources.db_datasource import DBDataSource


def _build_valid_config(tmp_path, extra: bool = False):
    cfg_dict = {
        "db_path": os.path.join(tmp_path, "schema_test.db"),
        "use_sqlite": True,
        "host": "localhost",
        "user": "user",
        "password": "pwd",
    }
    if extra:
        cfg_dict["extra_field"] = "extra_value"
    return OmegaConf.create(cfg_dict)


def test_valid_config_compliance(tmp_path):
    """A fully valid configuration should pass validation and construct the datasource."""
    cfg = _build_valid_config(tmp_path)
    ds = DBDataSource.from_config(cfg)
    try:
        assert ds.db_path == cfg.db_path
        assert ds.db_manager.use_sqlite is True
    finally:
        ds.db_manager.connection.close()


def test_unknown_field_allowed(tmp_path):
    """Unknown fields should be ignored by validation and not cause errors."""
    cfg = _build_valid_config(tmp_path, extra=True)
    ds = DBDataSource.from_config(cfg)
    try:
        assert ds.config.get("extra_field") == "extra_value"
    finally:
        ds.db_manager.connection.close() 