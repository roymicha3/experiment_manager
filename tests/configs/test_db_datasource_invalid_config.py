import os
import pytest
from omegaconf import OmegaConf

from experiment_manager.results.sources.db_datasource import DBDataSource


def _cfg_missing_db_path(tmp_path):
    return OmegaConf.create({"use_sqlite": True})


def _cfg_wrong_type_use_sqlite(tmp_path):
    return OmegaConf.create({"db_path": os.path.join(tmp_path, "invalid.db"), "use_sqlite": "yes"})


def _cfg_invalid_host(tmp_path):
    return OmegaConf.create({"db_path": os.path.join(tmp_path, "invalid.db"), "host": ["localhost"]})


def _cfg_numeric_password(tmp_path):
    return OmegaConf.create({"db_path": os.path.join(tmp_path, "invalid.db"), "password": 12345})


INVALID_CONFIGS = [
    (_cfg_missing_db_path, AttributeError),
    (_cfg_wrong_type_use_sqlite, TypeError),
    (_cfg_invalid_host, TypeError),
    (_cfg_numeric_password, TypeError),
]


@pytest.mark.parametrize("cfg_builder, expected_exc", INVALID_CONFIGS)
def test_invalid_configs_raise(cfg_builder, expected_exc, tmp_path):
    cfg = cfg_builder(tmp_path)
    cfg.readonly = False  # Needs write access for test DB creation
    with pytest.raises(expected_exc):
        # Attempt to construct from invalid config
        DBDataSource.from_config(cfg) 