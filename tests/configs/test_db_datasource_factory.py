import os
import pytest
from omegaconf import OmegaConf

from experiment_manager.results.sources.db_datasource import DBDataSource


def _build_config(tmp_path, minimal: bool):
    """Create DictConfig for DBDataSource with minimal or full parameters."""
    base = {
        "db_path": os.path.join(tmp_path, "factory_test.db"),
    }
    if not minimal:
        base.update(
            {
                "use_sqlite": True,
                "host": "127.0.0.1",
                "user": "admin",
                "password": "secret",
            }
        )
    return OmegaConf.create(base)


@pytest.mark.parametrize("minimal", [True, False])
def test_factory_reconstruction_from_yaml(minimal, tmp_path):
    """DBDataSource.from_config should correctly build an instance from YAML config."""
    # Prepare config and write to file for completeness (simulate YAML source)
    cfg = _build_config(tmp_path, minimal)
    yaml_path = os.path.join(tmp_path, "factory_config.yaml")
    OmegaConf.save(cfg, yaml_path)

    # Load YAML and construct instance via factory method
    cfg_loaded = OmegaConf.load(yaml_path)
    datasource = DBDataSource.from_config(cfg_loaded)

    try:
        # Basic attribute assertions
        assert datasource.db_path == cfg.db_path
        assert datasource.db_manager.use_sqlite == cfg.get("use_sqlite", True)

        # For full configuration, compare full config containers
        if not minimal:
            assert (
                OmegaConf.to_container(datasource.config, resolve=True)
                == OmegaConf.to_container(cfg, resolve=True)
            )
    finally:
        if hasattr(datasource, "db_manager") and hasattr(datasource.db_manager, "connection"):
            datasource.db_manager.connection.close() 