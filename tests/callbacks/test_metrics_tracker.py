import os
import pytest
from omegaconf import OmegaConf
from experiment_manager.experiment import Experiment, ConfigPaths
from experiment_manager.common.factory_registry import FactoryRegistry, FactoryType
from tests.pipelines.dummy_pipeline_factory import DummyPipelineFactory

@pytest.fixture
def config_dir(tmp_path):
    return os.path.join("tests", "configs", "test_callback")

@pytest.fixture
def prepare_env(tmp_path, config_dir):
    env_path = os.path.join(config_dir, ConfigPaths.ENV_CONFIG.value)
    env = OmegaConf.load(env_path)
    env.workspace = os.path.join(tmp_path, "test_outputs")
    OmegaConf.save(env, env_path)
    return env


def test_metrics_tracker_creates_metrics_log(config_dir, tmp_path, prepare_env):
    # Create custom factory registry
    registry = FactoryRegistry()
    registry.register(FactoryType.PIPELINE, DummyPipelineFactory())
    
    experiment = Experiment.create(config_dir, registry)
    experiment.env.workspace = os.path.join(tmp_path, "test_outputs")
    experiment.run()

    # Find metrics log files
    metrics_files = []
    for root, dirs, files in os.walk(experiment.env.workspace):
        for f in files:
            if f == "metrics.log":
                metrics_files.append(os.path.join(root, f))

    assert metrics_files, "No metrics.log files found!"
    for metrics_file in metrics_files:
        assert os.path.getsize(metrics_file) > 0, f"metrics.log is empty: {metrics_file}"
        with open(metrics_file, 'r', encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            assert lines, f"metrics.log is empty: {metrics_file}"
            header = lines[0].split(',')
            # Check 'type' column is first
            assert header[0] == "type", f"First column in header should be 'type', got {header[0]}"
            expected_metrics = {"train_loss", "val_loss", "val_acc", "test_acc"}
            assert expected_metrics.issubset(set(header[1:])), f"metrics.log header missing expected metrics: {metrics_file}"
            # Separate EPOCH and FINAL rows
            epoch_rows = []
            final_row = None
            for i, row in enumerate(lines[1:], 2):
                cols = row.split(',')
                assert len(cols) == len(header), f"Row {i} in {metrics_file} has {len(cols)} columns, expected {len(header)}: {row}"
                if cols[0] == "EPOCH":
                    epoch_rows.append(cols)
                    # All EPOCH values should be non-empty or 'nan'
                    assert all(c.strip() != '' or c.strip() == 'nan' for c in cols[1:]), f"EPOCH row {i} in {metrics_file} has empty values: {row}"
                elif cols[0] == "FINAL":
                    final_row = cols
                    # FINAL values should be non-empty or 'nan'
                    assert all(c.strip() != '' or c.strip() == 'nan' for c in cols[1:]), f"FINAL row in {metrics_file} has empty values: {row}"
            assert final_row is not None, f"No FINAL row found in {metrics_file}"
            assert len(epoch_rows) > 0, f"No EPOCH rows found in {metrics_file}"
