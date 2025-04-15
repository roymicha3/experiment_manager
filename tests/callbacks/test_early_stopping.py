import os
import pytest
from omegaconf import OmegaConf
from experiment_manager.experiment import Experiment, ConfigPaths
from tests.pipelines.dummy_pipeline_factory import DummyPipelineFactory

@pytest.fixture
def config_dir(tmp_path):
    return os.path.join("tests", "configs", "test_callback")


def test_early_stopping_stops_early(config_dir, tmp_path):
    experiment = Experiment.create(config_dir, DummyPipelineFactory)
    experiment.env.workspace = os.path.join(tmp_path, "test_outputs")
    experiment.run()

    # Find logs to check for early stopping
    log_files = []
    for root, dirs, files in os.walk(experiment.env.workspace):
        for f in files:
            if f.endswith(".log"):
                log_files.append(os.path.join(root, f))

    found_early_stopping = False
    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
            if "Early stopping triggered" in content or "early stopping" in content.lower():
                found_early_stopping = True
                break
    assert found_early_stopping, "Early stopping was not triggered according to logs."
