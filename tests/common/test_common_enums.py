import pytest

from experiment_manager.common.common import (
    RunStatus,
    Level,
    MetricCategory,
    Metric,
    ArtifactType,
)


def test_artifact_type_members():
    """Ensure ArtifactType enum contains expected members and values."""
    assert ArtifactType.MODEL.value == "model"
    assert ArtifactType.CHECKPOINT.name == "CHECKPOINT"
    assert ArtifactType.TENSORBOARD.value == "tensorboard"


def test_metric_name_property():
    """Metric.name property should return lowercase names."""
    assert Metric.TRAIN_LOSS.name == "train_loss"
    assert Metric.TEST_ACC.name == "test_acc"


def test_run_status_values():
    """RunStatus auto() values should increment starting from 0."""
    assert RunStatus.RUNNING.value == 0
    assert RunStatus.SUCCESS.value == 1
    # Ensure ordering is consistent
    assert RunStatus.SUCCESS.value < RunStatus.FINISHED.value 