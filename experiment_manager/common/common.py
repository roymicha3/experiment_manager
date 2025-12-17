from enum import Enum, IntEnum, auto

"""Common constants and enumerations shared across Experiment Manager.

This module centralises enums (RunStatus, Level, Metric, etc.) and other
constants so that the rest of the codebase can import them from a single
place, ensuring consistent values, type-safety, and IDE auto-completion.
Add new enums here whenever you introduce additional categorical fields
(e.g., for new database tables) to avoid scattering hard-coded strings
throughout the code.
"""

LOG_NAME = "log"

class RunStatus(Enum):
    RUNNING = 0
    SUCCESS = auto()
    FINISHED = auto()
    FAILED = auto()
    ABORTED = auto()
    STOPPED = auto()
    SKIPPED = auto()
class ConfigPaths(Enum):
    ENV_CONFIG     = "env.yaml"
    CONFIG_FILE    = "experiment.yaml"
    BASE_CONFIG    = "base.yaml"
    TRIALS_CONFIG  = "trials.yaml"

# enum for levels:
class Level(Enum):
    EXPERIMENT    = 0
    TRIAL         = 1
    TRIAL_RUN     = 2
    PIPELINE      = 3
    EPOCH         = 4
    BATCH         = 5
    

class MetricCategory(Enum):
    TRACKED = 0
    UNTRACKED = 1

class Metric(IntEnum):
    # Tracked metrics
    EPOCH = 1
    TEST_ACC = 2
    TEST_LOSS = 3
    VAL_ACC = 4
    VAL_LOSS = 5
    TRAIN_ACC = 6
    TRAIN_LOSS = 7
    LEARNING_RATE = 8
    CUSTOM = 9 # custom metric (name, value) - tracked by all trackers

    # Untracked metrics
    NETWORK = 10
    DATA = 11
    LABELS = 12
    STATUS = 13
    CONFUSION = 14
    CUSTOM_UNTRACKED = 15 # custom metric (name, value) - NOT tracked, only accessible in callbacks
    
    @property
    def name(self) -> str:
        """Get the name of the metric."""
        return self._name_.lower()

# Static mapping of metrics to categories
_metric_categories = {}

def _init_metric_categories():
    global _metric_categories
    _metric_categories = \
        {
    Metric.EPOCH: MetricCategory.TRACKED,
    Metric.TEST_ACC: MetricCategory.TRACKED,
    Metric.TEST_LOSS: MetricCategory.TRACKED,
    Metric.VAL_ACC: MetricCategory.TRACKED,
    Metric.VAL_LOSS: MetricCategory.TRACKED,
    Metric.TRAIN_ACC: MetricCategory.TRACKED,
    Metric.TRAIN_LOSS: MetricCategory.TRACKED,
    Metric.CONFUSION: MetricCategory.TRACKED,
    Metric.LEARNING_RATE: MetricCategory.TRACKED,
    Metric.CUSTOM: MetricCategory.TRACKED,
        
    Metric.NETWORK: MetricCategory.UNTRACKED,
    Metric.DATA: MetricCategory.UNTRACKED,
    Metric.LABELS: MetricCategory.UNTRACKED,
    Metric.STATUS: MetricCategory.UNTRACKED,
    Metric.CUSTOM_UNTRACKED: MetricCategory.UNTRACKED,
}

def get_metric_category(metric: Metric) -> MetricCategory:
    return _metric_categories[metric]

def get_metrics_by_category(category: MetricCategory) -> list[Metric]:
    return [m for m in Metric if _metric_categories[m] == category]

def get_tracked_metrics() -> list[Metric]:
    return get_metrics_by_category(MetricCategory.TRACKED)

# Initialize the categories after all classes are defined
_init_metric_categories()

class ArtifactType(Enum):
    """Category of artifact stored in the ARTIFACT table and on disk.

    The values are lowercase strings to match the ON-DISK/DATABASE values
    already used by the system.  When adding a new artifact category,
    update this enum and migrate any related database rows accordingly.
    """
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    FIGURE = "figure"
    LOG = "log"
    TENSORBOARD = "tensorboard"
    OTHER = "other"

# Public exports for `from experiment_manager.common.common import *`
__all__ = [
    "LOG_NAME",
    "RunStatus",
    "ConfigPaths",
    "Level",
    "MetricCategory",
    "Metric",
    "ArtifactType",
    "get_metric_category",
    "get_metrics_by_category",
    "get_tracked_metrics",
]