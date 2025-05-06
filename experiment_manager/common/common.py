from enum import Enum, IntEnum, auto

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
    CUSTOM = 9 # custom metric (name, value)

    # Untracked metrics
    NETWORK = 10
    DATA = 11
    LABELS = 12
    STATUS = 13
    CONFUSION = 14
    
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
}

def get_metric_category(metric: Metric) -> MetricCategory:
    return _metric_categories[metric]

def get_metrics_by_category(category: MetricCategory) -> list[Metric]:
    return [m for m in Metric if _metric_categories[m] == category]

def get_tracked_metrics() -> list[Metric]:
    return get_metrics_by_category(MetricCategory.TRACKED)

# Initialize the categories after all classes are defined
_init_metric_categories()