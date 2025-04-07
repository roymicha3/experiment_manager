import enum
from typing import Dict

LOG_NAME = "log"

# enum for levels:
class Level(enum.Enum):
    EXPERIMENT    = 0
    TRIAL         = 1
    TRIAL_RUN      = 2
    PIPELINE      = 3
    EPOCH         = 4
    

class MetricCategory(enum.Enum):
    TRACKED = 0
    UNTRACKED = 1

class Metric(enum.IntEnum):
    # Tracked metrics
    EPOCH = 1
    TEST_ACC = 2
    TEST_LOSS = 3
    VAL_ACC = 4
    VAL_LOSS = 5

    # Untracked metrics
    NETWORK = 6
    DATA = 7
    STATUS = 8

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
            Metric.NETWORK: MetricCategory.UNTRACKED,
            Metric.DATA: MetricCategory.UNTRACKED,
            Metric.STATUS: MetricCategory.UNTRACKED
        }

def get_metric_category(metric: Metric) -> MetricCategory:
    return _metric_categories[metric]

def get_metrics_by_category(category: MetricCategory) -> list[Metric]:
    return [m for m in Metric if _metric_categories[m] == category]

def get_tracked_metrics() -> list[Metric]:
    return get_metrics_by_category(MetricCategory.TRACKED)

# Initialize the categories after all classes are defined
_init_metric_categories()