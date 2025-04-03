from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any



class MetricCategory(Enum):
    TRACKED = auto()
    UNTRACKED = auto()

class Metric(Enum):
    # Tracked metrics
    EPOCH = ("epoch", MetricCategory.TRACKED)
    TEST_ACC = ("test_acc", MetricCategory.TRACKED)
    TEST_LOSS = ("test_loss", MetricCategory.TRACKED)
    VAL_ACC = ("val_acc", MetricCategory.TRACKED)
    VAL_LOSS = ("val_loss", MetricCategory.TRACKED)

    # Untracked metrics
    NETWORK = ("network", MetricCategory.UNTRACKED)
    DATA = ("data", MetricCategory.UNTRACKED)
    STATUS = ("status", MetricCategory.UNTRACKED)

    def __init__(self, value: str, category: MetricCategory):
        self._value_ = value
        self.category = category
        
    @classmethod
    def get(cls, value):
        for metric in cls:
            if metric.value == value:
                return metric
        
        raise ValueError

    @classmethod
    def get_metrics_by_category(cls, category: MetricCategory) -> Dict[str, 'Metric']:
        return {metric.value: metric for metric in cls if metric.category == category}

    @classmethod
    def tracked(cls) -> Dict[str, 'Metric']:
        return cls.get_metrics_by_category(MetricCategory.TRACKED)

    @classmethod
    def untracked(cls) -> Dict[str, 'Metric']:
        return cls.get_metrics_by_category(MetricCategory.UNTRACKED)


class Callback(ABC):
    @abstractmethod
    def on_start(self) -> None:
        """Called at the start of training."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch_idx, metrics) -> bool:
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_end(self, metrics):
        """Called at the end of training."""
        pass
