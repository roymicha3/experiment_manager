import os
import csv
from typing import Dict, List, Any
from omegaconf import DictConfig

from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.callback import Callback, Metric, MetricCategory
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("MetricsTracker")
class MetricsTracker(Callback, YAMLSerializable):
    """
    Metrics tracker for storing training statistics
    """
    LOG_NAME = "metrics.log"
    
    def __init__(self, env: Environment):
        super(MetricsTracker, self).__init__()
        super(YAMLSerializable, self).__init__()
        
        self.env = env
        
        self.log_path = os.path.join(self.env.log_dir, MetricsTracker.LOG_NAME)
        self.metrics: Dict[str, List[Any]] = {}
        
    def on_epoch_end(self, epoch_idx, metrics: Dict[str, Any]) -> bool:
        
        for key, value in metrics.items():
            
            if key.category == MetricCategory.TRACKED:
                # save it to the metrics dictionary
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
        
        return True

    def on_end(self, metrics: Dict[str, Any]):
        """Called at the end."""
        
        for key, value in metrics.items():
            metric = Metric(key)
            if metric.category == MetricCategory.TRACKED:
                self.metrics[key] = [value]
        
        # Save the metrics to the log path as a CSV file
        with open(self.log_path, 'w', newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(list(self.metrics.keys()))
            # Write the rows
            max_length = max(len(values) for values in self.metrics.values())
            for i in range(max_length):
                row = [values[i] if i < len(values) else '' for values in self.metrics.values()]
                writer.writerow(row)

    def get_latest(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, [default])[-1]
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        """
        Create an instance from a DictConfig.
        """
        return cls(env)
