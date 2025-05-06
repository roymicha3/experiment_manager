import os
import csv
from typing import Dict, List, Any
from omegaconf import DictConfig

from experiment_manager.common.common import Level, Metric
from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.pipelines.callbacks.callback import Callback
from experiment_manager.common.common import MetricCategory, get_metric_category


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
        
        self.log_path = os.path.join(self.env.artifact_dir, MetricsTracker.LOG_NAME)
        self.metrics: Dict[str, List[Any]] = {}
        self.env.logger.info(f"Initialized metrics tracker. Log file: {self.log_path}")
        
    def on_start(self) -> None:
        """Called when training starts."""
        self.env.logger.info("Starting metrics tracking")
        self.metrics.clear()  # Reset metrics at start
        
    def on_epoch_end(self, epoch_idx, metrics: Dict[str, Any]) -> bool:
        tracked_metrics = []
        for key, value in metrics.items():
            if get_metric_category(key) == MetricCategory.TRACKED:
                # save it to the metrics dictionary
                if key not in self.metrics:
                    if key == Metric.CUSTOM:
                        if value[0] not in self.metrics:
                            self.metrics[value[0]] = []
                            self.env.logger.debug(f"Started tracking custom metric: {value[0]}")
                    else:
                        self.metrics[key] = []
                    
                    self.env.logger.debug(f"Started tracking metric: {key.value}")
                
                if key == Metric.CUSTOM:
                    self.metrics[value[0]].append(value[1])
                    tracked_metrics.append(f"{value[0]}: {value[1]:.4f}")
                else:
                    self.metrics[key].append(value)
                    tracked_metrics.append(f"{key.value}: {value:.4f}")
        
        if tracked_metrics:
            self.env.logger.info(f"Epoch {epoch_idx} metrics - " + ", ".join(tracked_metrics))
        return True

    def on_end(self, metrics: Dict[str, Any]):
        """Called at the end."""
        self.env.logger.info("Finalizing metrics tracking")
        
        final_metrics = []
        for key, value in metrics.items():
            if get_metric_category(key) == MetricCategory.TRACKED:
                self.metrics[key] = [value]
                final_metrics.append(f"{key}: {value:.4f}")
        
        if final_metrics:
            self.env.logger.info("Final metrics - " + ", ".join(final_metrics))
        
        # Save the metrics to the log path as a CSV file
        try:
            self.env.logger.info(f"Saving metrics to {self.log_path}")
            with open(self.log_path, 'w', newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                # Write the header with a 'type' columnsss
                header = ["type"] + [h.name if isinstance(h, Metric) else h for h in self.metrics.keys()]
                writer.writerow(header)
                self.env.logger.debug(f"Wrote header: {', '.join(str(h) for h in header)}")
                
                # Write the rows
                max_length = max(len(values) for values in self.metrics.values())
                rows_written = 0
                for i in range(max_length):
                    row = ["EPOCH"] + [values[i] if i < len(values) else 'nan' for values in self.metrics.values()]
                    writer.writerow(row)
                    rows_written += 1
                
                self.env.logger.info(f"Successfully wrote {rows_written} rows of metrics data")
            
                # Write a final row with 'FINAL' marker and the latest value for each metric
                final_row = ["FINAL"] + [values[-1] if values else 'nan' for values in self.metrics.values()]
                writer.writerow(final_row)
                self.env.logger.debug(f"Wrote FINAL row: {final_row}")

            self.env.tracker_manager.on_add_artifact(
                level=Level.TRIAL_RUN,
                artifact_path=self.log_path,
                artifact_type="metrics")
        
        except Exception as e:
            self.env.logger.error(f"Failed to save metrics: {str(e)}")

    def get_latest(self, key: str, default: Any = None) -> Any:
        """Get the most recent value for a metric."""
        value = self.metrics.get(key, [default])[-1]
        self.env.logger.debug(f"Retrieved latest value for {key}: {value}")
        return value
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        """
        Create an instance from a DictConfig.
        """
        return cls(env)
