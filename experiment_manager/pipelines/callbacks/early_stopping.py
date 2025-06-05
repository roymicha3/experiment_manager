from typing import Dict, Any
from omegaconf import DictConfig

from experiment_manager.common.common import Metric
from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.pipelines.callbacks.callback import Callback

@YAMLSerializable.register("EarlyStopping")
class EarlyStopping(Callback, YAMLSerializable):
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, env: Environment, metric: str = Metric.VAL_LOSS, patience: int = 5, min_delta_percent: float = 0.0, mode: str = "min"):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta_percent (float): Minimum percentage change to qualify as an improvement.
            mode (str): One of "min" or "max". In "min" mode, training stops when metric stops decreasing. 
                       In "max" mode, training stops when metric stops increasing.
        """
        super(EarlyStopping, self).__init__()
        super(YAMLSerializable, self).__init__()
        
        self.env = env
        self.metric = metric
        self.patience = patience
        self.min_delta_percent = min_delta_percent
        self.mode = mode.lower()
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        
        if self.mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got '{mode}'")
        
        self.env.logger.info(f"Early stopping initialized with patience={patience}, min_delta_percent={min_delta_percent}%, mode={self.mode}")

    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        """
        Create an instance from a DictConfig.
        """
        if type(config.metric) == str:
            metric = Metric(config.metric)
        else:
            metric = config.metric
        
        return cls(
            env=env,
            metric=metric,
            patience=config.patience,
            min_delta_percent=config.min_delta_percent,
            mode=getattr(config, 'mode', 'min')  # Default to 'min' if not specified
        )

    def on_start(self) -> None:
        """Called when training starts."""
        self.env.logger.info(f"Starting training with {self.metric} monitoring")
        self.counter = 0
        self.best_metric = None

    def on_epoch_end(self, epoch_idx, metrics: Dict[str, Any]) -> bool:
        """
        Check if validation metric has improved; otherwise increase counter.
        Returns True if training should continue, False if it should stop.
        """
        if self.metric not in metrics:
            self.env.logger.warning(f"Metric {self.metric} not found in metrics")
            return True
        
        current_metric = metrics[self.metric]
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return True
            
        # Calculate improvement based on mode: 'min' means lower is better, 'max' means higher is better
        if self.mode == "min":
            delta = self.best_metric - current_metric  # Positive delta means improvement (metric decreased)
        else:  # mode == "max"
            delta = current_metric - self.best_metric  # Positive delta means improvement (metric increased)
        delta_percent = 100 * delta / abs(self.best_metric)
        
        if delta_percent > self.min_delta_percent:
            self.best_metric = current_metric
            self.counter = 0
            self.env.logger.info(f"Metric improved by {delta_percent:.2f}%")
        else:
            self.counter += 1
            self.env.logger.info(f"No improvement in {self.counter} epochs")
            
        if self.counter >= self.patience:
            self.env.logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            return False
            
        return True

    def on_end(self, metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
        if self.best_metric is not None:
            self.env.logger.info(f"Training finished. Best {self.metric}: {self.best_metric:.4f}")
        else:
            self.env.logger.info("Training finished without recording any metrics")
