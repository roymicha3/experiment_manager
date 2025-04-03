from omegaconf import DictConfig

from experiment_manager.pipelines.callbacks.callback import Callback, Metric
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment

@YAMLSerializable.register("EarlyStopping")
class EarlyStopping(Callback):
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, env: Environment, metric=Metric.VAL_LOSS, patience=5, min_delta_percent=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta_percent (float): Minimum percentage change to qualify as an improvement.
        """
        self.env = env
        self.metric = metric
        self.patience = patience
        self.min_delta_percent = min_delta_percent
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        self.env.logger.info(f"Early stopping initialized with patience={patience}, min_delta_percent={min_delta_percent}%")

    def on_epoch_end(self, epoch_idx, metrics) -> bool:
        """
        Check if validation metric has improved; otherwise increase counter.

        Args:
            metrics (dict): Current metrics including the monitored metric.
        """
        current_metric = metrics[self.metric]
        
        if self.best_metric is None:
            self.best_metric = current_metric
            self.env.logger.info(f"Initial {self.metric.value}: {current_metric:.4f}")
            return False

        percent_change = (self.best_metric - current_metric) / self.best_metric * 100

        if percent_change > self.min_delta_percent:
            self.env.logger.info(f"{self.metric.value} improved from {self.best_metric:.4f} to {current_metric:.4f} ({percent_change:.2f}% change)")
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            self.env.logger.info(f"No improvement in {self.metric.value}: {current_metric:.4f} vs best: {self.best_metric:.4f} ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                self.env.logger.warning(f"Early stopping triggered after {self.patience} epochs without improvement")
                    
        return self.early_stop

    def on_start(self) -> None:
        """Called when training starts."""
        self.env.logger.info(f"Starting training with {self.metric.value} monitoring")

    def on_end(self, metrics) -> None:
        """Called when training ends."""
        if self.early_stop:
            self.env.logger.info(f"Training stopped early. Best {self.metric.value}: {self.best_metric:.4f}")
        else:
            self.env.logger.info(f"Training completed normally. Best {self.metric.value}: {self.best_metric:.4f}")
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        return cls(env, Metric.get(config.metric), config.patience, config.min_delta)
