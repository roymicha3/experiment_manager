from omegaconf import DictConfig
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from experiment_manager.common.common import Metric, Level
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment
from experiment_manager.common.common import Level, Metric

@YAMLSerializable.register("SimpleClassifierPipeline")
class SimpleClassifierPipeline(Pipeline, YAMLSerializable):
    """A simple pipeline that trains a logistic regression classifier."""

    def __init__(self, config: DictConfig, env: Environment):
        super().__init__(env)
        self.config = config
        self.n_samples = config.get('n_samples', 1000)
        self.n_features = config.get('n_features', 20)
        self.n_classes = config.get('n_classes', 2)
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.env = env
        
        # Generate synthetic data
        self.X, self.y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_classes=self.n_classes,
            random_state=self.random_state
        )
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Initialize model
        self.model = LogisticRegression(random_state=self.random_state)
        
    def run(self, config: DictConfig):
        """Run the pipeline."""
        self.on_start()
        
        for epoch in range(self.config.pipeline.epochs):
            self.on_epoch_start()
            
            # Train model
            self.model.fit(self.X_train, self.y_train)
            
            # Get predictions
            y_pred_train = self.model.predict(self.X_train)
            y_pred_test = self.model.predict(self.X_test)
            
            # Calculate metrics
            train_acc = np.mean(y_pred_train == self.y_train)
            test_acc = np.mean(y_pred_test == self.y_test)
            
            # Track metrics
            self.env.tracker_manager.track(Metric.TRAIN_ACC, train_acc)
            self.env.tracker_manager.track(Metric.TEST_ACC, test_acc)
            
            metrics = {
                Metric.TRAIN_ACC: train_acc,
                Metric.TEST_ACC: test_acc,
                Metric.NETWORK: self.model
            }
            
            self.on_epoch_end(epoch, metrics)
        
        # Get predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_acc = np.mean(y_pred_train == self.y_train)
        test_acc = np.mean(y_pred_test == self.y_test)
        
        # Track metrics
        self.env.tracker_manager.track(Metric.TEST_ACC, test_acc)
        
        # Track per-class accuracy
        test_per_class = {}
        for c in range(self.n_classes):
            mask_test = self.y_test == c
            if np.any(mask_test):
                test_per_class[str(c)] = float(np.mean(y_pred_test[mask_test] == self.y_test[mask_test]))
        
        self.env.tracker_manager.track(Metric.TEST_ACC, test_acc, per_label_val=test_per_class)
        
        metrics = {
            Metric.TRAIN_ACC: train_acc,
            Metric.TEST_ACC: test_acc
        }
        self.on_end(metrics)
        return {"test_acc": test_acc}
    
    def save(self):
        pass
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls(config, env)