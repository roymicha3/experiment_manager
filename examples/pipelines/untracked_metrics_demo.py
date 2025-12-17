"""
Untracked Custom Metrics Demo Pipeline

This example demonstrates the difference between:
- Metric.CUSTOM (tracked by all trackers)
- Metric.CUSTOM_UNTRACKED (only accessible in callbacks, not tracked)

Use cases:
- CUSTOM: Important metrics for visualization, analysis, comparison
- CUSTOM_UNTRACKED: Debug info, temporary calculations, internal state
"""

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from typing import Dict, Any

from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.common.common import Metric, RunStatus
from experiment_manager.pipelines.callbacks.callback import Callback


@YAMLSerializable.register("UntrackedMetricsDemoPipeline")
class UntrackedMetricsDemoPipeline(Pipeline, YAMLSerializable):
    """
    Demo pipeline showing tracked vs untracked custom metrics.
    
    This pipeline demonstrates:
    1. Using CUSTOM for important metrics (tracked in DB, MLflow, TensorBoard)
    2. Using CUSTOM_UNTRACKED for debug/temporary data (only in callbacks)
    3. A custom callback that processes untracked metrics
    """
    
    def __init__(self, env: Environment, epochs: int = 10):
        super().__init__(env)
        self.epochs = epochs
        
        # Simple model for demo
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        # Register custom callback that uses untracked metrics
        self.register_callback(UntrackedMetricsProcessor(env))
        
        self.env.logger.info("📊 Untracked Metrics Demo Pipeline initialized")
        self.env.logger.info("   - CUSTOM metrics will be tracked by all trackers")
        self.env.logger.info("   - CUSTOM_UNTRACKED metrics will only be in callbacks")
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        epochs = config.pipeline.get("epochs", 10)
        return cls(env, epochs)
    
    @Pipeline.run_wrapper
    def run(self, config: DictConfig):
        """Run the demo pipeline."""
        self.env.logger.info(f"\n{'='*60}")
        self.env.logger.info("🚀 Starting Untracked Metrics Demo")
        self.env.logger.info(f"{'='*60}\n")
        
        # Generate dummy data
        X_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))
        
        # Training loop
        for epoch in range(self.epochs):
            self.run_epoch(epoch, X_train, y_train)
        
        self.env.logger.info(f"\n{'='*60}")
        self.env.logger.info("✅ Demo Complete!")
        self.env.logger.info(f"{'='*60}\n")
        self.env.logger.info("Check the following to see the difference:")
        self.env.logger.info("  📊 Tracked metrics → Database, MLflow, TensorBoard")
        self.env.logger.info("  🔍 Untracked metrics → Only in callback logs above")
        
        return RunStatus.SUCCESS
    
    @Pipeline.epoch_wrapper
    def run_epoch(self, epoch_idx: int, X_train, y_train):
        """Run a single epoch demonstrating both metric types."""
        
        # Simple training step
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(X_train)
        loss = self.criterion(outputs, y_train)
        
        loss.backward()
        self.optimizer.step()
        
        # Calculate predictions
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_train).float().mean().item()
        
        # ==================================================================
        # TRACKED CUSTOM METRICS (go to DB, MLflow, TensorBoard, etc.)
        # ==================================================================
        # These are important metrics you want to visualize and track
        self.epoch_metrics[Metric.CUSTOM] = [
            ("model_accuracy", accuracy),
            ("loss_value", loss.item()),
            ("learning_rate", self.optimizer.param_groups[0]['lr']),
            ("prediction_confidence", float(torch.softmax(outputs, dim=1).max(dim=1)[0].mean())),
        ]
        
        # ==================================================================
        # UNTRACKED CUSTOM METRICS (only accessible in callbacks)
        # ==================================================================
        # These are debug/temporary metrics you don't want cluttering your DB
        with torch.no_grad():
            # Debug info about gradients
            grad_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
            
            # Debug info about model internals
            weight_norms = [p.norm().item() for p in self.model.parameters()]
            
            # Temporary calculations
            output_std = outputs.std().item()
            output_mean = outputs.mean().item()
            
            # Internal state tracking
            batch_size = X_train.size(0)
            num_correct = (predictions == y_train).sum().item()
        
        self.epoch_metrics[Metric.CUSTOM_UNTRACKED] = [
            ("debug_grad_norm_mean", float(np.mean(grad_norms))),
            ("debug_grad_norm_max", float(np.max(grad_norms))),
            ("debug_weight_norm_mean", float(np.mean(weight_norms))),
            ("debug_output_std", output_std),
            ("debug_output_mean", output_mean),
            ("temp_batch_size", float(batch_size)),
            ("temp_num_correct", float(num_correct)),
            ("internal_epoch_progress", epoch_idx / self.epochs),
        ]
        
        # Also set standard metrics
        self.epoch_metrics[Metric.TRAIN_LOSS] = loss.item()
        self.epoch_metrics[Metric.TRAIN_ACC] = accuracy
        
        return RunStatus.FINISHED


class UntrackedMetricsProcessor(Callback):
    """
    Custom callback that processes untracked metrics.
    
    This demonstrates how to use CUSTOM_UNTRACKED metrics for:
    - Conditional logging based on debug values
    - Temporary calculations
    - Internal state tracking
    """
    
    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
    
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        """Process both tracked and untracked metrics."""
        
        # Process tracked custom metrics (these go to trackers)
        if Metric.CUSTOM in metrics:
            tracked = metrics[Metric.CUSTOM]
            self.env.logger.info(f"\n  📊 Tracked Custom Metrics (in database):")
            for name, value in tracked:
                self.env.logger.info(f"     • {name}: {value:.4f}")
        
        # Process untracked custom metrics (only here in callback)
        if Metric.CUSTOM_UNTRACKED in metrics:
            untracked = metrics[Metric.CUSTOM_UNTRACKED]
            self.env.logger.info(f"\n  🔍 Untracked Custom Metrics (debug only):")
            for name, value in untracked:
                self.env.logger.info(f"     • {name}: {value:.4f}")
            
            # Example: Use untracked metrics for conditional logic
            debug_metrics = dict(untracked)
            if debug_metrics.get("debug_grad_norm_max", 0) > 10.0:
                self.env.logger.warning("     ⚠️  High gradient norm detected!")
            
            if debug_metrics.get("debug_output_mean", 0) < -1.0:
                self.env.logger.warning("     ⚠️  Unusual output mean!")
        
        self.env.logger.info("")  # Blank line for readability
        
        return False  # Don't stop training
    
    def on_start(self) -> None:
        """Called at pipeline start."""
        self.env.logger.info("\n🎬 UntrackedMetricsProcessor callback started")
        self.env.logger.info("   Will process both CUSTOM and CUSTOM_UNTRACKED metrics\n")
    
    def on_end(self, metrics: Dict[str, Any]):
        """Called at pipeline end."""
        self.env.logger.info("\n🏁 UntrackedMetricsProcessor callback finished")
        self.env.logger.info("   Summary:")
        self.env.logger.info("   • CUSTOM metrics → Saved to all trackers")
        self.env.logger.info("   • CUSTOM_UNTRACKED metrics → Used for debug/logging only\n")


# Factory for the demo pipeline
from experiment_manager.pipelines.pipeline_factory import PipelineFactory

class UntrackedMetricsDemoFactory(PipelineFactory):
    """Factory for creating untracked metrics demo pipelines."""
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        return PipelineFactory.create(name, config, env, id)

