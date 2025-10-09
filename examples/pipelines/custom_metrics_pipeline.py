#!/usr/bin/env python3
"""
Custom Metrics Pipeline Example

This pipeline demonstrates how to use custom batch metrics with multiple key-value pairs.
It shows the proper way to implement custom metrics that work with all trackers.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.common import Metric, Level, RunStatus
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("CustomMetricsTestPipeline")
class CustomMetricsTestPipeline(Pipeline, YAMLSerializable):
    """
    Example pipeline that demonstrates custom batch metrics with multiple key-value pairs.
    
    This pipeline shows how to:
    1. Use the new list format for custom metrics
    2. Track multiple metrics per batch
    3. Work with all available trackers
    """
    
    def __init__(self, env: Environment, epochs: int = 3, batch_size: int = 32, 
                 learning_rate: float = 0.01, hidden_dim: int = 64):
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        
        # Create a simple model for testing
        self.model = self._create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Create synthetic data
        self._create_synthetic_data()
        
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls(
            env,
            epochs=config.pipeline.get('epochs', 3),
            batch_size=config.pipeline.get('batch_size', 32),
            learning_rate=config.pipeline.get('learning_rate', 0.01),
            hidden_dim=config.pipeline.get('hidden_dim', 64)
        )
    
    def _create_model(self):
        """Create a simple neural network for testing."""
        return nn.Sequential(
            nn.Linear(10, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        )
    
    def _create_synthetic_data(self):
        """Create synthetic training data."""
        # Generate random data
        np.random.seed(42)
        X = np.random.randn(1000, 10).astype(np.float32)
        y = np.random.randint(0, 2, 1000).astype(np.int64)
        
        # Split data
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        self.X_train = torch.tensor(X[:train_size])
        self.y_train = torch.tensor(y[:train_size])
        self.X_val = torch.tensor(X[train_size:train_size + val_size])
        self.y_val = torch.tensor(y[train_size:train_size + val_size])
        self.X_test = torch.tensor(X[train_size + val_size:])
        self.y_test = torch.tensor(y[train_size + val_size:])
    
    @Pipeline.run_wrapper
    def run(self, config: DictConfig):
        """Main run method with proper pipeline wrapper."""
        self.env.logger.info("Starting Custom Metrics Test Pipeline")
        self.env.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Training loop with all decorators
        for epoch in range(self.epochs):
            self.env.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Run epoch with proper decorator
            self.run_epoch(
                epoch, 
                self.model, 
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=torch.device('cpu')
            )
        
        # Final evaluation
        test_loss, test_acc = self._evaluate(self.model, test_loader, self.criterion)
        
        # Set final run metrics
        self.run_metrics = {
            Metric.TEST_LOSS: test_loss,
            Metric.TEST_ACC: test_acc,
            Metric.NETWORK: self.model
        }
        
        self.env.logger.info(f"Final test results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        return RunStatus.SUCCESS
    
    @Pipeline.epoch_wrapper
    def run_epoch(self, epoch_idx: int, model, *args, **kwargs):
        """Run a single epoch with proper epoch wrapper."""
        train_loader = kwargs["train_loader"]
        val_loader = kwargs["val_loader"]
        criterion = kwargs["criterion"]
        optimizer = kwargs["optimizer"]
        device = kwargs["device"]
        
        # Training phase
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Run batch with proper decorator
            self.run_batch(
                batch_idx,
                model,
                epoch_idx=epoch_idx,
                data=data,
                target=target,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )
            
            # Calculate loss for epoch metrics
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                total_train_loss += loss.item()
                num_batches += 1
        
        # Validation phase
        val_loss, val_acc = self._evaluate(model, val_loader, criterion)
        
        # Set epoch metrics
        self.epoch_metrics = {
            Metric.TRAIN_LOSS: total_train_loss / num_batches,
            Metric.VAL_LOSS: val_loss,
            Metric.VAL_ACC: val_acc,
            Metric.NETWORK: model
        }
        
        self.env.logger.info(f"Epoch {epoch_idx + 1} - Train Loss: {total_train_loss / num_batches:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return RunStatus.FINISHED
    
    @Pipeline.batch_wrapper
    def run_batch(self, batch_idx: int, model, *args, **kwargs):
        """Run a single batch with custom metrics using the new list format."""
        epoch_idx = kwargs["epoch_idx"]
        data = kwargs["data"]
        target = kwargs["target"]
        criterion = kwargs["criterion"]
        optimizer = kwargs["optimizer"]
        device = kwargs["device"]
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate custom metrics with multiple key-value pairs
        with torch.no_grad():
            # Get gradient statistics
            grad_norms = []
            grad_means = []
            grad_stds = []
            grad_mins = []
            grad_maxs = []
            
            for param in model.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
                    grad_means.append(param.grad.mean().item())
                    grad_stds.append(param.grad.std().item())
                    grad_mins.append(param.grad.min().item())
                    grad_maxs.append(param.grad.max().item())
            
            # Calculate predictions and accuracy
            predictions = torch.argmax(output, dim=1)
            correct = (predictions == target).sum().item()
            accuracy = correct / len(target)
            
            # Calculate additional custom metrics
            output_std = output.std().item()
            output_mean = output.mean().item()
            loss_std = loss.item()  # Single value for this batch
            
        # Set batch metrics using the new list format for custom metrics
        self.batch_metrics = {
            Metric.TRAIN_LOSS: loss.item(),
            Metric.NETWORK: model,
            Metric.CUSTOM: [
                ("gradient_norm_mean", np.mean(grad_norms)),
                ("gradient_norm_std", np.std(grad_norms)),
                ("gradient_mean_mean", np.mean(grad_means)),
                ("gradient_std_mean", np.mean(grad_stds)),
                ("gradient_min_mean", np.mean(grad_mins)),
                ("gradient_max_mean", np.mean(grad_maxs)),
                ("batch_accuracy", accuracy),
                ("output_std", output_std),
                ("output_mean", output_mean),
                ("loss_std", loss_std),
                ("epoch_batch", f"{epoch_idx}_{batch_idx}"),
                ("learning_rate", self.learning_rate)
            ]
        }
        
        return RunStatus.FINISHED
    
    def _evaluate(self, model, data_loader, criterion):
        """Evaluate the model on given data loader."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                pred = torch.argmax(output, dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy


def main():
    """Run the custom metrics example."""
    print("=== Custom Metrics Pipeline Example ===")
    print("This example demonstrates custom batch metrics with multiple key-value pairs.")
    print("It shows how to use the new list format for custom metrics that works with all trackers.")
    print()
    
    # This would typically be run through the experiment framework
    print("To run this example:")
    print("1. Use the configuration files in examples/configs/custom_metrics_test/")
    print("2. Run through the experiment framework with proper trackers")
    print("3. Check the database and MLflow logs for the custom metrics")
    print()
    print("The pipeline tracks the following custom metrics per batch:")
    print("- gradient_norm_mean: Mean gradient norm across all parameters")
    print("- gradient_norm_std: Standard deviation of gradient norms")
    print("- gradient_mean_mean: Mean of gradient means")
    print("- gradient_std_mean: Mean of gradient standard deviations")
    print("- gradient_min_mean: Mean of gradient minimums")
    print("- gradient_max_mean: Mean of gradient maximums")
    print("- batch_accuracy: Accuracy for this batch")
    print("- output_std: Standard deviation of model outputs")
    print("- output_mean: Mean of model outputs")
    print("- loss_std: Loss value for this batch")
    print("- epoch_batch: Identifier for epoch and batch")
    print("- learning_rate: Current learning rate")


if __name__ == "__main__":
    main()
