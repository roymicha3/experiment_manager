"""
MNIST Batch Gradient Tracking Pipeline for testing batch-level functionality.

This pipeline tracks detailed gradient statistics at the batch level:
- Max gradient magnitude per batch
- Min gradient magnitude per batch  
- Mean gradient magnitude per batch
- L2 norm of gradients per batch

Designed to demonstrate professional data science workflow with batch tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from omegaconf import DictConfig
from typing import Dict, Any, Tuple
import os

from experiment_manager.common.common import RunStatus, Metric
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("MNISTBatchGradientPipeline")
class MNISTBatchGradientPipeline(Pipeline, YAMLSerializable):
    """
    Professional MNIST pipeline with comprehensive batch-level gradient tracking.
    
    Features:
    - Perceptron model for MNIST classification
    - Batch-level gradient statistics tracking
    - Multiple optimizers support
    - Proper train/val/test splits
    - Artifact saving (model checkpoints)
    """
    
    def __init__(self, env: Environment, id: int = None):
        super().__init__(env)
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training configuration
        self.epochs = 3
        self.batch_size = 64
        self.learning_rate = 0.01
        self.optimizer_type = 'SGD'
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        pipeline = cls(env, id)
        
        # Override defaults with config values
        if hasattr(config, 'epochs'):
            pipeline.epochs = config.epochs
        if hasattr(config, 'batch_size'):
            pipeline.batch_size = config.batch_size
        if hasattr(config, 'learning_rate'):
            pipeline.learning_rate = config.learning_rate
        if hasattr(config, 'optimizer_type'):
            pipeline.optimizer_type = config.optimizer_type
            
        return pipeline
    
    def _create_real_mnist_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load real MNIST data from torchvision."""
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784 dimensions
        ])
        
        # Create data directory in workspace
        data_dir = os.path.join(self.env.workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Load MNIST datasets
        train_dataset = datasets.MNIST(
            root=data_dir, 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=data_dir, 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Create train/validation split from training data
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def _create_model(self) -> nn.Module:
        """Create a simple perceptron model for MNIST."""
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        return model.to(self.device)
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.optimizer_type.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'sgd':
            return optim.SGD(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def _compute_gradient_stats(self, model: nn.Module) -> Dict[str, float]:
        """Compute detailed gradient statistics for all model parameters."""
        all_grads = []
        
        for param in model.parameters():
            if param.grad is not None:
                all_grads.extend(param.grad.flatten().cpu().numpy())
        
        if not all_grads:
            return {
                "grad_max": 0.0, 
                "grad_min": 0.0, 
                "grad_mean": 0.0, 
                "grad_l2_norm": 0.0,
                "grad_std": 0.0,
                "grad_median": 0.0,
                "grad_99th_percentile": 0.0,
                "grad_sparsity": 0.0
            }
        
        all_grads = np.array(all_grads)
        abs_grads = np.abs(all_grads)
        
        return {
            "grad_max": float(np.max(abs_grads)),
            "grad_min": float(np.min(abs_grads)),
            "grad_mean": float(np.mean(abs_grads)),
            "grad_l2_norm": float(np.linalg.norm(all_grads)),
            "grad_std": float(np.std(abs_grads)),
            "grad_median": float(np.median(abs_grads)),
            "grad_99th_percentile": float(np.percentile(abs_grads, 99)),
            "grad_sparsity": float(np.sum(abs_grads < 1e-6) / len(abs_grads))  # Fraction of near-zero gradients
        }
    
    @Pipeline.batch_wrapper
    def _train_batch(self, batch_idx: int, data: torch.Tensor, target: torch.Tensor, step: int) -> RunStatus:
        """Train on a single batch and track gradient statistics."""
        self.model.train()
        
        # Move data to device
        data, target = data.to(self.device), target.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient statistics BEFORE optimizer step
        grad_stats = self._compute_gradient_stats(self.model)
        
        # Optimizer step
        self.optimizer.step()
        
        # Compute accuracy
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        accuracy = correct / len(target)
        
        # Track batch-level metrics using Metric.CUSTOM for each metric
        batch_metrics_data = {
            "batch_loss": float(loss.item()),
            "batch_acc": float(accuracy),
            **grad_stats  # Include all gradient statistics
        }
        
        # Store metrics for epoch aggregation and tracking
        self._current_batch_metrics = batch_metrics_data
        
        # Track multiple gradient parameters per batch using our new fix
        # This tests the multiple CUSTOM_METRICS fix
        custom_metrics_list = []
        for metric_name, metric_value in batch_metrics_data.items():
            custom_metrics_list.append((metric_name, metric_value))
        
        # Track all metrics at once using the new list approach
        self.env.tracker_manager.track(Metric.CUSTOM, custom_metrics_list, step)
        
        # Also populate batch_metrics for the wrapper (though we're not using track_dict anymore)
        self.batch_metrics.clear()
        self.batch_metrics.update(batch_metrics_data)
        
        return RunStatus.SUCCESS
    
    @Pipeline.epoch_wrapper  
    def _train_epoch(self, epoch_idx: int, model: nn.Module) -> RunStatus:
        """Train for one epoch with batch-level tracking."""
        self.env.logger.info(f"Training epoch {epoch_idx + 1}/{self.epochs}")
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_grad_stats = {
            "grad_max": [], "grad_min": [], "grad_mean": [], "grad_l2_norm": [],
            "grad_std": [], "grad_median": [], "grad_99th_percentile": [], "grad_sparsity": []
        }
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Train on batch (this will automatically create batch tracking)
            batch_status = self._train_batch(batch_idx, data, target, step=epoch_idx * len(self.train_loader) + batch_idx)
            
            if batch_status == RunStatus.FAILED:
                self.env.logger.error(f"Batch {batch_idx} failed")
                return RunStatus.FAILED
            
            # Since we're tracking metrics directly now, we need to store the values for epoch aggregation
            # Let's store the current batch metrics in a temporary variable
            if hasattr(self, '_current_batch_metrics'):
                # Accumulate for epoch statistics
                if "batch_loss" in self._current_batch_metrics:
                    epoch_loss += self._current_batch_metrics["batch_loss"]
                if "batch_acc" in self._current_batch_metrics:
                    epoch_acc += self._current_batch_metrics["batch_acc"]
                for stat in epoch_grad_stats:
                    if stat in self._current_batch_metrics:
                        epoch_grad_stats[stat].append(self._current_batch_metrics[stat])
        
        # Compute epoch averages
        num_batches = len(self.train_loader)
        epoch_metrics = {
            "train_loss": epoch_loss / num_batches,
            "train_acc": epoch_acc / num_batches,
            "train_grad_max_avg": float(np.mean(epoch_grad_stats["grad_max"])),
            "train_grad_min_avg": float(np.mean(epoch_grad_stats["grad_min"])), 
            "train_grad_mean_avg": float(np.mean(epoch_grad_stats["grad_mean"])),
            "train_grad_l2_norm_avg": float(np.mean(epoch_grad_stats["grad_l2_norm"])),
            "train_grad_std_avg": float(np.mean(epoch_grad_stats["grad_std"])),
            "train_grad_median_avg": float(np.mean(epoch_grad_stats["grad_median"])),
            "train_grad_99th_percentile_avg": float(np.mean(epoch_grad_stats["grad_99th_percentile"])),
            "train_grad_sparsity_avg": float(np.mean(epoch_grad_stats["grad_sparsity"]))
        }
        
        # Validation
        val_metrics = self._validate_epoch()
        epoch_metrics.update(val_metrics)
        
        # Track epoch-level metrics
        self.epoch_metrics.update(epoch_metrics)
        
        self.env.logger.info(f"Epoch {epoch_idx + 1} - Loss: {epoch_metrics['train_loss']:.4f}, "
                           f"Acc: {epoch_metrics['train_acc']:.4f}, "
                           f"Val Acc: {epoch_metrics['val_acc']:.4f}")
        
        return RunStatus.SUCCESS
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate the model on validation set."""
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_acc += pred.eq(target).sum().item() / len(target)
        
        return {
            "val_loss": val_loss / len(self.val_loader),
            "val_acc": val_acc / len(self.val_loader)
        }
    
    def _test_model(self) -> Dict[str, float]:
        """Test the final model."""
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                pred = output.argmax(dim=1)
                test_acc += pred.eq(target).sum().item() / len(target)
        
        return {
            "test_loss": test_loss / len(self.test_loader),
            "test_acc": test_acc / len(self.test_loader)
        }
    
    def _save_model_checkpoint(self, epoch: int) -> str:
        """Save model checkpoint as artifact."""
        checkpoint_path = f"model_epoch_{epoch}.pth"
        full_path = self.env.workspace / "artifacts" / checkpoint_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }, full_path)
        
        # Record artifact in tracking system
        artifact_location = str(full_path.relative_to(self.env.workspace))
        self.env.tracker_manager.record_artifact("model_checkpoint", artifact_location)
        
        return str(full_path)
    
    def _save_training_summary(self) -> str:
        """Save training summary as JSON artifact."""
        import json
        from pathlib import Path
        
        summary_path = Path(self.env.workspace) / "artifacts" / "training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "model_config": {
                "architecture": "perceptron",
                "input_size": 784,
                "hidden_size": 128,
                "output_size": 10,
                "parameters": sum(p.numel() for p in self.model.parameters())
            },
            "training_config": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "optimizer": self.optimizer_type,
                "device": str(self.device)
            },
            "data_info": {
                "dataset": "MNIST",
                "train_batches": len(self.train_loader),
                "val_batches": len(self.val_loader),
                "test_batches": len(self.test_loader)
            },
            "final_metrics": dict(self.run_metrics) if hasattr(self, 'run_metrics') else {}
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Record artifact in tracking system
        artifact_location = str(summary_path.relative_to(self.env.workspace))
        self.env.tracker_manager.record_artifact("training_summary", artifact_location)
        
        return str(summary_path)
    
    @Pipeline.run_wrapper
    def run(self, config: DictConfig) -> RunStatus:
        """Run the complete MNIST batch gradient tracking experiment."""
        try:
            self.env.logger.info("Starting MNIST Batch Gradient Tracking Pipeline")
            
            # Create data loaders with real MNIST data
            self.train_loader, self.val_loader, self.test_loader = self._create_real_mnist_data()
            self.env.logger.info(f"Loaded real MNIST data - Train: {len(self.train_loader)} batches, "
                               f"Val: {len(self.val_loader)} batches, Test: {len(self.test_loader)} batches")
            
            # Create model and optimizer
            self.model = self._create_model()
            self.optimizer = self._create_optimizer(self.model)
            self.env.logger.info(f"Created model with {sum(p.numel() for p in self.model.parameters())} parameters")
            self.env.logger.info(f"Using optimizer: {self.optimizer_type} with LR: {self.learning_rate}")
            
            # Training loop
            for epoch in range(self.epochs):
                status = self._train_epoch(epoch, self.model)
                if status == RunStatus.FAILED:
                    return RunStatus.FAILED
                
                # Save checkpoint every epoch
                checkpoint_path = self._save_model_checkpoint(epoch)
                self.env.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Final testing
            test_metrics = self._test_model()
            self.run_metrics.update(test_metrics)
            
            # Save training summary
            summary_path = self._save_training_summary()
            self.env.logger.info(f"Saved training summary: {summary_path}")
            
            self.env.logger.info(f"Final test results - Loss: {test_metrics['test_loss']:.4f}, "
                               f"Acc: {test_metrics['test_acc']:.4f}")
            
            return RunStatus.SUCCESS
            
        except Exception as e:
            self.env.logger.error(f"Pipeline failed: {e}")
            return RunStatus.FAILED
