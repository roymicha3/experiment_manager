import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiment_manager.common.common import Metric, RunStatus
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable


class SingleLayerPerceptron(nn.Module):
    """Simple single layer perceptron for MNIST."""
    
    def __init__(self, input_size=784, hidden_dim=128, num_classes=10):
        super(SingleLayerPerceptron, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


@YAMLSerializable.register("MNISTPerceptronPipeline")
class MNISTPerceptronPipeline(Pipeline, YAMLSerializable):
    """A lightweight MNIST single layer perceptron pipeline for testing."""

    def __init__(self, config: DictConfig, env: Environment):
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self, config)
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.001)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.epochs = config.pipeline.epochs
        self.batch_size = config.pipeline.batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load and prepare MNIST data (using sklearn's fetch_openml for simplicity)
        self._prepare_data()
        
        # Initialize model
        self.model = SingleLayerPerceptron(
            input_size=784, 
            hidden_dim=self.hidden_dim, 
            num_classes=10
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _prepare_data(self):
        """Load and prepare MNIST data."""
        # Use a subset of MNIST for faster testing
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        
        # Use only first 10000 samples for speed
        X = mnist.data[:10000].astype('float32') / 255.0  # Normalize to [0,1]
        y = mnist.target[:10000].astype('int64')
        
        # Split into train/val/test
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )
        
        # Convert to tensors
        self.X_train = torch.from_numpy(self.X_train)
        self.X_val = torch.from_numpy(self.X_val)
        self.X_test = torch.from_numpy(self.X_test)
        self.y_train = torch.from_numpy(self.y_train)
        self.y_val = torch.from_numpy(self.y_val)
        self.y_test = torch.from_numpy(self.y_test)

    @Pipeline.epoch_wrapper
    def run_epoch(self, epoch_idx, model, train_loader, val_loader, criterion, optimizer, device):
        """Train model for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Evaluate on validation set
        val_loss, val_acc = self._evaluate(model, val_loader, criterion, device)
        
        # Store epoch metrics
        self.epoch_metrics = {
            Metric.TRAIN_LOSS: train_loss,
            Metric.TRAIN_ACC: train_acc,
            Metric.VAL_LOSS: val_loss,
            Metric.VAL_ACC: val_acc,
            Metric.CUSTOM: ("epoch", epoch_idx)
        }
        
        return train_loss

    def _evaluate(self, model, data_loader, criterion, device):
        """Evaluate model on given dataset."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = running_loss / len(data_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    @Pipeline.run_wrapper
    def run(self, config: DictConfig):
        """Run the training pipeline."""
        self.env.logger.info("Starting MNIST perceptron training")
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        test_dataset = TensorDataset(self.X_test, self.y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(self.epochs):
            self.env.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            self.run_epoch(epoch, self.model, train_loader, val_loader, self.criterion, self.optimizer, self.device)
        
        # Final test evaluation
        test_loss, test_acc = self._evaluate(self.model, test_loader, self.criterion, self.device)
        
        # Store final metrics in run_metrics for automatic tracking by framework
        # These will be tracked automatically when _on_run_end is called
        self.run_metrics = {
            Metric.TEST_LOSS: test_loss,
            Metric.TEST_ACC: test_acc
        }
        
        self.env.logger.info(f"Final test accuracy: {test_acc:.4f}")
        
        return RunStatus.SUCCESS

    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls(config, env) 