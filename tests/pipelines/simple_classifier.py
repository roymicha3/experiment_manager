from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from experiment_manager.common.common import Metric, Level
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment

# 1. Define the Model (MLP)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    def save(self, file_path):
        torch.save(self, open(file_path, 'wb'))

@YAMLSerializable.register("SimpleClassifierPipeline")
class SimpleClassifierPipeline(Pipeline, YAMLSerializable):
    """A simple pipeline that trains a MLP classifier."""

    def __init__(self, config: DictConfig, env: Environment):
        super().__init__(env)
        self.config = config
        self.n_samples = config.get('n_samples', 1000)
        self.n_features = config.get('n_features', 20)
        self.n_classes = config.get('n_classes', 2)
        self.test_size = config.get('test_size', 0.2)
        self.val_size = config.get('val_size', 0.2) # Added validation size
        self.random_state = config.get('random_state', 42)
        self.batch_size = config.get('batch_size', 64)  # Added batch size
        self.learning_rate = config.get('learning_rate', 0.001) #Add learning rate
        self.epochs = config.pipeline.epochs #add epochs
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate synthetic data
        self.X, self.y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_classes=self.n_classes,
            random_state=self.random_state
        )
        self.X = torch.from_numpy(self.X).float() #convert to tensor
        self.y = torch.from_numpy(self.y).long()  #convert to tensor

        # Split data into train, validation, and test
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # Split train_val into train and val (use tensor slicing, not random_split)
        train_size = int((1 - self.val_size) * len(X_train_val))
        val_size = len(X_train_val) - train_size
        self.X_train = X_train_val[:train_size]
        self.X_val = X_train_val[train_size:]
        self.y_train = y_train_val[:train_size]
        self.y_val = y_train_val[train_size:]
        
        self.X_test = self.X_test.to(self.device) # Move test to device
        self.y_test = self.y_test.to(self.device)
        # Initialize model
        self.model = MLP(self.n_features, config.get('hidden_dims', [128, 64]), self.n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_one_epoch(self, model, train_loader, criterion, optimizer, device):
        """
        Trains the model for one epoch.

        Args:
            model: The neural network model to train.
            train_loader: The DataLoader for the training data.
            criterion: The loss function.
            optimizer: The optimizer.
            device: The device to train on (CPU or GPU).

        Returns:
            The average loss for the epoch.
        """
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss
    
    def evaluate(self, model, data_loader, criterion, device):
        """Evaluate the model on the given data loader

        Args:
            model (nn.Module): the model
            data_loader (DataLoader): dataloader
            criterion: loss
            device: cpu or cuda
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target).sum().item()
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = correct / len(data_loader.dataset)
        return epoch_loss, epoch_acc

    def run(self, config: DictConfig):
        """Run the pipeline."""
        self.env.logger.info("Starting pipeline run")
        self.on_start()
        
        # Create DataLoaders for training, validation, and testing
        train_loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(self.X_val, self.y_val), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(self.X_test, self.y_test), batch_size=self.batch_size, shuffle=False)
        for epoch in range(self.epochs):
            self.on_epoch_start()
            self.env.logger.info(f"Epoch {epoch+1}/{self.epochs}")

            # Train the model for one epoch
            train_loss = self.train_one_epoch(self.model, train_loader, self.criterion, self.optimizer, self.device)

            # Evaluate on validation set
            val_loss, val_acc = self.evaluate(self.model, val_loader, self.criterion, self.device)
            test_loss, test_acc = self.evaluate(self.model, test_loader, self.criterion, self.device)
            # Track metrics
            self.env.tracker_manager.track(Metric.TRAIN_LOSS, train_loss)
            self.env.tracker_manager.track(Metric.VAL_LOSS, val_loss)
            self.env.tracker_manager.track(Metric.VAL_ACC, val_acc)
            self.env.tracker_manager.track(Metric.TEST_ACC, test_acc)

            metrics = {
                Metric.TRAIN_LOSS: train_loss,
                Metric.VAL_LOSS: val_loss,
                Metric.VAL_ACC: val_acc,
                Metric.TEST_ACC: test_acc,
                Metric.NETWORK: self.model
            }
            self.on_epoch_end(epoch, metrics)

        #final test
        test_loss, test_acc = self.evaluate(self.model, test_loader, self.criterion, self.device)
        self.env.tracker_manager.track(Metric.TEST_ACC, test_acc)
        metrics = {
                Metric.TEST_ACC: test_acc,
                Metric.NETWORK: self.model
        }
        self.on_end(metrics)
        return {"test_acc": test_acc}

    def save(self):
        pass

    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls(config, env)
