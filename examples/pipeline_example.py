import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import numpy as np
from abc import ABC
from typing import Dict, Any, List
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.pipelines.callbacks.callback import Metric, Callback
from experiment_manager.pipelines.callbacks.callback_factory import CallbackFactory
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment

class SingleLayerPerceptron(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(SingleLayerPerceptron, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        return self.linear(x)

@YAMLSerializable.register("TrainingPipeline")
class TrainingPipeline(Pipeline, YAMLSerializable):
    """
    Class that is responsible for the training of the Network
    """
    def __init__(self, 
                 epochs: int, 
                 batch_size: int, 
                 validation_split: float, 
                 test_split: float,
                 env: Environment,
                 shuffle: bool = True,
                 id: int = None):
        
        # Initialize both parent classes
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.id = id
        
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        """
        Create a training pipeline from configuration.
        """
        pipeline = cls(
            config.epochs,
            config.batch_size, 
            config.validation_split,
            config.test_split,
            env,
            config.shuffle,
            id)
        
        # Register callbacks if specified in config
        if hasattr(config, 'callbacks'):
            for callback_config in config.callbacks:
                callback = CallbackFactory.create(callback_config.type, callback_config, env)
                pipeline.register_callback(callback)
            
        return pipeline
    
    def run(self, config: DictConfig):
        """
        Train the model using the provided data loader.
        """
        # Set device
        device = torch.device(self.env.config.device if torch.cuda.is_available() else "cpu")
        self.env.logger.info(f"Using device: {device}")
        
        # Load MNIST dataset
        data_dir = os.path.join(self.env.workspace, "outputs", "data")
        os.makedirs(data_dir, exist_ok=True)
        self.env.logger.info(f"Downloading MNIST dataset to {data_dir}")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        
        # Split into train and validation
        train_size = int((1 - self.validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle)
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False)
        
        # Create network
        network = SingleLayerPerceptron().to(device)
        self.env.logger.info(f"Created network: {network}")
        
        # Create optimizer and criterion
        optimizer = torch.optim.Adam(network.parameters())
        criterion = nn.CrossEntropyLoss()
        
        status = "completed"
        
        # Signal start to callbacks
        self.on_start()
        
        for epoch in range(self.epochs):
            network.train()
            correct_predictions = 0
            total_predictions = 0
            total_loss = 0.0
 
            # Training loop
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch [{epoch+1}/{self.epochs}]")
            
            for batch_idx, (inputs, labels) in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                running_loss = loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                total = labels.size(0)
                
                correct_predictions += correct
                total_predictions += total
                total_loss += running_loss
                
                # Update progress bar
                accuracy = 100 * correct_predictions / total_predictions
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)
            
            # Compute validation metrics
            val_loss, val_accuracy = self.evaluate(network, criterion, val_loader, device)
            
            # Send metrics to callbacks
            metrics = {
                Metric.VAL_LOSS: val_loss,
                Metric.VAL_ACC: val_accuracy,
                Metric.NETWORK: network
            }
            
            stop_flag = self.on_epoch_end(epoch, metrics)
            if stop_flag:
                status = "stopped"
                self.env.logger.info("Early stopping triggered")
                break
            
            # Print epoch summary
            self.env.logger.info(f"[Epoch {epoch + 1}] Loss: {val_loss:.3f}, Accuracy: {val_accuracy:.2f}%")

            if val_accuracy >= 99.9:
                self.env.logger.info(f"Early stopping at epoch {epoch + 1} due to 100% validation accuracy.")
                break
        
        # Signal end to callbacks
        self.on_end({Metric.STATUS: status})

    def evaluate(self, network, criterion, dataloader, device):
        """
        Compute the loss and accuracy metrics over the given dataset.

        Args:
            network: The neural network
            criterion: The loss function
            dataloader: The dataloader to evaluate on
            device: The device to run evaluation on
            
        Returns:
            Tuple of (average_loss, overall_accuracy)
        """
        network.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        label_correct = {}
        label_total = {}

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Update per-label accuracy metrics
                for label in labels.unique():
                    label_val = label.item()
                    label_mask = (labels == label_val)
                    
                    # Get current counts or initialize to 0
                    current_correct = label_correct.get(label_val, 0)
                    current_total = label_total.get(label_val, 0)
                    
                    # Update counts
                    matches = (predicted[label_mask] == label_val).sum().item()
                    total = label_mask.sum().item()
                    
                    label_correct[label_val] = current_correct + matches
                    label_total[label_val] = current_total + total

        # Calculate average loss and overall accuracy
        average_loss = total_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_predictions

        # Calculate and log per-label accuracy
        for label_val in label_correct:
            if label_total[label_val] > 0:
                label_accuracy = 100 * label_correct[label_val] / label_total[label_val]
                self.env.logger.info(f"Accuracy for label {label_val}: {label_accuracy:.2f}%")

        return average_loss, accuracy
    
    def save(self, file_path):
        """
        Save the training pipeline configuration to YAML.
        """
        config = DictConfig({
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'test_split': self.test_split,
            'shuffle': self.shuffle
        })
        OmegaConf.save(config, file_path)