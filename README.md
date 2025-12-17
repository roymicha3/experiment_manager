# Experiment Manager

A flexible and extensible framework for managing machine learning experiments, trials, and pipelines with comprehensive tracking, configuration management, and database integration.

## Why Use Experiment Manager?

Machine learning experimentation is inherently complex and iterative. As ML practitioners, we face numerous challenges:

- **Experiment Chaos**: Running multiple experiments with different hyperparameters becomes disorganized quickly
- **Configuration Sprawl**: Keeping track of what settings produced which results becomes overwhelming
- **Results Tracking**: Collecting and comparing metrics across experiments is time-consuming
- **Reproducibility Issues**: Recreating the exact conditions of a successful experiment is often difficult
- **Resource Management**: Organizing artifacts, checkpoints, and logs can become unwieldy
- **Pipeline Complexity**: Training workflows with early stopping, checkpointing, and metrics tracking require boilerplate code

**Experiment Manager solves these problems** by providing a structured, configurable framework that standardizes the entire experimentation workflow. Instead of writing custom code for each experiment, you define your configurations in YAML and let the framework handle the mechanics.

### Core Value Proposition

1. **Separation of Concerns**: 
   - Write your ML code once, then run it with different configurations
   - Focus on your models and algorithms, not the experimental infrastructure
   - Standardize tracking, logging, and artifact management across all experiments

2. **Experiment Organization**:
   - Hierarchical structure keeps trials, runs, and results organized
   - Automatic directory creation for clean separation of artifacts
   - Built-in support for repeating trials with different random seeds

3. **Comprehensive Tracking**:
   - Automatically capture metrics, parameters, and artifacts
   - Record both high-level results and fine-grained epoch metrics
   - Query and compare results across experiments

4. **Reproducibility**:
   - Configuration-driven approach ensures experiments can be repeated exactly
   - All parameters are versioned and stored with results
   - Database integration preserves complete experiment history

5. **Production Readiness**:
   - Scale from single-machine development to production clusters
   - Integrate with MLflow and other tracking systems
   - Database support for collaborative experiment management

## Real-World Applications

Experiment Manager is particularly valuable for:

- **Hyperparameter Optimization**: Systematically explore model configurations
- **Model Comparison**: Compare different architectures under the same conditions
- **Research Exploration**: Quickly iterate on ideas with minimal boilerplate
- **Team Collaboration**: Share experiment configurations and results
- **Reproducible Research**: Ensure your findings can be independently verified
- **ML Pipeline Development**: Standardize training workflows across projects

## Features

- **Experiment & Trial Management**
  - Hierarchical organization of experiments, trials, and runs
  - Support for trial repetitions with unique outputs
  - Resume capability for partially completed experiments
  - Structured workspace organization for all artifacts

- **Advanced Configuration System**
  - YAML-based configuration with inheritance and merging
  - Base configurations shared across trials
  - Trial-specific overrides and settings
  - Dynamic configuration generation (placeholder replacement, cartesian product)
  - Automatic validation of required fields

- **Comprehensive Tracking**
  - Multi-level logging system (experiment, trial, run)
  - Extensible tracking system with plugin architecture
  - MLflow integration for experiment tracking
  - Metric tracking at experiment, trial, and run levels
  - Artifact management (checkpoints, logs, plots)

- **Database Integration**
  - SQLite support for development
  - MySQL support for production
  - Complete schema for experiment data
  - Relationship tracking between entities
  - Comprehensive querying capabilities

- **Advanced Pipeline System**
  - Structured execution with automatic lifecycle management
  - Built-in decorators (@run_wrapper, @epoch_wrapper) for error handling
  - Comprehensive metric tracking with standardized Metric enum
  - **Tracked vs Untracked Custom Metrics** - Fine-grained control over what gets persisted
  - Extensible callback system with plugin architecture
  - Automatic database persistence and MLflow integration
  - Robust error handling and early stopping support

- **Built-in Training Callbacks**
  - **Early Stopping**: Configurable patience, delta thresholds, and monitoring modes
  - **Checkpoint Callback**: Automatic model saving at specified intervals
  - **Metrics Tracker**: CSV export of training progression and final results
  - **Custom Callbacks**: Easy extensibility via base callback interface
  - **Automatic Integration**: Seamless integration with pipeline decorators



## Installation

### From Source (Development)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/experiment_manager.git
cd experiment_manager
pip install -e .
```

### From Requirements

Install required packages directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create Configuration Files

Set up your experiment with YAML configuration files:

```yaml
# env.yaml - Environment configuration
workspace: "outputs/my_experiment"
verbose: true
debug: false
trackers:
  - type: "mlflow"
    tracking_uri: "sqlite:///mlruns.db"
    experiment_name: "my_experiment"

# experiment.yaml - Experiment configuration
name: my_experiment
id: 1
desc: "Training a model with different hyperparameters"
settings:
  model_type: mlp
  batch_size: 32
  epochs: 10

# base.yaml - Base configuration shared across trials
settings:
  model_type: mlp
  batch_size: 32
  optimizer: adam
  log_level: INFO

# trials.yaml - Trial-specific configurations
- name: small_network
  id: 1
  repeat: 3
  settings:
    hidden_layers: [64, 32]
    learning_rate: 0.001

- name: large_network
  id: 2
  repeat: 3
  settings:
    hidden_layers: [256, 128, 64]
    learning_rate: 0.0001
```



### 2. Create Your Pipeline

First, create a custom pipeline that implements your training logic:

```python
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.common import RunStatus, Metric
from omegaconf import DictConfig

class YourPipeline(Pipeline):
    def __init__(self, env, config: DictConfig):
        super().__init__(env)
        self.config = config
        # Initialize your model, data loaders, optimizers, etc.
        
    @Pipeline.run_wrapper
    def run(self, config):
        """Main training loop with automatic lifecycle management"""
        # Your training logic here
        for epoch in range(config.epochs):
            status = self.run_epoch(epoch)
            if status == RunStatus.FAILED:
                return RunStatus.FAILED
        
        # Store final results in run_metrics for automatic tracking
        final_test_loss, final_test_acc = self.final_test()
        self.run_metrics[Metric.TEST_LOSS] = final_test_loss
        self.run_metrics[Metric.TEST_ACC] = final_test_acc
        
        return RunStatus.COMPLETED
    
    @Pipeline.epoch_wrapper  
    def run_epoch(self, epoch_idx):
        """Single epoch with automatic metric tracking"""
        # Your epoch training logic
        train_loss = self.train_step()  # Your training code
        val_loss, val_acc = self.validate()  # Your validation code
        
        # Store in epoch_metrics - automatically tracked by @epoch_wrapper
        self.epoch_metrics[Metric.TRAIN_LOSS] = train_loss
        self.epoch_metrics[Metric.VAL_LOSS] = val_loss
        self.epoch_metrics[Metric.VAL_ACC] = val_acc
        
        # Tracked custom metrics (saved to DB, MLflow, TensorBoard)
        self.epoch_metrics[Metric.CUSTOM] = [
            ("learning_rate", 0.001),
            ("gradient_norm", 2.5),
        ]
        
        # Untracked custom metrics (only accessible in callbacks, NOT saved)
        self.epoch_metrics[Metric.CUSTOM_UNTRACKED] = [
            ("debug_info", 123.45),
            ("temp_calculation", 0.789),
        ]
        
        return RunStatus.COMPLETED
    
    def train_step(self):
        # Your actual training implementation
        return 0.5  # Example loss value
        
    def validate(self):
        # Your actual validation implementation  
        return 0.3, 0.85  # Example val_loss, val_acc
        
    def final_test(self):
        # Your actual final test implementation
        return 0.25, 0.90  # Example final_test_loss, final_test_acc
```

### 3. Create a Pipeline Factory

Now create a factory to instantiate your pipeline:

```python
from omegaconf import DictConfig
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.environment import Environment
from your_module.your_pipeline import YourPipeline

class YourPipelineFactory(PipelineFactory):
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        # Create and return your pipeline instance based on configuration
        if name == "your_pipeline_name":
            return YourPipeline(env, config)
        # Fall back to default factory for other pipelines
        return PipelineFactory.create(name, config, env, id)
```

### 4. Run Your Experiment

```python
from experiment_manager.experiment import Experiment
from your_module.your_pipeline_factory import YourPipelineFactory

# Configuration directory containing your YAML files
config_dir = "path/to/configs"

# Create experiment from configs
experiment = Experiment.create(config_dir, YourPipelineFactory)

# Run the entire experiment (all trials)
experiment.run()
```

## Core Components

### Environment

The `Environment` class manages workspace directories, logging, and tracking:

```python
from experiment_manager.environment import Environment
from omegaconf import OmegaConf

# Create environment config
env_config = OmegaConf.create({
    "workspace": "outputs/my_experiment",
    "verbose": True,
    "debug": False,
    "trackers": [
        {
            "type": "mlflow",
            "tracking_uri": "sqlite:///mlruns.db",
            "experiment_name": "my_experiment"
        }
    ]
})

# Create environment
env = Environment.from_config(env_config)

# Create child environments for trials
trial_env = env.create_child("trial_1")
```

### Experiment

The `Experiment` class manages experiment execution:

```python
from experiment_manager.experiment import Experiment
from your_module.your_pipeline_factory import YourPipelineFactory

# Create experiment from configuration directory
experiment = Experiment.create("path/to/configs", YourPipelineFactory)

# Run experiment
experiment.run()
```

### Tracker System

The tracking system provides a unified interface for different tracking backends with automatic lifecycle management:

```python
from experiment_manager.trackers.tracker_manager import TrackerManager, TrackScope
from experiment_manager.common.common import Metric, Level

# Access tracker from environment
tracker = env.tracker_manager

# Track individual metrics
tracker.track(Metric.TRAIN_LOSS, 0.456, step=1)
tracker.track(Metric.VAL_ACC, 0.921, step=1)

# Track multiple metrics at once (used by pipelines)
metrics_dict = {
    Metric.TRAIN_LOSS: 0.345,
    Metric.VAL_LOSS: 0.234,
    Metric.VAL_ACC: 0.892
}
tracker.track_dict(metrics_dict, step=2)

# Manual lifecycle management (usually handled by Pipeline decorators)
tracker.on_create(Level.TRIAL_RUN)  # Create tracking context
tracker.on_start(Level.TRIAL_RUN)   # Start tracking
# ... do work ...
tracker.on_end(Level.TRIAL_RUN)     # End tracking

# Context manager for scoped tracking
with TrackScope(tracker, Level.TRIAL_RUN):
    # Tracking context managed automatically
    tracker.track(Metric.TRAIN_LOSS, 0.345, step=2)
    # on_start called on enter, on_end called on exit

# Artifact tracking
tracker.on_add_artifact(
    Level.TRIAL_RUN, 
    artifact_path="/path/to/model.pth",
    artifact_type="checkpoint"
)

# Parameter logging (useful for hyperparameters)
tracker.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "adam"
})

# Checkpoint tracking with model
import torch
model = torch.nn.Linear(10, 1)
tracker.on_checkpoint(
    network=model,
    checkpoint_path="/path/to/checkpoint.pth",
    metrics={Metric.VAL_ACC: 0.95}
)
```

### Pipeline System

The Pipeline system provides a powerful foundation for structured experiment execution with built-in lifecycle management, metric tracking, and extensible callbacks.

#### Core Pipeline Features

- **Automatic Lifecycle Management**: Built-in decorators handle setup, teardown, and error handling
- **Metric Tracking**: Integrated metric collection with automatic database persistence
- **Callback System**: Extensible plugin architecture for custom behavior
- **Error Handling**: Robust error handling with proper cleanup and status reporting
- **Early Stopping**: Built-in support for training interruption via callbacks

#### Pipeline Decorators

The pipeline system uses two key decorators that handle lifecycle management automatically:

```python
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.common import RunStatus, Metric

class YourPipeline(Pipeline):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        
        # Pipeline automatically creates PIPELINE level tracking context
        # Register callbacks for enhanced functionality
        self.register_callback(EarlyStopping(
            env=env, 
            metric=Metric.VAL_LOSS,
            patience=5,
            min_delta_percent=1.0,
            mode="min"
        ))
    
    @Pipeline.run_wrapper
    def run(self, config):
        """
        @run_wrapper automatically handles:
        - Calling _on_run_start() before execution
        - Exception handling and status management
        - Calling _on_run_end() after completion
        - Proper cleanup even on failures
        """
        # Your main training loop
        for epoch in range(config.epochs):
            status = self.run_epoch(epoch, self.model)
            if status == RunStatus.FAILED:
                return RunStatus.FAILED
        return RunStatus.COMPLETED
    
    @Pipeline.epoch_wrapper  
    def run_epoch(self, epoch_idx, model):
        """
        @epoch_wrapper automatically handles:
        - Creating and starting EPOCH level tracking context
        - Tracking all metrics in self.epoch_metrics
        - Calling callback.on_epoch_end() for all registered callbacks
        - Early stopping via StopIteration exception
        - Clearing epoch_metrics for next epoch
        """
        # Your epoch implementation
        train_loss = self.train_step(model)
        val_loss, val_acc = self.validate(model)
        
        # Add metrics to be automatically tracked
        self.epoch_metrics[Metric.TRAIN_LOSS] = train_loss
        self.epoch_metrics[Metric.VAL_LOSS] = val_loss
        self.epoch_metrics[Metric.VAL_ACC] = val_acc
        
        # Metrics are automatically:
        # 1. Tracked to database with epoch context
        # 2. Passed to all callbacks
        # 3. Cleared after epoch completes
        
        return RunStatus.COMPLETED
```

#### Available Metrics

The system includes a comprehensive `Metric` enum for standardized tracking:

```python
from experiment_manager.common.common import Metric

# Tracked metrics (automatically saved to database)
Metric.TRAIN_LOSS      # Training loss
Metric.TRAIN_ACC       # Training accuracy  
Metric.VAL_LOSS        # Validation loss
Metric.VAL_ACC         # Validation accuracy
Metric.TEST_LOSS       # Test loss
Metric.TEST_ACC        # Test accuracy
Metric.LEARNING_RATE   # Learning rate
Metric.CONFUSION       # Confusion matrix
Metric.CUSTOM          # Custom metrics (name, value) tuple

# Untracked metrics (not automatically persisted)
Metric.NETWORK         # Model/network object
Metric.DATA           # Data objects
Metric.LABELS         # Label information
Metric.STATUS         # Status information
```

#### Built-in Callbacks

The system provides several powerful callbacks out of the box:

##### 1. Early Stopping Callback

```python
from experiment_manager.pipelines.callbacks.early_stopping import EarlyStopping

# Monitor validation loss with early stopping
early_stop = EarlyStopping(
    env=env,
    metric=Metric.VAL_LOSS,    # Metric to monitor
    patience=5,                # Epochs to wait without improvement
    min_delta_percent=1.0,     # Minimum % improvement required
    mode="min"                 # "min" for loss, "max" for accuracy
)

# Alternative: Monitor validation accuracy
early_stop_acc = EarlyStopping(
    env=env,
    metric=Metric.VAL_ACC,
    patience=10,
    min_delta_percent=0.5,
    mode="max"                 # Higher accuracy is better
)
```

##### 2. Metrics Tracker Callback

```python
from experiment_manager.pipelines.callbacks.metric_tracker import MetricsTracker

# Automatically save metrics to CSV file
metrics_tracker = MetricsTracker(env=env)
# Saves to {artifact_dir}/metrics.log
# Includes epoch-by-epoch progression and final values
```

##### 3. Checkpoint Callback

```python
from experiment_manager.pipelines.callbacks.checkpoint import CheckpointCallback

# Save model checkpoints at regular intervals
checkpoint = CheckpointCallback(
    interval=5,                # Save every 5 epochs
    env=env
)

# In your epoch_wrapper, include the model:
self.epoch_metrics[Metric.NETWORK] = model  # Model will be saved automatically
```

#### Custom Callbacks

Create custom callbacks by extending the base `Callback` class:

```python
from experiment_manager.pipelines.callbacks.callback import Callback
from typing import Dict, Any

class CustomCallback(Callback):
    def __init__(self, env):
        self.env = env
        
    def on_start(self) -> None:
        """Called when training starts."""
        self.env.logger.info("Training started!")
        
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        """Called after each epoch. Return False to stop training."""
        if epoch_idx > 0 and epoch_idx % 10 == 0:
            self.env.logger.info(f"Milestone: Completed {epoch_idx} epochs")
        return True  # Continue training
        
    def on_end(self, metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
        self.env.logger.info("Training completed!")
```

#### Pipeline Status Management

The pipeline system uses a comprehensive `RunStatus` enum:

```python
from experiment_manager.common.common import RunStatus

RunStatus.RUNNING      # Currently executing
RunStatus.COMPLETED    # Finished successfully  
RunStatus.STOPPED      # Stopped via early stopping
RunStatus.FAILED       # Failed due to error
RunStatus.ABORTED      # Manually aborted
RunStatus.SKIPPED      # Skipped execution
```

#### Error Handling and Early Stopping

The pipeline system handles interruptions gracefully:

```python
# Early stopping is triggered via StopIteration exception
# The @epoch_wrapper catches this and converts it to proper cleanup

# In your callback:
def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
    if should_stop_early():
        return False  # This triggers StopIteration in the pipeline
    return True

# Pipeline automatically:
# 1. Catches StopIteration 
# 2. Sets status to RunStatus.STOPPED
# 3. Calls _on_run_end() for proper cleanup
# 4. Calls callback.on_end() for all callbacks
```

#### Metric Tracking Integration

The pipeline integrates seamlessly with the tracking system:

```python
# Metrics are automatically tracked at the appropriate level
self.epoch_metrics[Metric.TRAIN_LOSS] = 0.345

# This results in:
# 1. Database record at EPOCH level
# 2. MLflow tracking (if configured)
# 3. Callback notification
# 4. Automatic cleanup after epoch
```

### Database Integration

The Experiment Manager includes a comprehensive database system for persistent experiment tracking and analysis. The database supports both SQLite (for development) and MySQL (for production) backends.

### Database Schema

The database consists of 7 core tables and 7 junction tables that provide a complete hierarchical structure for experiment tracking:

#### Core Tables
- **EXPERIMENT**: Stores high-level experiment metadata (id, title, description, timestamps)
- **TRIAL**: Represents specific configurations within experiments (id, name, experiment_id, timestamps)
- **TRIAL_RUN**: Individual executions of trials (id, trial_id, status, timestamps)
- **RESULTS**: Overall results for trial runs (trial_run_id, time)
- **EPOCH**: Individual epochs within trial runs (idx, trial_run_id, time)
- **METRIC**: Stores metric values with support for per-label metrics (id, type, total_val, per_label_val as JSON)
- **ARTIFACT**: Tracks files and objects generated during experiments (id, type, location)

#### Junction Tables
- **EXPERIMENT_ARTIFACT**: Links artifacts to experiments
- **TRIAL_ARTIFACT**: Links artifacts to trials
- **TRIAL_RUN_ARTIFACT**: Links artifacts to trial runs
- **RESULTS_METRIC**: Links metrics to results
- **RESULTS_ARTIFACT**: Links artifacts to results
- **EPOCH_METRIC**: Links metrics to epochs
- **EPOCH_ARTIFACT**: Links artifacts to epochs

This design enables artifact and metric tracking at every level of the experiment hierarchy.

### Database Manager API

The `DatabaseManager` class provides a comprehensive API for database operations:

```python
from experiment_manager.db.manager import DatabaseManager

# Initialize with SQLite for development
db = DatabaseManager(database_path="experiments.db", use_sqlite=True)

# Or MySQL for production
db = DatabaseManager(
    database_path="experiment_db",
    host="localhost",
    user="ml_user",
    password="secure_password"
)

# Create experiment hierarchy
experiment = db.create_experiment("CIFAR-10 Classification", 
                                "Comparing different architectures on CIFAR-10")
trial = db.create_trial(experiment.id, "ResNet-18")
trial_run = db.create_trial_run(trial.id, status="running")

# Record metrics at different levels
metric = db.record_metric(
    total_val=0.87, 
    metric_type="accuracy",
    per_label_val={"airplane": 0.85, "car": 0.89, "bird": 0.82}
)

# Create epoch and link metric
db.create_epoch(epoch_idx=1, trial_run_id=trial_run.id)
db.add_epoch_metric(epoch_idx=1, trial_run_id=trial_run.id, metric_id=metric.id)

# Record and link artifacts
model_artifact = db.record_artifact("model", "/path/to/model.pth")
plot_artifact = db.record_artifact("plot", "/path/to/loss_curve.png")

db.link_experiment_artifact(experiment.id, model_artifact.id)
db.link_epoch_artifact(1, trial_run.id, plot_artifact.id)

# Query data at different levels
experiment_metrics = db.get_experiment_metrics(experiment.id)
trial_artifacts = db.get_trial_artifacts(trial.id)
epoch_artifacts = db.get_epoch_artifacts(1, trial_run.id)

# Update trial status
db.update_trial_run_status(trial_run.id, "completed")
```

### Error Handling

The database module provides robust error handling with custom exceptions:

- `DatabaseError`: Base class for all database-related errors
- `ConnectionError`: Errors connecting to the database
- `QueryError`: Errors executing database queries

### Advanced Querying

The database structure supports complex queries for experiment analysis:

```python
# Get all metrics for experiments with specific criteria
experiment_metrics = db.get_experiment_metrics(experiment_id)

# Retrieve artifacts at any level of the hierarchy
experiment_artifacts = db.get_experiment_artifacts(experiment_id)
trial_run_artifacts = db.get_trial_run_artifacts(trial_run_id)

# Track experiment progress through trial run status
db.update_trial_run_status(trial_run_id, "completed")
```

### Flexible Metric Storage

The system supports both simple scalar metrics and complex per-label metrics:

```python
# Simple metric
accuracy_metric = db.record_metric(0.95, "accuracy")

# Complex per-label metric for multi-class classification
detailed_metric = db.record_metric(
    total_val=0.87,
    metric_type="f1_score",
    per_label_val={
        "class_0": 0.89,
        "class_1": 0.85,
        "class_2": 0.87
    }
)
```

### Multi-Level Artifact Management

Artifacts can be associated with any level of the experiment hierarchy:

```python
# Experiment-level artifacts (configs, final models)
config_artifact = db.record_artifact("config", "/exp/config.yaml")
db.link_experiment_artifact(experiment.id, config_artifact.id)

# Trial-level artifacts (trial-specific outputs)
trial_log = db.record_artifact("log", "/trial/output.log")
db.link_trial_artifact(trial.id, trial_log.id)

# Epoch-level artifacts (checkpoints, plots)
checkpoint = db.record_artifact("checkpoint", "/epoch/model_epoch_5.pth")
db.link_epoch_artifact(5, trial_run.id, checkpoint.id)
```

## Understanding the Key Systems

### Hierarchical Levels and Context Management

The experiment manager operates on a six-level hierarchy, each with specific tracking contexts:

```python
from experiment_manager.common.common import Level

Level.EXPERIMENT    # Top-level experiment (contains multiple trials)
Level.TRIAL         # Individual trial configuration (contains trial runs)
Level.TRIAL_RUN     # Single execution of a trial (contains epochs)
Level.PIPELINE      # Pipeline execution context
Level.EPOCH         # Individual training epoch (contains batches)
Level.BATCH         # Individual batch processing (optional fine-grained tracking)
```

#### Automatic Context Management

The pipeline system automatically manages contexts at appropriate levels:

```python
class YourPipeline(Pipeline):
    def __init__(self, env):
        super().__init__(env)
        # Automatically creates Level.PIPELINE context
        
    @Pipeline.run_wrapper
    def run(self, config):
        # @run_wrapper manages Level.PIPELINE start/end
        for epoch in range(config.epochs):
            self.run_epoch(epoch, model)
            
    @Pipeline.epoch_wrapper 
    def run_epoch(self, epoch_idx, model):
        # @epoch_wrapper manages Level.EPOCH create/start/end
        # Automatically tracks self.epoch_metrics at EPOCH level
        self.epoch_metrics[Metric.TRAIN_LOSS] = train_loss
```

#### Manual Context Management

For custom implementations, you can manually manage tracking contexts:

```python
# Create and start experiment-level tracking
tracker.on_create(Level.EXPERIMENT)
tracker.on_start(Level.EXPERIMENT)

# Trial-level tracking
tracker.on_create(Level.TRIAL)
tracker.on_start(Level.TRIAL)

# Track metrics at appropriate levels
tracker.track(Metric.TRAIN_LOSS, 0.345, step=epoch)

# End contexts (automatic cleanup)
tracker.on_end(Level.TRIAL)
tracker.on_end(Level.EXPERIMENT)
```

### The Tracker System in Depth

The tracking system is designed to record all aspects of your experiments across multiple levels:

- **Metrics Tracking**: 
  - Record scalar values (loss, accuracy, F1 score)
  - Store per-class or per-label breakdowns with JSON support
  - Track metrics over time (by epoch or step)
  - Automatic categorization (tracked vs untracked metrics)

- **Lifecycle Events**:
  - Automatically capture start/end of experiments, trials, runs, epochs
  - Create hierarchical relationships between entities
  - Enable proper organization in visualization tools
  - Support for nested contexts and proper cleanup

- **Artifact Management**:
  - Store models, checkpoints, plots, and other files
  - Link artifacts to the appropriate experiment level
  - Enable easy retrieval and comparison
  - Automatic artifact registration via callbacks

- **Plugin Architecture**:
  - Use MLflow for visualization and comparison
  - Store in database for persistence and querying
  - Extend with custom trackers for specific needs
  - Multiple tracker support with unified interface

The tracker uses both decorator-based automation and manual Context Manager patterns for flexible usage.

The Experiment Manager operates on a sophisticated hierarchical level system that provides structured context for both tracking and execution. Understanding these levels is crucial for effective use of the framework.

### The Six Hierarchical Levels

The system defines six distinct levels, each serving a specific purpose in the experiment lifecycle:

```python
class Level(Enum):
    EXPERIMENT    = 0  # Top-level experiment container
    TRIAL         = 1  # Individual experiment configurations  
    TRIAL_RUN     = 2  # Single execution of a trial (with repetition support)
    PIPELINE      = 3  # Execution context for training workflows
    EPOCH         = 4  # Individual training epochs within a pipeline
    BATCH         = 5  # Individual batch processing (finest granularity)
```

#### Level Purposes and Responsibilities

1. **EXPERIMENT Level (0)**:
   - **Purpose**: Overall experiment coordination and high-level metrics
   - **Responsibilities**: Experiment metadata, cross-trial comparisons, final results
   - **Tracked Items**: Experiment configuration, overall success/failure, summary metrics
   - **Directory**: `workspace/experiment_name/`

2. **TRIAL Level (1)**:
   - **Purpose**: Individual experimental configurations and parameter sets
   - **Responsibilities**: Trial-specific settings, parameter tracking, trial-level results
   - **Tracked Items**: Hyperparameters, trial configuration, trial outcomes
   - **Directory**: `workspace/experiment_name/trials/trial_1/`

3. **TRIAL_RUN Level (2)**:
   - **Purpose**: Individual executions of trials (supports repetition with different seeds)
   - **Responsibilities**: Run-specific metrics, status tracking, individual run artifacts
   - **Tracked Items**: Run status, random seeds, run-specific results
   - **Directory**: `workspace/experiment_name/trials/trial_1/run_1/`

4. **PIPELINE Level (3)**:
   - **Purpose**: Training workflow execution context
   - **Responsibilities**: Pipeline configuration, training process metrics, workflow status
   - **Tracked Items**: Training duration, pipeline configuration, workflow events
   - **Context**: Where callbacks operate and pipeline lifecycle is managed

5. **EPOCH Level (4)**:
   - **Purpose**: Individual training epochs and iterative learning progress
   - **Responsibilities**: Per-epoch metrics, learning curves, checkpoint management
   - **Tracked Items**: Loss values, accuracy metrics, learning rates, model checkpoints
   - **Context**: Fine-grained training progress and intermediate results

6. **BATCH Level (5)**:
   - **Purpose**: Finest granularity for batch-level processing (optional)
   - **Responsibilities**: Batch-specific metrics, detailed debugging information
   - **Tracked Items**: Batch loss, gradient information, detailed debugging data
   - **Context**: Detailed training diagnostics when needed

### Callbacks vs Trackers: Context Distinction

The key insight is that **callbacks and trackers operate in different contexts** with different responsibilities:

#### Callbacks: Pipeline Context

```python
class Callback(ABC):
    def on_start(self) -> None:
        """Called when training starts."""
        
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        """Called at the end of each epoch."""
        
    def on_end(self, metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
```

**Callbacks operate in PIPELINE context** and focus on:
- **Training workflow intervention**: Early stopping, learning rate scheduling, model pruning
- **Training-specific logic**: Checkpoint saving, model evaluation, metric computation
- **Pipeline lifecycle management**: Setup, epoch processing, cleanup
- **Immediate decision making**: Whether to continue training, adjust parameters
- **Training-specific artifacts**: Model weights, training plots, optimization state

#### Trackers: Experiment Context

```python
class Tracker(ABC):
    def on_create(self, level: Level, *args, **kwargs):
        """Called when an entity at the specified level is created."""
        
    def on_start(self, level: Level, *args, **kwargs):
        """Called when an entity at the specified level starts."""
        
    def on_end(self, level: Level, *args, **kwargs):
        """Called when an entity at the specified level ends."""
```

**Trackers operate across ALL levels** and focus on:
- **Experiment-wide data collection**: Metrics, artifacts, metadata across all levels
- **Hierarchical data organization**: Maintaining relationships between experiment entities
- **Persistent storage**: Database records, log files, external tracking systems
- **Cross-experiment analysis**: Comparing results across different experiments
- **Comprehensive audit trail**: Complete history of all experiment activities

### Design Choice: Separation of Concerns

This separation serves several important design purposes:

1. **Single Responsibility Principle**:
   - Callbacks focus on **training logic and intervention**
   - Trackers focus on **data collection and persistence**

2. **Different Lifecycles**:
   - Callbacks operate within the bounded pipeline execution
   - Trackers span the entire experiment hierarchy from creation to completion

3. **Different Data Requirements**:
   - Callbacks need immediate access to training state for decision making
   - Trackers need structured metadata for long-term storage and analysis

4. **Flexibility and Extensibility**:
   - Training logic can be customized without affecting tracking
   - Tracking backends can be changed without modifying training code

### Context Managers and Automatic Tracking

The system uses the `TrackScope` context manager to automatically handle level transitions:

```python
# Experiment level tracking
with TrackScope(tracker_manager, level=Level.EXPERIMENT):
    # Automatically calls tracker.on_start(Level.EXPERIMENT)
    for trial_config in trial_configs:
        # Trial level tracking
        with TrackScope(tracker_manager, level=Level.TRIAL, trial_name=trial.name):
            # Automatically calls tracker.on_start(Level.TRIAL)
            trial.run()
            # Automatically calls tracker.on_end(Level.TRIAL)
    # Automatically calls tracker.on_end(Level.EXPERIMENT)
```

This pattern ensures:
- **Proper lifecycle management**: Automatic start/end event tracking
- **Hierarchical consistency**: Correct level transitions and context maintenance
- **Error resilience**: Cleanup occurs even if exceptions are raised
- **Simplified user code**: No manual tracking lifecycle management required

### Level-Aware Artifact and Metric Association

Different levels can have associated artifacts and metrics:

```python
# Experiment-level artifacts (configs, final models)
tracker.on_add_artifact(Level.EXPERIMENT, "final_model.pth", artifact_type="model")

# Trial-level artifacts (trial-specific outputs)  
tracker.on_add_artifact(Level.TRIAL, "trial_config.yaml", artifact_type="config")

# Epoch-level artifacts (checkpoints, training plots)
tracker.on_add_artifact(Level.EPOCH, "checkpoint_epoch_5.pth", artifact_type="checkpoint")

# Epoch-level metrics (training progress)
tracker.track(Metric.TRAIN_LOSS, 0.345, step=epoch_idx)  # Links to current epoch
```

This hierarchical organization enables:
- **Precise artifact location**: Know exactly where each artifact belongs
- **Structured querying**: Retrieve artifacts/metrics by level and relationship
- **Logical organization**: Mirror the conceptual experiment structure in storage
- **Granular access control**: Manage permissions and access by experiment level

## Directory Structure

The experiment manager creates an organized directory structure:

```
workspace/
├── experiment_name/
│   ├── configs/           # Configuration files
│   │   ├── experiment.yaml
│   │   ├── base.yaml
│   │   └── trials.yaml
│   ├── logs/              # Experiment-level logs
│   │   └── metrics.log
│   ├── artifacts/         # Experiment-level artifacts


│   └── trials/            # Trial directories
│       ├── trial_1/
│       │   ├── configs/   # Trial-specific configs
│       │   ├── logs/      # Trial-specific logs
│       │   ├── artifacts/ # Trial-specific artifacts
│       │   └── run_1/     # Trial run directories
│       │       ├── logs/
│       │       └── artifacts/
│       └── trial_2/
│           ├── configs/
│           ├── logs/
│           ├── artifacts/
│           └── run_1/
│               ├── logs/
│               └── artifacts/
```

## Advanced Usage

### Factory-Serializable Pattern

The Experiment Manager uses a powerful factory-serializable pattern throughout the codebase to provide flexible, configuration-driven object creation. This pattern combines the Factory pattern with serialization support, enabling dynamic instantiation of components from configuration files.

#### Core Concept

1. **Registration**: Components register themselves with a type identifier
2. **Factory Creation**: Factory classes create instances based on type identifier from config
3. **Serialization**: Objects can be serialized to/from YAML configurations
4. **Configuration-Driven**: All components are configurable through YAML

#### Base Implementation

The pattern is implemented through these core components:

```python
# serializable.py
from omegaconf import DictConfig
from typing import Dict, Type, Any, ClassVar

class YAMLSerializable:
    """Base class for objects that can be serialized to/from YAML."""
    _registry: ClassVar[Dict[str, Type['YAMLSerializable']]] = {}
    
    def __init__(self, config: DictConfig = None):
        self.config = config or {}
    
    @classmethod
    def register(cls, type_name: str):
        """Decorator to register a class with a type name."""
        def decorator(subclass):
            cls._registry[type_name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def get_class(cls, type_name: str) -> Type['YAMLSerializable']:
        """Get class by type name."""
        if type_name not in cls._registry:
            raise ValueError(f"Unknown type: {type_name}")
        return cls._registry[type_name]
    
    @classmethod
    def from_config(cls, config: DictConfig):
        """Create instance from config."""
        return cls(config)
```

#### Component Implementation and Registration

Components register themselves with the serializable registry:

```python
# Import the base classes
from experiment_manager.common.serializable import YAMLSerializable
from omegaconf import DictConfig

# Register the component with a type name
@YAMLSerializable.register("MyComponent")
class MyComponent(YAMLSerializable):
    def __init__(self, config: DictConfig = None):
        super().__init__(config)
        self.param1 = config.get("param1", "default")
        self.param2 = config.get("param2", 0)
    
    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(config)
    
    def do_something(self):
        print(f"Doing something with {self.param1} and {self.param2}")
```

#### Factory Implementation

Factories typically specialize in creating specific types of components and ensure all needed classes are imported:

```python
# Import the base factory and all component types this factory will create
from experiment_manager.common.factory import Factory
from experiment_manager.common.serializable import YAMLSerializable
from omegaconf import DictConfig

# Import all component classes to ensure they're registered
from experiment_manager.components.my_component import MyComponent
from experiment_manager.components.other_component import OtherComponent

class ComponentFactory(Factory):
    """Factory for creating components."""
    
    @staticmethod
    def create(type_name: str, config: DictConfig, *args, **kwargs):
        """
        Create a component instance.
        
        The import statements above ensure all component classes 
        are registered before this method is called.
        """
        try:
            return Factory.create(type_name, config, *args, **kwargs)
        except ValueError as e:
            # Handle unknown types or provide custom error messages
            raise ValueError(f"Failed to create component of type {type_name}: {e}")
```

#### Usage Example

This pattern is used for pipelines, callbacks, trackers, and other extensible parts of the system:

```python
# Configuration (YAML or Dict)
component_config = {
    "type": "MyComponent",  # Type identifier
    "param1": "value1",     # Component-specific parameters
    "param2": 42
}

# Import the appropriate factory for your component type
from experiment_manager.components.component_factory import ComponentFactory

# Create component from config
component = ComponentFactory.create(
    component_config["type"],
    component_config
)

# Use the component
component.do_something()  # Outputs: Doing something with value1 and 42
```

#### Benefits

- **Extensibility**: Easily add new component types without modifying existing code
- **Configuration-Driven**: Components are created from configuration files
- **Serialization**: Objects can be serialized to/from YAML for persistence
- **Loose Coupling**: Factory pattern decouples component creation from usage
- **Consistency**: Unified approach to component creation throughout the codebase
- **Auto-Registration**: Components register themselves in the central registry
- **Type Safety**: Factory handles type resolution and validation

Most major components in Experiment Manager follow this pattern, including Pipelines, Callbacks, Trackers, and Environment components.

### YAML Serialization

The framework provides a `YAMLSerializable` base class:

```python
from experiment_manager.common.serializable import YAMLSerializable
from omegaconf import DictConfig

@YAMLSerializable.register("MyCallback")
class MyCallback(YAMLSerializable):
    def __init__(self, config: DictConfig = None):
        super().__init__(config)
        self.some_param = config.get("some_param", "default_value")

    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(config)
```

### Dynamic Configuration Generation

```python
from experiment_manager.common.yaml_utils import insert_value, multiply
from omegaconf import OmegaConf

# Replace placeholders
base_config = OmegaConf.create({
    "learning_rate": "?",
    "batch_size": 32
})
config = insert_value(base_config, 0.001)  # learning_rate becomes 0.001

# Create cartesian product of configurations
model_configs = [
    {"model": "mlp", "layers": [64, 32]},
    {"model": "cnn", "filters": [32, 64]}
]
lr_configs = [
    {"learning_rate": 0.1},
    {"learning_rate": 0.01}
]
# Creates 4 configurations combining each model with each learning rate
configs = multiply(model_configs, lr_configs)
```

## Common Workflows

### Hyperparameter Tuning

```python
# Generate trial configurations with different hyperparameters
from experiment_manager.common.yaml_utils import multiply
import yaml

# Define base configuration
base_config = {
    "name": "tuning_trial",
    "repeat": 3,  # Run each setting 3 times with different seeds
    "settings": {
        "model_type": "mlp",
        "batch_size": 32,
    }
}

# Define hyperparameter options
learning_rates = [{"settings": {"learning_rate": lr}} for lr in [0.1, 0.01, 0.001]]
hidden_layers = [{"settings": {"hidden_layers": layers}} for layers in [[64], [128, 64], [256, 128, 64]]]

# Generate all combinations
trial_configs = multiply(learning_rates, hidden_layers)

# Add trial IDs and names
for i, conf in enumerate(trial_configs):
    conf["id"] = i + 1
    conf["name"] = f"trial_{i+1}"

# Save to trials.yaml
with open("configs/trials.yaml", "w") as f:
    yaml.dump(trial_configs, f)

# Run the experiment as normal
from experiment_manager.experiment import Experiment
from your_module.your_pipeline_factory import YourPipelineFactory

experiment = Experiment.create("configs", YourPipelineFactory)
experiment.run()
```

### Model Comparison

```python
# Compare different model architectures
model_types = ["mlp", "cnn", "transformer"]

# Create a pipeline factory that handles different model types
class ModelComparisonFactory(PipelineFactory):
    @staticmethod
    def create(name, config, env, id=None):
        model_type = config.get("settings", {}).get("model_type", "mlp")
        
        if model_type == "mlp":
            return MLPPipeline(env, config)
        elif model_type == "cnn":
            return CNNPipeline(env, config)
        elif model_type == "transformer":
            return TransformerPipeline(env, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Create trials for each model type
trial_configs = []
for i, model_type in enumerate(model_types):
    trial_configs.append({
        "id": i + 1,
        "name": f"{model_type}_model",
        "repeat": 5,  # Run 5 times for statistical significance
        "settings": {
            "model_type": model_type,
            "learning_rate": 0.001,
            "batch_size": 32,
            # Other shared settings
        }
    })

# Save and run as above
```


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## License

MIT License
