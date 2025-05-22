# Experiment Manager: Design Document

## 1. Overview

Experiment Manager is a framework designed to address the challenges of machine learning experimentation, providing a structured approach to experiment configuration, execution, tracking, and analysis. This document outlines the architectural design and core principles that underpin the system.

## 2. Architecture

The Experiment Manager follows a layered architecture with clear separation of concerns between configuration, execution, tracking, and storage components. The design incorporates several patterns such as factory, decorator, and observer patterns to ensure flexibility and extensibility.

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│                     (Python API, CLI, etc.)                  │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                      Experiment Management                   │
│         (Experiment, Trial, Configuration Management)        │
└───────────────────────────────┬─────────────────────────────┘
                                │
           ┌───────────────────┐│┌───────────────────┐
           │                   ││                    │
┌──────────▼──────────┐ ┌──────▼▼──────────┐ ┌──────▼──────────┐
│  Pipeline Execution  │ │ Tracking System  │ │ Storage System  │
│  (Pipelines,         │ │ (Metrics, Events,│ │ (Database, File │
│   Callbacks)         │ │  Artifacts)      │ │  System)        │
└─────────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2.2 Core Design Principles

1. **Configurability**: Configuration-driven approach using YAML files
2. **Extensibility**: Factory pattern and abstract base classes for customization
3. **Separation of Concerns**: Clear boundaries between components
4. **Hierarchical Organization**: Structured approach to experiments, trials, and runs
5. **Reproducibility**: Versioning and storage of all parameters and artifacts

## 3. Core Components

### 3.1 Environment (`Environment`)

The Environment component serves as the central context for an experiment, managing:

- Workspace directories (logs, artifacts, configs)
- Logging facilities
- Tracking systems
- Device management

The Environment uses a hierarchical structure that mirrors the experiment organization, with child environments for trials and runs. It provides the primary interface for other components to access these shared resources.

#### Key Design Features:

- Property-based directory management (lazy creation)
- Hierarchical structure with parent-child relationships
- Configuration persistence
- Resource management

### 3.2 Experiment Management (`Experiment`, `Trial`)

The Experiment Management layer handles the orchestration of experiments and trials:

- **Experiment**: Represents a complete research investigation, containing multiple trials
- **Trial**: Represents a specific configuration or variant of an experiment

This layer is responsible for loading configurations, initializing the environment, setting up trials, and coordinating execution.

#### Key Design Features:

- Configuration inheritance and merging
- Trial repetition support
- Lifecycle management (creation, execution, completion)

### 3.3 Pipeline System (`Pipeline`, `Callback`)

The Pipeline system encapsulates the actual execution logic of experiments:

- **Pipeline**: Abstract class defining the execution flow
- **Callbacks**: Components that can respond to pipeline events

Pipelines implement the training/evaluation logic, while callbacks provide a way to extend this logic with additional behaviors (early stopping, checkpointing, etc.).

#### Key Design Features:

- Decorator-based wrapper methods for lifecycle management
- Event propagation system for callbacks
- Standardized interface for defining execution logic
- Error handling and status reporting

### 3.4 Tracking System (`Tracker`, `TrackerManager`)

The Tracking System records metrics, events, and artifacts during experiment execution:

- **Tracker**: Base class for tracking implementations
- **TrackerManager**: Coordinates multiple trackers

This system provides a unified interface for recording data, regardless of the underlying storage or visualization systems (MLflow, TensorBoard, etc.).

#### Key Design Features:

- Plugin architecture for multiple tracking backends
- Context manager for automatic lifecycle tracking
- Hierarchical tracking matching experiment structure
- Support for different metric types and artifact formats

### 3.5 Storage System (`DatabaseManager`)

The Storage System persists experiment data for later analysis using a comprehensive relational database schema:

- Database integration (SQLite for development, MySQL for production)
- Complete schema with 7 core tables and 7 junction tables
- Hierarchical data organization matching experiment structure
- Support for complex many-to-many relationships
- Comprehensive querying capabilities with proper indexing

#### Key Design Features:

- Abstraction over database backends (SQLite/MySQL)
- Entity relationship modeling with foreign key constraints
- Multi-level artifact and metric associations
- JSON support for complex metric data (per-label values)
- Comprehensive querying interface with custom exceptions
- Connection pooling and transaction management

## 4. Design Patterns

### 4.1 Factory-Serializable Pattern

A central pattern in the codebase is the Factory-Serializable pattern, which combines the Factory pattern with serialization capabilities. This pattern is used for creating components from configuration and enabling their persistence.

#### Implementation:

- `YAMLSerializable`: Base class providing serialization capabilities
- `Factory`: Base class for component creation
- Registration mechanism using decorators

#### Benefits:

- Configuration-driven component creation
- Dynamic loading of components
- Extensibility through custom implementations
- Loose coupling between component creation and usage

### 4.2 Decorator Pattern

Decorators are used extensively for cross-cutting concerns:

- `@Pipeline.run_wrapper`: Handles lifecycle management for pipeline runs
- `@Pipeline.epoch_wrapper`: Manages per-epoch operations
- `@YAMLSerializable.register`: Registers components with the factory system

### 4.3 Observer Pattern

The callback system implements the Observer pattern:

- Pipeline acts as the subject
- Callbacks act as observers
- Events propagate through the pipeline lifecycle

### 4.4 Context Manager Pattern

The tracking system uses context managers to ensure proper event registration:

- `with tracker.track_scope(Level.TRIAL)`: Automatically handles start/end events

## 5. Hierarchical Levels and Context Architecture

### 5.1 The Six-Level Hierarchy

The Experiment Manager implements a sophisticated six-level hierarchy that provides structured context and organization:

```python
class Level(Enum):
    EXPERIMENT    = 0  # Top-level experiment container
    TRIAL         = 1  # Individual experiment configurations  
    TRIAL_RUN     = 2  # Single execution of a trial (with repetition support)
    PIPELINE      = 3  # Execution context for training workflows
    EPOCH         = 4  # Individual training epochs within a pipeline
    BATCH         = 5  # Individual batch processing (finest granularity)
```

#### Level Responsibilities and Design Purpose

1. **EXPERIMENT Level**: 
   - **Purpose**: Overall coordination and experiment-wide metrics
   - **Design Choice**: Single source of truth for experiment configuration and metadata
   - **Tracked Entities**: Experiment config, cross-trial comparisons, final aggregated results

2. **TRIAL Level**: 
   - **Purpose**: Individual parameter configurations within an experiment
   - **Design Choice**: Enables systematic exploration of parameter spaces
   - **Tracked Entities**: Hyperparameters, trial-specific configurations, trial outcomes

3. **TRIAL_RUN Level**: 
   - **Purpose**: Individual executions supporting statistical significance through repetition
   - **Design Choice**: Separates configuration (trial) from execution (run) for clean abstraction
   - **Tracked Entities**: Run status, random seeds, run-specific metrics and artifacts

4. **PIPELINE Level**: 
   - **Purpose**: Training workflow execution context where callbacks operate
   - **Design Choice**: Bridges the gap between experiment management and training logic
   - **Tracked Entities**: Pipeline configuration, training duration, workflow events

5. **EPOCH Level**: 
   - **Purpose**: Individual training iterations for learning progress tracking
   - **Design Choice**: Enables fine-grained monitoring and intervention (early stopping)
   - **Tracked Entities**: Per-epoch metrics, learning curves, model checkpoints

6. **BATCH Level**: 
   - **Purpose**: Finest granularity for detailed debugging and analysis
   - **Design Choice**: Optional level for when detailed batch-level information is needed
   - **Tracked Entities**: Batch metrics, gradient information, detailed debugging data

### 5.2 Callbacks vs Trackers: Context Separation

A key architectural decision is the **context-based separation** between callbacks and trackers:

#### Callbacks: Pipeline-Context Components

```python
class Callback(ABC):
    def on_start(self) -> None:
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
    def on_end(self, metrics: Dict[str, Any]) -> None:
```

**Design Purpose**: Focus on **training workflow intervention and control**
- **Context**: Operate within PIPELINE level boundaries
- **Responsibilities**: Training decisions, immediate interventions, workflow management
- **Lifecycle**: Bounded to pipeline execution (start → epochs → end)
- **State**: Access to immediate training state for real-time decisions

#### Trackers: Multi-Level Context Components

```python
class Tracker(ABC):
    def on_create(self, level: Level, *args, **kwargs):
    def on_start(self, level: Level, *args, **kwargs):
    def on_end(self, level: Level, *args, **kwargs):
    def on_add_artifact(self, level: Level, artifact_path: str, *args, **kwargs):
```

**Design Purpose**: Focus on **comprehensive data collection and persistence**
- **Context**: Operate across ALL hierarchy levels
- **Responsibilities**: Data collection, persistent storage, cross-experiment analysis
- **Lifecycle**: Span entire experiment hierarchy from creation to completion
- **State**: Maintain hierarchical relationships and long-term data consistency

#### Design Benefits of This Separation

1. **Single Responsibility Principle**: 
   - Callbacks handle training logic and intervention
   - Trackers handle data collection and persistence

2. **Different Temporal Scopes**:
   - Callbacks: Real-time training decisions within pipeline execution
   - Trackers: Long-term data management across experiment lifecycle

3. **Orthogonal Concerns**:
   - Training logic can be modified without affecting tracking
   - Tracking backends can be changed without modifying training code

4. **Testability and Modularity**:
   - Callbacks can be unit tested with mock pipeline states
   - Trackers can be tested independently with mock level events

### 5.3 TrackScope: Context Manager for Level Transitions

The `TrackScope` context manager automates proper level lifecycle management:

```python
class TrackScope:
    def __init__(self, tracker_manager: TrackerManager, level: Level, *args, **kwargs):
        self.tracker_manager = tracker_manager
        self.level = level
    
    def __enter__(self):
        self.tracker_manager.on_start(self.level, *self.args, **self.kwargs)
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.tracker_manager.on_end(self.level)
```

**Design Benefits**:
- **Automatic lifecycle management**: Ensures proper start/end event tracking
- **Exception safety**: Cleanup occurs even when exceptions are raised
- **Nested context support**: Enables hierarchical level transitions
- **Simplified user code**: No manual lifecycle management required

## 6. Additional Design Patterns and Architectural Choices

### 6.1 Property-Based Lazy Directory Creation

The `Environment` class uses properties for lazy directory creation:

```python
@property
def log_dir(self):
    log_dir_path = os.path.join(self.workspace, ProductPaths.LOG_DIR.value)
    if not os.path.exists(log_dir_path):
        os.mkdir(log_dir_path)
    return log_dir_path
```

**Design Purpose**: 
- **Lazy Evaluation**: Directories created only when accessed
- **Automatic Management**: No manual directory setup required
- **Consistent Structure**: Ensures standard directory layout
- **Error Prevention**: Eliminates "directory not found" errors

### 6.2 Hierarchical Workspace Pattern

The framework implements a hierarchical workspace structure that mirrors the logical experiment organization:

```
workspace/
├── experiment_name/          # EXPERIMENT level
│   ├── configs/             
│   ├── logs/               
│   ├── artifacts/          
│   └── trials/             
│       ├── trial_1/         # TRIAL level
│       │   ├── configs/    
│       │   ├── logs/       
│       │   ├── artifacts/  
│       │   └── run_1/       # TRIAL_RUN level
│       │       ├── logs/   
│       │       └── artifacts/
```

**Design Benefits**:
- **Logical Organization**: Directory structure matches conceptual hierarchy
- **Isolation**: Each level has isolated workspace preventing conflicts
- **Discoverability**: Easy to locate artifacts and logs by experiment structure
- **Scalability**: Structure remains organized as experiments grow in complexity

### 6.3 Child Environment Creation Pattern

The system uses a child environment pattern for hierarchical resource management:

```python
def create_child(self, name: str, args: DictConfig = None) -> 'Environment':
    child_workspace = os.path.join(self.workspace, name)
    child_env = self.__class__(
        workspace=child_workspace,
        config=self.config,
        factory=self.factory,
        verbose=self.verbose,
        debug=self.debug,
        tracker_manager=self.tracker_manager.create_child(child_workspace),
        device=self.device,
        args=self.args)
```

**Design Purpose**:
- **Inheritance with Isolation**: Child inherits parent configuration but has isolated workspace
- **Resource Propagation**: Shared resources (factory, config) propagated to children
- **Tracker Hierarchy**: Tracker managers maintain parent-child relationships
- **Configuration Merging**: Child-specific arguments merged with inherited configuration

### 6.4 Decorator-Based Lifecycle Management

Pipelines use decorators for automatic lifecycle management:

```python
@staticmethod
def run_wrapper(run_function):
    @wraps(run_function)
    def wrapper(self, config: DictConfig):
        self._on_run_start()
        try:
            status = run_function(self, config)
        except Exception as e:
            status = RunStatus.FAILED
        finally:
            self._on_run_end(self.run_metrics)
        return status
    return wrapper
```

**Design Benefits**:
- **Automatic Resource Management**: Setup and cleanup handled automatically
- **Error Handling**: Consistent error handling across all pipeline implementations
- **Code Reuse**: Eliminates boilerplate lifecycle code in user implementations
- **Aspect-Oriented Programming**: Cross-cutting concerns handled declaratively

### 6.5 Composite Pattern for TrackerManager

The `TrackerManager` implements a composite pattern, treating collections of trackers as a single tracker:

```python
class TrackerManager(Tracker):
    def __init__(self, workspace: str = None) -> None:
        self.trackers: List[Tracker] = []
    
    def track(self, metric: Metric, value, step: int = None, *args, **kwargs):
        for tracker in self.trackers:
            tracker.track(metric, value, step, *args, **kwargs)
```

**Design Purpose**:
- **Unified Interface**: Multiple tracking backends accessed through single interface
- **Composition over Inheritance**: Flexible combination of different tracker types
- **Plugin Architecture**: Easy addition/removal of tracking backends
- **Consistent Behavior**: All trackers receive the same events simultaneously

### 6.6 Configuration Inheritance and Merging Pattern

The configuration system implements hierarchical inheritance:

```python
# Base configuration shared across trials
base_config = OmegaConf.load("base.yaml")

# Trial-specific overrides
trial_config = OmegaConf.load("trial.yaml")

# Merged configuration
final_config = OmegaConf.merge(base_config, trial_config)
```

**Design Benefits**:
- **DRY Principle**: Common configurations defined once and reused
- **Hierarchical Overrides**: More specific configurations override general ones
- **Composition**: Complex configurations built from simpler components
- **Maintainability**: Changes to base configurations propagate to all trials

## 5. Data Flow

### 5.1 Configuration Flow

1. User defines configuration files (env.yaml, experiment.yaml, base.yaml, trials.yaml)
2. `Experiment.create()` loads and validates configurations
3. Configurations are merged with inheritance (base → experiment → trial)
4. Configuration is passed to components during instantiation

### 5.2 Execution Flow

1. `Experiment.run()` initiates the experiment
2. For each trial configuration, a `Trial` is created
3. Trial creates a pipeline instance using the factory pattern
4. Pipeline executes the run with lifecycle hooks:
   - `_on_run_start()` → `run()` → `_on_run_end()`
   - For each epoch: `_on_epoch_start()` → `run_epoch()` → `_on_epoch_end()`
5. Callbacks are notified at each lifecycle point
6. Metrics are tracked throughout execution

### 5.3 Tracking Flow

1. `TrackerManager` initializes with configured trackers
2. During execution, components call tracking methods:
   - `tracker.track(metric, value, step)`
   - `tracker.on_start/on_end(level)`
3. `TrackerManager` delegates to appropriate trackers
4. Trackers record data to their respective backends (MLflow, TensorBoard, DB)

## 6. Directory Structure

The framework creates a hierarchical directory structure that mirrors the experiment organization:

```
workspace/
├── experiment_name/
│   ├── configs/           # Configuration files
│   ├── logs/              # Experiment-level logs
│   ├── artifacts/         # Experiment-level artifacts
│   └── trials/            # Trial directories
│       ├── trial_1/
│       │   ├── configs/   # Trial-specific configs
│       │   ├── logs/      # Trial-specific logs
│       │   ├── artifacts/ # Trial-specific artifacts
│       │   └── run_1/     # Trial run directories
│       └── trial_2/
│           └── ...
```

This structure ensures:
- Clear separation between experiments
- Organized storage of artifacts and logs
- Support for reproducibility and analysis

## 7. Extensibility Mechanisms

### 7.1 Custom Pipelines

Users can define custom pipeline implementations by:
1. Subclassing `Pipeline`
2. Implementing `run()` and `run_epoch()` methods
3. Creating a custom factory to instantiate the pipeline

### 7.2 Custom Callbacks

Users can extend functionality with custom callbacks by:
1. Subclassing `Callback`
2. Implementing lifecycle methods (`on_start()`, `on_epoch_end()`, `on_end()`)
3. Registering the callback with a pipeline

### 7.3 Custom Trackers

Users can integrate additional tracking backends by:
1. Subclassing `Tracker` and `YAMLSerializable`
2. Implementing tracker interface methods
3. Registering the tracker with the `@YAMLSerializable.register()` decorator

### 7.4 Configuration Extensions

The configuration system supports:
1. Custom configuration generators (`yaml_utils.py`)
2. Configuration merging and inheritance
3. Dynamic placeholder replacement

## 8. Known Limitations and Future Improvements

### 8.1 Current Limitations

- Tight coupling to specific ML frameworks (PyTorch)
- Limited support for distributed training
- Incomplete schema validation for configurations
- Some resource management issues (see report.txt)

### 8.2 Potential Improvements

- Enhanced configuration validation
- Adapter pattern for ML framework abstraction
- Improved resource management and cleanup
- Expanded test coverage
- Enhanced documentation

## 9. Conclusion

The Experiment Manager provides a robust architecture for managing machine learning experiments with a focus on configurability, extensibility, and reproducibility. The design patterns employed enable users to extend the framework to meet their specific needs while maintaining a consistent structure and approach.

The hierarchical organization of experiments, trials, and runs, combined with comprehensive tracking and storage capabilities, addresses the key challenges of machine learning experimentation: organization, reproducibility, and analysis.

## 4. Database Schema Design

The Experiment Manager uses a sophisticated relational database schema that mirrors the hierarchical structure of experiments while supporting complex relationships for metrics and artifacts.

### 4.1 Entity Relationship Model

```
┌────────────────┐       ┌────────────────┐       ┌────────────────┐
│   EXPERIMENT   │       │     TRIAL      │       │   TRIAL_RUN    │
├────────────────┤       ├────────────────┤       ├────────────────┤
│ id (PK)        │──┐    │ id (PK)        │──┐    │ id (PK)        │
│ title          │  │    │ name           │  │    │ trial_id (FK)  │◄┘
│ desc           │  │    │ experiment_id  │◄─┘    │ status         │
│ start_time     │  │    │ start_time     │       │ start_time     │
│ update_time    │  │    │ update_time    │       │ update_time    │
└────────────────┘  │    └────────────────┘       └────────────────┘
                    │                                      │
                    │    ┌────────────────┐                │
                    │    │    RESULTS     │                │
                    │    ├────────────────┤                │
                    │    │ trial_run_id   │◄───────────────┘
                    │    │ (PK, FK)       │
                    │    │ time           │
                    │    └────────────────┘
                    │                     │
┌───────────────────▼┐   ┌────────────────▼───┐   ┌────────────────┐
│     ARTIFACT       │   │      EPOCH         │   │     METRIC     │
├────────────────────┤   ├────────────────────┤   ├────────────────┤
│ id (PK)            │   │ idx (CK)           │   │ id (PK)        │
│ type               │   │ trial_run_id (CK)  │   │ type           │
│ loc                │   │ time               │   │ total_val      │
└────────────────────┘   └────────────────────┘   │ per_label_val  │
                                                   │ (JSON)         │
                                                   └────────────────┘
```

### 4.2 Core Tables

#### EXPERIMENT
Stores high-level experiment metadata and serves as the root of the hierarchy.
- `id` (INT, PK, AUTO_INCREMENT): Unique identifier
- `title` (VARCHAR(255), NOT NULL): Human-readable experiment name
- `desc` (TEXT): Detailed description of the experiment
- `start_time` (DATETIME, NOT NULL): When the experiment was created
- `update_time` (DATETIME, NOT NULL): Last modification timestamp

#### TRIAL
Represents specific configurations or variants within an experiment.
- `id` (INT, PK, AUTO_INCREMENT): Unique identifier
- `name` (VARCHAR(255), NOT NULL): Trial name or identifier
- `experiment_id` (INT, FK, NOT NULL): Reference to parent experiment
- `start_time` (DATETIME, NOT NULL): Trial creation time
- `update_time` (DATETIME, NOT NULL): Last modification timestamp

#### TRIAL_RUN
Individual executions of a trial (supports trial repetition).
- `id` (INT, PK, AUTO_INCREMENT): Unique identifier
- `trial_id` (INT, FK, NOT NULL): Reference to parent trial
- `status` (VARCHAR(50), NOT NULL): Current status (running, completed, failed, etc.)
- `start_time` (DATETIME, NOT NULL): Run start time
- `update_time` (DATETIME, NOT NULL): Last status update time

#### RESULTS
Stores overall results for a trial run (one-to-one relationship).
- `trial_run_id` (INT, PK, FK): Reference to trial run
- `time` (DATETIME, NOT NULL): When results were recorded

#### EPOCH
Represents individual epochs within a trial run.
- `idx` (INT): Epoch number/index
- `trial_run_id` (INT): Reference to trial run
- `time` (DATETIME, NOT NULL): Epoch completion time
- **Composite Primary Key**: (idx, trial_run_id)

#### METRIC
Stores metric values with support for both scalar and per-label metrics.
- `id` (INT, PK, AUTO_INCREMENT): Unique identifier
- `type` (VARCHAR(50), NOT NULL): Metric name/type (accuracy, loss, f1_score, etc.)
- `total_val` (FLOAT, NOT NULL): Overall metric value
- `per_label_val` (JSON): Per-class or per-label breakdown

#### ARTIFACT
Represents files, models, or other objects generated during experiments.
- `id` (INT, PK, AUTO_INCREMENT): Unique identifier
- `type` (VARCHAR(50), NOT NULL): Artifact type (model, plot, log, config, etc.)
- `loc` (VARCHAR(255), NOT NULL): File path or location

### 4.3 Junction Tables (Many-to-Many Relationships)

#### EXPERIMENT_ARTIFACT
Links artifacts to experiments (experiment-level outputs).
- `experiment_id` (INT, FK): Reference to experiment
- `artifact_id` (INT, FK): Reference to artifact
- **Composite Primary Key**: (experiment_id, artifact_id)

#### TRIAL_ARTIFACT
Links artifacts to trials (trial-specific outputs).
- `trial_id` (INT, FK): Reference to trial
- `artifact_id` (INT, FK): Reference to artifact
- **Composite Primary Key**: (trial_id, artifact_id)

#### TRIAL_RUN_ARTIFACT
Links artifacts to trial runs (run-specific outputs).
- `trial_run_id` (INT, FK): Reference to trial run
- `artifact_id` (INT, FK): Reference to artifact
- **Composite Primary Key**: (trial_run_id, artifact_id)

#### RESULTS_METRIC
Links metrics to results (overall trial metrics).
- `results_id` (INT, FK): Reference to results (trial_run_id)
- `metric_id` (INT, FK): Reference to metric
- **Composite Primary Key**: (results_id, metric_id)

#### RESULTS_ARTIFACT
Links artifacts to results (result-specific artifacts).
- `results_id` (INT, FK): Reference to results
- `artifact_id` (INT, FK): Reference to artifact
- **Composite Primary Key**: (results_id, artifact_id)

#### EPOCH_METRIC
Links metrics to specific epochs (per-epoch metrics).
- `epoch_idx` (INT): Epoch index
- `epoch_trial_run_id` (INT): Trial run ID
- `metric_id` (INT, FK): Reference to metric
- **Composite Primary Key**: (epoch_idx, epoch_trial_run_id, metric_id)

#### EPOCH_ARTIFACT
Links artifacts to specific epochs (per-epoch artifacts like checkpoints).
- `epoch_idx` (INT): Epoch index
- `epoch_trial_run_id` (INT): Trial run ID
- `artifact_id` (INT, FK): Reference to artifact
- **Composite Primary Key**: (epoch_idx, epoch_trial_run_id, artifact_id)

### 4.4 Database Access Patterns

The database design supports several common access patterns:

1. **Hierarchical Queries**: Retrieve all data for an experiment including all trials, runs, and metrics
2. **Metric Aggregation**: Aggregate metrics across trials or epochs for comparison
3. **Artifact Retrieval**: Find all artifacts associated with any level of the hierarchy
4. **Progress Tracking**: Monitor trial run status and completion
5. **Time Series Analysis**: Analyze metric evolution over epochs or time
6. **Cross-Experiment Analysis**: Compare metrics and artifacts across different experiments

### 4.5 Data Integrity and Constraints

- **Foreign Key Constraints**: Ensure referential integrity across the hierarchy
- **Composite Keys**: Support complex relationships (especially for epochs)
- **NOT NULL Constraints**: Ensure required fields are always populated
- **JSON Validation**: Per-label metric data stored as valid JSON
- **Timestamp Consistency**: Start and update times properly maintained 