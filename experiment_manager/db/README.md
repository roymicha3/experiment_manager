# Database Module

This module provides database management functionality for the experiment manager, supporting both SQLite and MySQL backends.

## Structure

- `docs/`: Contains database documentation
  - `diagram.md`: Entity-relationship diagram in Mermaid format
  - `init.sql`: SQL initialization script
  - `README.md`: Detailed database documentation
- `manager.py`: Database manager implementation
- `tables.py`: Data classes for database entities

## Database Schema

The database schema consists of 8 core tables that form a hierarchical structure:

1. `EXPERIMENT`: Stores experiment metadata (id, title, description, timestamps)
2. `TRIAL`: Stores trials associated with experiments (id, name, experiment_id, timestamps)
3. `TRIAL_RUN`: Stores individual runs of trials (id, trial_id, status, timestamps)
4. `RESULTS`: Stores overall results of trial runs (trial_run_id, time)
5. `EPOCH`: Stores epoch data for trial runs (idx, trial_run_id, time)
6. `BATCH`: Stores batch data within epochs (idx, epoch_idx, trial_run_id, time)
7. `METRIC`: Stores metric values (id, type, total_val, per_label_val as JSON)
8. `ARTIFACT`: Stores artifact metadata (id, type, location)

And 9 junction/relationship tables for many-to-many relationships:
- `EXPERIMENT_ARTIFACT`: Links artifacts to experiments
- `TRIAL_ARTIFACT`: Links artifacts to trials
- `TRIAL_RUN_ARTIFACT`: Links artifacts to trial runs
- `RESULTS_METRIC`: Links metrics to results
- `RESULTS_ARTIFACT`: Links artifacts to results
- `EPOCH_METRIC`: Links metrics to epochs
- `EPOCH_ARTIFACT`: Links artifacts to epochs
- `BATCH_METRIC`: Links metrics to batches (enables batch-level metric tracking)
- `BATCH_ARTIFACT`: Links artifacts to batches (enables batch-level artifact tracking)

## Usage

```python
from experiment_manager.db.manager import DatabaseManager

# Initialize SQLite manager (for local development)
db = DatabaseManager(database_path="experiment_manager.db", use_sqlite=True)

# Or initialize MySQL manager (for production)
db = DatabaseManager(
    database_path="experiment_manager",
    host="localhost",
    user="root",
    password="your_password"
)

# Create experiment
experiment = db.create_experiment("MNIST Training", "Training on MNIST dataset")

# Create trial
trial = db.create_trial(experiment.id, "SGD Optimizer")

# Create trial run
trial_run = db.create_trial_run(trial.id)

# Create epoch
db.create_epoch(1, trial_run.id)

# Create batch within epoch
batch = db.create_batch(0, 1, trial_run.id)

# Record and link metric to epoch
metric = db.record_metric(0.95, "accuracy", {"class_0": 0.93, "class_1": 0.97})

# Link metric to batch
db.add_batch_metric(0, 1, trial_run.id, metric.id)
db.add_epoch_metric(1, trial_run.id, metric.id)

# Record and link artifact to experiment
artifact = db.record_artifact("model", "/path/to/model.pt")
db.link_experiment_artifact(experiment.id, artifact.id)

# Get experiment metrics (includes both epoch and results metrics)
metrics = db.get_experiment_metrics(experiment.id)

# Get artifacts at different levels
exp_artifacts = db.get_experiment_artifacts(experiment.id)
trial_artifacts = db.get_trial_artifacts(trial.id)
trial_run_artifacts = db.get_trial_run_artifacts(trial_run.id)
epoch_artifacts = db.get_epoch_artifacts(1, trial_run.id)
```

## Error Handling

The module provides custom exceptions for error handling:
- `DatabaseError`: Base class for database errors
- `ConnectionError`: Error connecting to database
- `QueryError`: Error executing database queries

## Testing

Tests are located in `tests/db/test_manager.py`. To run tests:

```bash
pytest tests/db/test_manager.py -v
```

The tests use SQLite by default for easy testing without requiring a MySQL server.
