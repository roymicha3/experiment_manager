# Database Module

This module provides database management functionality for the experiment manager.

## Structure

- `docs/`: Contains database documentation
  - `diagram.md`: Entity-relationship diagram in Mermaid format
  - `init.sql`: SQL initialization script
- `manager.py`: Database manager implementation

## Database Schema

The database schema consists of the following main tables:

1. `EXPERIMENT`: Stores experiment metadata
2. `TRIAL`: Stores trials associated with experiments
3. `TRIAL_RUN`: Stores individual runs of trials
4. `RESULTS`: Stores results of trial runs
5. `EPOCH`: Stores epoch data for trial runs
6. `METRIC`: Stores metric values
7. `ARTIFACT`: Stores artifact metadata

And relationship tables:
- `EXPERIMENT_ARTIFACT`
- `TRIAL_ARTIFACT`
- `RESULTS_METRIC`
- `RESULTS_ARTIFACT`
- `EPOCH_METRIC`
- `EPOCH_ARTIFACT`
- `TRIAL_RUN_ARTIFACT`

## Usage

```python
from experiment_manager.db.manager import DatabaseManager

# Initialize manager
db = DatabaseManager(
    host="localhost",
    user="root",
    password="your_password",
    database="experiment_manager"
)

# Create experiment
exp_id = db.create_experiment("MNIST Training", "Training on MNIST dataset")

# Create trial
trial_id = db.create_trial(exp_id, "SGD Optimizer")

# Create trial run
run_id = db.create_trial_run(trial_id)

# Record metric
metric_id = db.record_metric(0.95, "accuracy", {"class_0": 0.93, "class_1": 0.97})

# Add metric to epoch
db.add_epoch_metric(1, run_id, metric_id)

# Record artifact
artifact_id = db.record_artifact("model", "/path/to/model.pt")

# Get experiment metrics
metrics = db.get_experiment_metrics(exp_id)

# Get trial artifacts
artifacts = db.get_trial_artifacts(trial_id)
```

## Testing

Tests are located in `tests/test_db_manager.py`. To run tests:

```bash
pytest tests/test_db_manager.py
```

Make sure you have MySQL running locally and have created the database using the initialization script in `docs/init.sql`.
