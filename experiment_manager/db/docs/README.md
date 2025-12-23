# Experiment Tracking Database

## Overview

This document outlines the structure of the Experiment Tracking Database, a comprehensive system designed to manage and track machine learning experiments, trials, and their associated data. This database is ideal for researchers, data scientists, and machine learning engineers who need to organize, monitor, and analyze their experimental results efficiently.

## Database Structure

### Core Tables

#### EXPERIMENT
Represents a high-level experiment.
- `id` (INT, PK): Unique identifier
- `title` (VARCHAR): Experiment title
- `desc` (TEXT): Experiment description
- `start_time` (DATETIME): Start time of the experiment
- `update_time` (DATETIME): Last update time

#### TRIAL
Represents a specific trial within an experiment.
- `id` (INT, PK): Unique identifier
- `name` (VARCHAR): Trial name
- `experiment_id` (INT, FK): Reference to EXPERIMENT
- `start_time` (DATETIME): Start time of the trial
- `update_time` (DATETIME): Last update time

#### TRIAL_RUN
Represents a single run of a trial.
- `id` (INT, PK): Unique identifier
- `trial_id` (INT, FK): Reference to TRIAL
- `status` (VARCHAR): Current status of the run
- `start_time` (DATETIME): Start time of the run
- `update_time` (DATETIME): Last update time

#### RESULTS
Stores the overall results of a trial run.
- `trial_run_id` (INT, PK, FK): Reference to TRIAL_RUN
- `time` (DATETIME): Time when results were recorded

#### EPOCH
Represents individual epochs within a trial run.
- `idx` (INT): Epoch index
- `trial_run_id` (INT): Reference to TRIAL_RUN
- `time` (DATETIME): Time of the epoch
(Composite PK: idx, trial_run_id)

#### BATCH
Represents individual batches within an epoch.
- `idx` (INT): Batch index within the epoch
- `epoch_idx` (INT): Reference to parent epoch
- `trial_run_id` (INT): Reference to TRIAL_RUN
- `time` (DATETIME): Time when batch processing started
(Composite PK: idx, epoch_idx, trial_run_id)

#### METRIC
Stores metric data for results and epochs.
- `id` (INT, PK): Unique identifier
- `type` (VARCHAR): Type of metric
- `total_val` (FLOAT): Total value of the metric
- `per_label_val` (JSON): Per-label values

#### ARTIFACT
Represents files or objects generated during the experiment.
- `id` (INT, PK): Unique identifier
- `type` (VARCHAR): Type of artifact
- `loc` (VARCHAR): Location or path of the artifact

### Junction Tables

To handle many-to-many relationships, the following junction tables are used:

#### EXPERIMENT_ARTIFACT
- `experiment_id` (INT, FK): Reference to EXPERIMENT
- `artifact_id` (INT, FK): Reference to ARTIFACT

#### TRIAL_ARTIFACT
- `trial_id` (INT, FK): Reference to TRIAL
- `artifact_id` (INT, FK): Reference to ARTIFACT

#### TRIAL_RUN_ARTIFACT
- `trial_run_id` (INT, FK): Reference to TRIAL_RUN
- `artifact_id` (INT, FK): Reference to ARTIFACT

#### RESULTS_METRIC
- `results_id` (INT, FK): Reference to RESULTS
- `metric_id` (INT, FK): Reference to METRIC

#### RESULTS_ARTIFACT
- `results_id` (INT, FK): Reference to RESULTS
- `artifact_id` (INT, FK): Reference to ARTIFACT

#### EPOCH_METRIC
- `epoch_idx` (INT): Reference to EPOCH.idx
- `epoch_trial_run_id` (INT): Reference to EPOCH.trial_run_id
- `metric_id` (INT, FK): Reference to METRIC

#### EPOCH_ARTIFACT
- `epoch_idx` (INT): Reference to EPOCH.idx
- `epoch_trial_run_id` (INT): Reference to EPOCH.trial_run_id
- `artifact_id` (INT, FK): Reference to ARTIFACT

#### BATCH_METRIC
- `batch_idx` (INT): Reference to BATCH.idx
- `epoch_idx` (INT): Reference to BATCH.epoch_idx
- `trial_run_id` (INT): Reference to BATCH.trial_run_id
- `metric_id` (INT, FK): Reference to METRIC

#### BATCH_ARTIFACT
- `batch_idx` (INT): Reference to BATCH.idx
- `epoch_idx` (INT): Reference to BATCH.epoch_idx
- `trial_run_id` (INT): Reference to BATCH.trial_run_id
- `artifact_id` (INT, FK): Reference to ARTIFACT

## Entity Relationships

1. An EXPERIMENT can have multiple TRIALs and ARTIFACTs
2. A TRIAL can have multiple TRIAL_RUNs and ARTIFACTs
3. A TRIAL_RUN is associated with one RESULTS entry and can have multiple ARTIFACTs
4. A TRIAL_RUN can have multiple EPOCHs
5. An EPOCH can have multiple BATCHes
6. RESULTS, EPOCHs, and BATCHes can be associated with multiple METRICs and ARTIFACTs

## Key Features

1. **Hierarchical Structure**: Experiments > Trials > Trial Runs > Epochs > Batches
2. **Flexible Metric Storage**: Supports both overall and per-label metric values (JSON)
3. **Comprehensive Artifact Tracking**: Ability to link artifacts (e.g., model files, plots) to experiments, trials, trial runs, results, epochs, and batches
4. **Temporal Tracking**: All major entities include timestamp information
5. **Batch-Level Granularity**: Fine-grained tracking for detailed training diagnostics

## Best Practices for Usage

1. Maintain consistent naming conventions for experiments, trials, and metrics
2. Regularly backup the database
3. Implement data validation before inserting or updating records
4. Use appropriate indexing for frequently queried columns
5. Consider partitioning large tables (e.g., EPOCH) for improved performance

## Potential Extensions

1. Add user authentication and access control
2. Implement versioning for experiments and trials
3. Add support for distributed experiments
4. Integrate with popular ML frameworks for automatic logging

## Conclusion

This database structure provides a robust foundation for tracking machine learning experiments. It offers flexibility to accommodate various experimental setups while maintaining a clear and organized data structure. The addition of artifact associations at multiple levels (experiment, trial, trial run) allows for more comprehensive tracking of resources throughout the experimental process.
