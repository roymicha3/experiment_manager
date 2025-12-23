# Experiment Data Fixtures Documentation

This document provides comprehensive guidance on using the MNIST experiment fixtures in the experiment_manager test suite.

## Overview

The experiment fixture system provides efficient access to shared MNIST baseline experiment data across all test modules. The system runs the experiment once per test session and provides various levels of data access to optimize test performance.

## Available Fixtures

### 1. Core Session Fixture

#### `shared_mnist_experiment`
- **Scope**: Session-scoped (runs once per test session)
- **Purpose**: Provides the foundation for all other experiment fixtures
- **Returns**: Dictionary with framework_experiment, db_path, and temp_dir
- **Usage**: Generally not used directly; use specialized fixtures instead

### 2. Standard Fixtures

#### `experiment_data` 
- **Purpose**: General-purpose experiment data access
- **Best for**: Most test cases that need experiment data
- **Returns**: Fresh DBDataSource with experiment, db_path, temp_dir, framework_experiment
- **Performance**: Medium (loads full experiment object)

```python
def test_experiment_functionality(experiment_data):
    experiment = experiment_data['experiment']
    assert experiment.name == "test_mnist_baseline"
    assert len(experiment.trials) == 3
```

### 3. Specialized Performance-Optimized Fixtures

#### `experiment_db_only`
- **Purpose**: Minimal overhead database access
- **Best for**: Tests that create custom DBDataSource instances
- **Returns**: String path to database file
- **Performance**: Fastest (just returns a string)

```python
def test_custom_database_operations(experiment_db_only):
    db_path = experiment_db_only
    with DBDataSource(db_path) as source:
        # Custom database operations
        data = source.get_experiment()
        assert data is not None
```

#### `experiment_metrics_only`
- **Purpose**: Pre-loaded metrics analysis
- **Best for**: Tests focusing on training curves, performance analysis
- **Returns**: Dictionary with db_path, metrics DataFrame, experiment_id, trial_count
- **Performance**: Medium (pre-loads metrics but not artifacts)

```python
def test_metrics_analysis(experiment_metrics_only):
    metrics_df = experiment_metrics_only['metrics']
    
    # Analyze training curves
    loss_data = metrics_df[metrics_df['metric'].str.contains('loss')]
    assert not loss_data.empty
    
    # Check for accuracy improvements
    acc_data = metrics_df[metrics_df['metric'].str.contains('acc')]
    assert not acc_data.empty
```

#### `experiment_lightweight`
- **Purpose**: Minimal metadata for fast unit tests
- **Best for**: Quick property validation, basic experiment checks
- **Returns**: Dictionary with experiment_name, trial_count, db_path, workspace_exists
- **Performance**: Fast (minimal database access)

```python
def test_basic_experiment_properties(experiment_lightweight):
    assert experiment_lightweight['experiment_name'] == "test_mnist_baseline"
    assert experiment_lightweight['trial_count'] == 3
    assert experiment_lightweight['workspace_exists'] is True
```

#### `experiment_full_artifacts`
- **Purpose**: Complete workspace access with artifact catalog
- **Best for**: Integration tests, artifact verification, complete workflow testing
- **Returns**: Dictionary with db_path, temp_dir, experiment, framework_experiment, artifacts
- **Performance**: Slowest (loads everything and catalogs artifacts)

```python
def test_complete_experiment_workflow(experiment_full_artifacts):
    workspace = Path(experiment_full_artifacts['temp_dir'])
    artifacts = experiment_full_artifacts['artifacts']
    
    # Check for model files
    model_files = artifacts['*.pth']
    assert len(model_files) > 0
    
    # Verify logs exist
    log_files = artifacts['*.log']
    assert len(log_files) > 0
```

### 4. Adaptive Configuration Fixtures

#### `experiment_config`
- **Purpose**: Provides configuration based on pytest markers
- **Best for**: Tests that need to specify their data requirements
- **Returns**: Configuration dictionary based on test markers

```python
@pytest.mark.experiment_data(scope='minimal', require_metrics=True)
def test_with_config(experiment_config):
    config = experiment_config
    assert config['data_scope'] == 'minimal'
    assert config['require_metrics'] is True
```

#### `adaptive_experiment_data`
- **Purpose**: Automatically provides appropriate data based on test markers
- **Best for**: Tests that want automatic optimization based on declared needs
- **Returns**: Data tailored to marker specifications

```python
@pytest.mark.experiment_data(scope='db_only')
def test_database_only(adaptive_experiment_data):
    # Automatically receives just the database path
    assert isinstance(adaptive_experiment_data, str)
    
@pytest.mark.experiment_data(scope='metrics')
def test_with_metrics(adaptive_experiment_data):
    # Automatically receives pre-loaded metrics data
    assert 'metrics' in adaptive_experiment_data
```

## Pytest Markers

### Custom Markers for Data Requirements

#### `@pytest.mark.experiment_data(scope=..., require_...=...)`
Specify exactly what experiment data your test needs:

```python
# Minimal data for fast tests
@pytest.mark.experiment_data(scope='minimal')
def test_basic_validation():
    pass

# Pre-loaded metrics for analysis
@pytest.mark.experiment_data(scope='metrics', require_metrics=True)
def test_training_curves():
    pass

# Full data for integration tests
@pytest.mark.experiment_data(scope='full', require_artifacts=True, require_workspace=True)
def test_complete_workflow():
    pass

# Database only for custom operations
@pytest.mark.experiment_data(scope='db_only')
def test_custom_queries():
    pass
```

#### Available marker parameters:
- `scope`: 'minimal', 'metrics', 'full', 'db_only'
- `require_metrics`: Boolean (default: True)
- `require_artifacts`: Boolean (default: False)
- `require_workspace`: Boolean (default: False)
- `performance_mode`: Boolean (default: False)

### Automatic Test Categorization Markers

The system automatically applies these markers based on fixture usage:

- `@pytest.mark.fast_test`: Applied to tests using lightweight fixtures
- `@pytest.mark.slow_test`: Applied to tests using full artifact fixtures
- `@pytest.mark.integration_test`: Applied to tests requiring full workspace access

## Performance Guidelines

### Choosing the Right Fixture

1. **For unit tests**: Use `experiment_lightweight` or `experiment_db_only`
2. **For metrics analysis**: Use `experiment_metrics_only`
3. **For integration tests**: Use `experiment_full_artifacts`
4. **For general testing**: Use `experiment_data`
5. **For adaptive behavior**: Use `adaptive_experiment_data` with markers

### Performance Characteristics

| Fixture | Setup Time | Memory Usage | Best Use Case |
|---------|------------|--------------|---------------|
| `experiment_db_only` | ~0ms | Minimal | Custom DB operations |
| `experiment_lightweight` | ~50ms | Low | Property validation |
| `experiment_metrics_only` | ~200ms | Medium | Metrics analysis |
| `experiment_data` | ~300ms | Medium | General testing |
| `experiment_full_artifacts` | ~500ms | High | Integration testing |

## Usage Patterns

### 1. Test Module Organization

```python
# tests/trackers/test_db_tracker.py
import pytest

class TestDBTrackerWithRealData:
    """Tests using real experiment data for DB tracker functionality."""
    
    def test_lightweight_validation(self, experiment_lightweight):
        # Fast validation tests
        pass
    
    def test_metrics_tracking(self, experiment_metrics_only):
        # Metrics-focused tests
        pass
    
    def test_full_integration(self, experiment_full_artifacts):
        # Complete workflow tests
        pass
```

### 2. Conditional Test Execution

```python
# Run only fast tests
pytest -m "fast_test"

# Run only integration tests
pytest -m "integration_test"

# Run tests that require metrics
pytest -m "experiment_data and require_metrics"

# Skip slow tests during development
pytest -m "not slow_test"
```

### 3. Custom Fixture Combinations

```python
@pytest.fixture
def enhanced_metrics_data(experiment_metrics_only):
    """Custom fixture that enhances metrics data with additional processing."""
    metrics_df = experiment_metrics_only['metrics']
    
    # Add computed columns
    metrics_df['epoch_normalized'] = metrics_df['epoch'] / metrics_df['epoch'].max()
    
    return {
        **experiment_metrics_only,
        'processed_metrics': metrics_df
    }

def test_enhanced_analysis(enhanced_metrics_data):
    processed_df = enhanced_metrics_data['processed_metrics']
    assert 'epoch_normalized' in processed_df.columns
```

## Best Practices

### 1. Fixture Selection

- **Start with the lightest fixture** that meets your needs
- **Use specialized fixtures** (`experiment_metrics_only`) over general ones when possible
- **Only use `experiment_full_artifacts`** for tests that truly need complete workspace access

### 2. Test Isolation

- **Never modify** the experiment data returned by fixtures
- **Create copies** if you need to modify data for testing
- **Use fresh DBDataSource instances** for tests that need custom database operations

### 3. Marker Usage

- **Always specify scope** in `@pytest.mark.experiment_data()` markers
- **Use descriptive markers** to make test requirements clear
- **Combine markers** for complex requirements

### 4. Error Handling

```python
def test_with_error_handling(experiment_data):
    try:
        # Test operations that might fail
        result = risky_operation(experiment_data)
        assert result is not None
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
    finally:
        # Cleanup is handled automatically by fixtures
        pass
```

## Migration Guide

### From Manual Experiment Creation

**Before:**
```python
def test_something():
    # Manual experiment setup
    temp_dir = tempfile.mkdtemp()
    framework_exp, db_path = run_mnist_baseline_experiment(temp_dir)
    # ... test code ...
    shutil.rmtree(temp_dir)  # Manual cleanup
```

**After:**
```python
def test_something(experiment_data):
    # Automatic experiment data provision
    experiment = experiment_data['experiment']
    db_path = experiment_data['db_path']
    # ... test code ...
    # Cleanup handled automatically
```

### From Synthetic Data

**Before:**
```python
def test_tracker():
    # Create mock experiment data
    mock_experiment = create_mock_experiment()
    # ... test with synthetic data ...
```

**After:**
```python
def test_tracker(experiment_lightweight):
    # Use real experiment metadata
    exp_name = experiment_lightweight['experiment_name']
    trial_count = experiment_lightweight['trial_count']
    # ... test with real data structure ...
```

## Troubleshooting

### Common Issues

1. **Fixture not found**: Ensure you're importing from the correct test module
2. **Slow test execution**: Use lighter fixtures or appropriate markers
3. **Data not as expected**: Check fixture documentation for exact return format
4. **Memory issues**: Use `experiment_db_only` for memory-constrained environments

### Debug Tips

```python
def test_debug_fixture_data(experiment_data):
    """Debug test to examine fixture contents."""
    print(f"Available keys: {list(experiment_data.keys())}")
    print(f"Experiment name: {experiment_data['experiment'].name}")
    print(f"Trial count: {len(experiment_data['experiment'].trials)}")
    print(f"DB path: {experiment_data['db_path']}")
```

## Examples by Test Category

### Tracker Tests
```python
def test_db_tracker_with_real_data(experiment_db_only):
    """Test DB tracker using real experiment database."""
    with DBDataSource(experiment_db_only) as source:
        experiment = source.get_experiment()
        # Test tracker functionality with real data
```

### Performance Tests
```python
@pytest.mark.experiment_data(scope='full', require_workspace=True)
def test_performance_monitoring(adaptive_experiment_data):
    """Test performance monitoring during actual training."""
    workspace = adaptive_experiment_data['temp_dir']
    # Monitor performance with real workspace
```

### Callback Tests
```python
def test_early_stopping_with_real_curves(experiment_metrics_only):
    """Test early stopping with actual training curves."""
    metrics_df = experiment_metrics_only['metrics']
    loss_data = metrics_df[metrics_df['metric'] == 'val_loss']
    # Test early stopping logic with real convergence patterns
```

This fixture system provides a robust foundation for testing all framework components with real MNIST experiment data while maintaining optimal performance characteristics.

---

## Framework Hierarchical Levels

The fixtures align with the framework's six-level hierarchy:

1. **EXPERIMENT** - Top-level experiment container
2. **TRIAL** - Individual experiment configurations
3. **TRIAL_RUN** - Single execution of a trial
4. **PIPELINE** - Execution context for training workflows
5. **EPOCH** - Individual training epochs
6. **BATCH** - Individual batch processing (finest granularity)

---

*Last Updated: December 2024* 