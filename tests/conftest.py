"""
Global test fixtures for the experiment_manager test suite.

This module contains session-scoped fixtures that are shared across all test modules,
particularly the MNIST baseline experiment fixture that provides real experiment data
for testing various framework components.
"""
import os
import tempfile
import pytest
import shutil
import atexit
import time
from pathlib import Path

from experiment_manager.results.sources.db_datasource import DBDataSource
from tests.run_mnist_baseline import run_mnist_baseline_experiment


@pytest.fixture(scope="session")
def shared_mnist_experiment():
    """
    Run MNIST baseline experiment once per test session and share the results.
    
    This fixture ensures the experiment runs only once, regardless of how many
    test classes or test files need the data. It provides real experiment data
    for testing various framework components including trackers, analytics,
    visualization, and database systems.
    
    Returns:
        dict: A dictionary containing:
            - framework_experiment: The experiment object from the framework
            - db_path: Path to the experiment database
            - temp_dir: Temporary directory containing all experiment artifacts
    
    Note:
        This is a session-scoped fixture that automatically handles cleanup
        at the end of the test session.
    """
    # Create a manually managed temporary directory
    temp_dir = tempfile.mkdtemp(prefix='shared_mnist_experiment_')
    
    print(f"\n🚀 Running shared MNIST experiment in: {temp_dir}")
    
    try:
        # Run the baseline experiment
        framework_experiment, db_path = run_mnist_baseline_experiment(temp_dir)
        
        # Verify database exists
        assert os.path.exists(db_path), f"Database not created at {db_path}"
        
        # Pre-load experiment data to verify it's valid
        data_source = DBDataSource(db_path)
        with data_source as source:
            experiment = source.get_experiment()
            
            # Basic validation
            assert experiment is not None
            assert experiment.name == "test_mnist_baseline"
            assert len(experiment.trials) == 3
            
        data_source.close()
        
        print(f"✅ MNIST experiment completed successfully!")
        print(f"📁 Workspace: {temp_dir}")
        print(f"🗄️  Database: {db_path}")
        
        # Return the shared data
        experiment_data = {
            'framework_experiment': framework_experiment,
            'db_path': db_path,
            'temp_dir': temp_dir
        }
        
        yield experiment_data
        
    finally:
        # Clean up at the end of the test session
        print(f"\n🧹 Cleaning up shared MNIST experiment: {temp_dir}")
        _cleanup_temp_dir(temp_dir)


def _cleanup_temp_dir(temp_dir):
    """Clean up temporary directory with Windows-compatible retry logic."""
    if not os.path.exists(temp_dir):
        return
        
    # Try multiple cleanup strategies
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Small delay to let handles close
            time.sleep(0.2 * (attempt + 1))
            
            # Try to remove the directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Check if it worked
            if not os.path.exists(temp_dir):
                print(f"✅ Cleanup successful on attempt {attempt + 1}")
                return  # Success!
                
        except Exception as e:
            print(f"⚠️  Cleanup attempt {attempt + 1} failed: {e}")
    
    # If we get here, cleanup failed - register for exit cleanup
    print(f"⚠️  Cleanup failed, registering for exit cleanup")
    def cleanup_on_exit():
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
    
    atexit.register(cleanup_on_exit)


@pytest.fixture
def experiment_data(shared_mnist_experiment):
    """
    Provide experiment data for individual tests.
    
    This fixture loads fresh data from the shared experiment database
    for each test, ensuring test isolation while sharing the underlying data.
    It's automatically available to all test modules and provides consistent
    access to MNIST experiment data.
    
    Args:
        shared_mnist_experiment: The session-scoped MNIST experiment fixture
        
    Returns:
        dict: A dictionary containing:
            - db_path: Path to the experiment database
            - experiment: Loaded experiment object with trials and runs
            - temp_dir: Temporary directory containing experiment artifacts
            - framework_experiment: The original framework experiment object
    
    Note:
        This fixture creates a fresh DBDataSource for each test to ensure
        test isolation while reusing the underlying experiment data.
    """
    db_path = shared_mnist_experiment['db_path']
    
    # Create a fresh DBDataSource for this test
    data_source = DBDataSource(db_path)
    
    try:
        with data_source as source:
            experiment = source.get_experiment()
            
            # Return test-specific data (read-only)
            yield {
                'db_path': db_path,
                'experiment': experiment,
                'temp_dir': shared_mnist_experiment['temp_dir'],
                'framework_experiment': shared_mnist_experiment['framework_experiment']
            }
    finally:
        # Ensure data source is closed for this test
        data_source.close()


# =============================================================================
# PARAMETERIZED EXPERIMENT FIXTURES
# =============================================================================
# These fixtures provide different levels of access to the MNIST experiment data,
# allowing tests to request only the data they need for optimal performance.

@pytest.fixture
def experiment_db_only(shared_mnist_experiment):
    """
    Lightweight fixture providing only the database path.
    
    This fixture is ideal for tests that need to create their own DBDataSource
    instances or perform custom database operations. It has minimal overhead
    and doesn't pre-load any experiment data.
    
    Args:
        shared_mnist_experiment: The session-scoped MNIST experiment fixture
        
    Returns:
        str: Path to the experiment database file
        
    Usage:
        def test_custom_db_operations(experiment_db_only):
            db_path = experiment_db_only
            # Create custom DBDataSource instances as needed
            with DBDataSource(db_path) as source:
                # Custom database operations
                pass
    """
    return shared_mnist_experiment['db_path']


@pytest.fixture
def experiment_metrics_only(shared_mnist_experiment):
    """
    Pre-loaded metrics data fixture for tests focusing on metrics analysis.
    
    This fixture pre-loads all metrics data from the experiment database,
    making it ideal for tests that analyze training curves, performance
    metrics, or validation results without needing access to artifacts.
    
    Args:
        shared_mnist_experiment: The session-scoped MNIST experiment fixture
        
    Returns:
        dict: A dictionary containing:
            - db_path: Path to the experiment database
            - metrics: Pre-loaded metrics DataFrame with all experiment metrics
            - experiment_id: The experiment ID for reference
            - trial_count: Number of trials in the experiment
            
    Usage:
        def test_metrics_analysis(experiment_metrics_only):
            metrics_df = experiment_metrics_only['metrics']
            # Analyze training curves, validation metrics, etc.
            assert not metrics_df.empty
    """
    db_path = shared_mnist_experiment['db_path']
    
    # Pre-load metrics data
    data_source = DBDataSource(db_path)
    
    try:
        with data_source as source:
            experiment = source.get_experiment()
            metrics_df = source.metrics_dataframe(experiment)
            
            yield {
                'db_path': db_path,
                'metrics': metrics_df,
                'experiment_id': experiment.id,
                'trial_count': len(experiment.trials)
            }
    finally:
        data_source.close()


@pytest.fixture
def experiment_full_artifacts(shared_mnist_experiment):
    """
    Complete experiment access with artifacts and workspace.
    
    This fixture provides full access to the experiment workspace, including
    all artifacts, model files, logs, and database. It's intended for
    comprehensive integration tests that need to verify the complete
    experiment lifecycle.
    
    Args:
        shared_mnist_experiment: The session-scoped MNIST experiment fixture
        
    Returns:
        dict: A dictionary containing:
            - db_path: Path to the experiment database
            - temp_dir: Full workspace directory with all artifacts
            - experiment: Loaded experiment object with all metadata
            - framework_experiment: The original framework experiment object
            - artifacts: Dictionary of artifact paths by type
            
    Usage:
        def test_full_integration(experiment_full_artifacts):
            workspace = Path(experiment_full_artifacts['temp_dir'])
            artifacts = experiment_full_artifacts['artifacts']
            # Test complete experiment lifecycle
    """
    db_path = shared_mnist_experiment['db_path']
    temp_dir = shared_mnist_experiment['temp_dir']
    
    # Load experiment and catalog artifacts
    data_source = DBDataSource(db_path)
    
    try:
        with data_source as source:
            experiment = source.get_experiment()
            
            # Catalog available artifacts
            artifacts = {}
            workspace_path = Path(temp_dir)
            
            # Look for common artifact types
            for pattern in ['*.pth', '*.pkl', '*.log', '*.json', '*.yaml', '*.csv']:
                artifacts[pattern] = list(workspace_path.rglob(pattern))
            
            yield {
                'db_path': db_path,
                'temp_dir': temp_dir,
                'experiment': experiment,
                'framework_experiment': shared_mnist_experiment['framework_experiment'],
                'artifacts': artifacts
            }
    finally:
        data_source.close()


@pytest.fixture
def experiment_lightweight(shared_mnist_experiment):
    """
    Minimal experiment data for fast tests.
    
    This fixture provides only the most essential experiment metadata
    without loading heavy data structures or artifacts. It's perfect for
    unit tests that need to verify basic experiment properties or
    perform simple validation checks.
    
    Args:
        shared_mnist_experiment: The session-scoped MNIST experiment fixture
        
    Returns:
        dict: A dictionary containing:
            - experiment_name: Name of the experiment
            - trial_count: Number of trials
            - db_path: Path to database (for lazy loading if needed)
            - workspace_exists: Boolean indicating if workspace directory exists
            
    Usage:
        def test_basic_experiment_properties(experiment_lightweight):
            assert experiment_lightweight['experiment_name'] == "test_mnist_baseline"
            assert experiment_lightweight['trial_count'] == 3
    """
    db_path = shared_mnist_experiment['db_path']
    temp_dir = shared_mnist_experiment['temp_dir']
    
    # Only load minimal metadata
    data_source = DBDataSource(db_path)
    
    try:
        with data_source as source:
            experiment = source.get_experiment()
            
            yield {
                'experiment_name': experiment.name,
                'trial_count': len(experiment.trials),
                'db_path': db_path,
                'workspace_exists': os.path.exists(temp_dir)
            }
    finally:
        data_source.close()


# =============================================================================
# FIXTURE CONFIGURATION SYSTEM
# =============================================================================
# These fixtures and markers allow tests to specify their experiment data needs

@pytest.fixture
def experiment_config(request):
    """
    Configuration fixture for experiment data requirements.
    
    This fixture allows tests to specify what level of experiment data
    they need using pytest markers. It examines the test's markers to
    determine the appropriate configuration.
    
    Args:
        request: pytest request object containing test metadata
        
    Returns:
        dict: Configuration options for experiment data access
        
    Usage:
        @pytest.mark.experiment_data(scope='minimal', require_metrics=True)
        def test_with_metrics(experiment_config):
            config = experiment_config
            assert config['data_scope'] == 'minimal'
            assert config['require_metrics'] is True
    """
    # Default configuration
    config = {
        'data_scope': 'full',  # Options: 'minimal', 'metrics', 'full', 'db_only'
        'require_artifacts': False,
        'require_metrics': True,
        'require_workspace': False,
        'cache_data': True,
        'performance_mode': False
    }
    
    # Check for experiment_data marker
    experiment_marker = request.node.get_closest_marker('experiment_data')
    if experiment_marker:
        # Update config with marker values
        marker_config = experiment_marker.kwargs
        config.update(marker_config)
        
        # Handle 'scope' parameter separately as it maps to 'data_scope'
        if 'scope' in marker_config:
            config['data_scope'] = marker_config['scope']
    
    return config


@pytest.fixture
def adaptive_experiment_data(request, shared_mnist_experiment):
    """
    Adaptive fixture that provides experiment data based on test requirements.
    
    This fixture examines pytest markers on the test to determine what level
    of experiment data to provide, optimizing performance by loading only
    what's needed.
    
    Args:
        request: pytest request object containing test metadata
        shared_mnist_experiment: The session-scoped MNIST experiment fixture
        
    Returns:
        dict: Experiment data tailored to test requirements
        
    Usage:
        @pytest.mark.experiment_data(scope='db_only')
        def test_database_only(adaptive_experiment_data):
            # Will receive only database path
            assert isinstance(adaptive_experiment_data, str)
            
        @pytest.mark.experiment_data(scope='metrics', require_metrics=True)
        def test_with_metrics(adaptive_experiment_data):
            # Will receive pre-loaded metrics data
            assert 'metrics' in adaptive_experiment_data
    """
    # Get configuration from marker
    experiment_marker = request.node.get_closest_marker('experiment_data')
    scope = 'full'  # default
    
    if experiment_marker:
        scope = experiment_marker.kwargs.get('scope', 'full')
    
    # Route to appropriate fixture based on scope
    if scope == 'db_only':
        return shared_mnist_experiment['db_path']
    elif scope == 'minimal':
        return _get_lightweight_data(shared_mnist_experiment)
    elif scope == 'metrics':
        return _get_metrics_data(shared_mnist_experiment)
    elif scope == 'full':
        return _get_full_data(shared_mnist_experiment)
    else:
        raise ValueError(f"Unknown experiment data scope: {scope}")


def _get_lightweight_data(shared_mnist_experiment):
    """Helper function to get lightweight experiment data."""
    db_path = shared_mnist_experiment['db_path']
    temp_dir = shared_mnist_experiment['temp_dir']
    
    data_source = DBDataSource(db_path)
    try:
        with data_source as source:
            experiment = source.get_experiment()
            return {
                'experiment_name': experiment.name,
                'trial_count': len(experiment.trials),
                'db_path': db_path,
                'workspace_exists': os.path.exists(temp_dir)
            }
    finally:
        data_source.close()


def _get_metrics_data(shared_mnist_experiment):
    """Helper function to get metrics-focused experiment data."""
    db_path = shared_mnist_experiment['db_path']
    
    data_source = DBDataSource(db_path)
    try:
        with data_source as source:
            experiment = source.get_experiment()
            metrics_df = source.metrics_dataframe(experiment)
            
            return {
                'db_path': db_path,
                'metrics': metrics_df,
                'experiment_id': experiment.id,
                'trial_count': len(experiment.trials)
            }
    finally:
        data_source.close()


def _get_full_data(shared_mnist_experiment):
    """Helper function to get complete experiment data."""
    db_path = shared_mnist_experiment['db_path']
    temp_dir = shared_mnist_experiment['temp_dir']
    
    data_source = DBDataSource(db_path)
    try:
        with data_source as source:
            experiment = source.get_experiment()
            
            # Catalog artifacts
            artifacts = {}
            workspace_path = Path(temp_dir)
            for pattern in ['*.pth', '*.pkl', '*.log', '*.json', '*.yaml', '*.csv']:
                artifacts[pattern] = list(workspace_path.rglob(pattern))
            
            return {
                'db_path': db_path,
                'temp_dir': temp_dir,
                'experiment': experiment,
                'framework_experiment': shared_mnist_experiment['framework_experiment'],
                'artifacts': artifacts
            }
    finally:
        data_source.close()


# =============================================================================
# PYTEST CONFIGURATION AND MARKERS
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers for experiment data management."""
    config.addinivalue_line(
        "markers", 
        "experiment_data(scope='full', require_metrics=True, require_artifacts=False, "
        "require_workspace=False, performance_mode=False): "
        "Specify experiment data requirements for the test. "
        "scope can be 'minimal', 'metrics', 'full', or 'db_only'."
    )
    config.addinivalue_line(
        "markers",
        "slow_test: Mark test as slow (requires full experiment data)"
    )
    config.addinivalue_line(
        "markers", 
        "fast_test: Mark test as fast (uses minimal experiment data)"
    )
    config.addinivalue_line(
        "markers",
        "integration_test: Mark test as integration test (may require full artifacts)"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their experiment data usage.
    
    This hook examines test functions and automatically applies appropriate
    markers based on the fixtures they use, helping with test categorization
    and execution optimization.
    """
    for item in items:
        # Get fixture names used by this test
        fixture_names = getattr(item, 'fixturenames', [])
        
        # Auto-mark based on fixture usage
        if 'experiment_full_artifacts' in fixture_names:
            item.add_marker(pytest.mark.slow_test)
            item.add_marker(pytest.mark.integration_test)
        elif 'experiment_lightweight' in fixture_names or 'experiment_db_only' in fixture_names:
            item.add_marker(pytest.mark.fast_test)
        elif 'experiment_metrics_only' in fixture_names:
            item.add_marker(pytest.mark.fast_test)
        
        # Mark any test using experiment data
        experiment_fixtures = [
            'experiment_data', 'experiment_db_only', 'experiment_metrics_only',
            'experiment_full_artifacts', 'experiment_lightweight', 'adaptive_experiment_data'
        ]
        
        if any(fixture in fixture_names for fixture in experiment_fixtures):
            # Add custom marker to indicate this test uses experiment data
            if not item.get_closest_marker('experiment_data'):
                # Apply default experiment_data marker if none exists
                item.add_marker(pytest.mark.experiment_data(scope='full')) 
