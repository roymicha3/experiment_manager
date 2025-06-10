"""
Shared fixtures for DBDataSource tests.
"""
import os
import tempfile
import pytest
import shutil
import atexit
import time
from pathlib import Path

from experiment_manager.results.sources.db_data_source import DBDataSource
from tests.run_mnist_baseline import run_mnist_baseline_experiment


@pytest.fixture(scope="session")
def shared_mnist_experiment():
    """
    Run MNIST baseline experiment once per test session and share the results.
    
    This fixture ensures the experiment runs only once, regardless of how many
    test classes or test files need the data.
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