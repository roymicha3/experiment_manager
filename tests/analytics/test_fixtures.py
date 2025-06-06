"""
Test fixtures and mock data for analytics module tests.
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

from experiment_manager.analytics import AnalyticsEngine, AnalyticsResult
from experiment_manager.common.common import RunStatus, Metric
from experiment_manager.analytics.processors.base import DataProcessor


@pytest.fixture
def mock_database_manager():
    """Create a mock database manager for testing."""
    mock_db = Mock()
    mock_db.get_connection = Mock()
    
    # Mock analytics methods to return proper DataFrames
    def get_analytics_data_side_effect(experiment_ids=None, filters=None):
        return pd.DataFrame(columns=[
            'experiment_id', 'experiment_title', 'experiment_description', 'experiment_start_time',
            'trial_id', 'trial_name', 'trial_start_time', 'trial_run_id', 'run_status',
            'run_start_time', 'run_update_time', 'metric_id', 'metric_type', 
            'metric_total_val', 'metric_per_label_val', 'epoch_idx', 'epoch_time'
        ])
    
    def get_aggregated_metrics_side_effect(experiment_ids=None, group_by='trial', functions=None):
        # Validate inputs like the real method
        valid_group_by = ['experiment', 'trial', 'trial_run']
        if group_by not in valid_group_by:
            raise ValueError(f"group_by must be one of {valid_group_by}")
        
        valid_functions = ['mean', 'std', 'min', 'max', 'count', 'sum']
        if functions:
            invalid_functions = [f for f in functions if f not in valid_functions]
            if invalid_functions:
                raise ValueError(f"Invalid aggregation functions: {invalid_functions}")
        
        return pd.DataFrame()
    
    def get_failure_data_side_effect(experiment_ids=None, include_configs=False):
        cols = ['experiment_id', 'trial_id', 'trial_run_id', 'run_status', 
                'run_start_time', 'run_update_time', 'duration_seconds']
        if include_configs:
            cols.extend(['config_location', 'config_type'])
        return pd.DataFrame(columns=cols)
    
    def get_epoch_series_side_effect(trial_run_ids, metric_types=None):
        if not trial_run_ids:
            return pd.DataFrame(columns=['trial_run_id', 'epoch_idx', 'epoch_time', 
                                       'metric_type', 'metric_total_val', 'metric_per_label_val'])
        return pd.DataFrame(columns=['trial_run_id', 'epoch_idx', 'epoch_time', 
                                   'metric_type', 'metric_total_val', 'metric_per_label_val'])
    
    def execute_query_side_effect(query, params=None):
        return pd.DataFrame()
    
    def create_analytics_indexes_side_effect():
        pass
    
    # Configure the mock with side effects
    mock_db.get_analytics_data.side_effect = get_analytics_data_side_effect
    mock_db.get_aggregated_metrics.side_effect = get_aggregated_metrics_side_effect
    mock_db.get_failure_data.side_effect = get_failure_data_side_effect
    mock_db.get_epoch_series.side_effect = get_epoch_series_side_effect
    mock_db.execute_query.side_effect = execute_query_side_effect
    mock_db.create_analytics_indexes.side_effect = create_analytics_indexes_side_effect
    
    return mock_db


@pytest.fixture
def sample_experiment_data():
    """Create sample experiment data for testing."""
    data = pd.DataFrame({
        'experiment_id': [1, 1, 1, 2, 2, 2, 3, 3],
        'experiment_name': ['exp1', 'exp1', 'exp1', 'exp2', 'exp2', 'exp2', 'exp3', 'exp3'],
        'trial_id': [1, 2, 3, 1, 2, 3, 1, 2],
        'trial_name': ['trial_1', 'trial_2', 'trial_3', 'trial_1', 'trial_2', 'trial_3', 'trial_1', 'trial_2'],
        'run_id': [1, 1, 1, 1, 1, 1, 1, 1],
        'metric_type': ['test_acc', 'test_acc', 'test_acc', 'test_acc', 'test_acc', 'test_acc', 'test_acc', 'test_acc'],
        'metric_value': [0.85, 0.87, 0.82, 0.89, 0.91, 0.78, 0.92, 0.88],
        'status': ['success', 'success', 'success', 'success', 'success', 'success', 'success', 'success'],
        'created_at': pd.to_datetime([
            '2023-01-01', '2023-01-02', '2023-01-03', 
            '2023-01-04', '2023-01-05', '2023-01-06',
            '2023-01-07', '2023-01-08'
        ])
    })
    return data


@pytest.fixture
def sample_experiment_data_with_failures():
    """Create sample experiment data including failures."""
    data = pd.DataFrame({
        'experiment_id': [1, 1, 1, 1, 1],
        'trial_id': [1, 2, 3, 4, 5],
        'run_id': [1, 1, 1, 1, 1],
        'metric_type': ['test_acc', 'test_acc', 'test_acc', 'test_acc', 'test_acc'],
        'metric_value': [0.85, None, 0.82, 0.89, None],  # None indicates failed runs
        'status': ['success', 'failed', 'success', 'success', 'timeout'],
        'created_at': pd.to_datetime([
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'
        ])
    })
    return data


@pytest.fixture
def sample_training_data():
    """Create sample training curve data."""
    data = pd.DataFrame({
        'experiment_id': [1, 1, 1, 1, 1, 1],
        'trial_id': [1, 1, 1, 1, 1, 1],
        'run_id': [1, 1, 1, 1, 1, 1],
        'epoch': [1, 2, 3, 4, 5, 6],
        'metric_type': ['train_loss', 'train_loss', 'train_loss', 'train_loss', 'train_loss', 'train_loss'],
        'metric_value': [2.5, 1.8, 1.2, 0.9, 0.7, 0.6],
        'context': ['training', 'training', 'training', 'training', 'training', 'training'],
        'created_at': pd.to_datetime([
            '2023-01-01 10:00:00', '2023-01-01 10:05:00', '2023-01-01 10:10:00',
            '2023-01-01 10:15:00', '2023-01-01 10:20:00', '2023-01-01 10:25:00'
        ])
    })
    return data


@pytest.fixture
def mock_analytics_engine(mock_database_manager):
    """Create a mock analytics engine for testing."""
    engine = AnalyticsEngine(mock_database_manager)
    return engine


@pytest.fixture
def sample_analytics_result(sample_experiment_data):
    """Create a sample AnalyticsResult for testing."""
    result = AnalyticsResult(
        data=sample_experiment_data,
        metadata={'test': True, 'source': 'fixture'},
        query_info={'complexity': 'low', 'filters': 1}
    )
    
    # Add some summary statistics
    result.add_summary_statistic('mean_accuracy', 0.865, 'Average test accuracy')
    result.add_summary_statistic('std_accuracy', 0.045, 'Standard deviation of test accuracy')
    
    # Add processing steps
    result.add_processing_step('mock_processor', {'param1': 'value1'}, 'Mock processing completed')
    
    return result


@pytest.fixture
def outlier_data():
    """Create data with outliers for testing outlier detection."""
    # Normal data points + outliers
    normal_values = [0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.85, 0.87]
    outliers = [0.45, 0.98]  # Clear outliers
    
    all_values = normal_values + outliers
    
    data = pd.DataFrame({
        'experiment_id': [1] * len(all_values),
        'trial_id': list(range(1, len(all_values) + 1)),
        'run_id': [1] * len(all_values),
        'metric_type': ['test_acc'] * len(all_values),
        'metric_value': all_values,
        'status': ['success'] * len(all_values)
    })
    
    return data


@pytest.fixture
def multi_experiment_data():
    """Create data for multiple experiments for comparison testing."""
    experiments = []
    
    for exp_id in [1, 2, 3]:
        for trial_id in [1, 2, 3]:
            experiments.append({
                'experiment_id': exp_id,
                'trial_id': trial_id,
                'run_id': 1,
                'metric_type': 'test_acc',
                'metric_value': 0.80 + (exp_id * 0.05) + (trial_id * 0.01),  # Varying performance
                'status': 'success'
            })
    
    return pd.DataFrame(experiments)


class MockProcessor:
    """Mock processor for testing processor functionality."""
    
    def __init__(self, name="MockProcessor"):
        self.name = name
    
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Mock processing that adds a processed column."""
        result = data.copy()
        result['processed_by'] = self.name
        return result
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Mock validation that always returns True."""
        return True
    
    def get_required_columns(self) -> List[str]:
        """Mock required columns."""
        return ['metric_value']


@pytest.fixture
def mock_processor():
    """Create a mock processor for testing."""
    return MockProcessor()


def create_mock_query_result(data: pd.DataFrame) -> AnalyticsResult:
    """Helper function to create mock query results."""
    return AnalyticsResult(
        data=data,
        metadata={'mock': True},
        query_info={'mock_query': True}
    )


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_time_series_data(experiment_id: int, 
                                 metric_type: str,
                                 num_points: int = 10,
                                 base_value: float = 0.8,
                                 noise_level: float = 0.05) -> pd.DataFrame:
        """Generate time series data for testing."""
        import numpy as np
        
        np.random.seed(42)  # For reproducible tests
        
        timestamps = [
            datetime.now() - timedelta(hours=num_points-i) 
            for i in range(num_points)
        ]
        
        values = [
            base_value + np.random.normal(0, noise_level) 
            for _ in range(num_points)
        ]
        
        return pd.DataFrame({
            'experiment_id': [experiment_id] * num_points,
            'trial_id': [1] * num_points,
            'run_id': [1] * num_points,
            'metric_type': [metric_type] * num_points,
            'metric_value': values,
            'created_at': timestamps,
            'status': ['success'] * num_points
        })
    
    @staticmethod
    def generate_multi_metric_data(experiment_id: int,
                                  metrics: List[str],
                                  num_trials: int = 3) -> pd.DataFrame:
        """Generate multi-metric data for testing."""
        data = []
        
        for trial_id in range(1, num_trials + 1):
            for metric in metrics:
                # Generate realistic values based on metric type
                if 'acc' in metric:
                    base_value = 0.85
                elif 'loss' in metric:
                    base_value = 0.25
                else:
                    base_value = 0.5
                
                data.append({
                    'experiment_id': experiment_id,
                    'trial_id': trial_id,
                    'run_id': 1,
                    'metric_type': metric,
                    'metric_value': base_value + (trial_id * 0.01),
                    'status': 'success'
                })
        
        return pd.DataFrame(data)


@pytest.fixture
def test_data_generator():
    """Provide TestDataGenerator for tests."""
    return TestDataGenerator 


# Add concrete processor implementations for testing
class ConcreteProcessor(DataProcessor):
    """Test processor implementation."""
    
    def __init__(self, name="TestProcessor"):
        super().__init__(name)
        self.required_columns = ['metric_value']
    
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Test implementation that adds a processed column."""
        result = data.copy()
        result['processed'] = True
        result['processor_name'] = self.name
        return result


class ErrorProcessor(DataProcessor):
    """Test processor that raises errors."""
    
    def __init__(self, name="ErrorProcessor"):
        super().__init__(name)
        self.required_columns = ['metric_value']
    
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Implementation that raises an error."""
        raise RuntimeError("Processing failed")


class ParameterizedProcessor(DataProcessor):
    """Test processor that uses parameters."""
    
    def __init__(self, name="ParameterizedProcessor"):
        super().__init__(name)
        self.required_columns = ['metric_value']
    
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Implementation that uses parameters."""
        result = data.copy()
        multiplier = kwargs.get('multiplier', 1.0)
        result['metric_value'] = result['metric_value'] * multiplier
        result['multiplier_used'] = multiplier
        return result 