import os
import pytest
import tempfile
import pandas as pd
from unittest.mock import Mock, patch
from omegaconf import OmegaConf

from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.metric_tracker import MetricsTracker
from experiment_manager.analytics.api import ExperimentAnalytics
from experiment_manager.analytics.results import AnalyticsResult
from experiment_manager.common.common import Metric, RunStatus


@pytest.fixture
def test_env():
    """Create a test environment for analytics testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = OmegaConf.create({
            "workspace": tmp_dir, 
            "verbose": False,
            "trackers": []  # No trackers to avoid complications
        })
        env = Environment(workspace=tmp_dir, config=config)
        yield env
        # Clean up file handles
        if hasattr(env.logger, 'close'):
            env.logger.close()
        elif hasattr(env.logger, 'handlers'):
            for handler in env.logger.handlers[:]:
                handler.close()
                env.logger.removeHandler(handler)


@pytest.fixture
def mock_database_manager():
    """Create a mock database manager for analytics testing."""
    mock_db = Mock()
    
    # Mock analytics data that would come from a real experiment
    sample_data = pd.DataFrame({
        'experiment_id': [1, 1, 1],
        'trial_id': [1, 1, 2],
        'trial_run_id': [1, 2, 3],
        'run_status': ['success', 'success', 'success'],
        'metric_type': ['test_acc', 'test_acc', 'test_acc'],
        'metric_total_val': [0.85, 0.87, 0.82],
        'epoch_idx': [4, 4, 4]
    })
    
    # Setup mock methods
    mock_db.get_analytics_data.return_value = sample_data
    mock_db.get_aggregated_metrics.return_value = pd.DataFrame({
        'experiment_id': [1],
        'metric_type': ['test_acc'],
        'mean': [0.85],
        'std': [0.025],
        'count': [3]
    })
    mock_db.get_failure_data.return_value = pd.DataFrame()
    mock_db.get_epoch_series.return_value = pd.DataFrame()
    
    return mock_db


class TestMetricsTrackingIntegration:
    """Test metrics tracking and analytics integration."""
    
    def test_metrics_tracker_initialization(self, test_env):
        """Test that MetricsTracker initializes correctly."""
        tracker = MetricsTracker(env=test_env)
        
        assert tracker.env == test_env
        assert len(tracker.metrics) == 0
        assert tracker.log_path.endswith("metrics.log")
    
    def test_metrics_tracker_on_start(self, test_env):
        """Test metrics tracker on_start functionality."""
        tracker = MetricsTracker(env=test_env)
        
        # Add some initial metrics
        tracker.metrics[Metric.TRAIN_LOSS] = [1.0, 0.8]
        
        # Call on_start - should clear metrics
        tracker.on_start()
        
        assert len(tracker.metrics) == 0
    
    def test_metrics_tracker_epoch_end(self, test_env):
        """Test metrics tracker on_epoch_end functionality."""
        tracker = MetricsTracker(env=test_env)
        
        # Simulate epoch end with metrics
        epoch_metrics = {
            Metric.TRAIN_LOSS: 0.5,
            Metric.VAL_LOSS: 0.6,
            Metric.VAL_ACC: 0.85
        }
        
        result = tracker.on_epoch_end(0, epoch_metrics)
        
        # Should return True to continue training
        assert result is True
        
        # Should track the metrics
        assert Metric.TRAIN_LOSS in tracker.metrics
        assert Metric.VAL_LOSS in tracker.metrics
        assert Metric.VAL_ACC in tracker.metrics
        
        # Check values
        assert tracker.metrics[Metric.TRAIN_LOSS][-1] == 0.5
        assert tracker.metrics[Metric.VAL_LOSS][-1] == 0.6
        assert tracker.metrics[Metric.VAL_ACC][-1] == 0.85
    
    def test_metrics_tracker_multiple_epochs(self, test_env):
        """Test metrics tracker across multiple epochs."""
        tracker = MetricsTracker(env=test_env)
        
        # Simulate multiple epochs
        for epoch in range(3):
            epoch_metrics = {
                Metric.TRAIN_LOSS: 1.0 - (epoch * 0.2),
                Metric.VAL_LOSS: 1.0 - (epoch * 0.15),
                Metric.VAL_ACC: 0.5 + (epoch * 0.1)
            }
            tracker.on_epoch_end(epoch, epoch_metrics)
        
        # Check that metrics were accumulated
        assert len(tracker.metrics[Metric.TRAIN_LOSS]) == 3
        assert len(tracker.metrics[Metric.VAL_LOSS]) == 3
        assert len(tracker.metrics[Metric.VAL_ACC]) == 3
        
        # Check progression
        assert tracker.metrics[Metric.TRAIN_LOSS] == [1.0, 0.8, 0.6]
        assert tracker.metrics[Metric.VAL_LOSS] == [1.0, 0.85, 0.7]
        assert tracker.metrics[Metric.VAL_ACC] == [0.5, 0.6, 0.7]
    
    def test_metrics_tracker_get_latest(self, test_env):
        """Test getting latest metric values."""
        tracker = MetricsTracker(env=test_env)
        
        # Add some metrics
        tracker.metrics[Metric.TRAIN_LOSS] = [1.0, 0.8, 0.6]
        tracker.metrics[Metric.VAL_ACC] = [0.5, 0.6, 0.7]
        
        # Get latest values
        latest_loss = tracker.get_latest(Metric.TRAIN_LOSS)
        latest_acc = tracker.get_latest(Metric.VAL_ACC)
        
        assert latest_loss == 0.6
        assert latest_acc == 0.7
        
        # Test default value for non-existent metric
        latest_unknown = tracker.get_latest("unknown_metric", default=0.0)
        assert latest_unknown == 0.0
    
    def test_metrics_tracker_on_end_saves_file(self, test_env):
        """Test that metrics are saved to file on training end."""
        tracker = MetricsTracker(env=test_env)
        
        # Add some metrics
        tracker.metrics[Metric.TRAIN_LOSS] = [1.0, 0.8, 0.6]
        tracker.metrics[Metric.VAL_ACC] = [0.5, 0.6, 0.7]
        
        # Call on_end
        final_metrics = {
            Metric.TEST_ACC: 0.75,
            Metric.TEST_LOSS: 0.4
        }
        tracker.on_end(final_metrics)
        
        # Check that metrics file was created
        assert os.path.exists(tracker.log_path)
        
        # Check file contents
        with open(tracker.log_path, 'r') as f:
            content = f.read()
            assert "train_loss" in content
            assert "val_acc" in content
    
    def test_analytics_api_with_mock_data(self, mock_database_manager):
        """Test ExperimentAnalytics API with mock data."""
        analytics = ExperimentAnalytics(mock_database_manager)
        
        # Test extract_results
        result = analytics.extract_results("test_experiment")
        
        assert isinstance(result, AnalyticsResult)
        assert not result.data.empty
        assert 'experiment_id' in result.data.columns
        assert 'metric_type' in result.data.columns
        assert 'metric_total_val' in result.data.columns
    
    def test_analytics_api_calculate_statistics(self, mock_database_manager):
        """Test statistics calculation."""
        analytics = ExperimentAnalytics(mock_database_manager)
        
        stats = analytics.calculate_statistics(1, metric_types=['test_acc'])
        
        assert isinstance(stats, dict)
        assert 'statistics_by_group' in stats
        
        # Verify database was called
        mock_database_manager.get_aggregated_metrics.assert_called()
    
    def test_analytics_api_analyze_failures(self, mock_database_manager):
        """Test failure analysis."""
        analytics = ExperimentAnalytics(mock_database_manager)
        
        # Mock failure data
        failure_data = pd.DataFrame({
            'experiment_id': [1],
            'trial_run_id': [1],
            'run_status': ['failed'],
            'error_message': ['Out of memory']
        })
        mock_database_manager.get_failure_data.return_value = failure_data
        
        failure_analysis = analytics.analyze_failures(1)
        
        assert isinstance(failure_analysis, dict)
        mock_database_manager.get_failure_data.assert_called_with([1], True)
    
    def test_analytics_result_processing(self):
        """Test AnalyticsResult data processing."""
        sample_data = pd.DataFrame({
            'experiment_id': [1, 1, 2],
            'metric_type': ['test_acc', 'test_acc', 'test_acc'],
            'metric_total_val': [0.85, 0.87, 0.82]
        })
        
        result = AnalyticsResult(sample_data)
        
        # Test summary generation
        summary = result.get_summary()
        assert isinstance(summary, dict)
        assert 'row_count' in summary
        assert 'column_count' in summary
        assert summary['row_count'] == 3
        assert summary['column_count'] == 3
        
        # Test dataframe conversion
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'experiment_id' in df.columns
    
    def test_available_metrics_list(self, mock_database_manager):
        """Test getting available metrics."""
        analytics = ExperimentAnalytics(mock_database_manager)
        
        metrics = analytics.get_available_metrics()
        
        assert isinstance(metrics, list)
        expected_metrics = ['test_acc', 'test_loss', 'val_acc', 'val_loss', 'train_acc', 'train_loss']
        for metric in expected_metrics:
            assert metric in metrics
    
    def test_metrics_integration_workflow(self, test_env):
        """Test a complete metrics tracking workflow."""
        tracker = MetricsTracker(env=test_env)
        
        # Start tracking
        tracker.on_start()
        assert len(tracker.metrics) == 0
        
        # Simulate training epochs
        epochs_data = []
        for epoch in range(5):
            epoch_metrics = {
                Metric.TRAIN_LOSS: 1.0 - (epoch * 0.15),
                Metric.VAL_LOSS: 1.0 - (epoch * 0.12),
                Metric.TRAIN_ACC: 0.5 + (epoch * 0.08),
                Metric.VAL_ACC: 0.5 + (epoch * 0.07)
            }
            epochs_data.append(epoch_metrics)
            
            # Track epoch
            result = tracker.on_epoch_end(epoch, epoch_metrics)
            assert result is True
        
        # Verify metrics were collected
        assert len(tracker.metrics[Metric.TRAIN_LOSS]) == 5
        assert len(tracker.metrics[Metric.VAL_LOSS]) == 5
        assert len(tracker.metrics[Metric.TRAIN_ACC]) == 5
        assert len(tracker.metrics[Metric.VAL_ACC]) == 5
        
        # End training
        final_metrics = {
            Metric.TEST_ACC: 0.88,
            Metric.TEST_LOSS: 0.25
        }
        tracker.on_end(final_metrics)
        
        # Verify file was created
        assert os.path.exists(tracker.log_path)
        
        # Verify metrics progression (should be improving)
        train_losses = tracker.metrics[Metric.TRAIN_LOSS]
        val_accuracies = tracker.metrics[Metric.VAL_ACC]
        
        # Loss should decrease
        assert train_losses[0] > train_losses[-1]
        
        # Accuracy should increase
        assert val_accuracies[0] < val_accuracies[-1]


if __name__ == "__main__":
    pytest.main([__file__]) 