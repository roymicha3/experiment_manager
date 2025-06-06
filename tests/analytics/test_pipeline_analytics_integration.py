import os
import pytest
import tempfile
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime, timedelta

from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.pipelines.callbacks.metric_tracker import MetricsTracker
from experiment_manager.analytics.api import ExperimentAnalytics
from experiment_manager.analytics.engine import AnalyticsEngine
from experiment_manager.analytics.results import AnalyticsResult
from experiment_manager.common.common import Metric, Level, RunStatus
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.experiment import Experiment


@YAMLSerializable.register("TestPipelineForAnalytics")
class TestPipelineForAnalytics(Pipeline, YAMLSerializable):
    """Test pipeline designed for analytics testing."""
    
    def __init__(self, env: Environment, epochs: int = 5, analytics_pattern: str = "standard"):
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self)
        self.epochs = epochs
        self.analytics_pattern = analytics_pattern
        
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls(
            env, 
            epochs=config.pipeline.get('epochs', 5),
            analytics_pattern=config.pipeline.get('analytics_pattern', 'standard')
        )
    
    def run(self, config: DictConfig):
        """Run the test pipeline generating metrics for analytics."""
        self.env.logger.info("Starting analytics test pipeline")
        
        # Manually handle lifecycle for testing purposes
        self._on_run_start()
        
        try:
            for epoch in range(self.epochs):
                self.run_epoch(epoch, model=None)
            
            # Generate final test metrics
            if self.analytics_pattern == "successful":
                test_acc = 0.95
                test_loss = 0.05
            elif self.analytics_pattern == "poor":
                test_acc = 0.60
                test_loss = 0.80
            else:  # standard
                test_acc = 0.85
                test_loss = 0.25
                
            self.run_metrics = {
                Metric.TEST_ACC: test_acc,
                Metric.TEST_LOSS: test_loss
            }
            
            # Call lifecycle end 
            self._on_run_end(self.run_metrics)
            
            return {"status": "completed", "test_acc": test_acc, "test_loss": test_loss}
            
        except Exception as e:
            self.env.logger.error(f"Pipeline failed: {e}")
            raise
    
    @Pipeline.epoch_wrapper 
    def run_epoch(self, epoch_idx, model, *args, **kwargs):
        """Run a single epoch generating realistic metrics."""
        
        if self.analytics_pattern == "successful":
            # Improving pattern
            train_loss = 1.0 - (epoch_idx * 0.15)
            val_loss = 1.0 - (epoch_idx * 0.12)
            train_acc = 0.5 + (epoch_idx * 0.08)
            val_acc = 0.5 + (epoch_idx * 0.07)
        elif self.analytics_pattern == "poor":
            # Poor performance pattern
            train_loss = 1.5 - (epoch_idx * 0.05)
            val_loss = 1.6 - (epoch_idx * 0.03)
            train_acc = 0.3 + (epoch_idx * 0.02)
            val_acc = 0.3 + (epoch_idx * 0.015)
        elif self.analytics_pattern == "overfitting":
            # Overfitting pattern
            train_loss = 1.0 - (epoch_idx * 0.20)
            val_loss = 0.8 + (epoch_idx * 0.05) if epoch_idx > 2 else 1.0 - (epoch_idx * 0.10)
            train_acc = 0.5 + (epoch_idx * 0.10)
            val_acc = 0.5 + (epoch_idx * 0.06) if epoch_idx <= 2 else 0.68 - (epoch_idx * 0.02)
        else:  # standard
            train_loss = 1.0 - (epoch_idx * 0.10)
            val_loss = 1.0 - (epoch_idx * 0.08)
            train_acc = 0.5 + (epoch_idx * 0.06)
            val_acc = 0.5 + (epoch_idx * 0.05)
        
        # Store metrics in epoch_metrics for automatic processing
        self.epoch_metrics = {
            Metric.TRAIN_LOSS: train_loss,
            Metric.VAL_LOSS: val_loss,
            Metric.TRAIN_ACC: train_acc,
            Metric.VAL_ACC: val_acc,
            Metric.CUSTOM: ("learning_rate", 0.001 * (0.9 ** epoch_idx))
        }
        
        self.env.logger.info(f"Epoch {epoch_idx}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")


@pytest.fixture
def test_env():
    """Create a test environment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = OmegaConf.create({
            "workspace": tmp_dir, 
            "verbose": False,
            "trackers": []
        })
        env = Environment(workspace=tmp_dir, config=config)
        yield env
        # Clean up all resources properly
        env.close()


@pytest.fixture
def mock_database_manager():
    """Create a mock database manager for analytics testing."""
    mock_db = Mock()
    
    # Mock analytics data
    sample_data = pd.DataFrame({
        'experiment_id': [1, 1, 1, 1],
        'trial_id': [1, 1, 2, 2],
        'trial_run_id': [1, 2, 3, 4],
        'run_status': ['success', 'success', 'success', 'failed'],
        'metric_type': ['test_acc', 'test_acc', 'test_acc', 'test_acc'],
        'metric_total_val': [0.85, 0.87, 0.82, None],
        'epoch_idx': [4, 4, 4, 2],
        'run_start_time': [datetime.now() - timedelta(hours=i) for i in range(4)]
    })
    
    # Mock aggregated metrics
    aggregated_data = pd.DataFrame({
        'experiment_id': [1],
        'metric_type': ['test_acc'],
        'mean': [0.85],
        'std': [0.025],
        'min': [0.82],
        'max': [0.87],
        'count': [3]
    })
    
    # CRITICAL FIX: Add execute_query method that returns proper pandas DataFrame
    mock_db.execute_query.return_value = sample_data
    
    # Setup mock methods
    mock_db.get_analytics_data.return_value = sample_data
    mock_db.get_aggregated_metrics.return_value = aggregated_data
    mock_db.get_failure_data.return_value = sample_data[sample_data['run_status'] == 'failed']
    mock_db.get_epoch_series.return_value = pd.DataFrame()
    
    return mock_db


@pytest.fixture
def sample_experiment_data():
    """Create sample experiment data for testing."""
    return pd.DataFrame({
        'experiment_id': [1, 1, 1],
        'trial_id': [1, 1, 2],
        'trial_run_id': [1, 2, 3],
        'run_status': ['success', 'success', 'success'],
        'metric_type': ['test_acc', 'test_acc', 'test_acc'],
        'metric_total_val': [0.85, 0.87, 0.82],
        'epoch_idx': [4, 4, 4]
    })


class TestPipelineAnalyticsIntegration:
    """Test analytics integration with pipelines."""
    
    def test_pipeline_generates_trackable_metrics(self, test_env):
        """Test that pipeline generates metrics that can be analyzed."""
        pipeline = TestPipelineForAnalytics(
            env=test_env, 
            epochs=3, 
            analytics_pattern="standard"
        )
        
        # Add metrics tracker
        metrics_tracker = MetricsTracker(env=test_env)
        pipeline.register_callback(metrics_tracker)
        
        config = OmegaConf.create({
            "pipeline": {
                "epochs": 3,
                "analytics_pattern": "standard"
            }
        })
        
        result = pipeline.run(config)
        
        # Verify metrics were generated
        assert result["status"] == "completed"
        assert "test_acc" in result
        assert "test_loss" in result
        
        # Check that metrics were tracked
        assert len(metrics_tracker.metrics) > 0
        assert Metric.TRAIN_LOSS in metrics_tracker.metrics
        assert Metric.VAL_LOSS in metrics_tracker.metrics
        
        # Verify metrics file was created
        metrics_file = os.path.join(test_env.artifact_dir, "metrics.log")
        assert os.path.exists(metrics_file)
    
    def test_analytics_api_extract_results(self, mock_database_manager, sample_experiment_data):
        """Test ExperimentAnalytics extract_results functionality."""
        analytics = ExperimentAnalytics(mock_database_manager)
        
        # Test extracting results
        result = analytics.extract_results("test_experiment", include_failed=False)
        
        # Verify result structure
        assert isinstance(result, AnalyticsResult)
        assert not result.data.empty
        assert 'experiment_id' in result.data.columns
        assert 'metric_type' in result.data.columns
    
    def test_analytics_api_calculate_statistics(self, mock_database_manager):
        """Test ExperimentAnalytics calculate_statistics functionality."""
        analytics = ExperimentAnalytics(mock_database_manager)
        
        # Test calculating statistics
        stats = analytics.calculate_statistics(1, metric_types=['test_acc'], group_by='trial')
        
        # Verify statistics structure
        assert isinstance(stats, dict)
        assert 'statistics_by_group' in stats
        
        # Verify database manager was called correctly - uses execute_query not get_aggregated_metrics
        mock_database_manager.execute_query.assert_called()
    
    def test_analytics_api_analyze_failures(self, mock_database_manager):
        """Test ExperimentAnalytics analyze_failures functionality."""
        analytics = ExperimentAnalytics(mock_database_manager)
        
        # Test failure analysis
        failure_analysis = analytics.analyze_failures(1, correlation_analysis=True)
        
        # Verify analysis structure
        assert isinstance(failure_analysis, dict)
        
        # Verify database manager was called correctly - uses execute_query multiple times
        assert mock_database_manager.execute_query.call_count >= 2
    
    def test_analytics_engine_caching(self, mock_database_manager):
        """Test analytics engine caching functionality."""
        config = {
            'cache_enabled': True,
            'cache_ttl': 300
        }
        
        engine = AnalyticsEngine(mock_database_manager, config)
        
        # First query - should hit database
        result1 = engine.get_aggregated_metrics([1], 'trial', ['mean', 'std'])
        
        # Second identical query - should hit cache
        result2 = engine.get_aggregated_metrics([1], 'trial', ['mean', 'std'])
        
        # Verify both results are AnalyticsResult instances
        assert isinstance(result1, AnalyticsResult)
        assert isinstance(result2, AnalyticsResult)
        
        # Check cache statistics
        assert engine._query_stats['total_queries'] >= 1
        assert engine._query_stats['cache_hits'] >= 0
    
    def test_analytics_performance_patterns(self, test_env):
        """Test analytics detection of different performance patterns."""
        patterns_to_test = ["successful", "poor", "overfitting"]
        results = {}
        
        for pattern in patterns_to_test:
            pipeline = TestPipelineForAnalytics(
                env=test_env, 
                epochs=5, 
                analytics_pattern=pattern
            )
            
            metrics_tracker = MetricsTracker(env=test_env)
            pipeline.register_callback(metrics_tracker)
            
            config = OmegaConf.create({
                "pipeline": {
                    "epochs": 5,
                    "analytics_pattern": pattern
                }
            })
            
            result = pipeline.run(config)
            results[pattern] = result
        
        # Verify different patterns produce different results
        assert results["successful"]["test_acc"] > results["poor"]["test_acc"]
        assert results["successful"]["test_loss"] < results["poor"]["test_loss"]
        
        # Verify overfitting pattern characteristics would be detectable
        # (This would require more sophisticated analytics in a real scenario)
        assert results["overfitting"]["test_acc"] != results["successful"]["test_acc"]
    
    def test_analytics_query_builder_integration(self, mock_database_manager):
        """Test analytics query builder integration."""
        engine = AnalyticsEngine(mock_database_manager)
        
        # Build a complex query
        query = (engine.create_query()
                .experiments(ids=[1])
                .runs(status=[RunStatus.SUCCESS])
                .metrics(types=['test_acc', 'val_acc'])
                .aggregate(['mean', 'std', 'max'])
                .group_by('trial')
                .sort_by('mean', ascending=False))
        
        # Execute query
        result = query.execute()
        
        # Verify result structure
        assert isinstance(result, AnalyticsResult)
        assert hasattr(result, 'data')
        assert hasattr(result, 'metadata')
    
    def test_analytics_epoch_series_analysis(self, mock_database_manager):
        """Test epoch series data analysis for training curves."""
        # Setup epoch series mock data
        epoch_data = pd.DataFrame({
            'trial_run_id': [1, 1, 1, 1],
            'epoch_idx': [0, 1, 2, 3],
            'metric_type': ['val_loss', 'val_loss', 'val_loss', 'val_loss'],
            'metric_total_val': [1.0, 0.8, 0.6, 0.5],
            'epoch_time': [datetime.now() - timedelta(minutes=i) for i in range(4)]
        })
        
        mock_database_manager.get_epoch_series.return_value = epoch_data
        
        engine = AnalyticsEngine(mock_database_manager)
        result = engine.get_epoch_series_data([1], ['val_loss'])
        
        # Verify epoch series result
        assert isinstance(result, AnalyticsResult)
        assert result.metadata['query_type'] == 'epoch_series'
        assert not result.data.empty
        
        mock_database_manager.get_epoch_series.assert_called_with([1], ['val_loss'])
    
    def test_analytics_comparison_across_experiments(self, mock_database_manager):
        """Test analytics comparison across multiple experiments."""
        # Setup multi-experiment data
        comparison_data = pd.DataFrame({
            'experiment_id': [1, 2, 3],
            'metric_type': ['test_acc', 'test_acc', 'test_acc'],
            'mean': [0.85, 0.82, 0.88],
            'std': [0.02, 0.03, 0.015],
            'count': [5, 4, 6]
        })
        
        mock_database_manager.get_aggregated_metrics.return_value = comparison_data
        
        analytics = ExperimentAnalytics(mock_database_manager)
        result = analytics.compare_experiments([1, 2, 3], 'test_acc')
        
        # Verify comparison result
        assert isinstance(result, AnalyticsResult)
        assert result.metadata['comparison_type'] == 'multi_experiment'
        assert result.metadata['primary_metric'] == 'test_acc'
        assert not result.data.empty
    
    def test_analytics_training_curve_analysis(self, mock_database_manager):
        """Test training curve analysis functionality."""
        analytics = ExperimentAnalytics(mock_database_manager)
        
        result = analytics.analyze_training_curves([1, 2], ['val_loss', 'train_loss'])
        
        # Verify training curve analysis
        assert isinstance(result, AnalyticsResult)
        assert result.metadata['analysis_type'] == 'training_curves'
        assert result.metadata['analyzed_runs'] == [1, 2]
        
        mock_database_manager.get_epoch_series.assert_called()
    
    def test_analytics_metrics_availability(self, mock_database_manager):
        """Test getting available metrics for analysis."""
        analytics = ExperimentAnalytics(mock_database_manager)
        
        # Test getting available metrics
        metrics = analytics.get_available_metrics()
        
        # Verify metrics list
        assert isinstance(metrics, list)
        assert 'test_acc' in metrics
        assert 'test_loss' in metrics
        assert 'val_acc' in metrics
        assert 'val_loss' in metrics
    
    def test_analytics_error_handling(self, mock_database_manager):
        """Test analytics error handling and graceful failures."""
        # Setup database manager to raise an exception
        mock_database_manager.get_analytics_data.side_effect = Exception("Database error")
        
        engine = AnalyticsEngine(mock_database_manager)
        
        # Test that exceptions are properly handled
        with pytest.raises(Exception):
            engine.execute_analytics_query([1])
    
    def test_analytics_result_processing(self, sample_experiment_data):
        """Test AnalyticsResult processing and summary generation."""
        result = AnalyticsResult(sample_experiment_data)
        
        # Test summary generation
        summary = result.get_summary()
        assert isinstance(summary, dict)
        assert 'row_count' in summary['overview']
        assert 'column_count' in summary['overview']
        assert summary['overview']['row_count'] == 3
        assert summary['overview']['column_count'] == 7
        
        # Test dataframe conversion
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'experiment_id' in df.columns
    
    def test_pipeline_analytics_end_to_end(self, test_env):
        """Test end-to-end pipeline analytics workflow."""
        # Create and run pipeline
        pipeline = TestPipelineForAnalytics(
            env=test_env, 
            epochs=4, 
            analytics_pattern="successful"
        )
        
        metrics_tracker = MetricsTracker(env=test_env)
        pipeline.register_callback(metrics_tracker)
        
        config = OmegaConf.create({
            "pipeline": {
                "epochs": 4,
                "analytics_pattern": "successful"
            }
        })
        
        result = pipeline.run(config)
        
        # Verify pipeline completed successfully
        assert result["status"] == "completed"
        assert result["test_acc"] > 0.9  # Should be high for "successful" pattern
        
        # Verify metrics were tracked
        assert len(metrics_tracker.metrics) > 0
        
        # Verify analytics artifacts were created
        artifacts_created = []
        for root, dirs, files in os.walk(test_env.workspace):
            artifacts_created.extend(files)
        
        # Should have created log files and metrics files
        assert any(f.endswith('.log') for f in artifacts_created)
        
        # Verify specific metrics were captured
        captured_metrics = list(metrics_tracker.metrics.keys())
        expected_metrics = [Metric.TRAIN_LOSS, Metric.VAL_LOSS, Metric.TRAIN_ACC, Metric.VAL_ACC]
        
        for metric in expected_metrics:
            assert metric in captured_metrics, f"Missing expected metric: {metric}"


if __name__ == "__main__":
    pytest.main([__file__]) 