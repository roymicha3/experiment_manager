"""
Integration tests for analytics engine with enhanced database manager.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from experiment_manager.analytics.engine import AnalyticsEngine
from experiment_manager.analytics.results import AnalyticsResult
from .test_fixtures import mock_database_manager


class TestAnalyticsEngineIntegration:
    """Test analytics engine integration with enhanced database methods."""
    
    @pytest.fixture
    def analytics_engine(self, mock_database_manager):
        """Create analytics engine with mock database."""
        config = {
            'cache_enabled': True,
            'cache_ttl': 300
        }
        return AnalyticsEngine(mock_database_manager, config)
    
    def test_execute_analytics_query(self, analytics_engine, mock_database_manager):
        """Test executing analytics query through engine."""
        # Mock database response with custom side effect to override fixture
        mock_data = pd.DataFrame({
            'experiment_id': [1, 2],
            'trial_id': [1, 2],
            'metric_type': ['test_acc', 'test_acc'],
            'metric_total_val': [0.85, 0.87]
        })
        
        def custom_side_effect(experiment_ids=None, filters=None):
            return mock_data
        
        mock_database_manager.get_analytics_data.side_effect = custom_side_effect
        
        result = analytics_engine.execute_analytics_query([1, 2])
        
        assert isinstance(result, AnalyticsResult)
        assert len(result.data) == 2
        assert result.metadata['query_type'] == 'analytics_data'
        
        # Verify database method was called
        mock_database_manager.get_analytics_data.assert_called_once_with([1, 2], None)
    
    def test_get_aggregated_metrics(self, analytics_engine, mock_database_manager):
        """Test aggregated metrics through engine."""
        mock_data = pd.DataFrame({
            'experiment_id': [1],
            'trial_id': [1],
            'metric_type': ['test_acc'],
            'mean': [0.85],
            'std': [0.02]
        })
        
        def custom_side_effect(experiment_ids=None, group_by='trial', functions=None):
            return mock_data
        
        mock_database_manager.get_aggregated_metrics.side_effect = custom_side_effect
        
        result = analytics_engine.get_aggregated_metrics([1], 'trial', ['mean', 'std'])
        
        assert isinstance(result, AnalyticsResult)
        assert 'mean' in result.data.columns
        assert result.metadata['query_type'] == 'aggregated_metrics'
        
        mock_database_manager.get_aggregated_metrics.assert_called_once_with([1], 'trial', ['mean', 'std'])
    
    def test_get_failure_analysis_data(self, analytics_engine, mock_database_manager):
        """Test failure analysis through engine."""
        mock_data = pd.DataFrame({
            'experiment_id': [1],
            'trial_run_id': [1],
            'run_status': ['failed'],
            'duration_seconds': [300]
        })
        mock_database_manager.get_failure_data.return_value = mock_data
        
        result = analytics_engine.get_failure_analysis_data([1], include_configs=True)
        
        assert isinstance(result, AnalyticsResult)
        assert result.metadata['query_type'] == 'failure_data'
        
        mock_database_manager.get_failure_data.assert_called_once_with([1], True)
    
    def test_get_epoch_series_data(self, analytics_engine, mock_database_manager):
        """Test epoch series through engine."""
        mock_data = pd.DataFrame({
            'trial_run_id': [1, 1],
            'epoch_idx': [1, 2],
            'metric_type': ['test_acc', 'test_acc'],
            'metric_total_val': [0.8, 0.85]
        })
        
        def custom_side_effect(trial_run_ids, metric_types=None):
            return mock_data
        
        mock_database_manager.get_epoch_series.side_effect = custom_side_effect
        
        result = analytics_engine.get_epoch_series_data([1], ['test_acc'])
        
        assert isinstance(result, AnalyticsResult)
        assert len(result.data) == 2
        assert result.metadata['query_type'] == 'epoch_series'
        
        mock_database_manager.get_epoch_series.assert_called_once_with([1], ['test_acc'])
    
    def test_caching_functionality(self, analytics_engine, mock_database_manager):
        """Test that caching works correctly."""
        mock_data = pd.DataFrame({'test': [1, 2, 3]})
        
        def custom_side_effect(experiment_ids=None, filters=None):
            return mock_data
        
        mock_database_manager.get_analytics_data.side_effect = custom_side_effect
        
        # First call - cache miss
        result1 = analytics_engine.execute_analytics_query([1])
        assert mock_database_manager.get_analytics_data.call_count == 1
        
        # Second call - cache hit
        result2 = analytics_engine.execute_analytics_query([1])
        assert mock_database_manager.get_analytics_data.call_count == 1  # No additional call
        
        # Verify cache stats (note: current implementation only counts cache misses as total_queries)
        stats = analytics_engine.get_cache_stats()
        assert stats['total_queries'] == 1  # Only cache miss is counted
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 1
    
    def test_initialize_database_indexes(self, analytics_engine, mock_database_manager):
        """Test database index initialization."""
        analytics_engine.initialize_database_indexes()
        
        mock_database_manager.create_analytics_indexes.assert_called_once()
    
    def test_custom_query_execution(self, analytics_engine, mock_database_manager):
        """Test custom query execution through engine."""
        mock_data = pd.DataFrame({'count': [5]})
        mock_database_manager.execute_query.return_value = mock_data
        
        query = "SELECT COUNT(*) as count FROM EXPERIMENT"
        result = analytics_engine.execute_custom_query(query)
        
        assert isinstance(result, AnalyticsResult)
        assert result.metadata['query_type'] == 'custom_query'
        
        mock_database_manager.execute_query.assert_called_once_with(query, None)


class TestAnalyticsEnginePerformance:
    """Test analytics engine performance features."""
    
    @pytest.fixture
    def analytics_engine(self, mock_database_manager):
        """Create analytics engine with performance config."""
        config = {
            'cache_enabled': True,
            'cache_ttl': 60  # Short TTL for testing
        }
        return AnalyticsEngine(mock_database_manager, config)
    
    def test_cache_expiration(self, analytics_engine, mock_database_manager):
        """Test cache expiration functionality."""
        mock_data = pd.DataFrame({'test': [1]})
        
        def custom_side_effect(experiment_ids=None, filters=None):
            return mock_data
        
        mock_database_manager.get_analytics_data.side_effect = custom_side_effect
        
        # First call
        result1 = analytics_engine.execute_analytics_query([1])
        
        # Manually expire cache
        for key in analytics_engine._query_cache:
            analytics_engine._query_cache[key]['timestamp'] = datetime.now() - pd.Timedelta(seconds=120)
        
        # Second call should be cache miss due to expiration
        result2 = analytics_engine.execute_analytics_query([1])
        
        assert mock_database_manager.get_analytics_data.call_count == 2
    
    def test_cache_disabled(self, mock_database_manager):
        """Test analytics engine with caching disabled."""
        config = {'cache_enabled': False}
        engine = AnalyticsEngine(mock_database_manager, config)
        
        mock_data = pd.DataFrame({'test': [1]})
        
        def custom_side_effect(experiment_ids=None, filters=None):
            return mock_data
        
        mock_database_manager.get_analytics_data.side_effect = custom_side_effect
        
        # Multiple calls should all hit database
        engine.execute_analytics_query([1])
        engine.execute_analytics_query([1])
        
        assert mock_database_manager.get_analytics_data.call_count == 2
        
        stats = engine.get_cache_stats()
        assert stats['cache_hits'] == 0  # No cache hits when disabled
    
    def test_performance_statistics(self, analytics_engine, mock_database_manager):
        """Test performance statistics tracking."""
        mock_data = pd.DataFrame({'test': [1]})
        
        def custom_side_effect(experiment_ids=None, filters=None):
            return mock_data
        
        mock_database_manager.get_analytics_data.side_effect = custom_side_effect
        
        # Execute some queries
        analytics_engine.execute_analytics_query([1])
        analytics_engine.execute_analytics_query([2])
        analytics_engine.execute_analytics_query([1])  # Cache hit
        
        stats = analytics_engine.get_cache_stats()
        
        # Note: current implementation only counts cache misses as total_queries
        assert stats['total_queries'] == 2  # Only cache misses are counted
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 2
        assert stats['cache_hit_rate_percent'] > 0
        assert stats['average_execution_time'] >= 0 