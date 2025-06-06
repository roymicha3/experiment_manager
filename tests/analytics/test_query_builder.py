"""
Tests for analytics query builder module.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from experiment_manager.analytics.query_builder import AnalyticsQuery
from experiment_manager.analytics.results import AnalyticsResult
from experiment_manager.common.common import RunStatus, Metric
from .test_fixtures import (
    mock_database_manager, sample_experiment_data, multi_experiment_data
)


class TestAnalyticsQuery:
    """Test cases for AnalyticsQuery class."""
    
    def test_query_creation(self, mock_database_manager):
        """Test creating a new query."""
        query = AnalyticsQuery(mock_database_manager)
        
        assert query.database_manager == mock_database_manager
        assert query.filters == {}
        assert query.processing_operations == []
        assert query.cache_enabled == True
        assert query.cache_ttl == 300
    
    def test_experiments_filter_by_ids(self, mock_database_manager):
        """Test filtering experiments by IDs."""
        query = AnalyticsQuery(mock_database_manager)
        
        filtered_query = query.experiments(ids=[1, 2, 3])
        
        assert 'experiments' in filtered_query.filters
        assert filtered_query.filters['experiments']['ids'] == [1, 2, 3]
        assert filtered_query is not query  # Should return new instance
    
    def test_experiments_filter_by_names(self, mock_database_manager):
        """Test filtering experiments by names."""
        query = AnalyticsQuery(mock_database_manager)
        
        filtered_query = query.experiments(names=['exp1', 'exp2'])
        
        assert 'experiments' in filtered_query.filters
        assert filtered_query.filters['experiments']['names'] == ['exp1', 'exp2']
    
    def test_experiments_filter_combined(self, mock_database_manager):
        """Test filtering experiments with multiple criteria."""
        query = AnalyticsQuery(mock_database_manager)
        
        filtered_query = query.experiments(
            ids=[1, 2],
            names=['exp1'],
            created_after=datetime(2023, 1, 1)
        )
        
        exp_filter = filtered_query.filters['experiments']
        assert exp_filter['ids'] == [1, 2]
        assert exp_filter['names'] == ['exp1']
        assert exp_filter['created_after'] == datetime(2023, 1, 1)
    
    def test_trials_filter(self, mock_database_manager):
        """Test filtering trials."""
        query = AnalyticsQuery(mock_database_manager)
        
        filtered_query = query.trials(
            ids=[1, 2],
            names=['trial1', 'trial2'],
            status=['completed']
        )
        
        assert 'trials' in filtered_query.filters
        trial_filter = filtered_query.filters['trials']
        assert trial_filter['ids'] == [1, 2]
        assert trial_filter['names'] == ['trial1', 'trial2']
        assert trial_filter['status'] == ['completed']
    
    def test_runs_filter(self, mock_database_manager):
        """Test filtering runs."""
        query = AnalyticsQuery(mock_database_manager)
        
        filtered_query = query.runs(
            run_ids=[1, 2, 3],
            status=[RunStatus.SUCCESS, RunStatus.FAILED]
        )
        
        assert 'runs' in filtered_query.filters
        run_filter = filtered_query.filters['runs']
        assert run_filter['run_ids'] == [1, 2, 3]
        assert run_filter['status'] == [RunStatus.SUCCESS, RunStatus.FAILED]
    
    def test_metrics_filter(self, mock_database_manager):
        """Test filtering metrics."""
        query = AnalyticsQuery(mock_database_manager)
        
        filtered_query = query.metrics(
            types=['test_acc', 'val_loss'],
            context='training',
            min_value=0.5,
            max_value=1.0
        )
        
        assert 'metrics' in filtered_query.filters
        metric_filter = filtered_query.filters['metrics']
        assert metric_filter['types'] == ['test_acc', 'val_loss']
        assert metric_filter['context'] == 'training'
        assert metric_filter['min_value'] == 0.5
        assert metric_filter['max_value'] == 1.0
    
    def test_date_range_filter(self, mock_database_manager):
        """Test date range filtering."""
        query = AnalyticsQuery(mock_database_manager)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        filtered_query = query.date_range(start_date, end_date)
        
        assert 'date_range' in filtered_query.filters
        date_filter = filtered_query.filters['date_range']
        assert date_filter['start'] == start_date
        assert date_filter['end'] == end_date
    
    def test_chained_filters(self, mock_database_manager):
        """Test chaining multiple filters."""
        query = AnalyticsQuery(mock_database_manager)
        
        filtered_query = (query
                         .experiments(ids=[1, 2])
                         .trials(status=['completed'])
                         .runs(status=[RunStatus.SUCCESS])
                         .metrics(types=['test_acc']))
        
        assert 'experiments' in filtered_query.filters
        assert 'trials' in filtered_query.filters
        assert 'runs' in filtered_query.filters
        assert 'metrics' in filtered_query.filters
    
    def test_exclude_outliers_operation(self, mock_database_manager):
        """Test exclude outliers processing operation."""
        query = AnalyticsQuery(mock_database_manager)
        
        processed_query = query.exclude_outliers(
            metric='test_acc',
            method='iqr',
            threshold=1.5
        )
        
        assert len(processed_query.processing_operations) == 1
        operation = processed_query.processing_operations[0]
        assert operation['type'] == 'exclude_outliers'
        assert operation['parameters']['metric'] == 'test_acc'
        assert operation['parameters']['method'] == 'iqr'
        assert operation['parameters']['threshold'] == 1.5
    
    def test_aggregate_operation(self, mock_database_manager):
        """Test aggregate processing operation."""
        query = AnalyticsQuery(mock_database_manager)
        
        aggregated_query = query.aggregate(['mean', 'std', 'count'])
        
        assert len(aggregated_query.processing_operations) == 1
        operation = aggregated_query.processing_operations[0]
        assert operation['type'] == 'aggregate'
        assert operation['parameters']['functions'] == ['mean', 'std', 'count']
    
    def test_group_by_operation(self, mock_database_manager):
        """Test group by processing operation."""
        query = AnalyticsQuery(mock_database_manager)
        
        grouped_query = query.group_by('experiment_id')
        
        assert len(grouped_query.processing_operations) == 1
        operation = grouped_query.processing_operations[0]
        assert operation['type'] == 'group_by'
        assert operation['parameters']['column'] == 'experiment_id'
    
    def test_sort_by_operation(self, mock_database_manager):
        """Test sort by processing operation."""
        query = AnalyticsQuery(mock_database_manager)
        
        sorted_query = query.sort_by('metric_value', ascending=False)
        
        assert len(sorted_query.processing_operations) == 1
        operation = sorted_query.processing_operations[0]
        assert operation['type'] == 'sort_by'
        assert operation['parameters']['column'] == 'metric_value'
        assert operation['parameters']['ascending'] == False
    
    def test_limit_operation(self, mock_database_manager):
        """Test limit processing operation."""
        query = AnalyticsQuery(mock_database_manager)
        
        limited_query = query.limit(100)
        
        assert len(limited_query.processing_operations) == 1
        operation = limited_query.processing_operations[0]
        assert operation['type'] == 'limit'
        assert operation['parameters']['count'] == 100
    
    def test_chained_operations(self, mock_database_manager):
        """Test chaining multiple processing operations."""
        query = AnalyticsQuery(mock_database_manager)
        
        processed_query = (query
                          .exclude_outliers('test_acc')
                          .aggregate(['mean', 'std'])
                          .group_by('experiment_id')
                          .sort_by('mean', ascending=False)
                          .limit(10))
        
        assert len(processed_query.processing_operations) == 5
        
        # Check operation order
        operations = processed_query.processing_operations
        assert operations[0]['type'] == 'exclude_outliers'
        assert operations[1]['type'] == 'aggregate'
        assert operations[2]['type'] == 'group_by'
        assert operations[3]['type'] == 'sort_by'
        assert operations[4]['type'] == 'limit'
    
    def test_cache_control(self, mock_database_manager):
        """Test cache control methods."""
        query = AnalyticsQuery(mock_database_manager)
        
        # Test disabling cache
        no_cache_query = query.no_cache()
        assert no_cache_query.cache_enabled == False
        
        # Test setting cache TTL
        cached_query = query.cache(ttl=600)
        assert cached_query.cache_enabled == True
        assert cached_query.cache_ttl == 600
    
    def test_validate_filters_success(self, mock_database_manager):
        """Test successful filter validation."""
        query = AnalyticsQuery(mock_database_manager)
        
        valid_query = (query
                      .experiments(ids=[1, 2])
                      .metrics(types=['test_acc']))
        
        # Should not raise any exception
        valid_query._validate_filters()
    
    def test_validate_filters_invalid_experiment_ids(self, mock_database_manager):
        """Test filter validation with invalid experiment IDs."""
        query = AnalyticsQuery(mock_database_manager)
        
        with pytest.raises(ValueError, match="Experiment IDs must be positive integers"):
            query.experiments(ids=[0, -1, 'invalid'])
    
    def test_validate_filters_invalid_date_range(self, mock_database_manager):
        """Test filter validation with invalid date range."""
        query = AnalyticsQuery(mock_database_manager)
        
        start_date = datetime(2023, 12, 31)
        end_date = datetime(2023, 1, 1)  # End before start
        
        with pytest.raises(ValueError, match="Start date must be before end date"):
            query.date_range(start_date, end_date)
    
    def test_estimate_complexity_simple(self, mock_database_manager):
        """Test complexity estimation for simple query."""
        query = AnalyticsQuery(mock_database_manager)
        
        simple_query = query.experiments(ids=[1])
        complexity = simple_query.estimate_complexity()
        
        assert complexity['level'] == 'low'
        assert complexity['estimated_rows'] <= 1000
        assert complexity['filter_count'] == 1
        assert complexity['operation_count'] == 0
    
    def test_estimate_complexity_complex(self, mock_database_manager):
        """Test complexity estimation for complex query."""
        query = AnalyticsQuery(mock_database_manager)
        
        complex_query = (query
                        .experiments(ids=list(range(100)))  # Many experiments
                        .trials(status=['completed'])
                        .runs(status=[RunStatus.SUCCESS])
                        .metrics(types=['test_acc', 'val_acc'])
                        .exclude_outliers('test_acc')
                        .aggregate(['mean', 'std', 'min', 'max'])
                        .group_by('experiment_id')
                        .sort_by('mean'))
        
        complexity = complex_query.estimate_complexity()
        
        assert complexity['level'] == 'high'
        assert complexity['filter_count'] == 4
        assert complexity['operation_count'] == 4
    
    def test_to_sql_generation(self, mock_database_manager):
        """Test SQL query generation."""
        query = AnalyticsQuery(mock_database_manager)
        
        filtered_query = (query
                         .experiments(ids=[1, 2])
                         .metrics(types=['test_acc'])
                         .sort_by('metric_value'))
        
        sql = filtered_query.to_sql()
        
        assert isinstance(sql, str)
        assert 'SELECT' in sql.upper()
        assert 'FROM' in sql.upper()
        assert 'WHERE' in sql.upper()
        assert 'ORDER BY' in sql.upper()
    
    def test_execute_with_mock_database(self, mock_database_manager, sample_experiment_data):
        """Test query execution with mocked database."""
        # Mock the database response using side effect to override fixture
        def custom_side_effect(query, params=None):
            return sample_experiment_data
        
        mock_database_manager.execute_query.side_effect = custom_side_effect
        
        query = AnalyticsQuery(mock_database_manager)
        filtered_query = query.experiments(ids=[1])
        
        result = filtered_query.execute()
        
        assert isinstance(result, AnalyticsResult)
        assert len(result.data) == len(sample_experiment_data)
        mock_database_manager.execute_query.assert_called_once()
    
    def test_execute_with_cache_miss(self, mock_database_manager, sample_experiment_data):
        """Test query execution with cache miss."""
        mock_database_manager.execute_query.return_value = sample_experiment_data
        
        query = AnalyticsQuery(mock_database_manager)
        filtered_query = query.experiments(ids=[1]).cache(ttl=300)
        
        # First execution - cache miss
        result1 = filtered_query.execute()
        
        # Second execution - should use cache
        result2 = filtered_query.execute()
        
        # Database should only be called once due to caching
        assert mock_database_manager.execute_query.call_count == 1
        assert isinstance(result1, AnalyticsResult)
        assert isinstance(result2, AnalyticsResult)
    
    def test_explain_query(self, mock_database_manager):
        """Test query explanation generation."""
        query = AnalyticsQuery(mock_database_manager)
        
        complex_query = (query
                        .experiments(ids=[1, 2])
                        .metrics(types=['test_acc'])
                        .exclude_outliers('test_acc')
                        .aggregate(['mean'])
                        .group_by('experiment_id'))
        
        explanation = complex_query.explain()
        
        assert isinstance(explanation, dict)
        assert 'filters' in explanation
        assert 'operations' in explanation
        assert 'complexity' in explanation
        assert 'estimated_execution_time' in explanation
    
    def test_query_yaml_serialization(self, mock_database_manager):
        """Test query YAML serialization."""
        query = AnalyticsQuery(mock_database_manager)
        
        configured_query = (query
                           .experiments(ids=[1, 2])
                           .metrics(types=['test_acc'])
                           .aggregate(['mean']))
        
        yaml_str = configured_query.to_yaml()
        
        assert isinstance(yaml_str, str)
        assert 'filters:' in yaml_str
        assert 'processing_operations:' in yaml_str
        assert 'cache_enabled:' in yaml_str
    
    def test_query_copy(self, mock_database_manager):
        """Test query copying."""
        original_query = AnalyticsQuery(mock_database_manager)
        configured_query = original_query.experiments(ids=[1])
        
        # Modifications should not affect original
        assert original_query.filters == {}
        assert 'experiments' in configured_query.filters
        
        # Test explicit copy
        copied_query = configured_query.copy()
        copied_query = copied_query.metrics(types=['test_acc'])
        
        # Original should not be affected
        assert 'metrics' not in configured_query.filters
        assert 'metrics' in copied_query.filters


class TestAnalyticsQueryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_filter_values(self, mock_database_manager):
        """Test handling of empty filter values."""
        query = AnalyticsQuery(mock_database_manager)
        
        # Empty lists should be handled gracefully
        filtered_query = query.experiments(ids=[])
        assert filtered_query.filters['experiments']['ids'] == []
    
    def test_invalid_aggregation_functions(self, mock_database_manager):
        """Test invalid aggregation functions."""
        query = AnalyticsQuery(mock_database_manager)
        
        with pytest.raises(ValueError, match="Invalid aggregation function"):
            query.aggregate(['invalid_function'])
    
    def test_invalid_sort_column(self, mock_database_manager):
        """Test invalid sort column."""
        query = AnalyticsQuery(mock_database_manager)
        
        with pytest.raises(ValueError, match="Invalid sort column"):
            query.sort_by('invalid_column')
    
    def test_invalid_limit_value(self, mock_database_manager):
        """Test invalid limit values."""
        query = AnalyticsQuery(mock_database_manager)
        
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            query.limit(-1)
        
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            query.limit(0)
    
    def test_duplicate_operations(self, mock_database_manager):
        """Test handling of duplicate operations."""
        query = AnalyticsQuery(mock_database_manager)
        
        # Should allow multiple operations of the same type
        processed_query = (query
                          .group_by('experiment_id')
                          .group_by('trial_id'))  # Second group_by should override
        
        # Only the last group_by should be present
        group_operations = [op for op in processed_query.processing_operations 
                           if op['type'] == 'group_by']
        assert len(group_operations) == 1
        assert group_operations[0]['parameters']['column'] == 'trial_id'
    
    def test_query_without_execution(self, mock_database_manager):
        """Test query building without execution."""
        query = AnalyticsQuery(mock_database_manager)
        
        # Should be able to build complex queries without executing
        complex_query = (query
                        .experiments(ids=list(range(100)))
                        .trials(status=['completed'])
                        .runs(status=[RunStatus.SUCCESS])
                        .metrics(types=['test_acc'])
                        .exclude_outliers('test_acc')
                        .aggregate(['mean', 'std'])
                        .group_by('experiment_id')
                        .sort_by('mean')
                        .limit(10))
        
        # Query should be valid and explainable
        explanation = complex_query.explain()
        assert explanation['complexity']['level'] == 'high'
        
        # SQL should be generated without errors
        sql = complex_query.to_sql()
        assert len(sql) > 0


class TestAnalyticsQueryPerformance:
    """Performance-related tests."""
    
    def test_query_building_performance(self, mock_database_manager):
        """Test performance of query building operations."""
        import time
        
        query = AnalyticsQuery(mock_database_manager)
        
        start_time = time.time()
        
        # Build a complex query
        for i in range(100):  # Build many similar queries
            complex_query = (query
                            .experiments(ids=[i])
                            .metrics(types=['test_acc'])
                            .aggregate(['mean'])
                            .group_by('experiment_id'))
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Query building should be fast (< 1 second for 100 queries)
        assert execution_time < 1.0
    
    def test_large_filter_handling(self, mock_database_manager):
        """Test handling of large filter sets."""
        query = AnalyticsQuery(mock_database_manager)
        
        # Large number of experiment IDs
        large_id_list = list(range(10000))
        
        filtered_query = query.experiments(ids=large_id_list)
        
        # Should handle large filters without issues
        assert len(filtered_query.filters['experiments']['ids']) == 10000
        
        # Complexity estimation should handle large filters
        complexity = filtered_query.estimate_complexity()
        assert complexity['level'] in ['medium', 'high'] 