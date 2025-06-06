"""
Tests for Analytics Query Builder

Tests the fluent query API including chainable methods, query state management,
validation, and execution capabilities.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd

from experiment_manager.analytics.query import AnalyticsQuery, ValidationError, QueryState
from experiment_manager.analytics.results import AnalyticsResult


class TestAnalyticsQueryFoundation:
    """Test the core foundation of AnalyticsQuery."""
    
    def test_query_initialization(self):
        """Test AnalyticsQuery initialization."""
        query = AnalyticsQuery()
        
        assert query._state is not None
        assert query._state.query_id is not None
        assert len(query._state.query_id) == 12  # MD5 hash truncated to 12 chars
        assert isinstance(query._state.created_at, datetime)
        assert query._execution_count == 0
    
    def test_query_id_generation(self):
        """Test that each query gets a unique ID."""
        query1 = AnalyticsQuery()
        query2 = AnalyticsQuery()
        
        assert query1._state.query_id != query2._state.query_id
    
    def test_query_state_copy(self):
        """Test that query state is properly copied when chaining."""
        original_query = AnalyticsQuery()
        original_query._state.filters.experiments['test'] = 'value'
        
        new_query = original_query._copy_state()
        
        # Should have different instances
        assert new_query is not original_query
        assert new_query._state is not original_query._state
        
        # But same content
        assert new_query._state.filters.experiments['test'] == 'value'
        
        # Modifying new query shouldn't affect original
        new_query._state.filters.experiments['test'] = 'new_value'
        assert original_query._state.filters.experiments['test'] == 'value'


class TestChainableFilterMethods:
    """Test the chainable filter methods."""
    
    def test_experiments_filter(self):
        """Test experiments filter method."""
        query = AnalyticsQuery()
        
        # Test with IDs
        filtered_query = query.experiments(ids=[1, 2, 3])
        assert filtered_query._state.filters.experiments['ids'] == [1, 2, 3]
        
        # Test with names
        filtered_query = query.experiments(names=['exp1', 'exp2'])
        assert filtered_query._state.filters.experiments['names'] == ['exp1', 'exp2']
        
        # Test with time range
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 12, 31)
        filtered_query = query.experiments(time_range=(start_time, end_time))
        assert filtered_query._state.filters.experiments['time_range'] == (start_time, end_time)
        
        # Test chaining returns new instance
        assert filtered_query is not query
    
    def test_experiments_filter_validation(self):
        """Test experiments filter validation."""
        query = AnalyticsQuery()
        
        # Invalid time_range should raise error
        with pytest.raises(ValidationError) as exc_info:
            query.experiments(time_range=(datetime.now(),))  # Only one element
        
        assert "time_range must be a tuple of (start_time, end_time)" in str(exc_info.value)
        assert exc_info.value.field == "time_range"