"""
Tests for analytics results module.
"""

import pytest
import pandas as pd
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, mock_open

from experiment_manager.analytics.results import AnalyticsResult
from .test_fixtures import (
    sample_experiment_data, sample_analytics_result
)


class TestAnalyticsResult:
    """Test cases for AnalyticsResult class."""
    
    def test_init_with_dataframe(self, sample_experiment_data):
        """Test AnalyticsResult initialization with DataFrame."""
        metadata = {'test': True}
        query_info = {'complexity': 'low'}
        
        result = AnalyticsResult(sample_experiment_data, metadata, query_info)
        
        assert isinstance(result.data, pd.DataFrame)
        assert result.metadata == metadata
        assert result.query_info == query_info
        assert isinstance(result.created_at, datetime)
        assert result.summary_statistics == {}
        assert result.processing_history == []
    
    def test_init_with_dict_data(self):
        """Test AnalyticsResult initialization with dict data."""
        data = {'values': [1, 2, 3], 'labels': ['a', 'b', 'c']}
        result = AnalyticsResult(data)
        
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 3
        assert 'values' in result.data.columns
        assert 'labels' in result.data.columns
    
    def test_add_summary_statistic(self, sample_analytics_result):
        """Test adding summary statistics."""
        result = sample_analytics_result
        
        result.add_summary_statistic('max_value', 0.92, 'Maximum test accuracy')
        
        assert 'max_value' in result.summary_statistics
        stat = result.summary_statistics['max_value']
        assert stat['value'] == 0.92
        assert stat['description'] == 'Maximum test accuracy'
        assert 'timestamp' in stat
    
    def test_add_processing_step(self, sample_analytics_result):
        """Test adding processing steps."""
        result = sample_analytics_result
        
        result.add_processing_step(
            'outlier_detector', 
            {'method': 'iqr', 'threshold': 1.5}, 
            'Outlier detection completed'
        )
        
        assert len(result.processing_history) == 2  # One from fixture + this one
        step = result.processing_history[-1]
        assert step['processor'] == 'outlier_detector'
        assert step['parameters']['method'] == 'iqr'
        assert step['description'] == 'Outlier detection completed'
    
    def test_get_summary(self, sample_analytics_result):
        """Test getting result summary."""
        result = sample_analytics_result
        summary = result.get_summary()
        
        assert 'overview' in summary
        assert 'summary_statistics' in summary
        assert 'processing_history' in summary
        assert 'metadata' in summary
        
        overview = summary['overview']
        assert 'row_count' in overview
        assert 'column_count' in overview
        assert 'columns' in overview
    
    def test_to_dataframe(self, sample_analytics_result):
        """Test converting to DataFrame."""
        result = sample_analytics_result
        df = result.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(result.data)
        assert list(df.columns) == list(result.data.columns)
    
    def test_len_operator(self, sample_analytics_result):
        """Test len() operator."""
        result = sample_analytics_result
        assert len(result) == len(result.data)
    
    def test_bool_operator(self, sample_analytics_result):
        """Test bool() operator."""
        result = sample_analytics_result
        assert bool(result) == True
        
        # Test with empty result
        empty_result = AnalyticsResult(pd.DataFrame())
        assert bool(empty_result) == False
    
    def test_iter_operator(self, sample_analytics_result):
        """Test iteration over result."""
        result = sample_analytics_result
        rows = list(result)
        
        assert len(rows) == len(result.data)
        # Each row should be a pandas Series
        assert all(isinstance(row, pd.Series) for _, row in enumerate(rows))
    
    def test_getitem_operator(self, sample_analytics_result):
        """Test indexing operator."""
        result = sample_analytics_result
        
        # Test column access
        if 'metric_value' in result.data.columns:
            metric_values = result['metric_value']
            assert isinstance(metric_values, pd.Series)
            assert len(metric_values) == len(result.data)
        
        # Test row access
        first_row = result[0]
        assert isinstance(first_row, pd.Series)
    
    def test_to_csv_export(self, sample_analytics_result):
        """Test CSV export functionality."""
        result = sample_analytics_result
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            returned_path = result.to_csv(filepath)
            
            assert returned_path == filepath
            assert os.path.exists(filepath)
            
            # Verify CSV content
            exported_df = pd.read_csv(filepath)
            assert len(exported_df) == len(result.data)
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_to_json_export(self, sample_analytics_result):
        """Test JSON export functionality."""
        result = sample_analytics_result
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            returned_path = result.to_json(filepath)
            
            assert returned_path == filepath
            assert os.path.exists(filepath)
            
            # Verify JSON content
            with open(filepath, 'r') as f:
                import json
                exported_data = json.load(f)
            
            assert 'data' in exported_data
            assert 'metadata' in exported_data
            assert 'summary_statistics' in exported_data
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_to_excel_export(self, sample_analytics_result):
        """Test Excel export functionality."""
        result = sample_analytics_result
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            filepath = f.name
        
        try:
            returned_path = result.to_excel(filepath)
            
            assert returned_path == filepath
            assert os.path.exists(filepath)
            
            # Verify Excel content
            exported_df = pd.read_excel(filepath, sheet_name='data')
            assert len(exported_df) == len(result.data)
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_to_html_report(self, sample_analytics_result):
        """Test HTML report generation."""
        result = sample_analytics_result
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            filepath = f.name
        
        try:
            returned_path = result.to_html_report(
                filepath, 
                title="Test Report",
                include_data_table=True
            )
            
            assert returned_path == filepath
            assert os.path.exists(filepath)
            
            # Verify HTML content
            with open(filepath, 'r') as f:
                html_content = f.read()
            
            assert 'Test Report' in html_content
            assert 'Summary Statistics' in html_content
            assert 'Processing History' in html_content
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_yaml_serialization(self, sample_analytics_result):
        """Test YAML serialization."""
        result = sample_analytics_result
        yaml_str = result.to_yaml()
        
        assert isinstance(yaml_str, str)
        assert 'metadata:' in yaml_str
        assert 'summary_statistics:' in yaml_str
        assert 'processing_history:' in yaml_str
        assert 'query_info:' in yaml_str
    
    def test_str_representation(self, sample_analytics_result):
        """Test string representation."""
        result = sample_analytics_result
        str_repr = str(result)
        
        assert 'AnalyticsResult' in str_repr
        assert f'{len(result)} rows' in str_repr
        assert f'{len(result.data.columns)} columns' in str_repr
    
    def test_repr_representation(self, sample_analytics_result):
        """Test detailed representation."""
        result = sample_analytics_result
        repr_str = repr(result)
        
        assert 'AnalyticsResult' in repr_str
        assert 'data=' in repr_str
        assert 'metadata=' in repr_str


class TestAnalyticsResultExportOptions:
    """Test cases for export options and parameters."""
    
    def test_csv_export_with_options(self, sample_analytics_result):
        """Test CSV export with custom options."""
        result = sample_analytics_result
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            result.to_csv(filepath, index=False, sep=';')
            
            # Verify custom separator
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert ';' in content
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_json_export_with_options(self, sample_analytics_result):
        """Test JSON export with custom options."""
        result = sample_analytics_result
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            result.to_json(filepath, indent=4, include_summary=False)
            
            # Verify indentation
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Check for indented JSON
            assert '    ' in content  # 4-space indentation
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_excel_export_with_multiple_sheets(self, sample_analytics_result):
        """Test Excel export with multiple sheets."""
        result = sample_analytics_result
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            filepath = f.name
        
        try:
            result.to_excel(filepath, include_summary_sheet=True)
            
            # Verify multiple sheets - properly close the ExcelFile
            with pd.ExcelFile(filepath) as xl_file:
                sheet_names = xl_file.sheet_names
            
            assert 'data' in sheet_names
            assert 'summary' in sheet_names
            
        finally:
            if os.path.exists(filepath):
                try:
                    os.unlink(filepath)
                except PermissionError:
                    # On Windows, sometimes we need to wait a moment
                    import time
                    time.sleep(0.1)
                    try:
                        os.unlink(filepath)
                    except PermissionError:
                        pass  # If still locked, just skip cleanup
    
    def test_html_report_customization(self, sample_analytics_result):
        """Test HTML report with custom styling."""
        result = sample_analytics_result
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            filepath = f.name
        
        try:
            result.to_html_report(
                filepath,
                title="Custom Report",
                include_data_table=False,
                custom_css="body { background-color: lightblue; }"
            )
            
            with open(filepath, 'r') as f:
                html_content = f.read()
            
            assert 'Custom Report' in html_content
            assert 'lightblue' in html_content
            # Should not include data table
            assert '<table' not in html_content or 'Data Table' not in html_content
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestAnalyticsResultEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = AnalyticsResult(empty_df)
        
        assert len(result) == 0
        assert bool(result) == False
        assert result.data.empty
    
    def test_invalid_export_path(self, sample_analytics_result):
        """Test export with invalid file path."""
        result = sample_analytics_result
        
        with pytest.raises(Exception):  # Should raise some OS/Permission error
            result.to_csv('/invalid/path/file.csv')
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create a larger dataset
        large_data = pd.DataFrame({
            'experiment_id': list(range(1000)),
            'metric_value': [0.8 + (i * 0.0001) for i in range(1000)],
            'status': ['success'] * 1000
        })
        
        result = AnalyticsResult(large_data)
        
        assert len(result) == 1000
        assert bool(result) == True
        
        # Test that summary works with large data
        summary = result.get_summary()
        assert summary['overview']['row_count'] == 1000
    
    def test_unicode_data_handling(self):
        """Test handling of Unicode data."""
        unicode_data = pd.DataFrame({
            'experiment_name': ['实验1', 'Exp_β', 'тест'],
            'metric_value': [0.85, 0.87, 0.82],
            'description': ['测试描述', 'Βeta test', 'Тестовое описание']
        })
        
        result = AnalyticsResult(unicode_data)
        
        assert len(result) == 3
        
        # Test that exports handle Unicode properly
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            result.to_json(filepath)
            
            # Should not raise encoding errors
            with open(filepath, 'r', encoding='utf-8') as f:
                import json
                exported_data = json.load(f)
            
            assert len(exported_data['data']) == 3
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_missing_data_handling(self):
        """Test handling of missing/NaN data."""
        data_with_nan = pd.DataFrame({
            'experiment_id': [1, 2, 3, 4],
            'metric_value': [0.85, None, 0.82, float('nan')],
            'status': ['success', 'failed', 'success', 'timeout']
        })
        
        result = AnalyticsResult(data_with_nan)
        
        assert len(result) == 4
        
        # Test that summary handles NaN values
        summary = result.get_summary()
        assert summary['overview']['row_count'] == 4
        
        # Test exports with NaN values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            result.to_csv(filepath)
            exported_df = pd.read_csv(filepath)
            assert len(exported_df) == 4
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestAnalyticsResultPerformance:
    """Performance-related tests."""
    
    def test_lazy_summary_computation(self, sample_analytics_result):
        """Test that summary is computed lazily."""
        result = sample_analytics_result
        
        # Summary should be computed on first access
        summary1 = result.get_summary()
        summary2 = result.get_summary()
        
        # Should return the same object (cached)
        assert summary1 is summary2
    
    def test_memory_efficient_iteration(self):
        """Test memory-efficient iteration over large results."""
        # Create a moderately large dataset
        large_data = pd.DataFrame({
            'id': list(range(500)),
            'value': [i * 0.01 for i in range(500)]
        })
        
        result = AnalyticsResult(large_data)
        
        # Iteration should not load all data into memory at once
        count = 0
        for row in result:
            count += 1
            if count > 10:  # Just test first few rows
                break
        
        assert count == 11 