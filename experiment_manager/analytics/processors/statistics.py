"""
Statistics Processor for Analytics

Provides comprehensive statistical analysis capabilities including basic statistics,
grouped calculations, confidence intervals, and missing data handling.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from omegaconf import DictConfig

from experiment_manager.analytics.processors.base import DataProcessor
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("StatisticsProcessor")
class StatisticsProcessor(DataProcessor):
    """
    Processor for calculating basic and advanced statistics on experiment data.
    
    Capabilities:
    - Basic statistics: mean, median, std, min, max, count
    - Grouped statistics by trial, experiment, or custom grouping
    - Confidence intervals and percentiles  
    - Missing data handling strategies
    """
    
    def __init__(self, name: str = "StatisticsProcessor", config: DictConfig = None):
        """Initialize the statistics processor."""
        super().__init__(name, config)
        # Core columns typically present in experiment data
        self.required_columns = ['metric_total_val']  # Minimum requirement
        self.optional_columns = ['experiment_name', 'trial_name', 'epoch', 'run_status']
    
    @classmethod
    def from_config(cls, config: DictConfig, **kwargs):
        """Create StatisticsProcessor from configuration."""
        # Get statistics-specific configuration
        stats_config = config.get('statistics', DictConfig({}))
        name = kwargs.get('name', 'StatisticsProcessor')
        return cls(name=name, config=stats_config)
    
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate statistics on the input data.
        
        Args:
            data: Input DataFrame with experiment data
            **kwargs: Processing parameters:
                - group_by: Column(s) to group statistics by (optional)
                - metric_columns: List of metric columns to analyze (optional)
                - confidence_level: Statistical confidence level (default: 0.95)
                - percentiles: List of percentiles to calculate (default: [25, 50, 75, 90, 95])
                - missing_strategy: How to handle missing values ('drop', 'fill_mean', 'fill_median', 'keep')
                - include_advanced: Include advanced statistics like skew, kurtosis (default: True)
                
        Returns:
            pd.DataFrame: Statistical analysis results
        """
        # Extract parameters
        group_by = kwargs.get('group_by', None)
        metric_columns = kwargs.get('metric_columns', ['metric_total_val'])
        confidence_level = kwargs.get('confidence_level', 0.95)
        percentiles = kwargs.get('percentiles', [25, 50, 75, 90, 95])
        missing_strategy = kwargs.get('missing_strategy', 'drop')
        include_advanced = kwargs.get('include_advanced', True)
        
        # Ensure metric_columns is a list
        if isinstance(metric_columns, str):
            metric_columns = [metric_columns]
        
        # Validate that metric columns exist
        missing_cols = set(metric_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Metric columns not found in data: {missing_cols}")
        
        # Handle missing values
        processed_data = self._handle_missing_values(data.copy(), metric_columns, missing_strategy)
        
        results = []
        
        if group_by:
            # Grouped statistics
            if isinstance(group_by, str):
                group_by = [group_by]
            
            # Validate group_by columns exist
            missing_group_cols = set(group_by) - set(processed_data.columns)
            if missing_group_cols:
                raise ValueError(f"Group-by columns not found in data: {missing_group_cols}")
            
            grouped = processed_data.groupby(group_by)
            
            for name, group_data in grouped:
                group_stats = self._calculate_group_statistics(
                    group_data, metric_columns, confidence_level, percentiles, include_advanced
                )
                
                # Add group identifiers
                if isinstance(name, tuple):
                    for i, col in enumerate(group_by):
                        group_stats[col] = name[i]
                else:
                    group_stats[group_by[0]] = name
                
                results.append(group_stats)
        else:
            # Overall statistics
            overall_stats = self._calculate_group_statistics(
                processed_data, metric_columns, confidence_level, percentiles, include_advanced
            )
            overall_stats['group'] = 'overall'
            results.append(overall_stats)
        
        return pd.DataFrame(results)
    
    def _handle_missing_values(self, data: pd.DataFrame, metric_columns: List[str], strategy: str) -> pd.DataFrame:
        """Handle missing values in metric columns according to the specified strategy."""
        if strategy == 'drop':
            # Drop rows with any missing values in metric columns
            return data.dropna(subset=metric_columns)
        elif strategy == 'fill_mean':
            # Fill with column means
            for col in metric_columns:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].mean())
            return data
        elif strategy == 'fill_median':
            # Fill with column medians
            for col in metric_columns:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].median())
            return data
        elif strategy == 'keep':
            # Keep missing values as-is
            return data
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
    
    def _calculate_group_statistics(self, data: pd.DataFrame, metric_columns: List[str], 
                                  confidence_level: float, percentiles: List[float],
                                  include_advanced: bool) -> Dict[str, Any]:
        """Calculate statistics for a group of data."""
        stats_result = {}
        
        for col in metric_columns:
            if col not in data.columns:
                continue
                
            values = data[col].dropna()
            if len(values) == 0:
                continue
            
            col_prefix = f"{col}_"
            
            # Basic statistics
            stats_result[f"{col_prefix}count"] = len(values)
            stats_result[f"{col_prefix}mean"] = float(values.mean())
            stats_result[f"{col_prefix}median"] = float(values.median())
            stats_result[f"{col_prefix}std"] = float(values.std())
            stats_result[f"{col_prefix}min"] = float(values.min())
            stats_result[f"{col_prefix}max"] = float(values.max())
            
            # Percentiles
            for p in percentiles:
                stats_result[f"{col_prefix}p{p}"] = float(values.quantile(p / 100))
            
            # Confidence interval
            if len(values) > 1:
                confidence_interval = stats.t.interval(
                    confidence_level, len(values) - 1,
                    loc=values.mean(), scale=stats.sem(values)
                )
                stats_result[f"{col_prefix}ci_lower"] = float(confidence_interval[0])
                stats_result[f"{col_prefix}ci_upper"] = float(confidence_interval[1])
            
            # Advanced statistics
            if include_advanced and len(values) > 2:
                stats_result[f"{col_prefix}skew"] = float(stats.skew(values))
                stats_result[f"{col_prefix}kurtosis"] = float(stats.kurtosis(values))
                
                # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
                if len(values) <= 5000:
                    _, p_value = stats.shapiro(values)
                    stats_result[f"{col_prefix}normality_p"] = float(p_value)
                    stats_result[f"{col_prefix}is_normal"] = p_value > 0.05
        
        return stats_result 