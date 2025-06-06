"""
Statistics Processor for Analytics

Provides comprehensive statistical analysis capabilities including basic statistics,
grouped calculations, confidence intervals, and missing data handling.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from .base import DataProcessor


class StatisticsProcessor(DataProcessor):
    """
    Processor for calculating basic and advanced statistics on experiment data.
    
    Capabilities:
    - Basic statistics: mean, median, std, min, max, count
    - Grouped statistics by trial, experiment, or custom grouping
    - Confidence intervals and percentiles  
    - Missing data handling strategies
    """
    
    def __init__(self, name: str = "StatisticsProcessor"):
        """Initialize the statistics processor."""
        super().__init__(name)
        # Core columns typically present in experiment data
        self.required_columns = ['metric_total_val']  # Minimum requirement
    
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate statistics on the input data.
        
        Args:
            data: Input DataFrame with experiment metrics
            **kwargs: Processing parameters:
                - group_by: Column(s) to group statistics by (str or list)
                - metrics: List of metric types to include (optional)
                - confidence_level: Confidence level for intervals (default: 0.95)
                - percentiles: List of percentiles to calculate (default: [25, 50, 75])
                - missing_strategy: How to handle missing data ('drop', 'fill_mean', 'fill_median', 'keep')
                - include_advanced: Include advanced statistics (default: True)
                
        Returns:
            pd.DataFrame: Statistics results
        """
        # Extract parameters
        group_by = kwargs.get('group_by', None)
        metrics = kwargs.get('metrics', None)
        confidence_level = kwargs.get('confidence_level', 0.95)
        percentiles = kwargs.get('percentiles', [25, 50, 75])
        missing_strategy = kwargs.get('missing_strategy', 'drop')
        include_advanced = kwargs.get('include_advanced', True)
        
        # Handle missing data
        processed_data = self._handle_missing_data(data.copy(), missing_strategy)
        
        # Filter by metrics if specified
        if metrics and 'metric_type' in processed_data.columns:
            processed_data = processed_data[processed_data['metric_type'].isin(metrics)]
        
        # Calculate statistics
        if group_by:
            stats_result = self._calculate_grouped_statistics(
                processed_data, group_by, confidence_level, percentiles, include_advanced
            )
        else:
            stats_result = self._calculate_overall_statistics(
                processed_data, confidence_level, percentiles, include_advanced
            )
        
        return stats_result
    
    def _handle_missing_data(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing data according to the specified strategy."""
        if strategy == 'drop':
            return data.dropna(subset=['metric_total_val'])
        elif strategy == 'fill_mean':
            data['metric_total_val'] = data['metric_total_val'].fillna(data['metric_total_val'].mean())
        elif strategy == 'fill_median':
            data['metric_total_val'] = data['metric_total_val'].fillna(data['metric_total_val'].median())
        elif strategy == 'keep':
            pass  # Keep missing values as-is
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")
        
        return data
    
    def _calculate_basic_statistics(self, values: pd.Series) -> Dict[str, float]:
        """Calculate basic statistics for a series of values."""
        # Filter out NaN values for calculations
        clean_values = values.dropna()
        
        if len(clean_values) == 0:
            return {
                'count': 0,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan
            }
        
        return {
            'count': len(clean_values),
            'mean': float(clean_values.mean()),
            'median': float(clean_values.median()),
            'std': float(clean_values.std()),
            'min': float(clean_values.min()),
            'max': float(clean_values.max())
        }
    
    def _calculate_percentiles(self, values: pd.Series, percentiles: List[float]) -> Dict[str, float]:
        """Calculate specified percentiles."""
        clean_values = values.dropna()
        if len(clean_values) == 0:
            return {f'p{p}': np.nan for p in percentiles}
        
        result = {}
        for p in percentiles:
            result[f'p{p}'] = float(np.percentile(clean_values, p))
        
        return result
    
    def _calculate_confidence_interval(self, values: pd.Series, confidence_level: float) -> Dict[str, float]:
        """Calculate confidence interval for the mean."""
        clean_values = values.dropna()
        if len(clean_values) < 2:
            return {
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'ci_margin': np.nan
            }
        
        mean = clean_values.mean()
        sem = stats.sem(clean_values)  # Standard error of the mean
        margin = sem * stats.t.ppf((1 + confidence_level) / 2, len(clean_values) - 1)
        
        return {
            'ci_lower': float(mean - margin),
            'ci_upper': float(mean + margin),
            'ci_margin': float(margin)
        }
    
    def _calculate_advanced_statistics(self, values: pd.Series) -> Dict[str, float]:
        """Calculate advanced statistics like skewness, kurtosis, etc."""
        clean_values = values.dropna()
        if len(clean_values) < 3:
            return {
                'skewness': np.nan,
                'kurtosis': np.nan,
                'variance': np.nan,
                'cv': np.nan  # Coefficient of variation
            }
        
        mean_val = clean_values.mean()
        std_val = clean_values.std()
        
        return {
            'skewness': float(stats.skew(clean_values)),
            'kurtosis': float(stats.kurtosis(clean_values)),
            'variance': float(clean_values.var()),
            'cv': float(std_val / mean_val) if mean_val != 0 else np.nan
        }
    
    def _calculate_overall_statistics(self, data: pd.DataFrame, confidence_level: float, 
                                   percentiles: List[float], include_advanced: bool) -> pd.DataFrame:
        """Calculate statistics for the entire dataset."""
        values = data['metric_total_val']
        
        # Start with basic statistics
        stats_dict = self._calculate_basic_statistics(values)
        
        # Add percentiles
        stats_dict.update(self._calculate_percentiles(values, percentiles))
        
        # Add confidence interval
        stats_dict.update(self._calculate_confidence_interval(values, confidence_level))
        
        # Add advanced statistics if requested
        if include_advanced:
            stats_dict.update(self._calculate_advanced_statistics(values))
        
        # Add metadata
        stats_dict.update({
            'confidence_level': confidence_level,
            'missing_count': data['metric_total_val'].isna().sum(),
            'total_observations': len(data)
        })
        
        # Convert to DataFrame
        return pd.DataFrame([stats_dict])
    
    def _calculate_grouped_statistics(self, data: pd.DataFrame, group_by: Union[str, List[str]], 
                                    confidence_level: float, percentiles: List[float],
                                    include_advanced: bool) -> pd.DataFrame:
        """Calculate statistics grouped by specified columns."""
        # Ensure group_by is a list
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Verify group columns exist
        missing_cols = set(group_by) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Grouping columns not found in data: {missing_cols}")
        
        results = []
        
        for group_key, group_data in data.groupby(group_by):
            values = group_data['metric_total_val']
            
            # Start with group identifiers
            if isinstance(group_key, tuple):
                group_dict = dict(zip(group_by, group_key))
            else:
                group_dict = {group_by[0]: group_key}
            
            # Calculate statistics for this group
            stats_dict = self._calculate_basic_statistics(values)
            stats_dict.update(self._calculate_percentiles(values, percentiles))
            stats_dict.update(self._calculate_confidence_interval(values, confidence_level))
            
            if include_advanced:
                stats_dict.update(self._calculate_advanced_statistics(values))
            
            # Add metadata
            stats_dict.update({
                'confidence_level': confidence_level,
                'missing_count': values.isna().sum(),
                'total_observations': len(group_data)
            })
            
            # Combine group identifiers with statistics
            result_dict = {**group_dict, **stats_dict}
            results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def calculate_summary_statistics(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate high-level summary statistics for quick insights.
        
        Args:
            data: Input DataFrame
            **kwargs: Same as _process_implementation
            
        Returns:
            Dict containing summary statistics
        """
        result_df = self._process_implementation(data, **kwargs)
        
        if len(result_df) == 1:
            # Single overall result
            return result_df.iloc[0].to_dict()
        else:
            # Multiple groups - provide summary
            return {
                'group_count': len(result_df),
                'overall_mean': result_df['mean'].mean(),
                'overall_std': result_df['std'].mean(),
                'best_group': result_df.loc[result_df['mean'].idxmax()].to_dict(),
                'worst_group': result_df.loc[result_df['mean'].idxmin()].to_dict()
            } 