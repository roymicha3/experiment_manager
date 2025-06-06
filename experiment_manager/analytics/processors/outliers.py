"""
Outlier Detection Processor for Analytics

Provides multiple methods for detecting outliers in experiment data including
statistical and custom threshold-based approaches.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from omegaconf import DictConfig

from experiment_manager.analytics.processors.base import DataProcessor
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("OutlierProcessor")
class OutlierProcessor(DataProcessor):
    """
    Processor for detecting outliers in experiment data using multiple methods.
    
    Detection Methods:
    - IQR Method: Interquartile range-based detection
    - Z-Score Method: Standard deviation-based detection  
    - Modified Z-Score: Median absolute deviation-based
    - Custom Thresholds: User-defined boundaries
    """
    
    def __init__(self, name: str = "OutlierProcessor", config: DictConfig = None):
        """Initialize the outlier detection processor."""
        super().__init__(name, config)
        self.required_columns = ['metric_total_val']
        self.optional_columns = ['experiment_name', 'trial_name', 'epoch', 'run_status']
    
    @classmethod
    def from_config(cls, config: DictConfig, **kwargs):
        """Create OutlierProcessor from configuration."""
        # Get outlier-specific configuration
        outlier_config = config.get('outliers', DictConfig({}))
        name = kwargs.get('name', 'OutlierProcessor')
        return cls(name=name, config=outlier_config)
    
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Detect outliers in the input data.
        
        Args:
            data: Input DataFrame with experiment data
            **kwargs: Processing parameters:
                - method: Detection method ('iqr', 'zscore', 'modified_zscore', 'custom', 'all')
                - metric_columns: List of metric columns to analyze (optional)
                - iqr_factor: Factor for IQR method (default: 1.5)
                - zscore_threshold: Threshold for Z-score method (default: 3.0)
                - modified_zscore_threshold: Threshold for modified Z-score (default: 3.5)
                - custom_thresholds: Dict of custom thresholds per column
                - action: What to do with outliers ('exclude', 'flag', 'keep')
                - group_by: Column(s) to group outlier detection by (optional)
                
        Returns:
            pd.DataFrame: Data with outlier information
        """
        # Extract parameters
        method = kwargs.get('method', 'iqr')
        metric_columns = kwargs.get('metric_columns', ['metric_total_val'])
        iqr_factor = kwargs.get('iqr_factor', 1.5)
        zscore_threshold = kwargs.get('zscore_threshold', 3.0)
        modified_zscore_threshold = kwargs.get('modified_zscore_threshold', 3.5)
        custom_thresholds = kwargs.get('custom_thresholds', {})
        action = kwargs.get('action', 'exclude')
        group_by = kwargs.get('group_by', None)
        
        # Ensure metric_columns is a list
        if isinstance(metric_columns, str):
            metric_columns = [metric_columns]
        
        # Validate that metric columns exist
        missing_cols = set(metric_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Metric columns not found in data: {missing_cols}")
        
        # Copy data to avoid modifying original
        result_data = data.copy()
        
        # Initialize outlier tracking columns
        for col in metric_columns:
            result_data[f'{col}_is_outlier'] = False
            result_data[f'{col}_outlier_method'] = ''
            result_data[f'{col}_outlier_score'] = np.nan
        
        if group_by:
            # Group-wise outlier detection
            if isinstance(group_by, str):
                group_by = [group_by]
            
            # Validate group_by columns exist
            missing_group_cols = set(group_by) - set(result_data.columns)
            if missing_group_cols:
                raise ValueError(f"Group-by columns not found in data: {missing_group_cols}")
            
            grouped = result_data.groupby(group_by)
            
            for name, group_data in grouped:
                group_indices = group_data.index
                outlier_info = self._detect_outliers_in_group(
                    group_data, metric_columns, method, iqr_factor, 
                    zscore_threshold, modified_zscore_threshold, custom_thresholds
                )
                
                # Update main dataframe with group results
                for col in metric_columns:
                    if col in outlier_info:
                        result_data.loc[group_indices, f'{col}_is_outlier'] = outlier_info[col]['is_outlier']
                        result_data.loc[group_indices, f'{col}_outlier_method'] = outlier_info[col]['method']
                        result_data.loc[group_indices, f'{col}_outlier_score'] = outlier_info[col]['score']
        else:
            # Overall outlier detection
            outlier_info = self._detect_outliers_in_group(
                result_data, metric_columns, method, iqr_factor,
                zscore_threshold, modified_zscore_threshold, custom_thresholds
            )
            
            # Update dataframe with results
            for col in metric_columns:
                if col in outlier_info:
                    result_data[f'{col}_is_outlier'] = outlier_info[col]['is_outlier']
                    result_data[f'{col}_outlier_method'] = outlier_info[col]['method']
                    result_data[f'{col}_outlier_score'] = outlier_info[col]['score']
        
        # Apply the specified action
        if action == 'exclude':
            # Remove rows that are outliers in any metric column
            outlier_cols = [f'{col}_is_outlier' for col in metric_columns]
            is_any_outlier = result_data[outlier_cols].any(axis=1)
            result_data = result_data[~is_any_outlier]
        elif action == 'flag':
            # Keep all data but add outlier flags (already done above)
            pass
        elif action == 'keep':
            # Remove outlier detection columns, just return original data with processing
            outlier_tracking_cols = []
            for col in metric_columns:
                outlier_tracking_cols.extend([
                    f'{col}_is_outlier', f'{col}_outlier_method', f'{col}_outlier_score'
                ])
            result_data = result_data.drop(columns=outlier_tracking_cols, errors='ignore')
        else:
            raise ValueError(f"Unknown action: {action}")
        
        return result_data
    
    def _detect_outliers_in_group(self, data: pd.DataFrame, metric_columns: List[str],
                                 method: str, iqr_factor: float, zscore_threshold: float,
                                 modified_zscore_threshold: float, 
                                 custom_thresholds: Dict[str, Dict]) -> Dict[str, Dict]:
        """Detect outliers in a specific group of data."""
        outlier_info = {}
        
        for col in metric_columns:
            if col not in data.columns:
                continue
            
            values = data[col].dropna()
            if len(values) < 3:  # Need at least 3 values for meaningful outlier detection
                outlier_info[col] = {
                    'is_outlier': np.full(len(data), False),
                    'method': 'insufficient_data',
                    'score': np.full(len(data), np.nan)
                }
                continue
            
            if method == 'iqr' or method == 'all':
                outlier_result = self._detect_iqr_outliers(data, col, iqr_factor)
            elif method == 'zscore':
                outlier_result = self._detect_zscore_outliers(data, col, zscore_threshold)
            elif method == 'modified_zscore':
                outlier_result = self._detect_modified_zscore_outliers(data, col, modified_zscore_threshold)
            elif method == 'custom':
                if col not in custom_thresholds:
                    raise ValueError(f"Custom thresholds not provided for column: {col}")
                outlier_result = self._detect_custom_outliers(data, col, custom_thresholds[col])
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            outlier_info[col] = outlier_result
        
        return outlier_info
    
    def _detect_iqr_outliers(self, data: pd.DataFrame, column: str, factor: float) -> Dict[str, Any]:
        """Detect outliers using the Interquartile Range (IQR) method."""
        values = data[column]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        is_outlier = (values < lower_bound) | (values > upper_bound)
        
        # Calculate outlier score as distance from nearest boundary
        outlier_score = np.where(
            values < lower_bound,
            (lower_bound - values) / iqr,
            np.where(
                values > upper_bound,
                (values - upper_bound) / iqr,
                0
            )
        )
        
        return {
            'is_outlier': is_outlier.values,
            'method': 'iqr',
            'score': outlier_score,
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
    
    def _detect_zscore_outliers(self, data: pd.DataFrame, column: str, threshold: float) -> Dict[str, Any]:
        """Detect outliers using the Z-score method."""
        values = data[column]
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val == 0:
            # No variation in data
            return {
                'is_outlier': np.full(len(data), False),
                'method': 'zscore',
                'score': np.full(len(data), 0.0)
            }
        
        z_scores = np.abs((values - mean_val) / std_val)
        is_outlier = z_scores > threshold
        
        return {
            'is_outlier': is_outlier.values,
            'method': 'zscore',
            'score': z_scores.values,
            'threshold': threshold
        }
    
    def _detect_modified_zscore_outliers(self, data: pd.DataFrame, column: str, threshold: float) -> Dict[str, Any]:
        """Detect outliers using the Modified Z-score method (median-based)."""
        values = data[column]
        median_val = values.median()
        mad = np.median(np.abs(values - median_val))  # Median Absolute Deviation
        
        if mad == 0:
            # Use fallback MAD calculation
            mad = np.median(np.abs(values - median_val)) or 1.0
        
        # Modified Z-score
        modified_z_scores = 0.6745 * (values - median_val) / mad
        modified_z_scores = np.abs(modified_z_scores)
        
        is_outlier = modified_z_scores > threshold
        
        return {
            'is_outlier': is_outlier.values,
            'method': 'modified_zscore',
            'score': modified_z_scores.values,
            'threshold': threshold
        }
    
    def _detect_custom_outliers(self, data: pd.DataFrame, column: str, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Detect outliers using custom thresholds."""
        values = data[column]
        
        lower_bound = thresholds.get('lower', -np.inf)
        upper_bound = thresholds.get('upper', np.inf)
        
        is_outlier = (values < lower_bound) | (values > upper_bound)
        
        # Calculate distance from boundaries as score
        outlier_score = np.maximum(
            np.maximum(lower_bound - values, 0),
            np.maximum(values - upper_bound, 0)
        )
        
        return {
            'is_outlier': is_outlier.values,
            'method': 'custom',
            'score': outlier_score.values,
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
    
    def get_outlier_summary(self, data: pd.DataFrame, metric_columns: List[str] = None) -> Dict[str, Any]:
        """
        Get a summary of outlier detection results.
        
        Args:
            data: DataFrame with outlier detection results
            metric_columns: Columns to summarize (optional)
            
        Returns:
            Dict with outlier summary statistics
        """
        if metric_columns is None:
            metric_columns = ['metric_total_val']
        
        summary = {}
        
        for col in metric_columns:
            outlier_col = f'{col}_is_outlier'
            if outlier_col in data.columns:
                total_count = len(data)
                outlier_count = data[outlier_col].sum()
                outlier_rate = outlier_count / total_count if total_count > 0 else 0
                
                summary[col] = {
                    'total_count': total_count,
                    'outlier_count': int(outlier_count),
                    'outlier_rate': float(outlier_rate),
                    'clean_count': int(total_count - outlier_count)
                }
        
        return summary 