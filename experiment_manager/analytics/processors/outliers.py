"""
Outlier Detection Processor for Analytics

Provides multiple methods for detecting outliers in experiment data including
statistical and custom threshold-based approaches.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from .base import DataProcessor


class OutlierProcessor(DataProcessor):
    """
    Processor for detecting outliers in experiment data using multiple methods.
    
    Detection Methods:
    - IQR Method: Interquartile range-based detection
    - Z-Score Method: Standard deviation-based detection  
    - Modified Z-Score: Median absolute deviation-based
    - Custom Thresholds: User-defined boundaries
    """
    
    def __init__(self, name: str = "OutlierProcessor"):
        """Initialize the outlier detection processor."""
        super().__init__(name)
        self.required_columns = ['metric_total_val']
    
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Detect outliers in the input data using specified methods.
        
        Args:
            data: Input DataFrame with experiment metrics
            **kwargs: Processing parameters:
                - method: Detection method ('iqr', 'zscore', 'modified_zscore', 'custom', 'all')
                - iqr_factor: Factor for IQR method (default: 1.5)
                - zscore_threshold: Threshold for Z-score method (default: 3.0)
                - modified_zscore_threshold: Threshold for modified Z-score (default: 3.5)
                - custom_lower: Lower bound for custom method
                - custom_upper: Upper bound for custom method
                - group_by: Column(s) to group detection by (optional)
                - include_scores: Include outlier scores in output (default: True)
                
        Returns:
            pd.DataFrame: Original data with outlier detection columns added
        """
        # Extract parameters
        method = kwargs.get('method', 'iqr')
        iqr_factor = kwargs.get('iqr_factor', 1.5)
        zscore_threshold = kwargs.get('zscore_threshold', 3.0)
        modified_zscore_threshold = kwargs.get('modified_zscore_threshold', 3.5)
        custom_lower = kwargs.get('custom_lower', None)
        custom_upper = kwargs.get('custom_upper', None)
        group_by = kwargs.get('group_by', None)
        include_scores = kwargs.get('include_scores', True)
        
        result_data = data.copy()
        
        # Apply outlier detection
        if group_by:
            result_data = self._detect_outliers_grouped(
                result_data, group_by, method, iqr_factor, zscore_threshold,
                modified_zscore_threshold, custom_lower, custom_upper, include_scores
            )
        else:
            result_data = self._detect_outliers_overall(
                result_data, method, iqr_factor, zscore_threshold,
                modified_zscore_threshold, custom_lower, custom_upper, include_scores
            )
        
        return result_data
    
    def _detect_outliers_overall(self, data: pd.DataFrame, method: str, iqr_factor: float,
                               zscore_threshold: float, modified_zscore_threshold: float,
                               custom_lower: Optional[float], custom_upper: Optional[float],
                               include_scores: bool) -> pd.DataFrame:
        """Detect outliers across the entire dataset."""
        values = data['metric_total_val'].dropna()
        
        if method == 'all':
            # Apply all methods
            self._apply_iqr_detection(data, values, iqr_factor, include_scores, suffix='_iqr')
            self._apply_zscore_detection(data, values, zscore_threshold, include_scores, suffix='_zscore')
            self._apply_modified_zscore_detection(data, values, modified_zscore_threshold, include_scores, suffix='_modified_zscore')
            if custom_lower is not None or custom_upper is not None:
                self._apply_custom_detection(data, values, custom_lower, custom_upper, include_scores, suffix='_custom')
            
            # Create combined outlier flag
            outlier_cols = [col for col in data.columns if col.startswith('is_outlier_')]
            data['is_outlier_any'] = data[outlier_cols].any(axis=1)
            data['outlier_method_count'] = data[outlier_cols].sum(axis=1)
            
        elif method == 'iqr':
            self._apply_iqr_detection(data, values, iqr_factor, include_scores)
        elif method == 'zscore':
            self._apply_zscore_detection(data, values, zscore_threshold, include_scores)
        elif method == 'modified_zscore':
            self._apply_modified_zscore_detection(data, values, modified_zscore_threshold, include_scores)
        elif method == 'custom':
            self._apply_custom_detection(data, values, custom_lower, custom_upper, include_scores)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return data
    
    def _detect_outliers_grouped(self, data: pd.DataFrame, group_by: Union[str, List[str]], 
                               method: str, iqr_factor: float, zscore_threshold: float,
                               modified_zscore_threshold: float, custom_lower: Optional[float],
                               custom_upper: Optional[float], include_scores: bool) -> pd.DataFrame:
        """Detect outliers within each group separately."""
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Initialize outlier columns
        if method == 'all':
            data['is_outlier_iqr'] = False
            data['is_outlier_zscore'] = False
            data['is_outlier_modified_zscore'] = False
            if custom_lower is not None or custom_upper is not None:
                data['is_outlier_custom'] = False
            if include_scores:
                data['outlier_score_iqr'] = 0.0
                data['outlier_score_zscore'] = 0.0
                data['outlier_score_modified_zscore'] = 0.0
                if custom_lower is not None or custom_upper is not None:
                    data['outlier_score_custom'] = 0.0
        else:
            data['is_outlier'] = False
            if include_scores:
                data['outlier_score'] = 0.0
        
        # Process each group
        for group_key, group_data in data.groupby(group_by):
            group_idx = group_data.index
            values = group_data['metric_total_val'].dropna()
            
            if len(values) < 3:  # Skip groups with insufficient data
                continue
            
            if method == 'all':
                # Apply all methods to this group
                group_result = group_data.copy()
                self._apply_iqr_detection(group_result, values, iqr_factor, include_scores, suffix='_iqr')
                self._apply_zscore_detection(group_result, values, zscore_threshold, include_scores, suffix='_zscore')
                self._apply_modified_zscore_detection(group_result, values, modified_zscore_threshold, include_scores, suffix='_modified_zscore')
                if custom_lower is not None or custom_upper is not None:
                    self._apply_custom_detection(group_result, values, custom_lower, custom_upper, include_scores, suffix='_custom')
                
                # Update main data
                for col in group_result.columns:
                    if col.startswith('is_outlier_') or col.startswith('outlier_score_'):
                        data.loc[group_idx, col] = group_result[col]
            else:
                # Apply single method to this group
                group_result = group_data.copy()
                if method == 'iqr':
                    self._apply_iqr_detection(group_result, values, iqr_factor, include_scores)
                elif method == 'zscore':
                    self._apply_zscore_detection(group_result, values, zscore_threshold, include_scores)
                elif method == 'modified_zscore':
                    self._apply_modified_zscore_detection(group_result, values, modified_zscore_threshold, include_scores)
                elif method == 'custom':
                    self._apply_custom_detection(group_result, values, custom_lower, custom_upper, include_scores)
                
                # Update main data
                data.loc[group_idx, 'is_outlier'] = group_result['is_outlier']
                if include_scores:
                    data.loc[group_idx, 'outlier_score'] = group_result['outlier_score']
        
        # Create combined flags for 'all' method
        if method == 'all':
            outlier_cols = [col for col in data.columns if col.startswith('is_outlier_') and not col.endswith('_any')]
            data['is_outlier_any'] = data[outlier_cols].any(axis=1)
            data['outlier_method_count'] = data[outlier_cols].sum(axis=1)
        
        return data
    
    def _apply_iqr_detection(self, data: pd.DataFrame, values: pd.Series, factor: float, 
                           include_scores: bool, suffix: str = '') -> None:
        """Apply IQR-based outlier detection."""
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        outlier_col = f'is_outlier{suffix}'
        data[outlier_col] = (data['metric_total_val'] < lower_bound) | (data['metric_total_val'] > upper_bound)
        
        if include_scores:
            # Calculate outlier score as distance from nearest bound
            score_col = f'outlier_score{suffix}'
            scores = np.zeros(len(data))
            
            # Lower outliers
            lower_mask = data['metric_total_val'] < lower_bound
            scores[lower_mask] = np.abs(data.loc[lower_mask, 'metric_total_val'] - lower_bound) / iqr
            
            # Upper outliers
            upper_mask = data['metric_total_val'] > upper_bound
            scores[upper_mask] = np.abs(data.loc[upper_mask, 'metric_total_val'] - upper_bound) / iqr
            
            data[score_col] = scores
    
    def _apply_zscore_detection(self, data: pd.DataFrame, values: pd.Series, threshold: float,
                              include_scores: bool, suffix: str = '') -> None:
        """Apply Z-score based outlier detection."""
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val == 0:
            # No variation in data
            outlier_col = f'is_outlier{suffix}'
            data[outlier_col] = False
            if include_scores:
                data[f'outlier_score{suffix}'] = 0.0
            return
        
        z_scores = np.abs((data['metric_total_val'] - mean_val) / std_val)
        
        outlier_col = f'is_outlier{suffix}'
        data[outlier_col] = z_scores > threshold
        
        if include_scores:
            data[f'outlier_score{suffix}'] = z_scores
    
    def _apply_modified_zscore_detection(self, data: pd.DataFrame, values: pd.Series, threshold: float,
                                       include_scores: bool, suffix: str = '') -> None:
        """Apply Modified Z-score (using median and MAD) outlier detection."""
        median_val = values.median()
        mad = np.median(np.abs(values - median_val))
        
        if mad == 0:
            # No variation in data
            outlier_col = f'is_outlier{suffix}'
            data[outlier_col] = False
            if include_scores:
                data[f'outlier_score{suffix}'] = 0.0
            return
        
        # Calculate modified Z-scores
        modified_z_scores = 0.6745 * np.abs((data['metric_total_val'] - median_val) / mad)
        
        outlier_col = f'is_outlier{suffix}'
        data[outlier_col] = modified_z_scores > threshold
        
        if include_scores:
            data[f'outlier_score{suffix}'] = modified_z_scores
    
    def _apply_custom_detection(self, data: pd.DataFrame, values: pd.Series, 
                              lower_bound: Optional[float], upper_bound: Optional[float],
                              include_scores: bool, suffix: str = '') -> None:
        """Apply custom threshold-based outlier detection."""
        outlier_col = f'is_outlier{suffix}'
        
        # Initialize outlier flags
        outliers = pd.Series(False, index=data.index)
        
        # Apply bounds
        if lower_bound is not None:
            outliers |= data['metric_total_val'] < lower_bound
        if upper_bound is not None:
            outliers |= data['metric_total_val'] > upper_bound
        
        data[outlier_col] = outliers
        
        if include_scores:
            # Calculate distance from nearest bound
            scores = np.zeros(len(data))
            
            if lower_bound is not None:
                lower_mask = data['metric_total_val'] < lower_bound
                scores[lower_mask] = np.abs(data.loc[lower_mask, 'metric_total_val'] - lower_bound)
            
            if upper_bound is not None:
                upper_mask = data['metric_total_val'] > upper_bound
                upper_distances = np.abs(data.loc[upper_mask, 'metric_total_val'] - upper_bound)
                scores[upper_mask] = np.maximum(scores[upper_mask], upper_distances)
            
            data[f'outlier_score{suffix}'] = scores
    
    def get_outlier_summary(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Get a summary of outlier detection results.
        
        Args:
            data: DataFrame with outlier detection results
            **kwargs: Additional parameters
            
        Returns:
            Dict containing outlier summary statistics
        """
        result_df = self._process_implementation(data, **kwargs)
        
        summary = {
            'total_observations': len(result_df),
            'missing_values': result_df['metric_total_val'].isna().sum()
        }
        
        # Find outlier columns
        outlier_cols = [col for col in result_df.columns if col.startswith('is_outlier')]
        
        for col in outlier_cols:
            method_name = col.replace('is_outlier', '').lstrip('_') or 'default'
            outlier_count = result_df[col].sum()
            summary[f'{method_name}_outliers'] = int(outlier_count)
            summary[f'{method_name}_outlier_rate'] = float(outlier_count / len(result_df))
        
        # Add score statistics if available
        score_cols = [col for col in result_df.columns if col.startswith('outlier_score')]
        for col in score_cols:
            method_name = col.replace('outlier_score', '').lstrip('_') or 'default'
            scores = result_df[col]
            summary[f'{method_name}_max_score'] = float(scores.max())
            summary[f'{method_name}_mean_score'] = float(scores.mean())
        
        return summary 