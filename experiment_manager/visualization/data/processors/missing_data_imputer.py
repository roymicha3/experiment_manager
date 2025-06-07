"""
Missing Data Imputer Processor

This module provides missing data imputation functionality for handling gaps
in time series data, training curves, and other sequential datasets commonly
found in machine learning experiments.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from scipy.interpolate import interp1d

from ..pipeline import DataProcessor, ProcessingContext, ProcessingResult, ProcessingStatus

logger = logging.getLogger(__name__)


class MissingDataImputer(DataProcessor):
    """
    Missing data imputation processor for handling gaps in experimental data.
    
    Supports multiple imputation strategies:
    - Forward fill (carry last observation forward)
    - Backward fill (carry next observation backward)
    - Linear interpolation
    - Polynomial interpolation
    - Spline interpolation
    - Mean/median/mode imputation
    - Zero fill
    - Custom value fill
    - Time-aware interpolation for time series
    
    Particularly useful for handling missing values in training logs,
    metric recordings, and experimental measurements.
    """
    
    @property
    def plugin_name(self) -> str:
        return "missing_data_imputer"
    
    @property
    def plugin_description(self) -> str:
        return "Imputes missing values in datasets using various strategies"
    
    @property
    def supported_capabilities(self) -> List[str]:
        return [
            "forward_fill",
            "backward_fill",
            "linear_interpolation",
            "polynomial_interpolation",
            "spline_interpolation",
            "mean_imputation",
            "median_imputation",
            "mode_imputation",
            "zero_fill",
            "custom_fill",
            "time_aware_interpolation",
            "pandas_support",
            "numpy_support",
            "multi_column_support"
        ]
    
    @property
    def supports_rollback(self) -> bool:
        return True
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize imputer with configuration."""
        self.config = config
        self.default_strategy = config.get('strategy', 'linear_interpolation')
        self.default_fill_value = config.get('fill_value', 0.0)
        self.max_gap_size = config.get('max_gap_size', None)
        
        logger.info(f"Initialized MissingDataImputer with strategy: {self.default_strategy}")
    
    def cleanup(self) -> None:
        """Cleanup imputer processor."""
        pass
    
    def validate_input(self, data: Any, context: ProcessingContext) -> bool:
        """Validate that input data can be imputed."""
        if data is None:
            return False
        
        # Support pandas DataFrames, Series, numpy arrays, and lists
        return isinstance(data, (pd.DataFrame, pd.Series, np.ndarray, list, dict))
    
    def process(self, data: Any, context: ProcessingContext) -> ProcessingResult:
        """Process data by imputing missing values."""
        context.start_time = datetime.now()
        context.status = ProcessingStatus.RUNNING
        
        result = ProcessingResult(data=data, context=context)
        
        try:
            if not self.validate_input(data, context):
                result.add_error("Invalid input data for missing data imputation")
                context.status = ProcessingStatus.FAILED
                return result
            
            # Get imputation configuration
            strategy = context.config.get('strategy', self.default_strategy)
            fill_value = context.config.get('fill_value', self.default_fill_value)
            max_gap_size = context.config.get('max_gap_size', self.max_gap_size)
            
            # Apply imputation based on data type
            imputed_data, original_data, missing_info = self._apply_imputation(
                data, strategy, fill_value, max_gap_size, context.config
            )
            
            context.end_time = datetime.now()
            context.status = ProcessingStatus.COMPLETED
            
            result.data = imputed_data
            result.original_data = original_data  # Store for rollback
            result.metrics = {
                'imputation_strategy': strategy,
                'missing_values_found': missing_info['total_missing'],
                'missing_values_imputed': missing_info['imputed'],
                'missing_percentage': missing_info['missing_percentage'],
                'gaps_found': missing_info['gaps_found'],
                'largest_gap_size': missing_info['largest_gap']
            }
            
            if missing_info['total_missing'] > 0:
                result.add_warning(f"Imputed {missing_info['imputed']} missing values "
                                 f"({missing_info['missing_percentage']:.1f}% of data)")
            
            return result
            
        except Exception as e:
            context.end_time = datetime.now()
            context.status = ProcessingStatus.FAILED
            result.add_error(f"Missing data imputation failed: {str(e)}")
            logger.error(f"MissingDataImputer error: {str(e)}")
            return result
    
    def _apply_imputation(self, data: Any, strategy: str, fill_value: Any,
                         max_gap_size: Optional[int], config: Dict[str, Any]) -> Tuple[Any, Any, Dict]:
        """Apply imputation strategy to data."""
        original_data = data
        
        if isinstance(data, pd.DataFrame):
            imputed_data, missing_info = self._impute_dataframe(
                data, strategy, fill_value, max_gap_size, config
            )
        elif isinstance(data, pd.Series):
            imputed_data, missing_info = self._impute_series(
                data, strategy, fill_value, max_gap_size, config
            )
        elif isinstance(data, np.ndarray):
            imputed_data, missing_info = self._impute_array(
                data, strategy, fill_value, max_gap_size, config
            )
        elif isinstance(data, list):
            imputed_data, missing_info = self._impute_list(
                data, strategy, fill_value, max_gap_size, config
            )
        elif isinstance(data, dict):
            imputed_data, missing_info = self._impute_dict(
                data, strategy, fill_value, max_gap_size, config
            )
        else:
            imputed_data = data
            missing_info = self._empty_missing_info()
        
        return imputed_data, original_data, missing_info
    
    def _impute_dataframe(self, df: pd.DataFrame, strategy: str, fill_value: Any,
                         max_gap_size: Optional[int], config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
        """Impute missing values in pandas DataFrame."""
        imputed_df = df.copy()
        total_missing = 0
        total_imputed = 0
        gaps_found = 0
        largest_gap = 0
        
        # Get columns to impute (default: all columns)
        columns_to_impute = config.get('columns', df.columns.tolist())
        
        for column in columns_to_impute:
            if column in df.columns:
                series_result, series_info = self._impute_series(
                    df[column], strategy, fill_value, max_gap_size, config
                )
                imputed_df[column] = series_result
                
                total_missing += series_info['total_missing']
                total_imputed += series_info['imputed']
                gaps_found += series_info['gaps_found']
                largest_gap = max(largest_gap, series_info['largest_gap'])
        
        missing_info = {
            'total_missing': total_missing,
            'imputed': total_imputed,
            'missing_percentage': (total_missing / df.size) * 100 if df.size > 0 else 0,
            'gaps_found': gaps_found,
            'largest_gap': largest_gap
        }
        
        return imputed_df, missing_info
    
    def _impute_series(self, series: pd.Series, strategy: str, fill_value: Any,
                      max_gap_size: Optional[int], config: Dict[str, Any]) -> Tuple[pd.Series, Dict]:
        """Impute missing values in pandas Series."""
        if series.isnull().sum() == 0:
            return series, self._empty_missing_info()
        
        imputed_series = series.copy()
        missing_mask = series.isnull()
        total_missing = missing_mask.sum()
        
        # Analyze gap structure
        gaps_info = self._analyze_gaps(missing_mask)
        
        # Filter gaps by max_gap_size if specified
        if max_gap_size is not None:
            gaps_to_fill = [gap for gap in gaps_info['gaps'] if gap['size'] <= max_gap_size]
            imputed_count = sum(gap['size'] for gap in gaps_to_fill)
        else:
            gaps_to_fill = gaps_info['gaps']
            imputed_count = total_missing
        
        # Apply imputation strategy
        if strategy == 'forward_fill':
            imputed_series = self._forward_fill_series(imputed_series, gaps_to_fill)
        elif strategy == 'backward_fill':
            imputed_series = self._backward_fill_series(imputed_series, gaps_to_fill)
        elif strategy == 'linear_interpolation':
            imputed_series = self._linear_interpolate_series(imputed_series, gaps_to_fill)
        elif strategy == 'polynomial_interpolation':
            imputed_series = self._polynomial_interpolate_series(imputed_series, gaps_to_fill, config)
        elif strategy == 'spline_interpolation':
            imputed_series = self._spline_interpolate_series(imputed_series, gaps_to_fill, config)
        elif strategy == 'mean_imputation':
            imputed_series = self._mean_impute_series(imputed_series, gaps_to_fill)
        elif strategy == 'median_imputation':
            imputed_series = self._median_impute_series(imputed_series, gaps_to_fill)
        elif strategy == 'mode_imputation':
            imputed_series = self._mode_impute_series(imputed_series, gaps_to_fill)
        elif strategy == 'zero_fill':
            imputed_series = self._zero_fill_series(imputed_series, gaps_to_fill)
        elif strategy == 'custom_fill':
            imputed_series = self._custom_fill_series(imputed_series, gaps_to_fill, fill_value)
        else:
            logger.warning(f"Unknown imputation strategy: {strategy}, using linear interpolation")
            imputed_series = self._linear_interpolate_series(imputed_series, gaps_to_fill)
        
        missing_info = {
            'total_missing': total_missing,
            'imputed': imputed_count,
            'missing_percentage': (total_missing / len(series)) * 100 if len(series) > 0 else 0,
            'gaps_found': len(gaps_info['gaps']),
            'largest_gap': gaps_info['largest_gap']
        }
        
        return imputed_series, missing_info
    
    def _impute_array(self, arr: np.ndarray, strategy: str, fill_value: Any,
                     max_gap_size: Optional[int], config: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Impute missing values in numpy array."""
        if arr.ndim == 1:
            # Convert to pandas Series for easier handling
            series = pd.Series(arr)
            imputed_series, missing_info = self._impute_series(
                series, strategy, fill_value, max_gap_size, config
            )
            return imputed_series.values, missing_info
        elif arr.ndim == 2:
            # Impute each column independently
            imputed_arr = arr.copy()
            total_missing = 0
            total_imputed = 0
            gaps_found = 0
            largest_gap = 0
            
            for i in range(arr.shape[1]):
                column_series = pd.Series(arr[:, i])
                imputed_series, column_info = self._impute_series(
                    column_series, strategy, fill_value, max_gap_size, config
                )
                imputed_arr[:, i] = imputed_series.values
                
                total_missing += column_info['total_missing']
                total_imputed += column_info['imputed']
                gaps_found += column_info['gaps_found']
                largest_gap = max(largest_gap, column_info['largest_gap'])
            
            missing_info = {
                'total_missing': total_missing,
                'imputed': total_imputed,
                'missing_percentage': (total_missing / arr.size) * 100 if arr.size > 0 else 0,
                'gaps_found': gaps_found,
                'largest_gap': largest_gap
            }
            
            return imputed_arr, missing_info
        else:
            logger.warning(f"Cannot impute array with {arr.ndim} dimensions")
            return arr, self._empty_missing_info()
    
    def _impute_list(self, data: List[Any], strategy: str, fill_value: Any,
                    max_gap_size: Optional[int], config: Dict[str, Any]) -> Tuple[List[Any], Dict]:
        """Impute missing values in list."""
        if not data:
            return data, self._empty_missing_info()
        
        # Convert to pandas Series for processing
        try:
            series = pd.Series(data)
            imputed_series, missing_info = self._impute_series(
                series, strategy, fill_value, max_gap_size, config
            )
            return imputed_series.tolist(), missing_info
        except Exception as e:
            logger.warning(f"Cannot impute list data: {e}")
            return data, self._empty_missing_info()
    
    def _impute_dict(self, data: Dict[str, Any], strategy: str, fill_value: Any,
                    max_gap_size: Optional[int], config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict]:
        """Impute missing values in dictionary."""
        imputed_dict = {}
        total_missing = 0
        total_imputed = 0
        gaps_found = 0
        largest_gap = 0
        
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray, pd.Series)):
                imputed_value, value_info = self._impute_list(
                    value, strategy, fill_value, max_gap_size, config
                )
                imputed_dict[key] = imputed_value
                
                total_missing += value_info['total_missing']
                total_imputed += value_info['imputed']
                gaps_found += value_info['gaps_found']
                largest_gap = max(largest_gap, value_info['largest_gap'])
            else:
                imputed_dict[key] = value
        
        missing_info = {
            'total_missing': total_missing,
            'imputed': total_imputed,
            'missing_percentage': 0,  # Cannot calculate for mixed dict
            'gaps_found': gaps_found,
            'largest_gap': largest_gap
        }
        
        return imputed_dict, missing_info
    
    def _analyze_gaps(self, missing_mask: pd.Series) -> Dict:
        """Analyze gap structure in missing data."""
        gaps = []
        gap_start = None
        largest_gap = 0
        
        for i, is_missing in enumerate(missing_mask):
            if is_missing and gap_start is None:
                gap_start = i
            elif not is_missing and gap_start is not None:
                gap_size = i - gap_start
                gaps.append({'start': gap_start, 'end': i - 1, 'size': gap_size})
                largest_gap = max(largest_gap, gap_size)
                gap_start = None
        
        # Handle gap at the end
        if gap_start is not None:
            gap_size = len(missing_mask) - gap_start
            gaps.append({'start': gap_start, 'end': len(missing_mask) - 1, 'size': gap_size})
            largest_gap = max(largest_gap, gap_size)
        
        return {
            'gaps': gaps,
            'largest_gap': largest_gap
        }
    
    def _forward_fill_series(self, series: pd.Series, gaps_to_fill: List[Dict]) -> pd.Series:
        """Apply forward fill to specified gaps."""
        result = series.copy()
        for gap in gaps_to_fill:
            if gap['start'] > 0:
                fill_value = result.iloc[gap['start'] - 1]
                result.iloc[gap['start']:gap['end'] + 1] = fill_value
        return result
    
    def _backward_fill_series(self, series: pd.Series, gaps_to_fill: List[Dict]) -> pd.Series:
        """Apply backward fill to specified gaps."""
        result = series.copy()
        for gap in gaps_to_fill:
            if gap['end'] < len(series) - 1:
                fill_value = result.iloc[gap['end'] + 1]
                result.iloc[gap['start']:gap['end'] + 1] = fill_value
        return result
    
    def _linear_interpolate_series(self, series: pd.Series, gaps_to_fill: List[Dict]) -> pd.Series:
        """Apply linear interpolation to specified gaps."""
        result = series.copy()
        
        # Use pandas interpolate method for efficiency
        result = result.interpolate(method='linear', limit_direction='both')
        
        return result
    
    def _polynomial_interpolate_series(self, series: pd.Series, gaps_to_fill: List[Dict],
                                     config: Dict[str, Any]) -> pd.Series:
        """Apply polynomial interpolation to specified gaps."""
        order = config.get('polynomial_order', 2)
        result = series.copy()
        
        try:
            result = result.interpolate(method='polynomial', order=order, limit_direction='both')
        except Exception as e:
            logger.warning(f"Polynomial interpolation failed: {e}, falling back to linear")
            result = self._linear_interpolate_series(result, gaps_to_fill)
        
        return result
    
    def _spline_interpolate_series(self, series: pd.Series, gaps_to_fill: List[Dict],
                                  config: Dict[str, Any]) -> pd.Series:
        """Apply spline interpolation to specified gaps."""
        order = config.get('spline_order', 3)
        result = series.copy()
        
        try:
            result = result.interpolate(method='spline', order=order, limit_direction='both')
        except Exception as e:
            logger.warning(f"Spline interpolation failed: {e}, falling back to linear")
            result = self._linear_interpolate_series(result, gaps_to_fill)
        
        return result
    
    def _mean_impute_series(self, series: pd.Series, gaps_to_fill: List[Dict]) -> pd.Series:
        """Apply mean imputation to specified gaps."""
        result = series.copy()
        mean_value = series.mean()
        
        for gap in gaps_to_fill:
            result.iloc[gap['start']:gap['end'] + 1] = mean_value
        
        return result
    
    def _median_impute_series(self, series: pd.Series, gaps_to_fill: List[Dict]) -> pd.Series:
        """Apply median imputation to specified gaps."""
        result = series.copy()
        median_value = series.median()
        
        for gap in gaps_to_fill:
            result.iloc[gap['start']:gap['end'] + 1] = median_value
        
        return result
    
    def _mode_impute_series(self, series: pd.Series, gaps_to_fill: List[Dict]) -> pd.Series:
        """Apply mode imputation to specified gaps."""
        result = series.copy()
        mode_values = series.mode()
        mode_value = mode_values.iloc[0] if len(mode_values) > 0 else series.mean()
        
        for gap in gaps_to_fill:
            result.iloc[gap['start']:gap['end'] + 1] = mode_value
        
        return result
    
    def _zero_fill_series(self, series: pd.Series, gaps_to_fill: List[Dict]) -> pd.Series:
        """Apply zero fill to specified gaps."""
        result = series.copy()
        
        for gap in gaps_to_fill:
            result.iloc[gap['start']:gap['end'] + 1] = 0.0
        
        return result
    
    def _custom_fill_series(self, series: pd.Series, gaps_to_fill: List[Dict],
                           fill_value: Any) -> pd.Series:
        """Apply custom value fill to specified gaps."""
        result = series.copy()
        
        for gap in gaps_to_fill:
            result.iloc[gap['start']:gap['end'] + 1] = fill_value
        
        return result
    
    def _empty_missing_info(self) -> Dict:
        """Return empty missing info structure."""
        return {
            'total_missing': 0,
            'imputed': 0,
            'missing_percentage': 0.0,
            'gaps_found': 0,
            'largest_gap': 0
        }
    
    def rollback(self, original_data: Any, processed_data: Any,
                context: ProcessingContext) -> Any:
        """Rollback imputation operation by returning original data."""
        logger.info("Rolling back missing data imputation")
        return original_data
    
    def get_cache_key(self, data: Any, context: ProcessingContext) -> str:
        """Generate cache key for imputation operation."""
        strategy = context.config.get('strategy', self.default_strategy)
        fill_value = context.config.get('fill_value', self.default_fill_value)
        max_gap_size = context.config.get('max_gap_size', self.max_gap_size)
        
        # Create a simple hash of the data
        data_hash = hash(str(data)[:100])  # Use first 100 chars for efficiency
        
        return f"impute_{strategy}_{fill_value}_{max_gap_size}_{data_hash}" 