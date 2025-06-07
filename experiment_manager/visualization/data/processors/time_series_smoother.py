"""
Time Series Smoother Processor

This module provides time series smoothing functionality for training curves
and other sequential data, supporting multiple smoothing algorithms and
configurable parameters.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from scipy import signal
from scipy.interpolate import interp1d

from ..pipeline import DataProcessor, ProcessingContext, ProcessingResult, ProcessingStatus

logger = logging.getLogger(__name__)


class TimeSeriesSmoother(DataProcessor):
    """
    Time series smoothing processor for training curves and sequential data.
    
    Supports multiple smoothing algorithms:
    - Exponential Moving Average (EMA)
    - Simple Moving Average (SMA)
    - Savitzky-Golay filter
    - LOWESS (Locally Weighted Scatterplot Smoothing)
    - Gaussian filter
    - Median filter
    
    Particularly useful for smoothing noisy training curves, loss functions,
    and other metrics that vary over time during model training.
    """
    
    @property
    def plugin_name(self) -> str:
        return "time_series_smoother"
    
    @property
    def plugin_description(self) -> str:
        return "Smooths time series data using various algorithms for training curve visualization"
    
    @property
    def supported_capabilities(self) -> List[str]:
        return [
            "exponential_moving_average",
            "simple_moving_average", 
            "savitzky_golay",
            "lowess",
            "gaussian_filter",
            "median_filter",
            "pandas_support",
            "numpy_support",
            "multi_series_support"
        ]
    
    @property
    def supports_rollback(self) -> bool:
        return True
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize smoother with configuration."""
        self.config = config
        self.default_method = config.get('method', 'exponential_moving_average')
        self.default_window = config.get('window', 10)
        self.default_alpha = config.get('alpha', 0.1)
        
        logger.info(f"Initialized TimeSeriesSmoother with method: {self.default_method}")
    
    def cleanup(self) -> None:
        """Cleanup smoother processor."""
        pass
    
    def validate_input(self, data: Any, context: ProcessingContext) -> bool:
        """Validate that input data can be smoothed."""
        if data is None:
            return False
        
        # Support pandas DataFrames, Series, numpy arrays, and lists
        if isinstance(data, (pd.DataFrame, pd.Series, np.ndarray, list)):
            return True
        
        # Support dictionaries with numeric values
        if isinstance(data, dict):
            return all(isinstance(v, (int, float, list, np.ndarray)) for v in data.values())
        
        return False
    
    def process(self, data: Any, context: ProcessingContext) -> ProcessingResult:
        """Process data by applying smoothing algorithms."""
        context.start_time = datetime.now()
        context.status = ProcessingStatus.RUNNING
        
        result = ProcessingResult(data=data, context=context)
        
        try:
            if not self.validate_input(data, context):
                result.add_error("Invalid input data for time series smoothing")
                context.status = ProcessingStatus.FAILED
                return result
            
            # Get smoothing configuration
            method = context.config.get('method', self.default_method)
            window = context.config.get('window', self.default_window)
            alpha = context.config.get('alpha', self.default_alpha)
            
            # Apply smoothing based on data type
            smoothed_data, original_data = self._apply_smoothing(
                data, method, window, alpha, context.config
            )
            
            context.end_time = datetime.now()
            context.status = ProcessingStatus.COMPLETED
            
            result.data = smoothed_data
            result.original_data = original_data  # Store for rollback
            result.metrics = {
                'smoothing_method': method,
                'window_size': window,
                'alpha': alpha,
                'data_points': self._get_data_size(data),
                'smoothing_factor': self._calculate_smoothing_factor(data, smoothed_data)
            }
            
            return result
            
        except Exception as e:
            context.end_time = datetime.now()
            context.status = ProcessingStatus.FAILED
            result.add_error(f"Time series smoothing failed: {str(e)}")
            logger.error(f"TimeSeriesSmoother error: {str(e)}")
            return result
    
    def _apply_smoothing(self, data: Any, method: str, window: int, alpha: float, 
                        config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Apply smoothing algorithm to data."""
        original_data = data
        
        if isinstance(data, pd.DataFrame):
            smoothed_data = self._smooth_dataframe(data, method, window, alpha, config)
        elif isinstance(data, pd.Series):
            smoothed_data = self._smooth_series(data, method, window, alpha, config)
        elif isinstance(data, np.ndarray):
            smoothed_data = self._smooth_array(data, method, window, alpha, config)
        elif isinstance(data, list):
            smoothed_data = self._smooth_list(data, method, window, alpha, config)
        elif isinstance(data, dict):
            smoothed_data = self._smooth_dict(data, method, window, alpha, config)
        else:
            smoothed_data = data
        
        return smoothed_data, original_data
    
    def _smooth_dataframe(self, df: pd.DataFrame, method: str, window: int, 
                         alpha: float, config: Dict[str, Any]) -> pd.DataFrame:
        """Smooth pandas DataFrame columns."""
        smoothed_df = df.copy()
        
        # Get columns to smooth (default: all numeric columns)
        columns_to_smooth = config.get('columns', None)
        if columns_to_smooth is None:
            columns_to_smooth = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns_to_smooth:
            if column in df.columns:
                smoothed_df[column] = self._smooth_series(
                    df[column], method, window, alpha, config
                )
        
        return smoothed_df
    
    def _smooth_series(self, series: pd.Series, method: str, window: int,
                      alpha: float, config: Dict[str, Any]) -> pd.Series:
        """Smooth pandas Series."""
        if len(series) < 2:
            return series
        
        # Handle missing values
        if series.isnull().any():
            series = series.interpolate()
        
        values = series.values
        smoothed_values = self._smooth_values(values, method, window, alpha, config)
        
        return pd.Series(smoothed_values, index=series.index, name=series.name)
    
    def _smooth_array(self, arr: np.ndarray, method: str, window: int,
                     alpha: float, config: Dict[str, Any]) -> np.ndarray:
        """Smooth numpy array."""
        if arr.ndim == 1:
            return self._smooth_values(arr, method, window, alpha, config)
        elif arr.ndim == 2:
            # Smooth each column independently
            smoothed_arr = np.zeros_like(arr)
            for i in range(arr.shape[1]):
                smoothed_arr[:, i] = self._smooth_values(
                    arr[:, i], method, window, alpha, config
                )
            return smoothed_arr
        else:
            logger.warning(f"Cannot smooth array with {arr.ndim} dimensions")
            return arr
    
    def _smooth_list(self, data: List[Any], method: str, window: int,
                    alpha: float, config: Dict[str, Any]) -> List[Any]:
        """Smooth list of numeric values."""
        if not data or len(data) < 2:
            return data
        
        # Convert to numpy array for processing
        try:
            arr = np.array(data, dtype=float)
            smoothed_arr = self._smooth_values(arr, method, window, alpha, config)
            return smoothed_arr.tolist()
        except (ValueError, TypeError):
            logger.warning("Cannot convert list to numeric array for smoothing")
            return data
    
    def _smooth_dict(self, data: Dict[str, Any], method: str, window: int,
                    alpha: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """Smooth dictionary values."""
        smoothed_dict = {}
        
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)):
                smoothed_dict[key] = self._smooth_list(value, method, window, alpha, config)
            elif isinstance(value, (int, float)):
                smoothed_dict[key] = value  # Single values don't need smoothing
            else:
                smoothed_dict[key] = value  # Keep non-numeric values as-is
        
        return smoothed_dict
    
    def _smooth_values(self, values: np.ndarray, method: str, window: int,
                      alpha: float, config: Dict[str, Any]) -> np.ndarray:
        """Apply smoothing algorithm to 1D array of values."""
        if len(values) < 2:
            return values
        
        # Handle NaN values
        mask = ~np.isnan(values)
        if not mask.any():
            return values
        
        clean_values = values[mask]
        
        if method == 'exponential_moving_average':
            smoothed = self._exponential_moving_average(clean_values, alpha)
        elif method == 'simple_moving_average':
            smoothed = self._simple_moving_average(clean_values, window)
        elif method == 'savitzky_golay':
            smoothed = self._savitzky_golay_filter(clean_values, window, config)
        elif method == 'lowess':
            smoothed = self._lowess_smoothing(clean_values, config)
        elif method == 'gaussian_filter':
            smoothed = self._gaussian_filter(clean_values, config)
        elif method == 'median_filter':
            smoothed = self._median_filter(clean_values, window)
        else:
            logger.warning(f"Unknown smoothing method: {method}, using EMA")
            smoothed = self._exponential_moving_average(clean_values, alpha)
        
        # Restore original shape with NaN values
        if len(smoothed) != len(values):
            # Interpolate back to original indices if needed
            result = np.full_like(values, np.nan)
            result[mask] = smoothed
        else:
            result = smoothed
        
        return result
    
    def _exponential_moving_average(self, values: np.ndarray, alpha: float) -> np.ndarray:
        """Apply exponential moving average smoothing."""
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed
    
    def _simple_moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply simple moving average smoothing."""
        if window >= len(values):
            return np.full_like(values, np.mean(values))
        
        # Use pandas for efficient rolling window
        series = pd.Series(values)
        smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
        return smoothed.values
    
    def _savitzky_golay_filter(self, values: np.ndarray, window: int, 
                              config: Dict[str, Any]) -> np.ndarray:
        """Apply Savitzky-Golay filter smoothing."""
        if len(values) < window:
            return values
        
        # Ensure window is odd
        if window % 2 == 0:
            window += 1
        
        polyorder = min(config.get('polyorder', 3), window - 1)
        
        try:
            return signal.savgol_filter(values, window, polyorder)
        except Exception as e:
            logger.warning(f"Savitzky-Golay filter failed: {e}, falling back to SMA")
            return self._simple_moving_average(values, window)
    
    def _lowess_smoothing(self, values: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Apply LOWESS smoothing."""
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            frac = config.get('frac', 0.3)  # Fraction of data to use for each local regression
            x = np.arange(len(values))
            
            smoothed = lowess(values, x, frac=frac, return_sorted=False)
            return smoothed
            
        except ImportError:
            logger.warning("statsmodels not available, falling back to SMA")
            window = max(1, int(len(values) * 0.1))
            return self._simple_moving_average(values, window)
    
    def _gaussian_filter(self, values: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Apply Gaussian filter smoothing."""
        try:
            from scipy.ndimage import gaussian_filter1d
            
            sigma = config.get('sigma', 1.0)
            return gaussian_filter1d(values, sigma)
            
        except ImportError:
            logger.warning("scipy.ndimage not available, falling back to SMA")
            window = max(1, int(sigma * 3))
            return self._simple_moving_average(values, window)
    
    def _median_filter(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply median filter smoothing."""
        try:
            from scipy.ndimage import median_filter
            
            # Ensure window is odd
            if window % 2 == 0:
                window += 1
            
            return median_filter(values, size=window)
            
        except ImportError:
            logger.warning("scipy.ndimage not available, falling back to SMA")
            return self._simple_moving_average(values, window)
    
    def _get_data_size(self, data: Any) -> int:
        """Get size of data for metrics."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return len(data)
        elif isinstance(data, np.ndarray):
            return data.size
        elif isinstance(data, (list, dict)):
            return len(data)
        return 1
    
    def _calculate_smoothing_factor(self, original: Any, smoothed: Any) -> float:
        """Calculate smoothing factor as variance reduction percentage."""
        try:
            if isinstance(original, pd.DataFrame):
                orig_var = original.select_dtypes(include=[np.number]).var().mean()
                smooth_var = smoothed.select_dtypes(include=[np.number]).var().mean()
            elif isinstance(original, (pd.Series, np.ndarray, list)):
                orig_values = np.array(original)
                smooth_values = np.array(smoothed)
                orig_var = np.var(orig_values[~np.isnan(orig_values)])
                smooth_var = np.var(smooth_values[~np.isnan(smooth_values)])
            else:
                return 0.0
            
            if orig_var == 0:
                return 0.0
            
            return ((orig_var - smooth_var) / orig_var) * 100.0
            
        except Exception:
            return 0.0
    
    def rollback(self, original_data: Any, processed_data: Any, 
                context: ProcessingContext) -> Any:
        """Rollback smoothing operation by returning original data."""
        logger.info("Rolling back time series smoothing")
        return original_data
    
    def get_cache_key(self, data: Any, context: ProcessingContext) -> str:
        """Generate cache key for smoothing operation."""
        method = context.config.get('method', self.default_method)
        window = context.config.get('window', self.default_window)
        alpha = context.config.get('alpha', self.default_alpha)
        
        # Create a simple hash of the data
        data_hash = hash(str(data)[:100])  # Use first 100 chars for efficiency
        
        return f"smooth_{method}_{window}_{alpha}_{data_hash}" 