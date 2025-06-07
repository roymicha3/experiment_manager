"""
Outlier Detector Processor

This module provides outlier detection and removal functionality for cleaning
experimental data, training curves, and other datasets commonly found in
machine learning workflows.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

from ..pipeline import DataProcessor, ProcessingContext, ProcessingResult, ProcessingStatus

logger = logging.getLogger(__name__)


class OutlierDetector(DataProcessor):
    """
    Outlier detection and removal processor for experimental data cleaning.
    
    Supports multiple outlier detection methods:
    - Z-score (standard deviation based)
    - Modified Z-score (median absolute deviation based)
    - Interquartile Range (IQR) method
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - One-Class SVM
    - Statistical percentile thresholds
    - Custom threshold functions
    
    Actions for detected outliers:
    - Remove (delete outlier points)
    - Cap (replace with threshold values)
    - Flag (mark but keep values)
    - Interpolate (replace with interpolated values)
    
    Particularly useful for cleaning noisy training metrics, removing
    anomalous measurements, and preprocessing experimental data.
    """
    
    @property
    def plugin_name(self) -> str:
        return "outlier_detector"
    
    @property
    def plugin_description(self) -> str:
        return "Detects and handles outliers in datasets using various statistical methods"
    
    @property
    def supported_capabilities(self) -> List[str]:
        return [
            "z_score_detection",
            "modified_z_score_detection",
            "iqr_detection",
            "isolation_forest",
            "local_outlier_factor",
            "one_class_svm",
            "percentile_thresholds",
            "custom_thresholds",
            "outlier_removal",
            "outlier_capping",
            "outlier_flagging",
            "outlier_interpolation",
            "pandas_support",
            "numpy_support",
            "multi_column_support"
        ]
    
    @property
    def supports_rollback(self) -> bool:
        return True
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize outlier detector with configuration."""
        self.config = config
        self.default_method = config.get('method', 'z_score')
        self.default_threshold = config.get('threshold', 3.0)
        self.default_action = config.get('action', 'remove')
        
        logger.info(f"Initialized OutlierDetector with method: {self.default_method}")
    
    def cleanup(self) -> None:
        """Cleanup outlier detector processor."""
        pass
    
    def validate_input(self, data: Any, context: ProcessingContext) -> bool:
        """Validate that input data can be processed for outliers."""
        if data is None:
            return False
        
        # Support pandas DataFrames, Series, numpy arrays, and lists
        return isinstance(data, (pd.DataFrame, pd.Series, np.ndarray, list, dict))
    
    def process(self, data: Any, context: ProcessingContext) -> ProcessingResult:
        """Process data by detecting and handling outliers."""
        context.start_time = datetime.now()
        context.status = ProcessingStatus.RUNNING
        
        result = ProcessingResult(data=data, context=context)
        
        try:
            if not self.validate_input(data, context):
                result.add_error("Invalid input data for outlier detection")
                context.status = ProcessingStatus.FAILED
                return result
            
            # Get outlier detection configuration
            method = context.config.get('method', self.default_method)
            threshold = context.config.get('threshold', self.default_threshold)
            action = context.config.get('action', self.default_action)
            
            # Apply outlier detection and handling
            processed_data, original_data, outlier_info = self._detect_and_handle_outliers(
                data, method, threshold, action, context.config
            )
            
            context.end_time = datetime.now()
            context.status = ProcessingStatus.COMPLETED
            
            result.data = processed_data
            result.original_data = original_data  # Store for rollback
            result.metrics = {
                'detection_method': method,
                'threshold': threshold,
                'action': action,
                'outliers_detected': outlier_info['total_outliers'],
                'outliers_percentage': outlier_info['outlier_percentage'],
                'outliers_by_column': outlier_info['outliers_by_column'],
                'data_points_processed': outlier_info['total_points']
            }
            
            if outlier_info['total_outliers'] > 0:
                result.add_warning(f"Detected {outlier_info['total_outliers']} outliers "
                                 f"({outlier_info['outlier_percentage']:.1f}% of data)")
            
            return result
            
        except Exception as e:
            context.end_time = datetime.now()
            context.status = ProcessingStatus.FAILED
            result.add_error(f"Outlier detection failed: {str(e)}")
            logger.error(f"OutlierDetector error: {str(e)}")
            return result
    
    def _detect_and_handle_outliers(self, data: Any, method: str, threshold: float,
                                   action: str, config: Dict[str, Any]) -> Tuple[Any, Any, Dict]:
        """Detect and handle outliers in data."""
        original_data = data
        
        if isinstance(data, pd.DataFrame):
            processed_data, outlier_info = self._process_dataframe(
                data, method, threshold, action, config
            )
        elif isinstance(data, pd.Series):
            processed_data, outlier_info = self._process_series(
                data, method, threshold, action, config
            )
        elif isinstance(data, np.ndarray):
            processed_data, outlier_info = self._process_array(
                data, method, threshold, action, config
            )
        elif isinstance(data, list):
            processed_data, outlier_info = self._process_list(
                data, method, threshold, action, config
            )
        elif isinstance(data, dict):
            processed_data, outlier_info = self._process_dict(
                data, method, threshold, action, config
            )
        else:
            processed_data = data
            outlier_info = self._empty_outlier_info()
        
        return processed_data, original_data, outlier_info
    
    def _process_dataframe(self, df: pd.DataFrame, method: str, threshold: float,
                          action: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
        """Process pandas DataFrame for outliers."""
        processed_df = df.copy()
        total_outliers = 0
        outliers_by_column = {}
        
        # Get columns to process (default: all numeric columns)
        columns_to_process = config.get('columns', None)
        if columns_to_process is None:
            columns_to_process = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns_to_process:
            if column in df.columns:
                series_result, series_info = self._process_series(
                    df[column], method, threshold, action, config
                )
                processed_df[column] = series_result
                
                column_outliers = series_info['total_outliers']
                total_outliers += column_outliers
                outliers_by_column[column] = column_outliers
        
        outlier_info = {
            'total_outliers': total_outliers,
            'outlier_percentage': (total_outliers / df.size) * 100 if df.size > 0 else 0,
            'outliers_by_column': outliers_by_column,
            'total_points': df.size
        }
        
        return processed_df, outlier_info
    
    def _process_series(self, series: pd.Series, method: str, threshold: float,
                       action: str, config: Dict[str, Any]) -> Tuple[pd.Series, Dict]:
        """Process pandas Series for outliers."""
        if len(series) == 0 or series.isnull().all():
            return series, self._empty_outlier_info()
        
        # Detect outliers
        outlier_mask = self._detect_outliers(series, method, threshold, config)
        outlier_count = outlier_mask.sum()
        
        # Handle outliers based on action
        processed_series = self._handle_outliers(series, outlier_mask, action, config)
        
        outlier_info = {
            'total_outliers': outlier_count,
            'outlier_percentage': (outlier_count / len(series)) * 100 if len(series) > 0 else 0,
            'outliers_by_column': {'series': outlier_count},
            'total_points': len(series)
        }
        
        return processed_series, outlier_info
    
    def _process_array(self, arr: np.ndarray, method: str, threshold: float,
                      action: str, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Process numpy array for outliers."""
        if arr.ndim == 1:
            # Convert to pandas Series for easier handling
            series = pd.Series(arr)
            processed_series, outlier_info = self._process_series(
                series, method, threshold, action, config
            )
            return processed_series.values, outlier_info
        elif arr.ndim == 2:
            # Process each column independently
            processed_arr = arr.copy()
            total_outliers = 0
            outliers_by_column = {}
            
            for i in range(arr.shape[1]):
                column_series = pd.Series(arr[:, i])
                processed_series, column_info = self._process_series(
                    column_series, method, threshold, action, config
                )
                processed_arr[:, i] = processed_series.values
                
                column_outliers = column_info['total_outliers']
                total_outliers += column_outliers
                outliers_by_column[f'column_{i}'] = column_outliers
            
            outlier_info = {
                'total_outliers': total_outliers,
                'outlier_percentage': (total_outliers / arr.size) * 100 if arr.size > 0 else 0,
                'outliers_by_column': outliers_by_column,
                'total_points': arr.size
            }
            
            return processed_arr, outlier_info
        else:
            logger.warning(f"Cannot process array with {arr.ndim} dimensions")
            return arr, self._empty_outlier_info()
    
    def _process_list(self, data: List[Any], method: str, threshold: float,
                     action: str, config: Dict[str, Any]) -> Tuple[List[Any], Dict]:
        """Process list for outliers."""
        if not data:
            return data, self._empty_outlier_info()
        
        # Convert to pandas Series for processing
        try:
            series = pd.Series(data)
            processed_series, outlier_info = self._process_series(
                series, method, threshold, action, config
            )
            return processed_series.tolist(), outlier_info
        except Exception as e:
            logger.warning(f"Cannot process list data for outliers: {e}")
            return data, self._empty_outlier_info()
    
    def _process_dict(self, data: Dict[str, Any], method: str, threshold: float,
                     action: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict]:
        """Process dictionary for outliers."""
        processed_dict = {}
        total_outliers = 0
        outliers_by_column = {}
        total_points = 0
        
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray, pd.Series)):
                processed_value, value_info = self._process_list(
                    value, method, threshold, action, config
                )
                processed_dict[key] = processed_value
                
                key_outliers = value_info['total_outliers']
                total_outliers += key_outliers
                outliers_by_column[key] = key_outliers
                total_points += value_info['total_points']
            else:
                processed_dict[key] = value
        
        outlier_info = {
            'total_outliers': total_outliers,
            'outlier_percentage': (total_outliers / total_points) * 100 if total_points > 0 else 0,
            'outliers_by_column': outliers_by_column,
            'total_points': total_points
        }
        
        return processed_dict, outlier_info
    
    def _detect_outliers(self, series: pd.Series, method: str, threshold: float,
                        config: Dict[str, Any]) -> pd.Series:
        """Detect outliers in a pandas Series."""
        if method == 'z_score':
            return self._z_score_detection(series, threshold)
        elif method == 'modified_z_score':
            return self._modified_z_score_detection(series, threshold)
        elif method == 'iqr':
            return self._iqr_detection(series, threshold)
        elif method == 'isolation_forest':
            return self._isolation_forest_detection(series, config)
        elif method == 'local_outlier_factor':
            return self._lof_detection(series, config)
        elif method == 'one_class_svm':
            return self._one_class_svm_detection(series, config)
        elif method == 'percentile':
            return self._percentile_detection(series, config)
        elif method == 'custom':
            return self._custom_detection(series, config)
        else:
            logger.warning(f"Unknown outlier detection method: {method}, using z_score")
            return self._z_score_detection(series, threshold)
    
    def _z_score_detection(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    def _modified_z_score_detection(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect outliers using Modified Z-score (MAD) method."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            # If MAD is 0, use standard deviation
            return self._z_score_detection(series, threshold)
        
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    def _iqr_detection(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect outliers using Interquartile Range method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _isolation_forest_detection(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            
            contamination = config.get('contamination', 0.1)
            random_state = config.get('random_state', 42)
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=random_state
            )
            outlier_labels = iso_forest.fit_predict(X)
            
            # Convert to boolean mask (outliers are labeled as -1)
            return pd.Series(outlier_labels == -1, index=series.index)
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to z_score")
            return self._z_score_detection(series, 3.0)
    
    def _lof_detection(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Detect outliers using Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            n_neighbors = config.get('n_neighbors', 20)
            contamination = config.get('contamination', 0.1)
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            lof = LocalOutlierFactor(
                n_neighbors=min(n_neighbors, len(series) - 1),
                contamination=contamination
            )
            outlier_labels = lof.fit_predict(X)
            
            # Convert to boolean mask (outliers are labeled as -1)
            return pd.Series(outlier_labels == -1, index=series.index)
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to z_score")
            return self._z_score_detection(series, 3.0)
    
    def _one_class_svm_detection(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Detect outliers using One-Class SVM."""
        try:
            from sklearn.svm import OneClassSVM
            
            nu = config.get('nu', 0.1)
            gamma = config.get('gamma', 'scale')
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            svm = OneClassSVM(nu=nu, gamma=gamma)
            outlier_labels = svm.fit_predict(X)
            
            # Convert to boolean mask (outliers are labeled as -1)
            return pd.Series(outlier_labels == -1, index=series.index)
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to z_score")
            return self._z_score_detection(series, 3.0)
    
    def _percentile_detection(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Detect outliers using percentile thresholds."""
        lower_percentile = config.get('lower_percentile', 1)
        upper_percentile = config.get('upper_percentile', 99)
        
        lower_bound = series.quantile(lower_percentile / 100)
        upper_bound = series.quantile(upper_percentile / 100)
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _custom_detection(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Detect outliers using custom function."""
        custom_function = config.get('custom_function')
        
        if custom_function and callable(custom_function):
            try:
                return custom_function(series)
            except Exception as e:
                logger.warning(f"Custom outlier detection function failed: {e}")
        
        # Fallback to z_score
        return self._z_score_detection(series, 3.0)
    
    def _handle_outliers(self, series: pd.Series, outlier_mask: pd.Series,
                        action: str, config: Dict[str, Any]) -> pd.Series:
        """Handle detected outliers based on specified action."""
        if not outlier_mask.any():
            return series
        
        if action == 'remove':
            return self._remove_outliers(series, outlier_mask)
        elif action == 'cap':
            return self._cap_outliers(series, outlier_mask, config)
        elif action == 'flag':
            return self._flag_outliers(series, outlier_mask)
        elif action == 'interpolate':
            return self._interpolate_outliers(series, outlier_mask)
        else:
            logger.warning(f"Unknown outlier action: {action}, using remove")
            return self._remove_outliers(series, outlier_mask)
    
    def _remove_outliers(self, series: pd.Series, outlier_mask: pd.Series) -> pd.Series:
        """Remove outliers from series."""
        return series[~outlier_mask]
    
    def _cap_outliers(self, series: pd.Series, outlier_mask: pd.Series,
                     config: Dict[str, Any]) -> pd.Series:
        """Cap outliers at threshold values."""
        result = series.copy()
        
        # Get capping bounds
        lower_percentile = config.get('cap_lower_percentile', 5)
        upper_percentile = config.get('cap_upper_percentile', 95)
        
        lower_bound = series.quantile(lower_percentile / 100)
        upper_bound = series.quantile(upper_percentile / 100)
        
        # Cap outliers
        result = result.clip(lower=lower_bound, upper=upper_bound)
        
        return result
    
    def _flag_outliers(self, series: pd.Series, outlier_mask: pd.Series) -> pd.Series:
        """Flag outliers but keep original values."""
        # For now, just return the original series
        # In a more advanced implementation, we could add metadata
        return series
    
    def _interpolate_outliers(self, series: pd.Series, outlier_mask: pd.Series) -> pd.Series:
        """Replace outliers with interpolated values."""
        result = series.copy()
        
        # Set outliers to NaN and interpolate
        result[outlier_mask] = np.nan
        result = result.interpolate(method='linear', limit_direction='both')
        
        return result
    
    def _empty_outlier_info(self) -> Dict:
        """Return empty outlier info structure."""
        return {
            'total_outliers': 0,
            'outlier_percentage': 0.0,
            'outliers_by_column': {},
            'total_points': 0
        }
    
    def rollback(self, original_data: Any, processed_data: Any,
                context: ProcessingContext) -> Any:
        """Rollback outlier detection operation by returning original data."""
        logger.info("Rolling back outlier detection")
        return original_data
    
    def get_cache_key(self, data: Any, context: ProcessingContext) -> str:
        """Generate cache key for outlier detection operation."""
        method = context.config.get('method', self.default_method)
        threshold = context.config.get('threshold', self.default_threshold)
        action = context.config.get('action', self.default_action)
        
        # Create a simple hash of the data
        data_hash = hash(str(data)[:100])  # Use first 100 chars for efficiency
        
        return f"outlier_{method}_{threshold}_{action}_{data_hash}" 