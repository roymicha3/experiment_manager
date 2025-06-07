"""
Metric Normalizer Processor

This module provides metric normalization and scaling functionality for
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


class MetricNormalizer(DataProcessor):
    """
    Metric normalization and scaling processor for experimental data.
    
    Supports multiple normalization strategies:
    - Min-Max scaling (0-1 normalization)
    - Z-score normalization (standardization)
    - Robust scaling (median and IQR based)
    - Unit vector scaling (L2 normalization)
    - Max absolute scaling
    - Quantile uniform transformation
    - Power transformation (Box-Cox, Yeo-Johnson)
    - Custom range scaling
    - Per-group normalization
    
    Particularly useful for:
    - Normalizing training metrics for comparison
    - Scaling features for visualization
    - Preparing data for machine learning models
    - Standardizing experimental measurements
    """
    
    @property
    def plugin_name(self) -> str:
        return "metric_normalizer"
    
    @property
    def plugin_description(self) -> str:
        return "Normalizes and scales metrics using various statistical methods"
    
    @property
    def supported_capabilities(self) -> List[str]:
        return [
            "min_max_scaling",
            "z_score_normalization",
            "robust_scaling",
            "unit_vector_scaling",
            "max_absolute_scaling",
            "quantile_uniform_transformation",
            "power_transformation",
            "custom_range_scaling",
            "per_group_normalization",
            "pandas_support",
            "numpy_support",
            "multi_column_support",
            "inverse_transform"
        ]
    
    @property
    def supports_rollback(self) -> bool:
        return True
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize normalizer with configuration."""
        self.config = config
        self.default_method = config.get('method', 'min_max')
        self.default_feature_range = config.get('feature_range', (0, 1))
        self.scaling_params = {}  # Store scaling parameters for inverse transform
        
        logger.info(f"Initialized MetricNormalizer with method: {self.default_method}")
    
    def cleanup(self) -> None:
        """Cleanup normalizer processor."""
        self.scaling_params.clear()
    
    def validate_input(self, data: Any, context: ProcessingContext) -> bool:
        """Validate that input data can be normalized."""
        if data is None:
            return False
        
        # Support pandas DataFrames, Series, numpy arrays, and lists
        return isinstance(data, (pd.DataFrame, pd.Series, np.ndarray, list, dict))
    
    def process(self, data: Any, context: ProcessingContext) -> ProcessingResult:
        """Process data by applying normalization."""
        context.start_time = datetime.now()
        context.status = ProcessingStatus.RUNNING
        
        result = ProcessingResult(data=data, context=context)
        
        try:
            if not self.validate_input(data, context):
                result.add_error("Invalid input data for metric normalization")
                context.status = ProcessingStatus.FAILED
                return result
            
            # Get normalization configuration
            method = context.config.get('method', self.default_method)
            feature_range = context.config.get('feature_range', self.default_feature_range)
            
            # Apply normalization based on data type
            normalized_data, original_data, norm_info = self._apply_normalization(
                data, method, feature_range, context.config
            )
            
            context.end_time = datetime.now()
            context.status = ProcessingStatus.COMPLETED
            
            result.data = normalized_data
            result.original_data = original_data  # Store for rollback
            result.metadata = {'scaling_params': self.scaling_params}  # Store for inverse transform
            result.metrics = {
                'normalization_method': method,
                'feature_range': feature_range,
                'columns_normalized': norm_info['columns_normalized'],
                'data_points_processed': norm_info['total_points'],
                'scaling_statistics': norm_info['scaling_stats']
            }
            
            return result
            
        except Exception as e:
            context.end_time = datetime.now()
            context.status = ProcessingStatus.FAILED
            result.add_error(f"Metric normalization failed: {str(e)}")
            logger.error(f"MetricNormalizer error: {str(e)}")
            return result
    
    def _apply_normalization(self, data: Any, method: str, feature_range: Tuple[float, float],
                           config: Dict[str, Any]) -> Tuple[Any, Any, Dict]:
        """Apply normalization method to data."""
        original_data = data
        
        if isinstance(data, pd.DataFrame):
            normalized_data, norm_info = self._normalize_dataframe(
                data, method, feature_range, config
            )
        elif isinstance(data, pd.Series):
            normalized_data, norm_info = self._normalize_series(
                data, method, feature_range, config
            )
        elif isinstance(data, np.ndarray):
            normalized_data, norm_info = self._normalize_array(
                data, method, feature_range, config
            )
        elif isinstance(data, list):
            normalized_data, norm_info = self._normalize_list(
                data, method, feature_range, config
            )
        elif isinstance(data, dict):
            normalized_data, norm_info = self._normalize_dict(
                data, method, feature_range, config
            )
        else:
            normalized_data = data
            norm_info = self._empty_norm_info()
        
        return normalized_data, original_data, norm_info
    
    def _normalize_dataframe(self, df: pd.DataFrame, method: str, feature_range: Tuple[float, float],
                           config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
        """Normalize pandas DataFrame columns."""
        normalized_df = df.copy()
        columns_normalized = []
        scaling_stats = {}
        
        # Get columns to normalize (default: all numeric columns)
        columns_to_normalize = config.get('columns', None)
        if columns_to_normalize is None:
            columns_to_normalize = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns_to_normalize:
            if column in df.columns:
                normalized_series, series_stats = self._normalize_series(
                    df[column], method, feature_range, config, column_name=column
                )
                normalized_df[column] = normalized_series
                columns_normalized.append(column)
                scaling_stats[column] = series_stats['scaling_stats']
        
        norm_info = {
            'columns_normalized': columns_normalized,
            'total_points': df.size,
            'scaling_stats': scaling_stats
        }
        
        return normalized_df, norm_info
    
    def _normalize_series(self, series: pd.Series, method: str, feature_range: Tuple[float, float],
                         config: Dict[str, Any], column_name: str = 'series') -> Tuple[pd.Series, Dict]:
        """Normalize pandas Series."""
        if len(series) == 0 or series.isnull().all():
            return series, self._empty_norm_info()
        
        # Handle missing values
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return series, self._empty_norm_info()
        
        # Apply normalization method
        normalized_values, scaling_params = self._normalize_values(
            clean_series.values, method, feature_range, config
        )
        
        # Store scaling parameters for inverse transform
        self.scaling_params[column_name] = scaling_params
        
        # Create normalized series with original index
        normalized_series = series.copy()
        normalized_series[clean_series.index] = normalized_values
        
        norm_info = {
            'columns_normalized': [column_name],
            'total_points': len(series),
            'scaling_stats': scaling_params
        }
        
        return normalized_series, norm_info
    
    def _normalize_array(self, arr: np.ndarray, method: str, feature_range: Tuple[float, float],
                        config: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Normalize numpy array."""
        if arr.ndim == 1:
            normalized_values, scaling_params = self._normalize_values(
                arr, method, feature_range, config
            )
            self.scaling_params['array'] = scaling_params
            
            norm_info = {
                'columns_normalized': ['array'],
                'total_points': arr.size,
                'scaling_stats': {'array': scaling_params}
            }
            
            return normalized_values, norm_info
        elif arr.ndim == 2:
            # Normalize each column independently
            normalized_arr = arr.copy()
            columns_normalized = []
            scaling_stats = {}
            
            for i in range(arr.shape[1]):
                column_values = arr[:, i]
                normalized_values, scaling_params = self._normalize_values(
                    column_values, method, feature_range, config
                )
                normalized_arr[:, i] = normalized_values
                
                column_name = f'column_{i}'
                self.scaling_params[column_name] = scaling_params
                columns_normalized.append(column_name)
                scaling_stats[column_name] = scaling_params
            
            norm_info = {
                'columns_normalized': columns_normalized,
                'total_points': arr.size,
                'scaling_stats': scaling_stats
            }
            
            return normalized_arr, norm_info
        else:
            logger.warning(f"Cannot normalize array with {arr.ndim} dimensions")
            return arr, self._empty_norm_info()
    
    def _normalize_list(self, data: List[Any], method: str, feature_range: Tuple[float, float],
                       config: Dict[str, Any]) -> Tuple[List[Any], Dict]:
        """Normalize list of numeric values."""
        if not data:
            return data, self._empty_norm_info()
        
        # Convert to numpy array for processing
        try:
            arr = np.array(data, dtype=float)
            normalized_arr, norm_info = self._normalize_array(
                arr, method, feature_range, config
            )
            return normalized_arr.tolist(), norm_info
        except (ValueError, TypeError):
            logger.warning("Cannot convert list to numeric array for normalization")
            return data, self._empty_norm_info()
    
    def _normalize_dict(self, data: Dict[str, Any], method: str, feature_range: Tuple[float, float],
                       config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict]:
        """Normalize dictionary values."""
        normalized_dict = {}
        columns_normalized = []
        scaling_stats = {}
        total_points = 0
        
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray, pd.Series)):
                normalized_value, value_info = self._normalize_list(
                    value, method, feature_range, config
                )
                normalized_dict[key] = normalized_value
                columns_normalized.append(key)
                scaling_stats[key] = value_info['scaling_stats'].get(key, {})
                total_points += value_info['total_points']
            else:
                normalized_dict[key] = value
        
        norm_info = {
            'columns_normalized': columns_normalized,
            'total_points': total_points,
            'scaling_stats': scaling_stats
        }
        
        return normalized_dict, norm_info
    
    def _normalize_values(self, values: np.ndarray, method: str, feature_range: Tuple[float, float],
                         config: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Apply normalization method to 1D array of values."""
        if len(values) == 0:
            return values, {}
        
        # Remove NaN values for calculation
        clean_values = values[~np.isnan(values)]
        if len(clean_values) == 0:
            return values, {}
        
        if method == 'min_max':
            normalized, params = self._min_max_scaling(clean_values, feature_range)
        elif method == 'z_score':
            normalized, params = self._z_score_normalization(clean_values)
        elif method == 'robust':
            normalized, params = self._robust_scaling(clean_values)
        elif method == 'unit_vector':
            normalized, params = self._unit_vector_scaling(clean_values)
        elif method == 'max_absolute':
            normalized, params = self._max_absolute_scaling(clean_values)
        elif method == 'quantile_uniform':
            normalized, params = self._quantile_uniform_transformation(clean_values, config)
        elif method == 'power':
            normalized, params = self._power_transformation(clean_values, config)
        elif method == 'custom_range':
            normalized, params = self._custom_range_scaling(clean_values, config)
        else:
            logger.warning(f"Unknown normalization method: {method}, using min_max")
            normalized, params = self._min_max_scaling(clean_values, feature_range)
        
        # Handle NaN values in original array
        result = values.copy()
        result[~np.isnan(values)] = normalized
        
        return result, params
    
    def _min_max_scaling(self, values: np.ndarray, feature_range: Tuple[float, float]) -> Tuple[np.ndarray, Dict]:
        """Apply min-max scaling to values."""
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val == min_val:
            # Handle constant values
            normalized = np.full_like(values, feature_range[0])
        else:
            # Scale to [0, 1] then to feature_range
            normalized = (values - min_val) / (max_val - min_val)
            normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        params = {
            'method': 'min_max',
            'min_val': min_val,
            'max_val': max_val,
            'feature_range': feature_range
        }
        
        return normalized, params
    
    def _z_score_normalization(self, values: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply z-score normalization to values."""
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            # Handle constant values
            normalized = np.zeros_like(values)
        else:
            normalized = (values - mean_val) / std_val
        
        params = {
            'method': 'z_score',
            'mean': mean_val,
            'std': std_val
        }
        
        return normalized, params
    
    def _robust_scaling(self, values: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply robust scaling using median and IQR."""
        median_val = np.median(values)
        q75 = np.percentile(values, 75)
        q25 = np.percentile(values, 25)
        iqr = q75 - q25
        
        if iqr == 0:
            # Handle constant values
            normalized = np.zeros_like(values)
        else:
            normalized = (values - median_val) / iqr
        
        params = {
            'method': 'robust',
            'median': median_val,
            'iqr': iqr,
            'q25': q25,
            'q75': q75
        }
        
        return normalized, params
    
    def _unit_vector_scaling(self, values: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply unit vector scaling (L2 normalization)."""
        norm = np.linalg.norm(values)
        
        if norm == 0:
            # Handle zero vector
            normalized = np.zeros_like(values)
        else:
            normalized = values / norm
        
        params = {
            'method': 'unit_vector',
            'norm': norm
        }
        
        return normalized, params
    
    def _max_absolute_scaling(self, values: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply max absolute scaling."""
        max_abs = np.max(np.abs(values))
        
        if max_abs == 0:
            # Handle zero values
            normalized = np.zeros_like(values)
        else:
            normalized = values / max_abs
        
        params = {
            'method': 'max_absolute',
            'max_abs': max_abs
        }
        
        return normalized, params
    
    def _quantile_uniform_transformation(self, values: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Apply quantile uniform transformation."""
        try:
            from sklearn.preprocessing import QuantileTransformer
            
            n_quantiles = config.get('n_quantiles', min(1000, len(values)))
            output_distribution = config.get('output_distribution', 'uniform')
            
            transformer = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution=output_distribution
            )
            
            normalized = transformer.fit_transform(values.reshape(-1, 1)).flatten()
            
            params = {
                'method': 'quantile_uniform',
                'n_quantiles': n_quantiles,
                'output_distribution': output_distribution,
                'transformer': transformer  # Store for inverse transform
            }
            
            return normalized, params
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to min_max")
            return self._min_max_scaling(values, (0, 1))
    
    def _power_transformation(self, values: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Apply power transformation (Box-Cox or Yeo-Johnson)."""
        try:
            from sklearn.preprocessing import PowerTransformer
            
            method = config.get('power_method', 'yeo-johnson')
            standardize = config.get('standardize', True)
            
            # Ensure positive values for Box-Cox
            if method == 'box-cox' and np.any(values <= 0):
                logger.warning("Box-Cox requires positive values, using Yeo-Johnson")
                method = 'yeo-johnson'
            
            transformer = PowerTransformer(method=method, standardize=standardize)
            normalized = transformer.fit_transform(values.reshape(-1, 1)).flatten()
            
            params = {
                'method': 'power',
                'power_method': method,
                'standardize': standardize,
                'transformer': transformer  # Store for inverse transform
            }
            
            return normalized, params
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to z_score")
            return self._z_score_normalization(values)
    
    def _custom_range_scaling(self, values: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Apply custom range scaling."""
        target_min = config.get('target_min', 0)
        target_max = config.get('target_max', 1)
        
        return self._min_max_scaling(values, (target_min, target_max))
    
    def _empty_norm_info(self) -> Dict:
        """Return empty normalization info structure."""
        return {
            'columns_normalized': [],
            'total_points': 0,
            'scaling_stats': {}
        }
    
    def inverse_transform(self, normalized_data: Any, column_name: str = None) -> Any:
        """Apply inverse transformation to get back original scale."""
        if not self.scaling_params:
            logger.warning("No scaling parameters available for inverse transform")
            return normalized_data
        
        if isinstance(normalized_data, pd.DataFrame):
            return self._inverse_transform_dataframe(normalized_data)
        elif isinstance(normalized_data, pd.Series):
            key = column_name or 'series'
            return self._inverse_transform_series(normalized_data, key)
        elif isinstance(normalized_data, np.ndarray):
            key = column_name or 'array'
            return self._inverse_transform_array(normalized_data, key)
        elif isinstance(normalized_data, list):
            key = column_name or 'array'
            arr = np.array(normalized_data)
            inverse_arr = self._inverse_transform_array(arr, key)
            return inverse_arr.tolist()
        else:
            return normalized_data
    
    def _inverse_transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transform to DataFrame."""
        result = df.copy()
        
        for column in df.columns:
            if column in self.scaling_params:
                result[column] = self._inverse_transform_series(df[column], column)
        
        return result
    
    def _inverse_transform_series(self, series: pd.Series, key: str) -> pd.Series:
        """Apply inverse transform to Series."""
        if key not in self.scaling_params:
            return series
        
        params = self.scaling_params[key]
        values = series.values
        
        inverse_values = self._inverse_transform_values(values, params)
        
        return pd.Series(inverse_values, index=series.index, name=series.name)
    
    def _inverse_transform_array(self, arr: np.ndarray, key: str) -> np.ndarray:
        """Apply inverse transform to array."""
        if key not in self.scaling_params:
            return arr
        
        params = self.scaling_params[key]
        return self._inverse_transform_values(arr, params)
    
    def _inverse_transform_values(self, values: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply inverse transformation to values."""
        method = params.get('method')
        
        if method == 'min_max':
            min_val = params['min_val']
            max_val = params['max_val']
            feature_range = params['feature_range']
            
            # Reverse min-max scaling
            normalized = (values - feature_range[0]) / (feature_range[1] - feature_range[0])
            return normalized * (max_val - min_val) + min_val
            
        elif method == 'z_score':
            mean_val = params['mean']
            std_val = params['std']
            return values * std_val + mean_val
            
        elif method == 'robust':
            median_val = params['median']
            iqr = params['iqr']
            return values * iqr + median_val
            
        elif method == 'unit_vector':
            norm = params['norm']
            return values * norm
            
        elif method == 'max_absolute':
            max_abs = params['max_abs']
            return values * max_abs
            
        elif method in ['quantile_uniform', 'power']:
            transformer = params.get('transformer')
            if transformer:
                return transformer.inverse_transform(values.reshape(-1, 1)).flatten()
            else:
                logger.warning(f"No transformer available for inverse {method}")
                return values
        else:
            logger.warning(f"Unknown method for inverse transform: {method}")
            return values
    
    def rollback(self, original_data: Any, processed_data: Any,
                context: ProcessingContext) -> Any:
        """Rollback normalization operation by returning original data."""
        logger.info("Rolling back metric normalization")
        return original_data
    
    def get_cache_key(self, data: Any, context: ProcessingContext) -> str:
        """Generate cache key for normalization operation."""
        method = context.config.get('method', self.default_method)
        feature_range = context.config.get('feature_range', self.default_feature_range)
        
        # Create a simple hash of the data
        data_hash = hash(str(data)[:100])  # Use first 100 chars for efficiency
        
        return f"normalize_{method}_{feature_range}_{data_hash}" 