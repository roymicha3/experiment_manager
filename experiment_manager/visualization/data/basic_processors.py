"""
Built-in Data Processors for Visualization Pipeline

This module provides common data processing implementations that demonstrate
the DataProcessor system and provide useful functionality for visualization
data preparation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .pipeline import DataProcessor, ProcessingContext, ProcessingResult, ProcessingStatus

logger = logging.getLogger(__name__)


class FilterProcessor(DataProcessor):
    """
    Generic data filtering processor.
    
    Filters data based on configurable criteria such as column values,
    data types, or custom filter functions.
    """
    
    @property
    def plugin_name(self) -> str:
        return "filter"
    
    @property
    def plugin_description(self) -> str:
        return "Filters data based on configurable criteria"
    
    @property
    def supported_capabilities(self) -> List[str]:
        return ["column_filter", "value_filter", "custom_filter", "pandas_support"]
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize filter processor with configuration."""
        self.filter_config = config
        logger.info(f"Initialized FilterProcessor with config: {config}")
    
    def cleanup(self) -> None:
        """Cleanup filter processor."""
        pass
    
    def validate_input(self, data: Any, context: ProcessingContext) -> bool:
        """Validate that input data can be filtered."""
        if data is None:
            return False
        
        # Support for pandas DataFrames, lists, and dictionaries
        return isinstance(data, (pd.DataFrame, list, dict, np.ndarray))
    
    def process(self, data: Any, context: ProcessingContext) -> ProcessingResult:
        """Process data by applying filters."""
        context.start_time = datetime.now()
        context.status = ProcessingStatus.RUNNING
        
        result = ProcessingResult(data=data, context=context)
        
        try:
            if not self.validate_input(data, context):
                result.add_error("Invalid input data for filtering")
                context.status = ProcessingStatus.FAILED
                return result
            
            # Apply filters based on configuration
            filtered_data = self._apply_filters(data, context.config)
            
            context.end_time = datetime.now()
            context.status = ProcessingStatus.COMPLETED
            
            result.data = filtered_data
            result.metrics = {
                'original_size': self._get_data_size(data),
                'filtered_size': self._get_data_size(filtered_data),
                'filter_efficiency': self._calculate_filter_efficiency(data, filtered_data)
            }
            
            return result
            
        except Exception as e:
            context.end_time = datetime.now()
            context.status = ProcessingStatus.FAILED
            result.add_error(f"Filter processing failed: {str(e)}")
            logger.error(f"FilterProcessor error: {str(e)}")
            return result
    
    def _apply_filters(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply configured filters to data."""
        filtered_data = data
        
        if isinstance(data, pd.DataFrame):
            filtered_data = self._filter_dataframe(data, config)
        elif isinstance(data, list):
            filtered_data = self._filter_list(data, config)
        elif isinstance(data, dict):
            filtered_data = self._filter_dict(data, config)
        
        return filtered_data
    
    def _filter_dataframe(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter pandas DataFrame based on configuration."""
        filtered_df = df.copy()
        
        # Column filters
        if 'columns' in config:
            columns_to_keep = config['columns']
            if isinstance(columns_to_keep, list):
                filtered_df = filtered_df[columns_to_keep]
        
        # Value filters
        if 'filters' in config:
            for filter_spec in config['filters']:
                column = filter_spec.get('column')
                operator = filter_spec.get('operator', '==')
                value = filter_spec.get('value')
                
                if column in filtered_df.columns:
                    if operator == '==':
                        filtered_df = filtered_df[filtered_df[column] == value]
                    elif operator == '!=':
                        filtered_df = filtered_df[filtered_df[column] != value]
                    elif operator == '>':
                        filtered_df = filtered_df[filtered_df[column] > value]
                    elif operator == '<':
                        filtered_df = filtered_df[filtered_df[column] < value]
                    elif operator == '>=':
                        filtered_df = filtered_df[filtered_df[column] >= value]
                    elif operator == '<=':
                        filtered_df = filtered_df[filtered_df[column] <= value]
                    elif operator == 'in':
                        filtered_df = filtered_df[filtered_df[column].isin(value)]
        
        return filtered_df
    
    def _filter_list(self, data: List[Any], config: Dict[str, Any]) -> List[Any]:
        """Filter list based on configuration."""
        if 'predicate' in config:
            predicate = config['predicate']
            if callable(predicate):
                return [item for item in data if predicate(item)]
        
        if 'max_items' in config:
            return data[:config['max_items']]
            
        return data
    
    def _filter_dict(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter dictionary based on configuration."""
        if 'keys' in config:
            keys_to_keep = config['keys']
            return {k: v for k, v in data.items() if k in keys_to_keep}
        
        return data
    
    def _get_data_size(self, data: Any) -> int:
        """Get size of data for metrics."""
        if isinstance(data, pd.DataFrame):
            return len(data)
        elif isinstance(data, (list, dict)):
            return len(data)
        elif isinstance(data, np.ndarray):
            return data.size
        return 1
    
    def _calculate_filter_efficiency(self, original: Any, filtered: Any) -> float:
        """Calculate filter efficiency as percentage of data retained."""
        original_size = self._get_data_size(original)
        filtered_size = self._get_data_size(filtered)
        
        if original_size == 0:
            return 0.0
        
        return (filtered_size / original_size) * 100.0


class AggregationProcessor(DataProcessor):
    """
    Data aggregation processor.
    
    Aggregates data using various functions like sum, mean, count, etc.
    Supports groupby operations for pandas DataFrames.
    """
    
    @property
    def plugin_name(self) -> str:
        return "aggregation"
    
    @property
    def plugin_description(self) -> str:
        return "Aggregates data using various functions and groupby operations"
    
    @property
    def supported_capabilities(self) -> List[str]:
        return ["sum", "mean", "count", "min", "max", "std", "groupby", "pandas_support"]
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize aggregation processor."""
        self.config = config
        logger.info(f"Initialized AggregationProcessor with config: {config}")
    
    def cleanup(self) -> None:
        """Cleanup aggregation processor."""
        pass
    
    def validate_input(self, data: Any, context: ProcessingContext) -> bool:
        """Validate input data for aggregation."""
        return isinstance(data, (pd.DataFrame, np.ndarray, list))
    
    def process(self, data: Any, context: ProcessingContext) -> ProcessingResult:
        """Process data by applying aggregations."""
        context.start_time = datetime.now()
        context.status = ProcessingStatus.RUNNING
        
        result = ProcessingResult(data=data, context=context)
        
        try:
            if not self.validate_input(data, context):
                result.add_error("Invalid input data for aggregation")
                context.status = ProcessingStatus.FAILED
                return result
            
            aggregated_data = self._apply_aggregations(data, context.config)
            
            context.end_time = datetime.now()
            context.status = ProcessingStatus.COMPLETED
            
            result.data = aggregated_data
            result.metrics = {
                'original_size': self._get_data_size(data),
                'aggregated_size': self._get_data_size(aggregated_data),
                'aggregation_operations': list(context.config.get('operations', []))
            }
            
            return result
            
        except Exception as e:
            context.end_time = datetime.now()
            context.status = ProcessingStatus.FAILED
            result.add_error(f"Aggregation processing failed: {str(e)}")
            logger.error(f"AggregationProcessor error: {str(e)}")
            return result
    
    def _apply_aggregations(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply aggregation operations to data."""
        if isinstance(data, pd.DataFrame):
            return self._aggregate_dataframe(data, config)
        elif isinstance(data, np.ndarray):
            return self._aggregate_array(data, config)
        elif isinstance(data, list):
            return self._aggregate_list(data, config)
        
        return data
    
    def _aggregate_dataframe(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Aggregate pandas DataFrame."""
        result_df = df.copy()
        
        # Groupby aggregation
        if 'groupby' in config:
            groupby_cols = config['groupby']
            operations = config.get('operations', {'value': 'sum'})
            
            if isinstance(groupby_cols, str):
                groupby_cols = [groupby_cols]
            
            grouped = result_df.groupby(groupby_cols)
            result_df = grouped.agg(operations).reset_index()
        
        # Column-wise aggregations
        elif 'operations' in config:
            operations = config['operations']
            aggregated_data = {}
            
            for column, operation in operations.items():
                if column in result_df.columns:
                    if operation == 'sum':
                        aggregated_data[f"{column}_sum"] = result_df[column].sum()
                    elif operation == 'mean':
                        aggregated_data[f"{column}_mean"] = result_df[column].mean()
                    elif operation == 'count':
                        aggregated_data[f"{column}_count"] = result_df[column].count()
                    elif operation == 'min':
                        aggregated_data[f"{column}_min"] = result_df[column].min()
                    elif operation == 'max':
                        aggregated_data[f"{column}_max"] = result_df[column].max()
                    elif operation == 'std':
                        aggregated_data[f"{column}_std"] = result_df[column].std()
            
            result_df = pd.DataFrame([aggregated_data])
        
        return result_df
    
    def _aggregate_array(self, arr: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Aggregate numpy array."""
        operation = config.get('operation', 'mean')
        axis = config.get('axis', None)
        
        if operation == 'sum':
            return np.sum(arr, axis=axis)
        elif operation == 'mean':
            return np.mean(arr, axis=axis)
        elif operation == 'min':
            return np.min(arr, axis=axis)
        elif operation == 'max':
            return np.max(arr, axis=axis)
        elif operation == 'std':
            return np.std(arr, axis=axis)
        
        return arr
    
    def _aggregate_list(self, data: List[Any], config: Dict[str, Any]) -> Union[Any, List[Any]]:
        """Aggregate list data."""
        operation = config.get('operation', 'sum')
        
        if operation == 'sum':
            return sum(data)
        elif operation == 'mean':
            return sum(data) / len(data) if data else 0
        elif operation == 'count':
            return len(data)
        elif operation == 'min':
            return min(data) if data else None
        elif operation == 'max':
            return max(data) if data else None
        
        return data
    
    def _get_data_size(self, data: Any) -> int:
        """Get size of data for metrics."""
        if isinstance(data, pd.DataFrame):
            return len(data)
        elif isinstance(data, (list, np.ndarray)):
            return len(data)
        return 1


class NormalizationProcessor(DataProcessor):
    """
    Data normalization processor.
    
    Normalizes numerical data using various techniques like min-max scaling,
    z-score normalization, or custom scaling functions.
    """
    
    @property
    def plugin_name(self) -> str:
        return "normalization"
    
    @property
    def plugin_description(self) -> str:
        return "Normalizes numerical data using various scaling techniques"
    
    @property
    def supported_capabilities(self) -> List[str]:
        return ["min_max", "z_score", "robust", "unit_vector", "pandas_support"]
    
    @property
    def supports_rollback(self) -> bool:
        return True
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize normalization processor."""
        self.config = config
        self.scaling_params = {}
        logger.info(f"Initialized NormalizationProcessor with config: {config}")
    
    def cleanup(self) -> None:
        """Cleanup normalization processor."""
        pass
    
    def validate_input(self, data: Any, context: ProcessingContext) -> bool:
        """Validate input data for normalization."""
        return isinstance(data, (pd.DataFrame, np.ndarray, list))
    
    def process(self, data: Any, context: ProcessingContext) -> ProcessingResult:
        """Process data by applying normalization."""
        context.start_time = datetime.now()
        context.status = ProcessingStatus.RUNNING
        
        result = ProcessingResult(data=data, context=context)
        
        try:
            if not self.validate_input(data, context):
                result.add_error("Invalid input data for normalization")
                context.status = ProcessingStatus.FAILED
                return result
            
            normalized_data, scaling_params = self._apply_normalization(data, context.config)
            
            context.end_time = datetime.now()
            context.status = ProcessingStatus.COMPLETED
            
            result.data = normalized_data
            result.rollback_data = scaling_params
            result.metrics = {
                'normalization_method': context.config.get('method', 'min_max'),
                'scaling_parameters': scaling_params
            }
            
            return result
            
        except Exception as e:
            context.end_time = datetime.now()
            context.status = ProcessingStatus.FAILED
            result.add_error(f"Normalization processing failed: {str(e)}")
            logger.error(f"NormalizationProcessor error: {str(e)}")
            return result
    
    def _apply_normalization(self, data: Any, config: Dict[str, Any]) -> tuple:
        """Apply normalization to data."""
        method = config.get('method', 'min_max')
        
        if isinstance(data, pd.DataFrame):
            return self._normalize_dataframe(data, method, config)
        elif isinstance(data, np.ndarray):
            return self._normalize_array(data, method, config)
        elif isinstance(data, list):
            return self._normalize_list(data, method, config)
        
        return data, {}
    
    def _normalize_dataframe(self, df: pd.DataFrame, method: str, config: Dict[str, Any]) -> tuple:
        """Normalize pandas DataFrame."""
        normalized_df = df.copy()
        scaling_params = {}
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = config.get('columns', numeric_cols)
        
        for col in target_cols:
            if col in df.columns and df[col].dtype in [np.int64, np.float64]:
                normalized_values, params = self._normalize_column(df[col].values, method)
                normalized_df[col] = normalized_values
                scaling_params[col] = params
        
        return normalized_df, scaling_params
    
    def _normalize_array(self, arr: np.ndarray, method: str, config: Dict[str, Any]) -> tuple:
        """Normalize numpy array."""
        normalized_arr, params = self._normalize_column(arr.flatten(), method)
        return normalized_arr.reshape(arr.shape), params
    
    def _normalize_list(self, data: List[Any], method: str, config: Dict[str, Any]) -> tuple:
        """Normalize list data."""
        arr = np.array(data)
        normalized_arr, params = self._normalize_column(arr, method)
        return normalized_arr.tolist(), params
    
    def _normalize_column(self, values: np.ndarray, method: str) -> tuple:
        """Normalize a single column of values."""
        if method == 'min_max':
            min_val = np.min(values)
            max_val = np.max(values)
            if max_val - min_val != 0:
                normalized = (values - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(values)
            return normalized, {'min': min_val, 'max': max_val, 'method': 'min_max'}
        
        elif method == 'z_score':
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val != 0:
                normalized = (values - mean_val) / std_val
            else:
                normalized = np.zeros_like(values)
            return normalized, {'mean': mean_val, 'std': std_val, 'method': 'z_score'}
        
        elif method == 'robust':
            median_val = np.median(values)
            mad_val = np.median(np.abs(values - median_val))
            if mad_val != 0:
                normalized = (values - median_val) / mad_val
            else:
                normalized = np.zeros_like(values)
            return normalized, {'median': median_val, 'mad': mad_val, 'method': 'robust'}
        
        elif method == 'unit_vector':
            norm = np.linalg.norm(values)
            if norm != 0:
                normalized = values / norm
            else:
                normalized = np.zeros_like(values)
            return normalized, {'norm': norm, 'method': 'unit_vector'}
        
        return values, {'method': method}
    
    def rollback(self, original_data: Any, processed_data: Any, context: ProcessingContext) -> Any:
        """Rollback normalization operation."""
        # In a full implementation, we would reverse the normalization
        # using the stored scaling parameters
        logger.info(f"Rolling back normalization for processor {self.plugin_name}")
        return original_data


# Built-in processor registry
BUILTIN_PROCESSORS = {
    'filter': FilterProcessor,
    'aggregation': AggregationProcessor,
    'normalization': NormalizationProcessor,
}


def get_builtin_processors() -> Dict[str, type]:
    """Get dictionary of built-in data processors."""
    return BUILTIN_PROCESSORS.copy()


def register_builtin_processors(pipeline):
    """Register all built-in processors with a pipeline."""
    for name, processor_class in BUILTIN_PROCESSORS.items():
        processor = processor_class()
        pipeline.add_processor(name, processor)
    
    logger.info(f"Registered {len(BUILTIN_PROCESSORS)} built-in processors") 