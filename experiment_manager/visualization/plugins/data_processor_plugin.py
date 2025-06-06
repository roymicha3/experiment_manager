"""
Data processor plugin interface for data processing and transformation.

This module defines the abstract interface that all data processor plugins must implement.
Data processor plugins are responsible for transforming, filtering, aggregating,
and preprocessing data before visualization.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from experiment_manager.visualization.plugins.base import BasePlugin, PluginType


class ProcessingContext:
    """
    Context information for data processing operations.
    
    This class encapsulates information about the processing environment,
    requirements, and constraints.
    """
    
    def __init__(self,
                 operation_type: str,
                 input_schema: Optional[Dict[str, Any]] = None,
                 output_schema: Optional[Dict[str, Any]] = None,
                 constraints: Optional[Dict[str, Any]] = None,
                 memory_limit: Optional[int] = None,
                 processing_mode: str = "batch"):
        """
        Initialize processing context.
        
        Args:
            operation_type: Type of processing operation (e.g., 'filter', 'aggregate', 'transform')
            input_schema: Expected input data schema
            output_schema: Expected output data schema
            constraints: Processing constraints (time, memory, etc.)
            memory_limit: Maximum memory usage in bytes
            processing_mode: Processing mode ('batch', 'streaming', 'incremental')
        """
        self.operation_type = operation_type
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}
        self.constraints = constraints or {}
        self.memory_limit = memory_limit
        self.processing_mode = processing_mode
        
    def get_constraint(self, key: str, default: Any = None) -> Any:
        """Get processing constraint by key."""
        return self.constraints.get(key, default)


class ProcessingResult:
    """
    Container for data processing results.
    
    This class encapsulates the result of a data processing operation,
    including the processed data and processing metadata.
    """
    
    def __init__(self,
                 data: Any,
                 metadata: Optional[Dict[str, Any]] = None,
                 processing_stats: Optional[Dict[str, Any]] = None,
                 success: bool = True,
                 warnings: Optional[List[str]] = None,
                 error_message: Optional[str] = None):
        """
        Initialize processing result.
        
        Args:
            data: Processed data
            metadata: Metadata about the processed data
            processing_stats: Statistics about the processing operation
            success: Whether processing was successful
            warnings: List of warning messages
            error_message: Error message if processing failed
        """
        self.data = data
        self.metadata = metadata or {}
        self.processing_stats = processing_stats or {}
        self.success = success
        self.warnings = warnings or []
        self.error_message = error_message
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def get_stat(self, key: str, default: Any = None) -> Any:
        """Get processing statistic by key."""
        return self.processing_stats.get(key, default)
    
    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Get shape of processed data if applicable."""
        if hasattr(self.data, 'shape'):
            return self.data.shape
        elif isinstance(self.data, (list, tuple)):
            return (len(self.data),)
        return None


class DataProcessorPlugin(BasePlugin):
    """
    Abstract base class for data processor plugins.
    
    Data processor plugins handle various data transformation operations
    including filtering, aggregation, normalization, feature extraction,
    and other preprocessing tasks required for visualization.
    """
    
    @property
    def plugin_type(self) -> PluginType:
        """Data processor plugins always return DATA_PROCESSOR type."""
        return PluginType.DATA_PROCESSOR
    
    @property
    @abstractmethod
    def supported_operations(self) -> List[str]:
        """
        List of processing operations this plugin supports.
        
        Returns:
            List of operation identifiers (e.g., ['filter', 'aggregate', 'normalize'])
        """
        pass
    
    @property
    @abstractmethod
    def supported_input_types(self) -> List[str]:
        """
        List of input data types this plugin can handle.
        
        Returns:
            List of data type identifiers (e.g., ['dataframe', 'array', 'timeseries'])
        """
        pass
    
    @property
    @abstractmethod
    def supported_output_types(self) -> List[str]:
        """
        List of output data types this plugin can produce.
        
        Returns:
            List of data type identifiers
        """
        pass
    
    @property
    def supports_streaming(self) -> bool:
        """
        Whether this processor supports streaming data.
        
        Returns:
            True if processor can handle streaming data
        """
        return False
    
    @property
    def requires_full_dataset(self) -> bool:
        """
        Whether this processor requires the full dataset.
        
        Returns:
            True if processor needs to see all data at once
        """
        return True
    
    @property
    def is_stateful(self) -> bool:
        """
        Whether this processor maintains state between operations.
        
        Returns:
            True if processor is stateful
        """
        return False
    
    @abstractmethod
    def can_process(self, 
                   data: Any,
                   operation: str,
                   context: Optional[ProcessingContext] = None) -> bool:
        """
        Check if this processor can handle the given data and operation.
        
        Args:
            data: Input data to check
            operation: Processing operation to perform
            context: Optional processing context
            
        Returns:
            True if processor can handle this combination
        """
        pass
    
    @abstractmethod
    def process(self,
                data: Any,
                operation: str,
                context: Optional[ProcessingContext] = None,
                config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process data using the specified operation.
        
        Args:
            data: Input data to process
            operation: Processing operation to perform
            context: Optional processing context
            config: Optional operation-specific configuration
            
        Returns:
            ProcessingResult containing processed data
            
        Raises:
            ValueError: If data or operation is not supported
            RuntimeError: If processing fails
        """
        pass
    
    def validate_input(self, 
                      data: Any,
                      operation: str,
                      context: Optional[ProcessingContext] = None) -> bool:
        """
        Validate input data for the specified operation.
        
        Args:
            data: Input data to validate
            operation: Processing operation
            context: Optional processing context
            
        Returns:
            True if input is valid
        """
        return self.can_process(data, operation, context)
    
    def get_operation_info(self, operation: str) -> Dict[str, Any]:
        """
        Get information about a supported operation.
        
        Args:
            operation: Operation to get information about
            
        Returns:
            Dictionary with operation information
        """
        if operation not in self.supported_operations:
            raise ValueError(f"Operation '{operation}' not supported by this processor")
            
        return {
            "operation": operation,
            "description": f"Perform {operation} operation on data",
            "input_types": self.supported_input_types,
            "output_types": self.supported_output_types,
            "requires_full_dataset": self.requires_full_dataset,
            "supports_streaming": self.supports_streaming,
        }
    
    def estimate_memory_usage(self,
                            data: Any,
                            operation: str,
                            config: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Estimate memory usage for processing operation.
        
        Args:
            data: Input data
            operation: Processing operation
            config: Optional configuration
            
        Returns:
            Estimated memory usage in bytes, or None if cannot estimate
        """
        # Default implementation cannot estimate
        return None
    
    def estimate_processing_time(self,
                               data: Any,
                               operation: str,
                               config: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Estimate processing time for operation.
        
        Args:
            data: Input data
            operation: Processing operation
            config: Optional configuration
            
        Returns:
            Estimated processing time in seconds, or None if cannot estimate
        """
        # Default implementation cannot estimate
        return None
    
    def get_schema_requirements(self, operation: str) -> Dict[str, Any]:
        """
        Get data schema requirements for an operation.
        
        Args:
            operation: Processing operation
            
        Returns:
            Dictionary describing schema requirements
        """
        return {
            "operation": operation,
            "required_columns": [],
            "optional_columns": [],
            "data_types": {},
            "constraints": {},
        }
    
    def process_batch(self,
                     data_items: List[Any],
                     operation: str,
                     context: Optional[ProcessingContext] = None,
                     config: Optional[Dict[str, Any]] = None) -> List[ProcessingResult]:
        """
        Process multiple data items in batch.
        
        Args:
            data_items: List of data items to process
            operation: Processing operation
            context: Optional processing context
            config: Optional configuration
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        for item in data_items:
            try:
                result = self.process(item, operation, context, config)
                results.append(result)
            except Exception as e:
                error_result = ProcessingResult(
                    data=None,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
        return results
    
    def create_pipeline(self, operations: List[str]) -> Callable:
        """
        Create a processing pipeline from multiple operations.
        
        Args:
            operations: List of operations to chain together
            
        Returns:
            Callable that processes data through the pipeline
        """
        def pipeline(data: Any, config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
            current_data = data
            all_warnings = []
            all_stats = {}
            
            for op in operations:
                result = self.process(current_data, op, config=config)
                if not result.success:
                    return result
                    
                current_data = result.data
                all_warnings.extend(result.warnings)
                all_stats[op] = result.processing_stats
            
            return ProcessingResult(
                data=current_data,
                processing_stats=all_stats,
                warnings=all_warnings,
                success=True
            )
        
        return pipeline 