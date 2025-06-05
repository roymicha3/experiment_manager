"""
Base classes for data processing in the analytics module.

This module provides the foundational classes for implementing data processors
that can transform and analyze experiment data.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime
import pandas as pd
import yaml
from omegaconf import DictConfig

from experiment_manager.common.serializable import YAMLSerializable


class ProcessedData(YAMLSerializable):
    """
    Container for processed analytics data with metadata tracking.
    
    This class wraps processed data with information about the processing
    steps applied and maintains metadata about the data's lifecycle.
    """
    
    def __init__(self, 
                 data: Union[pd.DataFrame, Dict[str, Any]], 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize processed data container.
        
        Args:
            data: The processed data as DataFrame or dict
            metadata: Additional metadata about the data
        """
        # Handle dict data
        if isinstance(data, dict):
            self.data = pd.DataFrame(data)
        else:
            self.data = data if data is not None else pd.DataFrame()
            
        self.metadata = metadata or {}
        self.processing_steps = []
        
        # Add default metadata
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.now().isoformat()
    
    def add_processing_step(self, processor_name: str, parameters: Dict[str, Any], description: str = None):
        """Add information about a processing step."""
        step = {
            'processor': processor_name,
            'processor_name': processor_name,  # For backward compatibility with tests
            'parameters': parameters,
            'timestamp': datetime.now().isoformat()
        }
        if description:
            step['description'] = description
        self.processing_steps.append(step)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the processed data."""
        summary = {
            'row_count': len(self.data),
            'column_count': len(self.data.columns),
            'columns': list(self.data.columns),
            'processing_steps': len(self.processing_steps),
            'metadata': self.metadata
        }
        
        # Add memory usage if data exists
        if not self.data.empty:
            summary['memory_usage'] = self.data.memory_usage(deep=True).sum()
            
        return summary
    
    def to_yaml(self) -> str:
        """Serialize processed data to YAML string."""
        data_dict = {
            'data_shape': self.data.shape,
            'columns': list(self.data.columns),
            'metadata': self.metadata,
            'processing_steps': self.processing_steps,
            'data_summary': self.get_summary()
        }
        return yaml.dump(data_dict, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'ProcessedData':
        """Create ProcessedData from YAML string."""
        data_dict = yaml.safe_load(yaml_str)
        # Create empty DataFrame for structure
        empty_data = pd.DataFrame(columns=data_dict.get('columns', []))
        processed = cls(empty_data, data_dict.get('metadata', {}))
        processed.processing_steps = data_dict.get('processing_steps', [])
        return processed


class DataProcessor(YAMLSerializable, ABC):
    """
    Abstract base class for data processors.
    
    Data processors transform input DataFrames according to specific algorithms
    and return ProcessedData containers with tracking information.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the data processor.
        
        Args:
            name: Name identifier for this processor
        """
        self.name = name or self.__class__.__name__
        self.required_columns = []
    
    @abstractmethod
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Implement the actual data processing logic.
        
        Args:
            data: Input DataFrame to process
            **kwargs: Additional processing parameters
            
        Returns:
            pd.DataFrame: Processed data
        """
        pass
    
    def process(self, data: pd.DataFrame, **kwargs) -> ProcessedData:
        """
        Process the input data and return a ProcessedData container.
        
        Args:
            data: Input DataFrame to process
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessedData: Container with processed data and metadata
            
        Raises:
            ValueError: If input data validation fails
        """
        # Validate input
        if not self.validate_input(data):
            missing_cols = set(self.required_columns) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Process the data
        processed_df = self._process_implementation(data, **kwargs)
        
        # Create result container
        result = ProcessedData(processed_df, metadata={'processor': self.name})
        result.add_processing_step(
            processor_name=self.name,
            parameters=kwargs,
            description=f"Processed by {self.name}"
        )
        
        return result
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data contains all required columns.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        return all(col in data.columns for col in self.required_columns)
    
    def get_required_columns(self) -> List[str]:
        """
        Get the list of required columns for this processor.
        
        Returns:
            List[str]: Required column names
        """
        return self.required_columns.copy()
    
    def to_yaml(self) -> str:
        """Serialize processor to YAML string."""
        processor_dict = {
            'name': self.name,
            'class': self.__class__.__name__,
            'required_columns': self.required_columns
        }
        return yaml.dump(processor_dict, default_flow_style=False)


class ProcessorManager(YAMLSerializable):
    """
    Manager for registering and executing data processors.
    
    Provides functionality to register processors by name and execute them
    individually or in chains.
    """
    
    def __init__(self):
        """Initialize the processor manager."""
        self.processors: Dict[str, DataProcessor] = {}
    
    def register_processor(self, name: str, processor: DataProcessor):
        """
        Register a processor with a given name.
        
        Args:
            name: Name to register the processor under
            processor: DataProcessor instance to register
            
        Raises:
            ValueError: If a processor with the same name is already registered
        """
        if name in self.processors:
            raise ValueError(f"Processor '{name}' is already registered")
        
        self.processors[name] = processor
    
    def get_processor(self, name: str) -> DataProcessor:
        """
        Get a registered processor by name.
        
        Args:
            name: Name of the processor to retrieve
            
        Returns:
            DataProcessor: The registered processor
            
        Raises:
            KeyError: If no processor with the given name is found
        """
        if name not in self.processors:
            raise KeyError(f"Processor '{name}' not found")
        
        return self.processors[name]
    
    def remove_processor(self, name: str):
        """
        Remove a registered processor.
        
        Args:
            name: Name of the processor to remove
            
        Raises:
            KeyError: If no processor with the given name is found
        """
        if name not in self.processors:
            raise KeyError(f"Processor '{name}' not found")
        
        del self.processors[name]
    
    def list_processors(self) -> List[str]:
        """
        Get a list of all registered processor names.
        
        Returns:
            List[str]: Names of registered processors
        """
        return list(self.processors.keys())
    
    def execute(self, processor_name: str, data: pd.DataFrame, **kwargs) -> ProcessedData:
        """
        Execute a single processor on the given data.
        
        Args:
            processor_name: Name of the processor to execute
            data: Input data to process
            **kwargs: Additional parameters for the processor
            
        Returns:
            ProcessedData: Result of the processing
            
        Raises:
            KeyError: If the processor is not found
        """
        processor = self.get_processor(processor_name)
        return processor.process(data, **kwargs)
    
    def execute_chain(self, processor_names: List[str], data: pd.DataFrame, **kwargs) -> ProcessedData:
        """
        Execute a chain of processors in sequence.
        
        Args:
            processor_names: List of processor names to execute in order
            data: Input data to process
            **kwargs: Additional parameters for processors
            
        Returns:
            ProcessedData: Result after all processors have been applied
            
        Raises:
            KeyError: If any processor in the chain is not found
        """
        current_data = data
        final_result = None
        
        for processor_name in processor_names:
            processor = self.get_processor(processor_name)
            result = processor.process(current_data, **kwargs)
            
            if final_result is None:
                final_result = result
            else:
                # Merge processing steps
                final_result.data = result.data
                final_result.processing_steps.extend(result.processing_steps)
            
            current_data = result.data
        
        return final_result
    
    def execute_parallel(self, processor_names: List[str], data: pd.DataFrame, **kwargs) -> Dict[str, ProcessedData]:
        """
        Execute multiple processors in parallel on the same data.
        
        Args:
            processor_names: List of processor names to execute
            data: Input data to process
            **kwargs: Additional parameters for processors
            
        Returns:
            Dict[str, ProcessedData]: Results keyed by processor name
            
        Raises:
            KeyError: If any processor is not found
        """
        results = {}
        
        for processor_name in processor_names:
            processor = self.get_processor(processor_name)
            results[processor_name] = processor.process(data, **kwargs)
        
        return results
    
    def clear_processors(self):
        """Remove all registered processors."""
        self.processors.clear()
    
    def clear(self):
        """Remove all registered processors (alias for clear_processors)."""
        self.clear_processors()
    
    def to_yaml(self) -> str:
        """Serialize manager to YAML string."""
        processor_details = {}
        for name, processor in self.processors.items():
            processor_details[f'{name}:'] = {
                'name': processor.name,
                'class': processor.__class__.__name__,
                'required_columns': processor.required_columns
            }
        
        manager_dict = {
            'processor_count': len(self.processors),
            'processors:': processor_details
        }
        
        return yaml.dump(manager_dict, default_flow_style=False) 