"""
Base Data Processor for Analytics

Provides the base class and interface for all analytics processors, integrating
with the YAMLSerializable framework for configuration-driven instantiation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from omegaconf import DictConfig

from experiment_manager.common.serializable import YAMLSerializable


class DataProcessor(YAMLSerializable, ABC):
    """
    Base class for all analytics data processors.
    
    This class provides the common interface and functionality for processing
    experiment data. All analytics processors should inherit from this class
    and implement the required abstract methods.
    
    The class integrates with the YAMLSerializable framework to enable
    configuration-driven instantiation through factory patterns.
    """
    
    def __init__(self, name: str = None, config: DictConfig = None):
        """
        Initialize the data processor.
        
        Args:
            name: Name identifier for this processor
            config: Configuration for the processor
        """
        super().__init__(config)
        self.name = name or self.__class__.__name__
        self.required_columns: List[str] = []
        self.optional_columns: List[str] = []
        self._config = config or DictConfig({})
        
    @classmethod
    def from_config(cls, config: DictConfig, **kwargs):
        """
        Create a processor instance from configuration.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            DataProcessor: Configured processor instance
        """
        # Extract processor-specific configuration
        processor_config = config.get(cls.__name__.lower().replace('processor', ''), {})
        if isinstance(processor_config, dict):
            processor_config = DictConfig(processor_config)
        
        # Create instance with name if provided
        name = kwargs.get('name', cls.__name__)
        return cls(name=name, config=processor_config)
    
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process the input data.
        
        This method validates the input data and calls the implementation-specific
        processing method.
        
        Args:
            data: Input DataFrame to process
            **kwargs: Additional processing parameters
            
        Returns:
            pd.DataFrame: Processed data
            
        Raises:
            ValueError: If required columns are missing
        """
        # Validate input data
        self._validate_input(data)
        
        # Merge configuration with runtime parameters
        processing_params = self._merge_config_params(**kwargs)
        
        # Call implementation-specific processing
        return self._process_implementation(data, **processing_params)
    
    @abstractmethod
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Implementation-specific processing logic.
        
        Subclasses must implement this method to define their processing behavior.
        
        Args:
            data: Input DataFrame to process
            **kwargs: Processing parameters
            
        Returns:
            pd.DataFrame: Processed data
        """
        pass
    
    def _validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate input data requirements.
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"{self.name} requires columns {self.required_columns}, "
                f"but missing: {missing_columns}"
            )
    
    def _merge_config_params(self, **kwargs) -> Dict[str, Any]:
        """
        Merge configuration parameters with runtime parameters.
        
        Runtime parameters take precedence over configuration parameters.
        
        Args:
            **kwargs: Runtime parameters
            
        Returns:
            Dict[str, Any]: Merged parameters
        """
        # Start with configuration parameters
        params = dict(self._config) if self._config else {}
        
        # Override with runtime parameters
        params.update(kwargs)
        
        return params
    
    def get_required_columns(self) -> List[str]:
        """
        Get the list of required columns for this processor.
        
        Returns:
            List[str]: Required column names
        """
        return self.required_columns.copy()
    
    def get_optional_columns(self) -> List[str]:
        """
        Get the list of optional columns for this processor.
        
        Returns:
            List[str]: Optional column names
        """
        return self.optional_columns.copy()
    
    def get_name(self) -> str:
        """
        Get the processor name.
        
        Returns:
            str: Processor name
        """
        return self.name
    
    def get_config(self) -> DictConfig:
        """
        Get the processor configuration.
        
        Returns:
            DictConfig: Processor configuration
        """
        return self._config
    
    def __str__(self) -> str:
        """String representation of the processor."""
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the processor."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"required_columns={self.required_columns}, "
            f"config_keys={list(self._config.keys()) if self._config else []}"
            f")"
        ) 