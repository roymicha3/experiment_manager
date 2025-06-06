"""
Plot plugin interface for creating different types of plots.

This module defines the abstract interface that all plot plugins must implement.
Plot plugins are responsible for generating specific types of visualizations
from experiment data.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from .base import BasePlugin, PluginType


class PlotData:
    """
    Container for plot data with metadata.
    
    This class encapsulates the data to be plotted along with metadata
    about the data structure and requirements.
    """
    
    def __init__(self, 
                 data: Union[np.ndarray, pd.DataFrame, Dict[str, Any]],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize plot data container.
        
        Args:
            data: The actual data to be plotted
            metadata: Optional metadata about the data
        """
        self.data = data
        self.metadata = metadata or {}
        
    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Get the shape of the data if applicable."""
        if hasattr(self.data, 'shape'):
            return self.data.shape
        elif isinstance(self.data, dict):
            return None
        return None
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)


class PlotResult:
    """
    Container for plot generation results.
    
    This class encapsulates the result of a plot generation operation,
    including the plot object and any associated metadata.
    """
    
    def __init__(self,
                 plot_object: Any,
                 metadata: Optional[Dict[str, Any]] = None,
                 success: bool = True,
                 error_message: Optional[str] = None):
        """
        Initialize plot result.
        
        Args:
            plot_object: The generated plot object (matplotlib figure, plotly figure, etc.)
            metadata: Optional metadata about the plot
            success: Whether the plot generation was successful
            error_message: Error message if generation failed
        """
        self.plot_object = plot_object
        self.metadata = metadata or {}
        self.success = success
        self.error_message = error_message
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)


class PlotPlugin(BasePlugin):
    """
    Abstract base class for plot plugins.
    
    Plot plugins generate specific types of visualizations from experiment data.
    Each plugin should implement a specific plot type (e.g., line plots, scatter plots,
    heatmaps, etc.) and handle the conversion from raw data to plot objects.
    """
    
    @property
    def plugin_type(self) -> PluginType:
        """Plot plugins always return PLOT type."""
        return PluginType.PLOT
    
    @property
    @abstractmethod
    def supported_data_types(self) -> List[str]:
        """
        List of data types this plot plugin can handle.
        
        Returns:
            List of supported data type identifiers (e.g., ['timeseries', 'scalar', 'image'])
        """
        pass
    
    @property
    @abstractmethod
    def plot_dimensions(self) -> str:
        """
        Dimensionality of plots this plugin creates.
        
        Returns:
            One of: '1D', '2D', '3D', 'ND'
        """
        pass
    
    @property
    def required_data_columns(self) -> List[str]:
        """
        List of required column names for structured data.
        
        Returns:
            List of column names that must be present in DataFrame inputs
        """
        return []
    
    @property
    def optional_data_columns(self) -> List[str]:
        """
        List of optional column names for structured data.
        
        Returns:
            List of column names that enhance the plot if present
        """
        return []
    
    @abstractmethod
    def can_handle_data(self, data: PlotData) -> bool:
        """
        Check if this plugin can handle the given data.
        
        Args:
            data: Plot data to check
            
        Returns:
            True if plugin can create a plot from this data
        """
        pass
    
    @abstractmethod
    def generate_plot(self, 
                     data: PlotData,
                     config: Optional[Dict[str, Any]] = None) -> PlotResult:
        """
        Generate a plot from the provided data.
        
        Args:
            data: Data to create the plot from
            config: Optional plot-specific configuration
            
        Returns:
            PlotResult containing the generated plot
            
        Raises:
            ValueError: If data is incompatible with this plot type
            RuntimeError: If plot generation fails
        """
        pass
    
    def validate_data(self, data: PlotData) -> bool:
        """
        Validate that the data meets requirements for this plot type.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid for this plot type
        """
        # Default implementation checks if plugin can handle the data
        return self.can_handle_data(data)
    
    def get_data_requirements(self) -> Dict[str, Any]:
        """
        Get detailed data requirements for this plot type.
        
        Returns:
            Dictionary describing data requirements
        """
        return {
            "supported_data_types": self.supported_data_types,
            "plot_dimensions": self.plot_dimensions,
            "required_columns": self.required_data_columns,
            "optional_columns": self.optional_data_columns,
        }
    
    def preprocess_data(self, data: PlotData) -> PlotData:
        """
        Preprocess data before plotting.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data ready for plotting
        """
        # Default implementation returns data unchanged
        return data
    
    def get_default_style(self) -> Dict[str, Any]:
        """
        Get default styling configuration for this plot type.
        
        Returns:
            Dictionary with default style settings
        """
        return {
            "width": 800,
            "height": 600,
            "title": "",
            "theme": "default"
        } 