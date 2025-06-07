"""
Plot Specification Classes

This module defines the core PlotSpec class for declarative plot configuration,
providing a structured way to define plots with validation and type safety.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Type, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class PlotType(Enum):
    """Enumeration of supported plot types."""
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"
    AREA = "area"
    PIE = "pie"
    RADAR = "radar"
    SURFACE = "surface"
    CONTOUR = "contour"
    TRAINING_CURVES = "training_curves"
    EXPERIMENT_COMPARISON = "experiment_comparison"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    CUSTOM = "custom"


class PlotTheme(BaseModel):
    """Plot theme configuration."""
    name: str = "default"
    colors: List[str] = Field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    background_color: str = "#ffffff"
    grid_color: str = "#e0e0e0"
    text_color: str = "#333333"
    font_family: str = "sans-serif"
    font_size: int = 12
    figure_size: Tuple[float, float] = (10.0, 6.0)
    dpi: int = 100


class PlotLayout(BaseModel):
    """Plot layout configuration."""
    title: Optional[str] = None
    subtitle: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    zlabel: Optional[str] = None
    legend: bool = True
    legend_position: str = "best"
    grid: bool = True
    show_axes: bool = True
    tight_layout: bool = True
    margin: Dict[str, float] = Field(default_factory=lambda: {
        "left": 0.1, "right": 0.9, "top": 0.9, "bottom": 0.1
    })


class PlotStyling(BaseModel):
    """Plot styling configuration."""
    line_width: float = 2.0
    line_style: str = "-"
    marker: Optional[str] = None
    marker_size: float = 6.0
    alpha: float = 1.0
    color: Optional[str] = None
    colors: Optional[List[str]] = None
    fill_alpha: float = 0.7
    edge_color: Optional[str] = None
    edge_width: float = 1.0
    
    @field_validator('line_style')
    @classmethod
    def validate_line_style(cls, v):
        valid_styles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
        if v not in valid_styles:
            raise ValueError(f"Invalid line style: {v}. Must be one of {valid_styles}")
        return v


class PlotAxes(BaseModel):
    """Plot axes configuration."""
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    zlim: Optional[Tuple[float, float]] = None
    xscale: str = "linear"
    yscale: str = "linear"
    zscale: str = "linear"
    invert_x: bool = False
    invert_y: bool = False
    aspect_ratio: Optional[str] = None
    
    @field_validator('xscale', 'yscale', 'zscale')
    @classmethod
    def validate_scale(cls, v):
        valid_scales = ['linear', 'log', 'symlog', 'logit']
        if v not in valid_scales:
            raise ValueError(f"Invalid scale: {v}. Must be one of {valid_scales}")
        return v


class PlotAnnotations(BaseModel):
    """Plot annotations configuration."""
    text_annotations: List[Dict[str, Any]] = Field(default_factory=list)
    arrow_annotations: List[Dict[str, Any]] = Field(default_factory=list)
    shape_annotations: List[Dict[str, Any]] = Field(default_factory=list)
    reference_lines: List[Dict[str, Any]] = Field(default_factory=list)


class PlotInteractivity(BaseModel):
    """Plot interactivity configuration."""
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_select: bool = False
    enable_hover: bool = True
    enable_crossfilter: bool = False
    hover_template: Optional[str] = None
    click_callback: Optional[str] = None


class PlotExport(BaseModel):
    """Plot export configuration."""
    format: str = "png"
    dpi: int = 300
    transparent: bool = False
    bbox_inches: str = "tight"
    quality: int = 95
    optimize: bool = True
    
    @field_validator('format')
    @classmethod
    def validate_format(cls, v):
        valid_formats = ['png', 'jpg', 'jpeg', 'pdf', 'svg', 'eps', 'ps', 'tiff']
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid format: {v}. Must be one of {valid_formats}")
        return v.lower()


class PlotSpecError(Exception):
    """Base exception for plot specification errors."""
    pass


class PlotSpecValidationError(PlotSpecError):
    """Exception raised when plot specification validation fails."""
    pass


class PlotSpec(BaseModel):
    """
    Declarative plot specification class.
    
    This class provides a comprehensive, structured way to define plots
    with validation, type safety, and integration with the visualization system.
    """
    
    # Core specification
    plot_type: PlotType
    title: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    
    # Data specification reference
    data_spec_id: Optional[str] = None
    data_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Plot configuration
    layout: PlotLayout = Field(default_factory=PlotLayout)
    styling: PlotStyling = Field(default_factory=PlotStyling)
    theme: PlotTheme = Field(default_factory=PlotTheme)
    axes: PlotAxes = Field(default_factory=PlotAxes)
    annotations: PlotAnnotations = Field(default_factory=PlotAnnotations)
    interactivity: PlotInteractivity = Field(default_factory=PlotInteractivity)
    export: PlotExport = Field(default_factory=PlotExport)
    
    # Plugin configuration
    plugin_name: Optional[str] = None
    plugin_config: Dict[str, Any] = Field(default_factory=dict)
    renderer: str = "matplotlib"
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
    
    @field_validator('renderer')
    @classmethod
    def validate_renderer(cls, v):
        valid_renderers = ['matplotlib', 'plotly', 'bokeh', 'seaborn']
        if v not in valid_renderers:
            raise ValueError(f"Invalid renderer: {v}. Must be one of {valid_renderers}")
        return v
    
    @model_validator(mode='after')
    def validate_plot_spec(self):
        """Validate the complete plot specification."""
        # Update modified timestamp (bypass Pydantic validation to avoid recursion)
        object.__setattr__(self, 'modified_at', datetime.now())
        
        # Validate plot type specific requirements
        self._validate_plot_type_requirements()
        
        # Validate theme and styling consistency
        self._validate_theme_styling_consistency()
        
        # Validate axes configuration
        self._validate_axes_configuration()
        
        return self
    
    def _validate_plot_type_requirements(self) -> None:
        """Validate requirements specific to the plot type."""
        if self.plot_type == PlotType.HEATMAP:
            if not self.data_requirements.get('z_column'):
                raise PlotSpecValidationError("Heatmap requires z_column in data_requirements")
        
        elif self.plot_type == PlotType.SURFACE:
            required = ['x_column', 'y_column', 'z_column']
            missing = [col for col in required if col not in self.data_requirements]
            if missing:
                raise PlotSpecValidationError(f"Surface plot requires {missing} in data_requirements")
        
        elif self.plot_type == PlotType.PIE:
            if not self.data_requirements.get('value_column'):
                raise PlotSpecValidationError("Pie chart requires value_column in data_requirements")
    
    def _validate_theme_styling_consistency(self) -> None:
        """Validate consistency between theme and styling."""
        # If styling has custom colors, ensure they're consistent with theme
        if self.styling.colors and len(self.styling.colors) > len(self.theme.colors):
            logger.warning("More styling colors than theme colors - theme colors will be extended")
    
    def _validate_axes_configuration(self) -> None:
        """Validate axes configuration."""
        # Check for logical inconsistencies
        if self.axes.xlim and self.axes.xlim[0] >= self.axes.xlim[1]:
            raise PlotSpecValidationError("xlim: lower bound must be less than upper bound")
        
        if self.axes.ylim and self.axes.ylim[0] >= self.axes.ylim[1]:
            raise PlotSpecValidationError("ylim: lower bound must be less than upper bound")
        
        # Validate log scale with positive limits
        if self.axes.xscale == "log" and self.axes.xlim:
            if any(x <= 0 for x in self.axes.xlim):
                raise PlotSpecValidationError("Log scale requires positive axis limits")
        
        if self.axes.yscale == "log" and self.axes.ylim:
            if any(y <= 0 for y in self.axes.ylim):
                raise PlotSpecValidationError("Log scale requires positive axis limits")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plot specification to dictionary."""
        return self.model_dump(mode='json')
    
    def to_json(self, indent: int = 2) -> str:
        """Convert plot specification to JSON string."""
        return self.model_dump_json(indent=indent)
    
    def to_yaml(self) -> str:
        """Convert plot specification to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlotSpec':
        """Create plot specification from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as e:
            raise PlotSpecValidationError(f"Failed to create PlotSpec from dict: {e}")
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PlotSpec':
        """Create plot specification from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise PlotSpecValidationError(f"Invalid JSON: {e}")
        except Exception as e:
            raise PlotSpecValidationError(f"Failed to create PlotSpec from JSON: {e}")
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'PlotSpec':
        """Create plot specification from YAML string."""
        try:
            data = yaml.safe_load(yaml_str)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise PlotSpecValidationError(f"Invalid YAML: {e}")
        except Exception as e:
            raise PlotSpecValidationError(f"Failed to create PlotSpec from YAML: {e}")
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'PlotSpec':
        """Load plot specification from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise PlotSpecError(f"File not found: {file_path}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return cls.from_yaml(content)
            elif file_path.suffix.lower() == '.json':
                return cls.from_json(content)
            else:
                raise PlotSpecError(f"Unsupported file format: {file_path.suffix}")
        
        except Exception as e:
            raise PlotSpecError(f"Failed to load plot spec from {file_path}: {e}")
    
    def save_to_file(self, file_path: Union[str, Path], format: Optional[str] = None) -> None:
        """Save plot specification to file."""
        file_path = Path(file_path)
        
        # Determine format from extension if not specified
        if format is None:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                format = 'yaml'
            elif file_path.suffix.lower() == '.json':
                format = 'json'
            else:
                raise PlotSpecError(f"Cannot determine format from extension: {file_path.suffix}")
        
        # Generate content based on format
        if format.lower() == 'yaml':
            content = self.to_yaml()
        elif format.lower() == 'json':
            content = self.to_json()
        else:
            raise PlotSpecError(f"Unsupported format: {format}")
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        try:
            file_path.write_text(content, encoding='utf-8')
        except Exception as e:
            raise PlotSpecError(f"Failed to save plot spec to {file_path}: {e}")
    
    def clone(self) -> 'PlotSpec':
        """Create a deep copy of the plot specification."""
        return self.model_copy(deep=True)
    
    def merge(self, other: 'PlotSpec') -> 'PlotSpec':
        """Merge this plot specification with another."""
        # Create a copy of current spec
        merged = self.clone()
        
        # Update with non-None values from other spec
        other_dict = other.to_dict()
        merged_dict = merged.to_dict()
        
        def deep_merge(base: Dict, update: Dict) -> Dict:
            """Recursively merge dictionaries."""
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                elif value is not None:
                    base[key] = value
            return base
        
        merged_dict = deep_merge(merged_dict, other_dict)
        return PlotSpec.from_dict(merged_dict)
    
    def get_required_data_columns(self) -> List[str]:
        """Get list of required data columns for this plot."""
        columns = []
        
        # Add columns from data requirements
        for key, value in self.data_requirements.items():
            if key.endswith('_column') and value:
                columns.append(value)
        
        # Add plot type specific requirements
        if self.plot_type in [PlotType.LINE, PlotType.SCATTER]:
            columns.extend(['x', 'y'])
        elif self.plot_type == PlotType.HEATMAP:
            columns.extend(['x', 'y', 'z'])
        elif self.plot_type == PlotType.HISTOGRAM:
            columns.append('value')
        
        return list(set(columns))  # Remove duplicates
    
    def is_compatible_with_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
        """Check if this plot specification is compatible with given data."""
        required_columns = self.get_required_data_columns()
        
        if isinstance(data, pd.DataFrame):
            available_columns = set(data.columns)
        elif isinstance(data, dict):
            available_columns = set(data.keys())
        else:
            return False
        
        return set(required_columns).issubset(available_columns)
    
    def estimate_complexity(self) -> int:
        """Estimate the complexity of this plot specification (1-10 scale)."""
        complexity = 1
        
        # Base complexity by plot type
        complex_types = [PlotType.SURFACE, PlotType.RADAR, PlotType.TRAINING_CURVES]
        if self.plot_type in complex_types:
            complexity += 3
        elif self.plot_type in [PlotType.HEATMAP, PlotType.CONTOUR]:
            complexity += 2
        
        # Add complexity for features
        if self.annotations.text_annotations or self.annotations.arrow_annotations:
            complexity += 1
        
        if self.interactivity.enable_hover or self.interactivity.enable_select:
            complexity += 1
        
        if len(self.styling.colors or []) > 5:
            complexity += 1
        
        if self.axes.xscale != "linear" or self.axes.yscale != "linear":
            complexity += 1
        
        return min(complexity, 10)
    
    def validate_with_plugin(self, plugin_class: Type) -> bool:
        """Validate compatibility with a specific plugin class."""
        try:
            # This would be implemented to check against actual plugin requirements
            # For now, return True as a placeholder
            return True
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of plot specification."""
        return f"PlotSpec(type={self.plot_type.value}, title='{self.title}', renderer={self.renderer})"
    
    def __repr__(self) -> str:
        """Developer representation of plot specification."""
        return (f"PlotSpec(plot_type={self.plot_type}, title={self.title}, "
                f"renderer={self.renderer}, version={self.version})") 