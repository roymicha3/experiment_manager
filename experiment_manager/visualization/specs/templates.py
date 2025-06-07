"""
Plot Template System

This module provides a template system for common plot configurations,
allowing users to quickly create standard plots with predefined settings.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Type, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime
import copy

from .plot_spec import PlotSpec, PlotType, PlotSpecError
from .data_spec import DataSpec, DataType, DataColumn, DataMapping, AggregationFunction

logger = logging.getLogger(__name__)


class TemplateCategory(Enum):
    """Categories of plot templates."""
    BASIC = "basic"
    SCIENTIFIC = "scientific"
    BUSINESS = "business"
    ANALYTICS = "analytics"
    COMPARISON = "comparison"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    TIMESERIES = "timeseries"
    CUSTOM = "custom"


class TemplateError(Exception):
    """Base exception for template errors."""
    pass


@dataclass
class PlotTemplate:
    """
    Plot template definition with metadata and configuration.
    """
    
    # Template identification
    template_id: str
    name: str
    description: str
    category: TemplateCategory
    plot_spec: PlotSpec
    version: str = "1.0.0"
    
    # Template configuration
    data_spec: Optional[DataSpec] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    # Customization
    customizable_fields: List[str] = field(default_factory=list)
    required_parameters: List[str] = field(default_factory=list)
    
    def apply_parameters(self, parameters: Dict[str, Any]) -> PlotSpec:
        """
        Apply parameters to the template and return a customized PlotSpec.
        
        Args:
            parameters: Dictionary of parameter values to apply
            
        Returns:
            Customized PlotSpec instance
            
        Raises:
            TemplateError: If required parameters are missing or invalid
        """
        # Check required parameters
        missing_params = set(self.required_parameters) - set(parameters.keys())
        if missing_params:
            raise TemplateError(f"Missing required parameters: {missing_params}")
        
        # Clone the plot spec to avoid modifying the template
        custom_spec = self.plot_spec.clone()
        
        # Apply parameters to customizable fields
        for field_path in self.customizable_fields:
            if field_path in parameters:
                self._set_nested_field(custom_spec, field_path, parameters[field_path])
        
        # Apply direct parameter mappings
        if 'title' in parameters:
            custom_spec.title = parameters['title']
        
        if 'description' in parameters:
            custom_spec.description = parameters['description']
        
        if 'theme_name' in parameters:
            custom_spec.theme.name = parameters['theme_name']
        
        if 'renderer' in parameters:
            custom_spec.renderer = parameters['renderer']
        
        # Apply plot-specific parameters
        self._apply_plot_specific_parameters(custom_spec, parameters)
        
        return custom_spec
    
    def _set_nested_field(self, obj: Any, field_path: str, value: Any) -> None:
        """Set a nested field value using dot notation."""
        fields = field_path.split('.')
        current = obj
        
        for field in fields[:-1]:
            if hasattr(current, field):
                current = getattr(current, field)
            else:
                raise TemplateError(f"Invalid field path: {field_path}")
        
        final_field = fields[-1]
        if hasattr(current, final_field):
            setattr(current, final_field, value)
        else:
            raise TemplateError(f"Invalid field path: {field_path}")
    
    def _apply_plot_specific_parameters(self, spec: PlotSpec, parameters: Dict[str, Any]) -> None:
        """Apply plot type specific parameters."""
        if spec.plot_type == PlotType.LINE:
            if 'line_style' in parameters:
                spec.styling.line_style = parameters['line_style']
            if 'line_width' in parameters:
                spec.styling.line_width = parameters['line_width']
            if 'marker' in parameters:
                spec.styling.marker = parameters['marker']
        
        elif spec.plot_type == PlotType.SCATTER:
            if 'marker_size' in parameters:
                spec.styling.marker_size = parameters['marker_size']
            if 'alpha' in parameters:
                spec.styling.alpha = parameters['alpha']
        
        elif spec.plot_type == PlotType.BAR:
            if 'edge_color' in parameters:
                spec.styling.edge_color = parameters['edge_color']
            if 'edge_width' in parameters:
                spec.styling.edge_width = parameters['edge_width']
        
        # Apply axis parameters
        if 'xlabel' in parameters:
            spec.layout.xlabel = parameters['xlabel']
        if 'ylabel' in parameters:
            spec.layout.ylabel = parameters['ylabel']
        if 'xscale' in parameters:
            spec.axes.xscale = parameters['xscale']
        if 'yscale' in parameters:
            spec.axes.yscale = parameters['yscale']
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get schema for template parameters."""
        schema = {
            "type": "object",
            "properties": {},
            "required": self.required_parameters
        }
        
        # Add common parameters
        schema["properties"]["title"] = {"type": "string", "description": "Plot title"}
        schema["properties"]["description"] = {"type": "string", "description": "Plot description"}
        schema["properties"]["xlabel"] = {"type": "string", "description": "X-axis label"}
        schema["properties"]["ylabel"] = {"type": "string", "description": "Y-axis label"}
        schema["properties"]["theme_name"] = {"type": "string", "description": "Theme name"}
        schema["properties"]["renderer"] = {
            "type": "string", 
            "enum": ["matplotlib", "plotly", "bokeh"],
            "description": "Renderer to use"
        }
        
        # Add plot-specific parameters
        if self.plot_spec.plot_type == PlotType.LINE:
            schema["properties"]["line_style"] = {
                "type": "string",
                "enum": ["-", "--", "-.", ":"],
                "description": "Line style"
            }
            schema["properties"]["line_width"] = {
                "type": "number",
                "minimum": 0.1,
                "maximum": 10,
                "description": "Line width"
            }
            schema["properties"]["marker"] = {
                "type": "string",
                "description": "Marker style"
            }
        
        elif self.plot_spec.plot_type == PlotType.SCATTER:
            schema["properties"]["marker_size"] = {
                "type": "number",
                "minimum": 1,
                "maximum": 100,
                "description": "Marker size"
            }
            schema["properties"]["alpha"] = {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Transparency"
            }
        
        return schema
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "plot_spec": self.plot_spec.to_dict(),
            "data_spec": self.data_spec.to_dict() if self.data_spec else None,
            "tags": self.tags,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "customizable_fields": self.customizable_fields,
            "required_parameters": self.required_parameters,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlotTemplate':
        """Create template from dictionary."""
        plot_spec = PlotSpec.from_dict(data["plot_spec"])
        data_spec = DataSpec.from_dict(data["data_spec"]) if data["data_spec"] else None
        
        return cls(
            template_id=data["template_id"],
            name=data["name"],
            description=data["description"],
            category=TemplateCategory(data["category"]),
            version=data.get("version", "1.0.0"),
            plot_spec=plot_spec,
            data_spec=data_spec,
            tags=data.get("tags", []),
            author=data.get("author"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            customizable_fields=data.get("customizable_fields", []),
            required_parameters=data.get("required_parameters", []),
        )


class TemplateManager:
    """
    Manager for plot templates with registration, discovery, and caching.
    """
    
    def __init__(self):
        self._templates: Dict[str, PlotTemplate] = {}
        self._templates_by_category: Dict[TemplateCategory, List[PlotTemplate]] = {}
        self._templates_by_plot_type: Dict[PlotType, List[PlotTemplate]] = {}
        self._template_paths: List[Path] = []
        
        # Initialize with builtin templates
        self._register_builtin_templates()
    
    def register_template(self, template: PlotTemplate) -> None:
        """Register a template."""
        if template.template_id in self._templates:
            logger.warning(f"Overriding existing template: {template.template_id}")
        
        self._templates[template.template_id] = template
        
        # Update category index
        if template.category not in self._templates_by_category:
            self._templates_by_category[template.category] = []
        self._templates_by_category[template.category].append(template)
        
        # Update plot type index
        plot_type = template.plot_spec.plot_type
        if plot_type not in self._templates_by_plot_type:
            self._templates_by_plot_type[plot_type] = []
        self._templates_by_plot_type[plot_type].append(template)
        
        logger.info(f"Registered template: {template.template_id}")
    
    def unregister_template(self, template_id: str) -> bool:
        """Unregister a template."""
        if template_id not in self._templates:
            return False
        
        template = self._templates.pop(template_id)
        
        # Remove from category index
        if template.category in self._templates_by_category:
            self._templates_by_category[template.category] = [
                t for t in self._templates_by_category[template.category] 
                if t.template_id != template_id
            ]
        
        # Remove from plot type index
        plot_type = template.plot_spec.plot_type
        if plot_type in self._templates_by_plot_type:
            self._templates_by_plot_type[plot_type] = [
                t for t in self._templates_by_plot_type[plot_type]
                if t.template_id != template_id
            ]
        
        logger.info(f"Unregistered template: {template_id}")
        return True
    
    def get_template(self, template_id: str) -> Optional[PlotTemplate]:
        """Get template by ID."""
        return self._templates.get(template_id)
    
    def list_templates(self, 
                      category: Optional[TemplateCategory] = None,
                      plot_type: Optional[PlotType] = None,
                      tags: Optional[List[str]] = None) -> List[PlotTemplate]:
        """List templates with optional filtering."""
        templates = list(self._templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if plot_type:
            templates = [t for t in templates if t.plot_spec.plot_type == plot_type]
        
        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]
        
        return templates
    
    def create_plot_from_template(self, 
                                 template_id: str, 
                                 parameters: Optional[Dict[str, Any]] = None) -> PlotSpec:
        """Create a plot specification from template."""
        template = self.get_template(template_id)
        if not template:
            raise TemplateError(f"Template not found: {template_id}")
        
        return template.apply_parameters(parameters or {})
    
    def load_templates_from_directory(self, directory: Union[str, Path]) -> int:
        """Load templates from a directory."""
        directory = Path(directory)
        if not directory.exists():
            raise TemplateError(f"Directory not found: {directory}")
        
        loaded_count = 0
        
        for file_path in directory.glob("*.yaml"):
            try:
                content = file_path.read_text(encoding='utf-8')
                data = yaml.safe_load(content)
                template = PlotTemplate.from_dict(data)
                self.register_template(template)
                loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load template from {file_path}: {e}")
        
        for file_path in directory.glob("*.json"):
            try:
                content = file_path.read_text(encoding='utf-8')
                data = json.loads(content)
                template = PlotTemplate.from_dict(data)
                self.register_template(template)
                loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load template from {file_path}: {e}")
        
        logger.info(f"Loaded {loaded_count} templates from {directory}")
        return loaded_count
    
    def save_template(self, template: PlotTemplate, file_path: Union[str, Path]) -> None:
        """Save template to file."""
        file_path = Path(file_path)
        
        # Determine format from extension
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            content = yaml.dump(template.to_dict(), default_flow_style=False, indent=2)
        elif file_path.suffix.lower() == '.json':
            content = json.dumps(template.to_dict(), indent=2)
        else:
            raise TemplateError(f"Unsupported file format: {file_path.suffix}")
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"Saved template {template.template_id} to {file_path}")
    
    def _register_builtin_templates(self) -> None:
        """Register built-in templates."""
        # This will be populated with actual templates
        builtin_templates = BuiltinTemplates.get_all_templates()
        
        for template in builtin_templates:
            self.register_template(template)
    
    def get_template_info(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template information including parameter schema."""
        template = self.get_template(template_id)
        if not template:
            return None
        
        return {
            "template_id": template.template_id,
            "name": template.name,
            "description": template.description,
            "category": template.category.value,
            "plot_type": template.plot_spec.plot_type.value,
            "tags": template.tags,
            "parameter_schema": template.get_parameter_schema(),
            "customizable_fields": template.customizable_fields,
            "required_parameters": template.required_parameters,
        }


class BuiltinTemplates:
    """Factory for built-in plot templates."""
    
    @staticmethod
    def get_all_templates() -> List[PlotTemplate]:
        """Get all built-in templates."""
        return [
            BuiltinTemplates.create_basic_line_plot(),
            BuiltinTemplates.create_basic_scatter_plot(),
            BuiltinTemplates.create_basic_bar_chart(),
            BuiltinTemplates.create_basic_histogram(),
            BuiltinTemplates.create_training_curves(),
            BuiltinTemplates.create_experiment_comparison(),
            BuiltinTemplates.create_correlation_heatmap(),
            BuiltinTemplates.create_distribution_plot(),
        ]
    
    @staticmethod
    def create_basic_line_plot() -> PlotTemplate:
        """Create basic line plot template."""
        plot_spec = PlotSpec(
            plot_type=PlotType.LINE,
            title="Line Plot",
            description="Basic line plot for time series or continuous data",
            data_requirements={"x_column": "x", "y_column": "y"},
        )
        
        return PlotTemplate(
            template_id="basic_line_plot",
            name="Basic Line Plot",
            description="Simple line plot with customizable styling",
            category=TemplateCategory.BASIC,
            plot_spec=plot_spec,
            customizable_fields=[
                "styling.line_style", "styling.line_width", "styling.marker",
                "styling.color", "layout.xlabel", "layout.ylabel"
            ],
            required_parameters=["title"],
            tags=["line", "basic", "timeseries"]
        )
    
    @staticmethod
    def create_basic_scatter_plot() -> PlotTemplate:
        """Create basic scatter plot template."""
        plot_spec = PlotSpec(
            plot_type=PlotType.SCATTER,
            title="Scatter Plot",
            description="Basic scatter plot for correlation analysis",
            data_requirements={"x_column": "x", "y_column": "y"},
        )
        
        return PlotTemplate(
            template_id="basic_scatter_plot",
            name="Basic Scatter Plot",
            description="Simple scatter plot with customizable markers",
            category=TemplateCategory.BASIC,
            plot_spec=plot_spec,
            customizable_fields=[
                "styling.marker_size", "styling.alpha", "styling.color",
                "layout.xlabel", "layout.ylabel"
            ],
            required_parameters=["title"],
            tags=["scatter", "basic", "correlation"]
        )
    
    @staticmethod
    def create_basic_bar_chart() -> PlotTemplate:
        """Create basic bar chart template."""
        plot_spec = PlotSpec(
            plot_type=PlotType.BAR,
            title="Bar Chart",
            description="Basic bar chart for categorical data",
            data_requirements={"x_column": "category", "y_column": "value"},
        )
        
        return PlotTemplate(
            template_id="basic_bar_chart",
            name="Basic Bar Chart",
            description="Simple bar chart with customizable styling",
            category=TemplateCategory.BASIC,
            plot_spec=plot_spec,
            customizable_fields=[
                "styling.color", "styling.edge_color", "styling.edge_width",
                "layout.xlabel", "layout.ylabel"
            ],
            required_parameters=["title"],
            tags=["bar", "basic", "categorical"]
        )
    
    @staticmethod
    def create_basic_histogram() -> PlotTemplate:
        """Create basic histogram template."""
        plot_spec = PlotSpec(
            plot_type=PlotType.HISTOGRAM,
            title="Histogram",
            description="Basic histogram for distribution analysis",
            data_requirements={"value_column": "value"},
        )
        
        return PlotTemplate(
            template_id="basic_histogram",
            name="Basic Histogram",
            description="Simple histogram with customizable bins",
            category=TemplateCategory.DISTRIBUTION,
            plot_spec=plot_spec,
            customizable_fields=[
                "styling.color", "styling.alpha", "styling.edge_color",
                "layout.xlabel", "layout.ylabel"
            ],
            required_parameters=["title"],
            tags=["histogram", "distribution", "basic"]
        )
    
    @staticmethod
    def create_training_curves() -> PlotTemplate:
        """Create training curves template."""
        plot_spec = PlotSpec(
            plot_type=PlotType.TRAINING_CURVES,
            title="Training Curves",
            description="Training and validation curves for machine learning",
            data_requirements={
                "epoch_column": "epoch",
                "train_metric_column": "train_loss",
                "val_metric_column": "val_loss"
            },
            plugin_name="TrainingCurvesPlotPlugin",
        )
        
        return PlotTemplate(
            template_id="training_curves",
            name="Training Curves",
            description="ML training progress visualization",
            category=TemplateCategory.ANALYTICS,
            plot_spec=plot_spec,
            customizable_fields=[
                "layout.xlabel", "layout.ylabel", "styling.line_width"
            ],
            required_parameters=["title"],
            tags=["training", "ml", "analytics", "curves"]
        )
    
    @staticmethod
    def create_experiment_comparison() -> PlotTemplate:
        """Create experiment comparison template."""
        plot_spec = PlotSpec(
            plot_type=PlotType.EXPERIMENT_COMPARISON,
            title="Experiment Comparison",
            description="Side-by-side comparison of multiple experiments",
            data_requirements={
                "experiment_column": "experiment_name",
                "metric_column": "accuracy",
                "value_column": "value"
            },
        )
        
        return PlotTemplate(
            template_id="experiment_comparison",
            name="Experiment Comparison",
            description="Compare multiple experiments side by side",
            category=TemplateCategory.COMPARISON,
            plot_spec=plot_spec,
            customizable_fields=[
                "layout.xlabel", "layout.ylabel", "styling.colors"
            ],
            required_parameters=["title"],
            tags=["comparison", "experiments", "analytics"]
        )
    
    @staticmethod
    def create_correlation_heatmap() -> PlotTemplate:
        """Create correlation heatmap template."""
        plot_spec = PlotSpec(
            plot_type=PlotType.HEATMAP,
            title="Correlation Heatmap",
            description="Correlation matrix visualization",
            data_requirements={
                "x_column": "variable1",
                "y_column": "variable2",
                "z_column": "correlation"
            },
        )
        
        return PlotTemplate(
            template_id="correlation_heatmap",
            name="Correlation Heatmap",
            description="Visualize correlations between variables",
            category=TemplateCategory.CORRELATION,
            plot_spec=plot_spec,
            customizable_fields=[
                "layout.xlabel", "layout.ylabel", "theme.colors"
            ],
            required_parameters=["title"],
            tags=["correlation", "heatmap", "matrix"]
        )
    
    @staticmethod
    def create_distribution_plot() -> PlotTemplate:
        """Create distribution plot template."""
        plot_spec = PlotSpec(
            plot_type=PlotType.DISTRIBUTION,
            title="Distribution Plot",
            description="Statistical distribution visualization",
            data_requirements={"value_column": "value", "group_column": "group"},
        )
        
        return PlotTemplate(
            template_id="distribution_plot",
            name="Distribution Plot",
            description="Visualize data distributions with statistics",
            category=TemplateCategory.DISTRIBUTION,
            plot_spec=plot_spec,
            customizable_fields=[
                "layout.xlabel", "layout.ylabel", "styling.alpha"
            ],
            required_parameters=["title"],
            tags=["distribution", "statistics", "groups"]
        )


# Global template manager instance
template_manager = TemplateManager() 