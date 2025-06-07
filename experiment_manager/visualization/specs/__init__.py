"""
Plot Specification System for Visualization Components

This module provides declarative plot configuration and validation through
PlotSpec and DataSpec classes, with schema validation, serialization, and
template system for common plot configurations.
"""

from .plot_spec import (
    PlotSpec,
    PlotType,
    PlotSpecError,
    PlotSpecValidationError
)

from .data_spec import (
    DataSpec,
    DataType,
    DataMapping,
    DataColumn,
    DataSpecError,
    DataSpecValidationError
)

from .templates import (
    PlotTemplate,
    TemplateCategory,
    TemplateManager,
    BuiltinTemplates,
    TemplateError,
    template_manager
)

from .serialization import (
    SpecSerializer,
    SerializationFormat,
    SerializationError,
    spec_serializer
)

from .validation import (
    SpecValidator,
    ValidationResult,
    ValidationRule,
    ValidationSeverity,
    ValidationError as SpecValidationError,
    spec_validator
)

__all__ = [
    # Core specifications
    'PlotSpec',
    'PlotType',
    'PlotSpecError',
    'PlotSpecValidationError',
    
    # Data specifications
    'DataSpec',
    'DataType',
    'DataMapping',
    'DataColumn',
    'DataSpecError',
    'DataSpecValidationError',
    
    # Template system
    'PlotTemplate',
    'TemplateCategory',
    'TemplateManager',
    'BuiltinTemplates',
    'TemplateError',
    'template_manager',
    
    # Serialization
    'SpecSerializer',
    'SerializationFormat',
    'SerializationError',
    'spec_serializer',
    
    # Validation
    'SpecValidator',
    'ValidationResult',
    'ValidationRule',
    'ValidationSeverity',
    'SpecValidationError',
    'spec_validator',
] 