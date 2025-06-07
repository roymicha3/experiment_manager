"""
Validation System for Plot and Data Specifications

This module provides comprehensive validation capabilities for PlotSpec and DataSpec
objects, including custom rules, cross-validation, and extensible validation framework.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
import re
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from .plot_spec import PlotSpec, PlotType, PlotSpecError
from .data_spec import DataSpec, DataType, DataColumn, DataMapping

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    issues: List['ValidationIssue'] = field(default_factory=list)
    warnings: List['ValidationIssue'] = field(default_factory=list)
    info: List['ValidationIssue'] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List['ValidationIssue']:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def has_errors(self) -> bool:
        """Check if there are any errors or critical issues."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'has_errors': self.has_errors(),
            'has_warnings': self.has_warnings(),
            'issue_count': len(self.issues),
            'issues': [issue.to_dict() for issue in self.issues],
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    message: str
    severity: ValidationSeverity
    field_path: Optional[str] = None
    rule_name: Optional[str] = None
    suggested_fix: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'message': self.message,
            'severity': self.severity.value,
            'field_path': self.field_path,
            'rule_name': self.rule_name,
            'suggested_fix': self.suggested_fix,
            'context': self.context
        }


class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    def __init__(self, name: str, description: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.description = description
        self.severity = severity
    
    @abstractmethod
    def validate(self, spec: Union[PlotSpec, DataSpec]) -> List[ValidationIssue]:
        """Validate specification and return list of issues."""
        pass
    
    def create_issue(self, message: str, field_path: Optional[str] = None, 
                     suggested_fix: Optional[str] = None, **context) -> ValidationIssue:
        """Create validation issue with rule context."""
        return ValidationIssue(
            message=message,
            severity=self.severity,
            field_path=field_path,
            rule_name=self.name,
            suggested_fix=suggested_fix,
            context=context
        )


class PlotTypeCompatibilityRule(ValidationRule):
    """Validate plot type compatibility with data and configuration."""
    
    def __init__(self):
        super().__init__(
            "plot_type_compatibility",
            "Validate plot type compatibility with axes and data requirements",
            ValidationSeverity.ERROR
        )
    
    def validate(self, spec: Union[PlotSpec, DataSpec]) -> List[ValidationIssue]:
        if not isinstance(spec, PlotSpec):
            return []
        
        issues = []
        
        # Check scatter plot requirements
        if spec.plot_type == PlotType.SCATTER:
            if not spec.axes or not spec.axes.x_axis:
                issues.append(self.create_issue(
                    "Scatter plots require x-axis configuration",
                    field_path="axes.x_axis",
                    suggested_fix="Add x_axis configuration"
                ))
            if not spec.axes or not spec.axes.y_axis:
                issues.append(self.create_issue(
                    "Scatter plots require y-axis configuration",
                    field_path="axes.y_axis",
                    suggested_fix="Add y_axis configuration"
                ))
        
        # Check 3D plot requirements
        if spec.plot_type in [PlotType.SURFACE, PlotType.VOLUME]:
            if not spec.layout or not getattr(spec.layout, 'is_3d', False):
                issues.append(self.create_issue(
                    f"{spec.plot_type.value} plots require 3D layout",
                    field_path="layout.is_3d",
                    suggested_fix="Set layout.is_3d=True"
                ))
        
        return issues


class DataMappingRule(ValidationRule):
    """Validate data mapping consistency."""
    
    def __init__(self):
        super().__init__(
            "data_mapping_consistency",
            "Validate data mapping between columns and plot dimensions",
            ValidationSeverity.ERROR
        )
    
    def validate(self, spec: Union[PlotSpec, DataSpec]) -> List[ValidationIssue]:
        if not isinstance(spec, DataSpec):
            return []
        
        issues = []
        column_names = {col.name for col in spec.columns}
        
        # Check mapping references
        for i, mapping in enumerate(spec.mappings):
            if mapping.source_column not in column_names:
                issues.append(self.create_issue(
                    f"Mapping references non-existent column: {mapping.source_column}",
                    field_path=f"mappings[{i}].source_column",
                    suggested_fix=f"Add column '{mapping.source_column}' or update mapping"
                ))
            
            # Check group by columns
            if mapping.group_by:
                missing_groups = set(mapping.group_by) - column_names
                if missing_groups:
                    issues.append(self.create_issue(
                        f"Group by references non-existent columns: {missing_groups}",
                        field_path=f"mappings[{i}].group_by",
                        suggested_fix=f"Add columns {missing_groups} or update group_by"
                    ))
        
        return issues


class DataTypeConsistencyRule(ValidationRule):
    """Validate data type consistency."""
    
    def __init__(self):
        super().__init__(
            "data_type_consistency",
            "Validate data type consistency with aggregations and transformations",
            ValidationSeverity.WARNING
        )
    
    def validate(self, spec: Union[PlotSpec, DataSpec]) -> List[ValidationIssue]:
        if not isinstance(spec, DataSpec):
            return []
        
        issues = []
        
        for i, mapping in enumerate(spec.mappings):
            # Find source column
            source_col = None
            for col in spec.columns:
                if col.name == mapping.source_column:
                    source_col = col
                    break
            
            if not source_col:
                continue
            
            # Check aggregation compatibility
            if mapping.aggregation:
                numeric_aggregations = ['mean', 'median', 'sum', 'std', 'var']
                if (mapping.aggregation.value in numeric_aggregations and 
                    source_col.data_type not in [DataType.NUMERIC]):
                    issues.append(self.create_issue(
                        f"Aggregation '{mapping.aggregation.value}' requires numeric data type",
                        field_path=f"mappings[{i}].aggregation",
                        suggested_fix="Use compatible aggregation or change data type"
                    ))
        
        return issues


class PerformanceRule(ValidationRule):
    """Validate performance implications."""
    
    def __init__(self):
        super().__init__(
            "performance_implications",
            "Check for potential performance issues",
            ValidationSeverity.WARNING
        )
    
    def validate(self, spec: Union[PlotSpec, DataSpec]) -> List[ValidationIssue]:
        issues = []
        
        if isinstance(spec, PlotSpec):
            # Check for complex plot types
            complex_types = [PlotType.HEATMAP, PlotType.SURFACE, PlotType.VOLUME]
            if spec.plot_type in complex_types:
                issues.append(self.create_issue(
                    f"Plot type '{spec.plot_type.value}' may have performance implications with large datasets",
                    field_path="plot_type",
                    suggested_fix="Consider data sampling or progressive rendering"
                ))
            
            # Check for too many annotations
            if spec.annotations and len(spec.annotations.text_annotations or []) > 50:
                issues.append(self.create_issue(
                    "Large number of text annotations may impact performance",
                    field_path="annotations.text_annotations",
                    suggested_fix="Consider reducing annotations or using selective display"
                ))
        
        elif isinstance(spec, DataSpec):
            # Check for potential memory issues
            if len(spec.columns) > 100:
                issues.append(self.create_issue(
                    "Large number of columns may impact memory usage",
                    field_path="columns",
                    suggested_fix="Consider column selection or chunked processing"
                ))
        
        return issues


class ValidationError(Exception):
    """Validation error."""
    pass


class SpecValidator:
    """
    Comprehensive specification validator.
    
    Provides validation for PlotSpec and DataSpec objects using configurable
    rules and custom validation logic.
    """
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.custom_validators: Dict[str, Callable] = {}
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules."""
        self.rules.extend([
            PlotTypeCompatibilityRule(),
            DataMappingRule(),
            DataTypeConsistencyRule(),
            PerformanceRule()
        ])
    
    def add_rule(self, rule: ValidationRule):
        """Add custom validation rule."""
        self.rules.append(rule)
        logger.debug(f"Added validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove validation rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                logger.debug(f"Removed validation rule: {rule_name}")
                return True
        return False
    
    def add_custom_validator(self, name: str, validator: Callable):
        """Add custom validation function."""
        self.custom_validators[name] = validator
        logger.debug(f"Added custom validator: {name}")
    
    def validate_spec(self, spec: Union[PlotSpec, DataSpec]) -> ValidationResult:
        """Validate specification using all registered rules."""
        all_issues = []
        
        try:
            # Run Pydantic validation first
            spec.model_validate(spec.model_dump())
        except PydanticValidationError as e:
            for error in e.errors():
                issue = ValidationIssue(
                    message=f"Pydantic validation error: {error['msg']}",
                    severity=ValidationSeverity.ERROR,
                    field_path=".".join(str(loc) for loc in error.get('loc', [])),
                    rule_name="pydantic_validation"
                )
                all_issues.append(issue)
        
        # Run custom rules
        for rule in self.rules:
            try:
                issues = rule.validate(spec)
                all_issues.extend(issues)
            except Exception as e:
                logger.error(f"Error in validation rule {rule.name}: {e}")
                issue = ValidationIssue(
                    message=f"Validation rule error: {e}",
                    severity=ValidationSeverity.CRITICAL,
                    rule_name=rule.name
                )
                all_issues.append(issue)
        
        # Run custom validators
        for name, validator in self.custom_validators.items():
            try:
                issues = validator(spec)
                if isinstance(issues, list):
                    all_issues.extend(issues)
                elif isinstance(issues, ValidationIssue):
                    all_issues.append(issues)
            except Exception as e:
                logger.error(f"Error in custom validator {name}: {e}")
                issue = ValidationIssue(
                    message=f"Custom validator error: {e}",
                    severity=ValidationSeverity.CRITICAL,
                    rule_name=name
                )
                all_issues.append(issue)
        
        # Determine overall validity
        has_errors = any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
                        for issue in all_issues)
        
        result = ValidationResult(
            is_valid=not has_errors,
            issues=all_issues
        )
        
        logger.info(f"Validation completed: {len(all_issues)} issues found, valid={result.is_valid}")
        return result
    
    def validate_plot_spec(self, plot_spec: PlotSpec) -> ValidationResult:
        """Validate plot specification."""
        return self.validate_spec(plot_spec)
    
    def validate_data_spec(self, data_spec: DataSpec) -> ValidationResult:
        """Validate data specification."""
        return self.validate_spec(data_spec)
    
    def validate_compatibility(self, plot_spec: PlotSpec, data_spec: DataSpec) -> ValidationResult:
        """Validate compatibility between plot and data specifications."""
        issues = []
        
        # Check if data mappings match plot requirements
        required_dimensions = self._get_required_dimensions(plot_spec.plot_type)
        mapped_dimensions = {mapping.target_dimension for mapping in data_spec.mappings}
        
        missing_dimensions = required_dimensions - mapped_dimensions
        if missing_dimensions:
            issues.append(ValidationIssue(
                message=f"Missing required dimensions for {plot_spec.plot_type.value}: {missing_dimensions}",
                severity=ValidationSeverity.ERROR,
                rule_name="plot_data_compatibility",
                suggested_fix=f"Add mappings for dimensions: {missing_dimensions}"
            ))
        
        # Check data type compatibility
        for mapping in data_spec.mappings:
            source_col = data_spec.get_column(mapping.source_column)
            if source_col:
                compatibility_issues = self._check_dimension_compatibility(
                    mapping.target_dimension, source_col.data_type, plot_spec.plot_type
                )
                issues.extend(compatibility_issues)
        
        has_errors = any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
                        for issue in issues)
        
        return ValidationResult(is_valid=not has_errors, issues=issues)
    
    def _get_required_dimensions(self, plot_type: PlotType) -> set:
        """Get required dimensions for plot type."""
        dimension_map = {
            PlotType.LINE: {'x', 'y'},
            PlotType.SCATTER: {'x', 'y'},
            PlotType.BAR: {'x', 'y'},
            PlotType.HEATMAP: {'x', 'y', 'z'},
            PlotType.SURFACE: {'x', 'y', 'z'},
            PlotType.BOX: {'x', 'y'},
            PlotType.HISTOGRAM: {'x'},
            PlotType.PIE: {'values', 'labels'},
        }
        return dimension_map.get(plot_type, set())
    
    def _check_dimension_compatibility(self, dimension: str, data_type: DataType, 
                                     plot_type: PlotType) -> List[ValidationIssue]:
        """Check if data type is compatible with plot dimension."""
        issues = []
        
        # Define compatibility rules
        numeric_dimensions = {'x', 'y', 'z', 'size', 'values'}
        categorical_dimensions = {'color', 'group', 'labels', 'category'}
        
        if dimension in numeric_dimensions and data_type not in [DataType.NUMERIC]:
            if not (dimension == 'x' and data_type == DataType.DATETIME):  # Allow datetime for x-axis
                issues.append(ValidationIssue(
                    message=f"Dimension '{dimension}' typically requires numeric data, got {data_type.value}",
                    severity=ValidationSeverity.WARNING,
                    rule_name="dimension_type_compatibility",
                    suggested_fix=f"Consider data transformation or use appropriate dimension"
                ))
        
        return issues
    
    def get_rule_info(self) -> List[Dict[str, Any]]:
        """Get information about registered rules."""
        return [
            {
                'name': rule.name,
                'description': rule.description,
                'severity': rule.severity.value
            }
            for rule in self.rules
        ]


# Global validator instance
spec_validator = SpecValidator() 