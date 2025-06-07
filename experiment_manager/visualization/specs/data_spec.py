"""
Data Specification Classes

This module defines the DataSpec class for declarative data configuration,
providing structured data mapping, validation, and transformation specifications.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Type, Tuple
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


class DataType(Enum):
    """Enumeration of supported data types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    ARRAY = "array"
    IMAGE = "image"
    TIMESERIES = "timeseries"
    GEOSPATIAL = "geospatial"
    CUSTOM = "custom"


class AggregationFunction(Enum):
    """Enumeration of aggregation functions."""
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    FIRST = "first"
    LAST = "last"
    MODE = "mode"
    PERCENTILE = "percentile"
    CUSTOM = "custom"


class DataColumn(BaseModel):
    """Specification for a single data column."""
    name: str
    display_name: Optional[str] = None
    data_type: DataType
    required: bool = True
    nullable: bool = False
    default_value: Optional[Any] = None
    
    # Validation rules
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for string validation
    
    # Transformation rules
    transformation: Optional[str] = None
    format_string: Optional[str] = None
    scale_factor: Optional[float] = None
    
    # Metadata
    description: Optional[str] = None
    unit: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('transformation')
    @classmethod
    def validate_transformation(cls, v):
        if v is not None:
            valid_transformations = [
                'log', 'log10', 'sqrt', 'square', 'abs', 'normalize',
                'standardize', 'min_max_scale', 'robust_scale', 'custom'
            ]
            if v not in valid_transformations:
                raise ValueError(f"Invalid transformation: {v}. Must be one of {valid_transformations}")
        return v
    
    def validate_value(self, value: Any) -> bool:
        """Validate a value against this column specification."""
        if value is None:
            return self.nullable
        
        # Type-specific validation
        if self.data_type == DataType.NUMERIC:
            try:
                float_val = float(value)
                if self.min_value is not None and float_val < self.min_value:
                    return False
                if self.max_value is not None and float_val > self.max_value:
                    return False
            except (ValueError, TypeError):
                return False
        
        elif self.data_type == DataType.CATEGORICAL:
            if self.allowed_values and value not in self.allowed_values:
                return False
        
        elif self.data_type == DataType.BOOLEAN:
            if not isinstance(value, bool):
                try:
                    # Allow string representations of booleans
                    bool(value)
                except (ValueError, TypeError):
                    return False
        
        return True


class DataMapping(BaseModel):
    """Mapping specification between data columns and plot axes/dimensions."""
    source_column: str
    target_dimension: str  # e.g., 'x', 'y', 'z', 'color', 'size', 'text'
    aggregation: Optional[AggregationFunction] = None
    aggregation_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Group by specifications
    group_by: Optional[List[str]] = None
    
    # Filtering
    filter_condition: Optional[str] = None  # Pandas query string
    
    # Transformation
    transformation: Optional[str] = None
    transformation_params: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('target_dimension')
    @classmethod
    def validate_target_dimension(cls, v):
        valid_dimensions = [
            'x', 'y', 'z', 'color', 'size', 'shape', 'alpha', 'text',
            'tooltip', 'facet_row', 'facet_col', 'animation_frame'
        ]
        if v not in valid_dimensions:
            raise ValueError(f"Invalid target dimension: {v}. Must be one of {valid_dimensions}")
        return v


class DataSource(BaseModel):
    """Data source specification."""
    source_type: str  # 'file', 'database', 'api', 'memory', 'analytics'
    connection_string: Optional[str] = None
    query: Optional[str] = None
    file_path: Optional[str] = None
    
    # Caching
    cache_enabled: bool = True
    cache_ttl: Optional[int] = None  # seconds
    
    # Streaming
    streaming: bool = False
    batch_size: Optional[int] = None
    
    @field_validator('source_type')
    @classmethod
    def validate_source_type(cls, v):
        valid_types = ['file', 'database', 'api', 'memory', 'analytics', 'custom']
        if v not in valid_types:
            raise ValueError(f"Invalid source type: {v}. Must be one of {valid_types}")
        return v


class DataPreprocessing(BaseModel):
    """Data preprocessing specification."""
    # Missing data handling
    handle_missing: str = "drop"  # 'drop', 'fill', 'interpolate', 'keep'
    fill_method: Optional[str] = None  # 'mean', 'median', 'mode', 'forward', 'backward'
    fill_value: Optional[Any] = None
    
    # Outlier handling
    detect_outliers: bool = False
    outlier_method: str = "iqr"  # 'iqr', 'zscore', 'isolation_forest'
    outlier_action: str = "remove"  # 'remove', 'clip', 'flag'
    
    # Sampling
    sample_size: Optional[int] = None
    sample_method: str = "random"  # 'random', 'systematic', 'stratified'
    
    # Sorting
    sort_by: Optional[List[str]] = None
    sort_ascending: bool = True
    
    @field_validator('handle_missing')
    @classmethod
    def validate_handle_missing(cls, v):
        valid_methods = ['drop', 'fill', 'interpolate', 'keep']
        if v not in valid_methods:
            raise ValueError(f"Invalid missing data method: {v}. Must be one of {valid_methods}")
        return v
    
    @field_validator('outlier_method')
    @classmethod
    def validate_outlier_method(cls, v):
        valid_methods = ['iqr', 'zscore', 'isolation_forest', 'custom']
        if v not in valid_methods:
            raise ValueError(f"Invalid outlier method: {v}. Must be one of {valid_methods}")
        return v


class DataValidation(BaseModel):
    """Data validation rules and constraints."""
    # Schema validation
    enforce_schema: bool = True
    allow_extra_columns: bool = False
    
    # Data quality checks
    check_data_types: bool = True
    check_ranges: bool = True
    check_uniqueness: List[str] = Field(default_factory=list)  # Columns that should be unique
    
    # Statistical validation
    min_rows: Optional[int] = None
    max_rows: Optional[int] = None
    completeness_threshold: float = 0.8  # Minimum fraction of non-null values
    
    # Custom validation functions
    custom_validators: List[str] = Field(default_factory=list)


class DataSpecError(Exception):
    """Base exception for data specification errors."""
    pass


class DataSpecValidationError(DataSpecError):
    """Exception raised when data specification validation fails."""
    pass


class DataSpec(BaseModel):
    """
    Declarative data specification class.
    
    This class provides a comprehensive way to define data requirements,
    mappings, transformations, and validation rules for visualization data.
    """
    
    # Core specification
    spec_id: str
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    
    # Data source
    source: DataSource
    
    # Column specifications
    columns: List[DataColumn] = Field(default_factory=list)
    
    # Data mappings for visualization
    mappings: List[DataMapping] = Field(default_factory=list)
    
    # Preprocessing
    preprocessing: DataPreprocessing = Field(default_factory=DataPreprocessing)
    
    # Validation
    validation: DataValidation = Field(default_factory=DataValidation)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
    
    @model_validator(mode='after')
    def validate_data_spec(self):
        """Validate the complete data specification."""
        # Update modified timestamp (bypass Pydantic validation to avoid recursion)
        object.__setattr__(self, 'modified_at', datetime.now())
        
        # Validate column consistency
        self._validate_column_consistency()
        
        # Validate mapping consistency
        self._validate_mapping_consistency()
        
        # Validate preprocessing configuration
        self._validate_preprocessing_configuration()
        
        return self
    
    def _validate_column_consistency(self) -> None:
        """Validate column specification consistency."""
        column_names = [col.name for col in self.columns]
        
        # Check for duplicate column names
        if len(column_names) != len(set(column_names)):
            duplicates = [name for name in column_names if column_names.count(name) > 1]
            raise DataSpecValidationError(f"Duplicate column names: {duplicates}")
        
        # Validate required columns
        required_columns = [col.name for col in self.columns if col.required]
        if not required_columns:
            logger.warning("No required columns specified in data spec")
    
    def _validate_mapping_consistency(self) -> None:
        """Validate data mapping consistency."""
        column_names = {col.name for col in self.columns}
        
        # Check that all mapped columns exist
        for mapping in self.mappings:
            if mapping.source_column not in column_names:
                raise DataSpecValidationError(
                    f"Mapping references non-existent column: {mapping.source_column}"
                )
            
            # Check group by columns
            if mapping.group_by:
                missing_groups = set(mapping.group_by) - column_names
                if missing_groups:
                    raise DataSpecValidationError(
                        f"Group by references non-existent columns: {missing_groups}"
                    )
    
    def _validate_preprocessing_configuration(self) -> None:
        """Validate preprocessing configuration."""
        # Validate fill method consistency
        if (self.preprocessing.handle_missing == "fill" and 
            not self.preprocessing.fill_method and 
            self.preprocessing.fill_value is None):
            raise DataSpecValidationError(
                "When handle_missing='fill', must specify fill_method or fill_value"
            )
        
        # Validate sort columns
        if self.preprocessing.sort_by:
            column_names = {col.name for col in self.columns}
            missing_sort = set(self.preprocessing.sort_by) - column_names
            if missing_sort:
                raise DataSpecValidationError(
                    f"Sort by references non-existent columns: {missing_sort}"
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert data specification to dictionary."""
        return self.model_dump(mode='json')
    
    def to_json(self, indent: int = 2) -> str:
        """Convert data specification to JSON string."""
        return self.model_dump_json(indent=indent)
    
    def to_yaml(self) -> str:
        """Convert data specification to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSpec':
        """Create data specification from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as e:
            raise DataSpecValidationError(f"Failed to create DataSpec from dict: {e}")
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DataSpec':
        """Create data specification from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise DataSpecValidationError(f"Invalid JSON: {e}")
        except Exception as e:
            raise DataSpecValidationError(f"Failed to create DataSpec from JSON: {e}")
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'DataSpec':
        """Create data specification from YAML string."""
        try:
            data = yaml.safe_load(yaml_str)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise DataSpecValidationError(f"Invalid YAML: {e}")
        except Exception as e:
            raise DataSpecValidationError(f"Failed to create DataSpec from YAML: {e}")
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'DataSpec':
        """Load data specification from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DataSpecError(f"File not found: {file_path}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return cls.from_yaml(content)
            elif file_path.suffix.lower() == '.json':
                return cls.from_json(content)
            else:
                raise DataSpecError(f"Unsupported file format: {file_path.suffix}")
        
        except Exception as e:
            raise DataSpecError(f"Failed to load data spec from {file_path}: {e}")
    
    def save_to_file(self, file_path: Union[str, Path], format: Optional[str] = None) -> None:
        """Save data specification to file."""
        file_path = Path(file_path)
        
        # Determine format from extension if not specified
        if format is None:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                format = 'yaml'
            elif file_path.suffix.lower() == '.json':
                format = 'json'
            else:
                raise DataSpecError(f"Cannot determine format from extension: {file_path.suffix}")
        
        # Generate content based on format
        if format.lower() == 'yaml':
            content = self.to_yaml()
        elif format.lower() == 'json':
            content = self.to_json()
        else:
            raise DataSpecError(f"Unsupported format: {format}")
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        try:
            file_path.write_text(content, encoding='utf-8')
        except Exception as e:
            raise DataSpecError(f"Failed to save data spec to {file_path}: {e}")
    
    def clone(self) -> 'DataSpec':
        """Create a deep copy of the data specification."""
        return self.model_copy(deep=True)
    
    def get_column(self, name: str) -> Optional[DataColumn]:
        """Get column specification by name."""
        for column in self.columns:
            if column.name == name:
                return column
        return None
    
    def get_required_columns(self) -> List[str]:
        """Get list of required column names."""
        return [col.name for col in self.columns if col.required]
    
    def get_mappings_for_dimension(self, dimension: str) -> List[DataMapping]:
        """Get mappings for a specific target dimension."""
        return [mapping for mapping in self.mappings if mapping.target_dimension == dimension]
    
    def validate_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
        """Validate data against this specification."""
        try:
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            
            # Check required columns
            required_cols = self.get_required_columns()
            missing_cols = set(required_cols) - set(data.columns)
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check extra columns if not allowed
            if not self.validation.allow_extra_columns:
                expected_cols = {col.name for col in self.columns}
                extra_cols = set(data.columns) - expected_cols
                if extra_cols:
                    logger.warning(f"Unexpected columns: {extra_cols}")
            
            # Validate data types and ranges
            if self.validation.check_data_types or self.validation.check_ranges:
                for column in self.columns:
                    if column.name in data.columns:
                        valid_values = data[column.name].apply(column.validate_value)
                        if not valid_values.all():
                            invalid_count = (~valid_values).sum()
                            logger.error(f"Column {column.name} has {invalid_count} invalid values")
                            return False
            
            # Check minimum/maximum rows
            if self.validation.min_rows and len(data) < self.validation.min_rows:
                logger.error(f"Data has {len(data)} rows, minimum required: {self.validation.min_rows}")
                return False
            
            if self.validation.max_rows and len(data) > self.validation.max_rows:
                logger.error(f"Data has {len(data)} rows, maximum allowed: {self.validation.max_rows}")
                return False
            
            # Check completeness
            for column in self.columns:
                if column.name in data.columns and not column.nullable:
                    completeness = data[column.name].notna().mean()
                    if completeness < self.validation.completeness_threshold:
                        logger.error(f"Column {column.name} completeness {completeness:.2%} < threshold {self.validation.completeness_threshold:.2%}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    def apply_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to data."""
        processed_data = data.copy()
        
        # Handle missing data
        if self.preprocessing.handle_missing == "drop":
            processed_data = processed_data.dropna()
        elif self.preprocessing.handle_missing == "fill":
            if self.preprocessing.fill_method == "mean":
                processed_data = processed_data.fillna(processed_data.mean())
            elif self.preprocessing.fill_method == "median":
                processed_data = processed_data.fillna(processed_data.median())
            elif self.preprocessing.fill_method == "mode":
                processed_data = processed_data.fillna(processed_data.mode().iloc[0])
            elif self.preprocessing.fill_value is not None:
                processed_data = processed_data.fillna(self.preprocessing.fill_value)
        
        # Apply sampling
        if self.preprocessing.sample_size and len(processed_data) > self.preprocessing.sample_size:
            if self.preprocessing.sample_method == "random":
                processed_data = processed_data.sample(n=self.preprocessing.sample_size)
            # Additional sampling methods could be implemented here
        
        # Apply sorting
        if self.preprocessing.sort_by:
            processed_data = processed_data.sort_values(
                by=self.preprocessing.sort_by,
                ascending=self.preprocessing.sort_ascending
            )
        
        return processed_data
    
    def get_transformed_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get data with column transformations applied."""
        transformed_data = data.copy()
        
        for column in self.columns:
            if column.transformation and column.name in transformed_data.columns:
                col_data = transformed_data[column.name]
                
                if column.transformation == "log":
                    transformed_data[column.name] = np.log(col_data)
                elif column.transformation == "log10":
                    transformed_data[column.name] = np.log10(col_data)
                elif column.transformation == "sqrt":
                    transformed_data[column.name] = np.sqrt(col_data)
                elif column.transformation == "square":
                    transformed_data[column.name] = col_data ** 2
                elif column.transformation == "abs":
                    transformed_data[column.name] = np.abs(col_data)
                elif column.transformation == "normalize":
                    transformed_data[column.name] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                elif column.transformation == "standardize":
                    transformed_data[column.name] = (col_data - col_data.mean()) / col_data.std()
                
                # Apply scale factor if specified
                if column.scale_factor:
                    transformed_data[column.name] *= column.scale_factor
        
        return transformed_data
    
    def estimate_data_size(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate memory usage and performance characteristics."""
        return {
            "rows": len(data),
            "columns": len(data.columns),
            "memory_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(data.select_dtypes(include=['object', 'category']).columns),
            "has_missing": data.isnull().any().any(),
            "completeness": data.notna().mean().mean()
        }
    
    def __str__(self) -> str:
        """String representation of data specification."""
        return f"DataSpec(id={self.spec_id}, name='{self.name}', columns={len(self.columns)})"
    
    def __repr__(self) -> str:
        """Developer representation of data specification."""
        return (f"DataSpec(spec_id={self.spec_id}, name={self.name}, "
                f"columns={len(self.columns)}, mappings={len(self.mappings)})") 