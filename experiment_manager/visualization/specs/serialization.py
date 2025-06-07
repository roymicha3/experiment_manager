"""
Serialization System for Plot and Data Specifications

This module provides comprehensive serialization support for PlotSpec and DataSpec
objects, including multiple formats, validation, and conversion utilities.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Type, IO
from pathlib import Path
import json
import yaml
import pickle
import base64
import logging
from datetime import datetime
import io
import zipfile
import gzip

from .plot_spec import PlotSpec, PlotSpecError
from .data_spec import DataSpec, DataSpecError

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    YAML = "yaml"
    PICKLE = "pickle"
    BINARY = "binary"
    XML = "xml"
    CSV = "csv"  # For data specs only
    COMPRESSED = "compressed"


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    BZIP2 = "bzip2"


class SerializationError(Exception):
    """Base exception for serialization errors."""
    pass


class SerializationOptions:
    """Configuration options for serialization."""
    
    def __init__(self,
                 format: SerializationFormat = SerializationFormat.JSON,
                 compression: CompressionType = CompressionType.NONE,
                 indent: int = 2,
                 include_metadata: bool = True,
                 include_timestamps: bool = True,
                 validate_on_load: bool = True,
                 encoding: str = 'utf-8'):
        self.format = format
        self.compression = compression
        self.indent = indent
        self.include_metadata = include_metadata
        self.include_timestamps = include_timestamps
        self.validate_on_load = validate_on_load
        self.encoding = encoding


class SpecSerializer:
    """
    Universal serializer for plot and data specifications.
    
    Supports multiple formats, compression, validation, and batch operations.
    """
    
    def __init__(self, options: Optional[SerializationOptions] = None):
        self.options = options or SerializationOptions()
        self._format_handlers = {
            SerializationFormat.JSON: self._handle_json,
            SerializationFormat.YAML: self._handle_yaml,
            SerializationFormat.PICKLE: self._handle_pickle,
            SerializationFormat.BINARY: self._handle_binary,
        }
    
    def serialize_plot_spec(self, 
                           spec: PlotSpec, 
                           target: Optional[Union[str, Path, IO]] = None,
                           options: Optional[SerializationOptions] = None) -> Union[str, bytes]:
        """
        Serialize a PlotSpec to the specified format.
        
        Args:
            spec: PlotSpec to serialize
            target: Target file path, file object, or None for return value
            options: Serialization options (overrides instance options)
            
        Returns:
            Serialized data as string or bytes (if target is None)
            
        Raises:
            SerializationError: If serialization fails
        """
        opts = options or self.options
        
        try:
            # Prepare data for serialization
            data = self._prepare_plot_spec_data(spec, opts)
            
            # Serialize to specified format
            handler = self._format_handlers.get(opts.format)
            if not handler:
                raise SerializationError(f"Unsupported format: {opts.format}")
            
            serialized_data = handler(data, serialize=True, options=opts)
            
            # Apply compression if specified
            if opts.compression != CompressionType.NONE:
                serialized_data = self._compress_data(serialized_data, opts.compression)
            
            # Write to target or return
            if target is not None:
                self._write_to_target(serialized_data, target, opts)
            else:
                return serialized_data
                
        except Exception as e:
            raise SerializationError(f"Failed to serialize PlotSpec: {e}")
    
    def deserialize_plot_spec(self, 
                             source: Union[str, bytes, Path, IO],
                             options: Optional[SerializationOptions] = None) -> PlotSpec:
        """
        Deserialize a PlotSpec from the specified source.
        
        Args:
            source: Source data, file path, or file object
            options: Serialization options (overrides instance options)
            
        Returns:
            Deserialized PlotSpec instance
            
        Raises:
            SerializationError: If deserialization fails
        """
        opts = options or self.options
        
        try:
            # Read data from source
            data = self._read_from_source(source, opts)
            
            # Decompress if needed
            if opts.compression != CompressionType.NONE:
                data = self._decompress_data(data, opts.compression)
            
            # Deserialize from format
            handler = self._format_handlers.get(opts.format)
            if not handler:
                raise SerializationError(f"Unsupported format: {opts.format}")
            
            deserialized_data = handler(data, serialize=False, options=opts)
            
            # Create PlotSpec from data
            spec = PlotSpec.from_dict(deserialized_data)
            
            # Validate if requested
            if opts.validate_on_load:
                spec.model_validate(spec.model_dump())
            
            return spec
            
        except Exception as e:
            raise SerializationError(f"Failed to deserialize PlotSpec: {e}")
    
    def serialize_data_spec(self, 
                           spec: DataSpec, 
                           target: Optional[Union[str, Path, IO]] = None,
                           options: Optional[SerializationOptions] = None) -> Union[str, bytes]:
        """
        Serialize a DataSpec to the specified format.
        
        Args:
            spec: DataSpec to serialize
            target: Target file path, file object, or None for return value
            options: Serialization options (overrides instance options)
            
        Returns:
            Serialized data as string or bytes (if target is None)
            
        Raises:
            SerializationError: If serialization fails
        """
        opts = options or self.options
        
        try:
            # Prepare data for serialization
            data = self._prepare_data_spec_data(spec, opts)
            
            # Serialize to specified format
            handler = self._format_handlers.get(opts.format)
            if not handler:
                raise SerializationError(f"Unsupported format: {opts.format}")
            
            serialized_data = handler(data, serialize=True, options=opts)
            
            # Apply compression if specified
            if opts.compression != CompressionType.NONE:
                serialized_data = self._compress_data(serialized_data, opts.compression)
            
            # Write to target or return
            if target is not None:
                self._write_to_target(serialized_data, target, opts)
            else:
                return serialized_data
                
        except Exception as e:
            raise SerializationError(f"Failed to serialize DataSpec: {e}")
    
    def deserialize_data_spec(self, 
                             source: Union[str, bytes, Path, IO],
                             options: Optional[SerializationOptions] = None) -> DataSpec:
        """
        Deserialize a DataSpec from the specified source.
        
        Args:
            source: Source data, file path, or file object
            options: Serialization options (overrides instance options)
            
        Returns:
            Deserialized DataSpec instance
            
        Raises:
            SerializationError: If deserialization fails
        """
        opts = options or self.options
        
        try:
            # Read data from source
            data = self._read_from_source(source, opts)
            
            # Decompress if needed
            if opts.compression != CompressionType.NONE:
                data = self._decompress_data(data, opts.compression)
            
            # Deserialize from format
            handler = self._format_handlers.get(opts.format)
            if not handler:
                raise SerializationError(f"Unsupported format: {opts.format}")
            
            deserialized_data = handler(data, serialize=False, options=opts)
            
            # Create DataSpec from data
            spec = DataSpec.from_dict(deserialized_data)
            
            # Validate if requested
            if opts.validate_on_load:
                spec.model_validate(spec.model_dump())
            
            return spec
            
        except Exception as e:
            raise SerializationError(f"Failed to deserialize DataSpec: {e}")
    
    def serialize_batch(self, 
                       specs: List[Union[PlotSpec, DataSpec]], 
                       target: Union[str, Path],
                       options: Optional[SerializationOptions] = None) -> None:
        """
        Serialize multiple specifications to a batch file.
        
        Args:
            specs: List of specifications to serialize
            target: Target file path
            options: Serialization options
        """
        opts = options or self.options
        target_path = Path(target)
        
        try:
            batch_data = {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "count": len(specs),
                "specs": []
            }
            
            for i, spec in enumerate(specs):
                if isinstance(spec, PlotSpec):
                    spec_data = {
                        "type": "PlotSpec",
                        "index": i,
                        "data": self._prepare_plot_spec_data(spec, opts)
                    }
                elif isinstance(spec, DataSpec):
                    spec_data = {
                        "type": "DataSpec", 
                        "index": i,
                        "data": self._prepare_data_spec_data(spec, opts)
                    }
                else:
                    raise SerializationError(f"Unsupported spec type: {type(spec)}")
                
                batch_data["specs"].append(spec_data)
            
            # Serialize batch data
            self.serialize_plot_spec(PlotSpec(**batch_data), target_path, opts)
            
        except Exception as e:
            raise SerializationError(f"Failed to serialize batch: {e}")
    
    def convert_format(self, 
                      source: Union[str, Path], 
                      target: Union[str, Path],
                      source_format: SerializationFormat,
                      target_format: SerializationFormat) -> None:
        """
        Convert specification from one format to another.
        
        Args:
            source: Source file path
            target: Target file path
            source_format: Source format
            target_format: Target format
        """
        try:
            # Load with source format
            source_opts = SerializationOptions(format=source_format)
            
            # Try to determine spec type by loading as PlotSpec first
            try:
                spec = self.deserialize_plot_spec(source, source_opts)
                spec_type = "plot"
            except:
                # If that fails, try DataSpec
                spec = self.deserialize_data_spec(source, source_opts)
                spec_type = "data"
            
            # Save with target format
            target_opts = SerializationOptions(format=target_format)
            
            if spec_type == "plot":
                self.serialize_plot_spec(spec, target, target_opts)
            else:
                self.serialize_data_spec(spec, target, target_opts)
                
        except Exception as e:
            raise SerializationError(f"Failed to convert format: {e}")
    
    def _prepare_plot_spec_data(self, spec: PlotSpec, options: SerializationOptions) -> Dict[str, Any]:
        """Prepare PlotSpec data for serialization."""
        data = spec.to_dict()
        
        if not options.include_timestamps:
            data.pop('created_at', None)
            data.pop('modified_at', None)
        
        if not options.include_metadata:
            data.pop('metadata', None)
            data.pop('tags', None)
        
        return data
    
    def _prepare_data_spec_data(self, spec: DataSpec, options: SerializationOptions) -> Dict[str, Any]:
        """Prepare DataSpec data for serialization."""
        data = spec.to_dict()
        
        if not options.include_timestamps:
            data.pop('created_at', None)
            data.pop('modified_at', None)
        
        if not options.include_metadata:
            data.pop('metadata', None)
            data.pop('tags', None)
        
        return data
    
    def _handle_json(self, data: Any, serialize: bool, options: SerializationOptions) -> Union[str, Dict]:
        """Handle JSON serialization/deserialization."""
        if serialize:
            return json.dumps(data, indent=options.indent, default=str)
        else:
            if isinstance(data, (str, bytes)):
                return json.loads(data)
            return data
    
    def _handle_yaml(self, data: Any, serialize: bool, options: SerializationOptions) -> Union[str, Dict]:
        """Handle YAML serialization/deserialization."""
        if serialize:
            return yaml.dump(data, default_flow_style=False, indent=options.indent)
        else:
            if isinstance(data, (str, bytes)):
                return yaml.safe_load(data)
            return data
    
    def _handle_pickle(self, data: Any, serialize: bool, options: SerializationOptions) -> Union[bytes, Any]:
        """Handle Pickle serialization/deserialization."""
        if serialize:
            return pickle.dumps(data)
        else:
            if isinstance(data, bytes):
                return pickle.loads(data)
            return data
    
    def _handle_binary(self, data: Any, serialize: bool, options: SerializationOptions) -> Union[bytes, Any]:
        """Handle binary serialization/deserialization."""
        if serialize:
            # Convert to JSON first, then encode
            json_data = json.dumps(data, default=str)
            return base64.b64encode(json_data.encode(options.encoding))
        else:
            if isinstance(data, bytes):
                decoded = base64.b64decode(data).decode(options.encoding)
                return json.loads(decoded)
            return data
    
    def _compress_data(self, data: Union[str, bytes], compression: CompressionType) -> bytes:
        """Compress data using specified compression type."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if compression == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression == CompressionType.ZIP:
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('data', data)
            return bio.getvalue()
        elif compression == CompressionType.BZIP2:
            import bz2
            return bz2.compress(data)
        else:
            return data
    
    def _decompress_data(self, data: bytes, compression: CompressionType) -> Union[str, bytes]:
        """Decompress data using specified compression type."""
        if compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.ZIP:
            with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
                return zf.read('data')
        elif compression == CompressionType.BZIP2:
            import bz2
            return bz2.decompress(data)
        else:
            return data
    
    def _read_from_source(self, source: Union[str, bytes, Path, IO], options: SerializationOptions) -> Union[str, bytes]:
        """Read data from various source types."""
        if isinstance(source, (str, bytes)):
            return source
        elif isinstance(source, Path):
            if options.format in [SerializationFormat.PICKLE, SerializationFormat.BINARY]:
                return source.read_bytes()
            else:
                return source.read_text(encoding=options.encoding)
        elif hasattr(source, 'read'):
            return source.read()
        else:
            raise SerializationError(f"Unsupported source type: {type(source)}")
    
    def _write_to_target(self, data: Union[str, bytes], target: Union[str, Path, IO], options: SerializationOptions) -> None:
        """Write data to various target types."""
        if isinstance(target, Path):
            if isinstance(data, bytes):
                target.write_bytes(data)
            else:
                target.write_text(data, encoding=options.encoding)
        elif hasattr(target, 'write'):
            target.write(data)
        else:
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, bytes):
                target_path.write_bytes(data)
            else:
                target_path.write_text(data, encoding=options.encoding)
    
    def get_format_info(self, format: SerializationFormat) -> Dict[str, Any]:
        """Get information about a serialization format."""
        format_info = {
            SerializationFormat.JSON: {
                "name": "JSON",
                "description": "JavaScript Object Notation",
                "binary": False,
                "human_readable": True,
                "file_extension": ".json",
                "mime_type": "application/json"
            },
            SerializationFormat.YAML: {
                "name": "YAML",
                "description": "YAML Ain't Markup Language",
                "binary": False,
                "human_readable": True,
                "file_extension": ".yaml",
                "mime_type": "application/x-yaml"
            },
            SerializationFormat.PICKLE: {
                "name": "Pickle",
                "description": "Python pickle format",
                "binary": True,
                "human_readable": False,
                "file_extension": ".pkl",
                "mime_type": "application/octet-stream"
            },
            SerializationFormat.BINARY: {
                "name": "Binary",
                "description": "Base64 encoded binary format",
                "binary": True,
                "human_readable": False,
                "file_extension": ".bin",
                "mime_type": "application/octet-stream"
            }
        }
        
        return format_info.get(format, {})


class ValidationResult:
    """Result of specification validation."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


class SpecValidator:
    """Validator for specifications with comprehensive checks."""
    
    def validate_plot_spec(self, spec: PlotSpec) -> ValidationResult:
        """Validate a PlotSpec."""
        result = ValidationResult(True)
        
        try:
            # Validate with Pydantic
            spec.model_validate(spec.model_dump())
        except Exception as e:
            result.add_error(f"Pydantic validation failed: {e}")
        
        # Custom validation rules
        self._validate_plot_spec_custom(spec, result)
        
        return result
    
    def validate_data_spec(self, spec: DataSpec) -> ValidationResult:
        """Validate a DataSpec."""
        result = ValidationResult(True)
        
        try:
            # Validate with Pydantic
            spec.model_validate(spec.model_dump())
        except Exception as e:
            result.add_error(f"Pydantic validation failed: {e}")
        
        # Custom validation rules
        self._validate_data_spec_custom(spec, result)
        
        return result
    
    def _validate_plot_spec_custom(self, spec: PlotSpec, result: ValidationResult) -> None:
        """Apply custom validation rules to PlotSpec."""
        # Check for missing required fields
        if not spec.plot_type:
            result.add_error("Plot type is required")
        
        # Check theme consistency
        if spec.styling.colors and spec.theme.colors:
            if len(spec.styling.colors) > len(spec.theme.colors):
                result.add_warning("More styling colors than theme colors")
        
        # Check axes configuration
        if spec.axes.xlim and len(spec.axes.xlim) != 2:
            result.add_error("xlim must contain exactly 2 values")
        
        if spec.axes.ylim and len(spec.axes.ylim) != 2:
            result.add_error("ylim must contain exactly 2 values")
    
    def _validate_data_spec_custom(self, spec: DataSpec, result: ValidationResult) -> None:
        """Apply custom validation rules to DataSpec."""
        # Check for duplicate column names
        column_names = [col.name for col in spec.columns]
        if len(column_names) != len(set(column_names)):
            result.add_error("Duplicate column names found")
        
        # Check mapping consistency
        column_name_set = {col.name for col in spec.columns}
        for mapping in spec.mappings:
            if mapping.source_column not in column_name_set:
                result.add_error(f"Mapping references unknown column: {mapping.source_column}")


# Global serializer instance
spec_serializer = SpecSerializer() 