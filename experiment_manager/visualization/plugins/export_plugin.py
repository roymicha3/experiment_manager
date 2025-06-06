"""
Export plugin interface for exporting visualization data and results.

This module defines the abstract interface that all export plugins must implement.
Export plugins are responsible for exporting visualization data, configurations,
and results to various external formats and systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Iterator
from pathlib import Path
from datetime import datetime
import json

from experiment_manager.visualization.plugins.base import BasePlugin, PluginType


class ExportData:
    """
    Container for data to be exported.
    
    This class encapsulates the data, metadata, and export options
    for a visualization export operation.
    """
    
    def __init__(self,
                 data: Any,
                 data_type: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[datetime] = None):
        """
        Initialize export data container.
        
        Args:
            data: The data to be exported
            data_type: Type identifier for the data (e.g., 'plot', 'dataset', 'config')
            metadata: Optional metadata about the data
            timestamp: Optional timestamp for the data
        """
        self.data = data
        self.data_type = data_type
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()
        
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata key-value pair."""
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert export data to dictionary representation."""
        return {
            "data": self.data if isinstance(self.data, (dict, list, str, int, float, bool)) else str(self.data),
            "data_type": self.data_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class ExportOptions:
    """
    Configuration options for export operations.
    
    This class encapsulates various options that control how
    the export operation is performed.
    """
    
    def __init__(self,
                 output_path: Optional[Path] = None,
                 format_options: Optional[Dict[str, Any]] = None,
                 compression: Optional[str] = None,
                 include_metadata: bool = True,
                 include_timestamp: bool = True,
                 overwrite: bool = False,
                 create_backup: bool = False):
        """
        Initialize export options.
        
        Args:
            output_path: Path where to save exported data
            format_options: Format-specific options
            compression: Compression method to use ('gzip', 'zip', etc.)
            include_metadata: Whether to include metadata in export
            include_timestamp: Whether to include timestamp information
            overwrite: Whether to overwrite existing files
            create_backup: Whether to create backup of existing files
        """
        self.output_path = output_path
        self.format_options = format_options or {}
        self.compression = compression
        self.include_metadata = include_metadata
        self.include_timestamp = include_timestamp
        self.overwrite = overwrite
        self.create_backup = create_backup


class ExportResult:
    """
    Container for export operation results.
    
    This class encapsulates the result of an export operation,
    including success status, output location, and metadata.
    """
    
    def __init__(self,
                 success: bool,
                 output_path: Optional[Path] = None,
                 exported_items: int = 0,
                 file_size: Optional[int] = None,
                 export_format: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 error_message: Optional[str] = None,
                 warnings: Optional[List[str]] = None):
        """
        Initialize export result.
        
        Args:
            success: Whether the export was successful
            output_path: Path where data was exported
            exported_items: Number of items successfully exported
            file_size: Size of exported file in bytes
            export_format: Format used for export
            metadata: Optional metadata about the export
            error_message: Error message if export failed
            warnings: List of warning messages
        """
        self.success = success
        self.output_path = output_path
        self.exported_items = exported_items
        self.file_size = file_size
        self.export_format = export_format
        self.metadata = metadata or {}
        self.error_message = error_message
        self.warnings = warnings or []
        self.timestamp = datetime.now()
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert export result to dictionary."""
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "exported_items": self.exported_items,
            "file_size": self.file_size,
            "export_format": self.export_format,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }


class ExportPlugin(BasePlugin):
    """
    Abstract base class for export plugins.
    
    Export plugins handle exporting visualization data, configurations,
    and results to various external formats and systems. This includes
    saving to files, uploading to cloud services, sending to databases, etc.
    """
    
    @property
    def plugin_type(self) -> PluginType:
        """Export plugins always return EXPORTER type."""
        return PluginType.EXPORTER
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """
        List of export formats this plugin supports.
        
        Returns:
            List of format identifiers (e.g., ['json', 'csv', 'parquet', 'hdf5'])
        """
        pass
    
    @property
    @abstractmethod
    def supported_data_types(self) -> List[str]:
        """
        List of data types this plugin can export.
        
        Returns:
            List of data type identifiers (e.g., ['plot', 'dataset', 'config', 'results'])
        """
        pass
    
    @property
    def requires_authentication(self) -> bool:
        """
        Whether this export plugin requires authentication.
        
        Returns:
            True if plugin needs authentication credentials
        """
        return False
    
    @property
    def supports_streaming(self) -> bool:
        """
        Whether this plugin supports streaming exports.
        
        Returns:
            True if plugin can handle streaming data
        """
        return False
    
    @property
    def supports_compression(self) -> bool:
        """
        Whether this plugin supports data compression.
        
        Returns:
            True if plugin can compress exported data
        """
        return False
    
    @abstractmethod
    def can_export(self, data: ExportData, export_format: str) -> bool:
        """
        Check if this plugin can export the given data in the specified format.
        
        Args:
            data: Data to check for export compatibility
            export_format: Target export format
            
        Returns:
            True if plugin can handle this export operation
        """
        pass
    
    @abstractmethod
    def export(self,
               data: Union[ExportData, List[ExportData]],
               options: ExportOptions,
               config: Optional[Dict[str, Any]] = None) -> ExportResult:
        """
        Export data using the specified options.
        
        Args:
            data: Data to export (single item or list of items)
            options: Export options and configuration
            config: Optional plugin-specific configuration
            
        Returns:
            ExportResult containing information about the export operation
            
        Raises:
            ValueError: If data or options are invalid
            RuntimeError: If export operation fails
        """
        pass
    
    def export_batch(self,
                    data_items: List[ExportData],
                    options: ExportOptions,
                    config: Optional[Dict[str, Any]] = None) -> List[ExportResult]:
        """
        Export multiple data items in batch.
        
        Args:
            data_items: List of data items to export
            options: Export options
            config: Optional plugin-specific configuration
            
        Returns:
            List of ExportResult objects, one for each item
        """
        results = []
        for item in data_items:
            try:
                result = self.export(item, options, config)
                results.append(result)
            except Exception as e:
                error_result = ExportResult(
                    success=False,
                    error_message=str(e),
                    export_format=options.format_options.get('format', 'unknown')
                )
                results.append(error_result)
        return results
    
    def validate_options(self, options: ExportOptions) -> bool:
        """
        Validate export options for this plugin.
        
        Args:
            options: Export options to validate
            
        Returns:
            True if options are valid for this plugin
        """
        # Default implementation performs basic validation
        if not options.output_path and not self.supports_streaming:
            return False
        return True
    
    def get_format_info(self, export_format: str) -> Dict[str, Any]:
        """
        Get information about a supported export format.
        
        Args:
            export_format: Format to get information about
            
        Returns:
            Dictionary with format information
        """
        if export_format not in self.supported_formats:
            raise ValueError(f"Format '{export_format}' not supported by this exporter")
            
        return {
            "format": export_format,
            "description": f"Export data in {export_format.upper()} format",
            "supports_compression": self.supports_compression,
            "supports_streaming": self.supports_streaming,
        }
    
    def estimate_export_size(self, 
                           data: Union[ExportData, List[ExportData]],
                           options: ExportOptions) -> Optional[int]:
        """
        Estimate the size of exported data in bytes.
        
        Args:
            data: Data to estimate export size for
            options: Export options
            
        Returns:
            Estimated size in bytes, or None if cannot estimate
        """
        # Default implementation cannot estimate
        return None
    
    def get_export_preview(self,
                          data: ExportData,
                          options: ExportOptions,
                          max_items: int = 10) -> str:
        """
        Get a preview of how the data would be exported.
        
        Args:
            data: Data to preview
            options: Export options
            max_items: Maximum number of items to include in preview
            
        Returns:
            String representation of export preview
        """
        # Default implementation returns JSON preview
        preview_data = data.to_dict()
        if isinstance(preview_data.get('data'), list) and len(preview_data['data']) > max_items:
            preview_data['data'] = preview_data['data'][:max_items] + [f"... and {len(preview_data['data']) - max_items} more items"]
        
        return json.dumps(preview_data, indent=2, default=str) 