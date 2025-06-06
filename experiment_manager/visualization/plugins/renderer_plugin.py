"""
Renderer plugin interface for rendering plots in different formats.

This module defines the abstract interface that all renderer plugins must implement.
Renderer plugins are responsible for taking plot objects and rendering them
to specific output formats (HTML, PNG, SVG, PDF, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, BinaryIO, TextIO
from pathlib import Path
import io

from .base import BasePlugin, PluginType


class RenderContext:
    """
    Context information for rendering operations.
    
    This class encapsulates information about the rendering environment
    and target output format.
    """
    
    def __init__(self,
                 output_format: str,
                 output_path: Optional[Path] = None,
                 dpi: Optional[int] = None,
                 quality: Optional[int] = None,
                 interactive: bool = False,
                 embed_data: bool = False):
        """
        Initialize render context.
        
        Args:
            output_format: Target output format (e.g., 'png', 'svg', 'html', 'pdf')
            output_path: Optional path where output should be saved
            dpi: Dots per inch for raster formats
            quality: Quality setting (0-100) for lossy formats
            interactive: Whether to include interactive features
            embed_data: Whether to embed data in the output
        """
        self.output_format = output_format.lower()
        self.output_path = output_path
        self.dpi = dpi or 300
        self.quality = quality or 90
        self.interactive = interactive
        self.embed_data = embed_data
        
    def get_mime_type(self) -> str:
        """Get MIME type for the output format."""
        mime_types = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'svg': 'image/svg+xml',
            'pdf': 'application/pdf',
            'html': 'text/html',
            'json': 'application/json',
        }
        return mime_types.get(self.output_format, 'application/octet-stream')


class RenderResult:
    """
    Container for rendering results.
    
    This class encapsulates the result of a rendering operation,
    including the rendered data and metadata.
    """
    
    def __init__(self,
                 data: Union[bytes, str],
                 context: RenderContext,
                 metadata: Optional[Dict[str, Any]] = None,
                 success: bool = True,
                 error_message: Optional[str] = None):
        """
        Initialize render result.
        
        Args:
            data: Rendered data (bytes for binary formats, str for text)
            context: Render context used for this result
            metadata: Optional metadata about the rendering
            success: Whether rendering was successful
            error_message: Error message if rendering failed
        """
        self.data = data
        self.context = context
        self.metadata = metadata or {}
        self.success = success
        self.error_message = error_message
    
    @property
    def size(self) -> int:
        """Get size of rendered data in bytes."""
        if isinstance(self.data, bytes):
            return len(self.data)
        elif isinstance(self.data, str):
            return len(self.data.encode('utf-8'))
        return 0
    
    def save_to_file(self, path: Path) -> None:
        """
        Save rendered data to file.
        
        Args:
            path: Path where to save the file
        """
        if isinstance(self.data, bytes):
            with open(path, 'wb') as f:
                f.write(self.data)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.data)
    
    def get_stream(self) -> Union[BinaryIO, TextIO]:
        """Get stream containing the rendered data."""
        if isinstance(self.data, bytes):
            return io.BytesIO(self.data)
        else:
            return io.StringIO(self.data)


class RendererPlugin(BasePlugin):
    """
    Abstract base class for renderer plugins.
    
    Renderer plugins take plot objects and convert them to specific output formats.
    Each plugin should implement rendering for one or more output formats
    and handle format-specific options and optimizations.
    """
    
    @property
    def plugin_type(self) -> PluginType:
        """Renderer plugins always return RENDERER type."""
        return PluginType.RENDERER
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """
        List of output formats this renderer supports.
        
        Returns:
            List of format identifiers (e.g., ['png', 'svg', 'pdf'])
        """
        pass
    
    @property
    @abstractmethod
    def supported_plot_types(self) -> List[str]:
        """
        List of plot object types this renderer can handle.
        
        Returns:
            List of plot type identifiers (e.g., ['matplotlib', 'plotly', 'bokeh'])
        """
        pass
    
    @property
    def default_format(self) -> str:
        """
        Default output format for this renderer.
        
        Returns:
            Default format identifier
        """
        return self.supported_formats[0] if self.supported_formats else "png"
    
    @property
    def supports_interactive(self) -> bool:
        """
        Whether this renderer supports interactive output.
        
        Returns:
            True if renderer can produce interactive output
        """
        return False
    
    @abstractmethod
    def can_render(self, plot_object: Any, output_format: str) -> bool:
        """
        Check if this renderer can handle the given plot and format.
        
        Args:
            plot_object: Plot object to check
            output_format: Target output format
            
        Returns:
            True if renderer can handle this combination
        """
        pass
    
    @abstractmethod
    def render(self, 
               plot_object: Any,
               context: RenderContext,
               config: Optional[Dict[str, Any]] = None) -> RenderResult:
        """
        Render a plot object to the specified format.
        
        Args:
            plot_object: Plot object to render
            context: Rendering context and options
            config: Optional renderer-specific configuration
            
        Returns:
            RenderResult containing the rendered output
            
        Raises:
            ValueError: If plot object or format is not supported
            RuntimeError: If rendering fails
        """
        pass
    
    def get_format_info(self, output_format: str) -> Dict[str, Any]:
        """
        Get information about a supported format.
        
        Args:
            output_format: Format to get info about
            
        Returns:
            Dictionary with format information
        """
        if output_format not in self.supported_formats:
            raise ValueError(f"Format '{output_format}' not supported by this renderer")
            
        return {
            "format": output_format,
            "mime_type": RenderContext(output_format).get_mime_type(),
            "is_binary": output_format in ['png', 'jpg', 'jpeg', 'pdf'],
            "supports_transparency": output_format in ['png', 'svg'],
            "supports_animation": output_format in ['gif', 'html'],
        }
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for this renderer.
        
        Returns:
            Dictionary with default renderer settings
        """
        return {
            "dpi": 300,
            "quality": 90,
            "optimize": True,
            "embed_fonts": True,
        }
    
    def estimate_output_size(self, 
                           plot_object: Any,
                           context: RenderContext) -> Optional[int]:
        """
        Estimate the size of rendered output in bytes.
        
        Args:
            plot_object: Plot object to estimate for
            context: Rendering context
            
        Returns:
            Estimated size in bytes, or None if cannot estimate
        """
        # Default implementation cannot estimate
        return None
    
    def validate_context(self, context: RenderContext) -> bool:
        """
        Validate that the render context is supported.
        
        Args:
            context: Render context to validate
            
        Returns:
            True if context is valid for this renderer
        """
        return context.output_format in self.supported_formats 