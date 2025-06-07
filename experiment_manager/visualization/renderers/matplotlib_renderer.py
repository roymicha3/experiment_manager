"""
Matplotlib renderer plugin for the visualization system.

This module provides a comprehensive matplotlib-based renderer that supports
various plot types, themes, and output formats.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

from ..plugins.renderer_plugin import RendererPlugin, RenderContext, RenderResult
from ..themes.registry import ThemeRegistry
from ..plugins.theme_plugin import ThemeConfig

logger = logging.getLogger(__name__)


class MatplotlibRendererPlugin(RendererPlugin):
    """
    Matplotlib-based renderer plugin supporting various plot types and themes.
    
    Features:
    - Multiple plot types (line, scatter, bar, histogram, box, heatmap)
    - Theme integration with custom styling
    - Multiple output formats (PNG, SVG, PDF, EPS, TIFF)
    - High-quality output with configurable DPI
    - Memory-efficient rendering with proper cleanup
    """
    
    def __init__(self, theme_registry: Optional[ThemeRegistry] = None):
        """Initialize the matplotlib renderer plugin."""
        self.theme_registry = theme_registry or ThemeRegistry()
        self._current_theme: Optional[ThemeConfig] = None
        self._figure_cache: Dict[str, Figure] = {}
        
        # Default configuration
        self.default_dpi = 300
        self.default_figsize = (10, 6)
        self.supported_formats_set = {'png', 'svg', 'pdf', 'eps', 'tiff'}
        
        logger.info("MatplotlibRendererPlugin initialized")
    
    @property
    def plugin_name(self) -> str:
        """Get the plugin name."""
        return "matplotlib"
    
    @property 
    def name(self) -> str:
        """Get the renderer name."""
        return "matplotlib"
    
    @property
    def supported_formats(self) -> List[str]:
        """Get supported output formats."""
        return list(self.supported_formats_set)
    
    @property
    def supported_plot_types(self) -> List[str]:
        """Get supported plot types."""
        return [
            'line', 'scatter', 'bar', 'histogram', 'box', 'heatmap',
            'area', 'pie', 'violin', 'density'
        ]
    
    def set_theme(self, theme: ThemeConfig) -> None:
        """Set the current theme for rendering."""
        self._current_theme = theme
        self._apply_theme_to_matplotlib()
        logger.debug(f"Theme set to: {theme.name}")
    
    def _apply_theme_to_matplotlib(self) -> None:
        """Apply current theme settings to matplotlib."""
        if not self._current_theme:
            return
        
        # Apply matplotlib style parameters using ThemeConfig
        style_params = {
            'figure.facecolor': self._current_theme.get_color('background', '#ffffff'),
            'axes.facecolor': self._current_theme.get_color('background', '#ffffff'),
            'axes.edgecolor': self._current_theme.get_color('text', '#000000'),
            'axes.labelcolor': self._current_theme.get_color('text', '#000000'),
            'text.color': self._current_theme.get_color('text', '#000000'),
            'xtick.color': self._current_theme.get_color('text', '#000000'),
            'ytick.color': self._current_theme.get_color('text', '#000000'),
            'grid.color': self._current_theme.get_color('grid', '#cccccc'),
            'grid.alpha': 0.3,  # Default grid alpha
        }
        
        # Apply font settings
        font_family = self._current_theme.get_font('family', 'sans-serif')
        font_size = self._current_theme.fonts.get('size', 10)
        
        style_params['font.family'] = font_family
        style_params['font.size'] = font_size
        
        plt.rcParams.update(style_params)
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the renderer with configuration."""
        # Apply any configuration settings
        if 'default_dpi' in config:
            self.default_dpi = config['default_dpi']
        if 'default_figsize' in config:
            self.default_figsize = tuple(config['default_figsize'])
        
        logger.info(f"MatplotlibRendererPlugin initialized with config: {config}")
    
    def can_render(self, plot_object: Any, output_format: str) -> bool:
        """Check if this renderer can handle the given plot and format."""
        # For matplotlib renderer, we accept plot specifications (dicts) and matplotlib figures
        if isinstance(plot_object, dict):
            plot_type = plot_object.get('type', 'line')
            return plot_type in self.supported_plot_types and output_format in self.supported_formats_set
        
        # Check if it's a matplotlib figure
        try:
            from matplotlib.figure import Figure
            if isinstance(plot_object, Figure):
                return output_format in self.supported_formats_set
        except ImportError:
            pass
        
        return False
    
    def render(self, 
               plot_object: Any,
               context: Optional[RenderContext] = None,
               config: Optional[Dict[str, Any]] = None,
               output_path: Optional[str] = None) -> Union[RenderResult, bytes, str]:
        """
        Render a plot object to the specified format.
        
        Args:
            plot_object: Plot specification dict or matplotlib figure
            context: Rendering context and options (optional for backward compatibility)
            config: Optional renderer-specific configuration
            output_path: Optional output path for backward compatibility
            
        Returns:
            RenderResult if context provided, otherwise bytes/str for backward compatibility
        """
        # Backward compatibility: if no context provided, use legacy behavior
        if context is None:
            if isinstance(plot_object, dict):
                return self.render_legacy(plot_object, output_path)
            else:
                raise ValueError("Context required for non-dict plot objects")
        
        try:
            # Handle different plot object types
            if isinstance(plot_object, dict):
                # Plot specification dictionary
                return self._render_from_spec(plot_object, context, config)
            else:
                # Assume matplotlib figure
                return self._render_figure(plot_object, context, config)
        except Exception as e:
            logger.error(f"Error rendering plot: {e}")
            return RenderResult(
                data=b"",
                context=context,
                success=False,
                error_message=str(e)
            )
    
    def _render_from_spec(self, plot_spec: Dict[str, Any], context: RenderContext, config: Optional[Dict[str, Any]]) -> RenderResult:
        """Render from plot specification dictionary."""
        try:
            # Extract plot configuration
            plot_type = plot_spec.get('type', 'line')
            data = plot_spec.get('data', {})
            plot_config = plot_spec.get('config', {})
            
            # Merge with renderer config
            if config:
                plot_config.update(config)
            
            # Create figure
            figsize = plot_config.get('figsize', self.default_figsize)
            dpi = context.dpi or plot_config.get('dpi', self.default_dpi)
            
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Apply theme if available
            if self._current_theme:
                self._apply_theme_to_matplotlib()
            
            # Render based on plot type
            self._render_plot_type(ax, plot_type, data, plot_config)
            
            # Apply styling and labels
            self._apply_plot_styling(ax, plot_config)
            
            # Convert to output format
            return self._figure_to_result(fig, context, plot_config)
                
        except Exception as e:
            logger.error(f"Error rendering plot from spec: {e}")
            raise
        finally:
            plt.close('all')  # Clean up memory
    
    def _render_figure(self, figure: Any, context: RenderContext, config: Optional[Dict[str, Any]]) -> RenderResult:
        """Render existing matplotlib figure."""
        try:
            plot_config = config or {}
            return self._figure_to_result(figure, context, plot_config)
        except Exception as e:
            logger.error(f"Error rendering matplotlib figure: {e}")
            raise
    
    def _figure_to_result(self, fig: Any, context: RenderContext, config: Dict[str, Any]) -> RenderResult:
        """Convert matplotlib figure to RenderResult."""
        # Get format-specific settings
        save_kwargs = self._get_format_kwargs(context.output_format, config)
        save_kwargs['dpi'] = context.dpi
        
        if context.output_path:
            # Save to file
            context.output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(context.output_path, format=context.output_format, **save_kwargs)
            
            # Read file back as bytes for result
            with open(context.output_path, 'rb') as f:
                data = f.read()
            
            logger.info(f"Plot saved to: {context.output_path}")
        else:
            # Save to bytes
            buffer = io.BytesIO()
            fig.savefig(buffer, format=context.output_format, **save_kwargs)
            buffer.seek(0)
            data = buffer.getvalue()
        
        return RenderResult(
            data=data,
            context=context,
            metadata={
                'format': context.output_format,
                'size': len(data),
                'dpi': context.dpi
            },
            success=True
        )
    
    def render_legacy(self, plot_spec: Dict[str, Any], output_path: Optional[str] = None) -> Union[str, bytes]:
        """
        Render a plot based on the specification.
        
        Args:
            plot_spec: Plot specification dictionary
            output_path: Optional output file path
            
        Returns:
            File path if output_path provided, otherwise bytes data
        """
        try:
            # Extract plot configuration
            plot_type = plot_spec.get('type', 'line')
            data = plot_spec.get('data', {})
            config = plot_spec.get('config', {})
            
            # Create figure
            figsize = config.get('figsize', self.default_figsize)
            dpi = config.get('dpi', self.default_dpi)
            
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Apply theme if available
            if self._current_theme:
                self._apply_theme_to_matplotlib()
            
            # Render based on plot type
            self._render_plot_type(ax, plot_type, data, config)
            
            # Apply styling and labels
            self._apply_plot_styling(ax, config)
            
            # Handle output
            if output_path:
                return self._save_to_file(fig, output_path, config)
            else:
                return self._save_to_bytes(fig, config)
                
        except Exception as e:
            logger.error(f"Error rendering plot: {e}")
            raise
        finally:
            plt.close('all')  # Clean up memory
    
    def _render_plot_type(self, ax: Axes, plot_type: str, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render specific plot type."""
        if plot_type == 'line':
            self._render_line_plot(ax, data, config)
        elif plot_type == 'scatter':
            self._render_scatter_plot(ax, data, config)
        elif plot_type == 'bar':
            self._render_bar_plot(ax, data, config)
        elif plot_type == 'histogram':
            self._render_histogram(ax, data, config)
        elif plot_type == 'box':
            self._render_box_plot(ax, data, config)
        elif plot_type == 'heatmap':
            self._render_heatmap(ax, data, config)
        elif plot_type == 'area':
            self._render_area_plot(ax, data, config)
        elif plot_type == 'pie':
            self._render_pie_chart(ax, data, config)
        elif plot_type == 'violin':
            self._render_violin_plot(ax, data, config)
        elif plot_type == 'density':
            self._render_density_plot(ax, data, config)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def _render_line_plot(self, ax: Axes, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render line plot."""
        x = data.get('x', [])
        y = data.get('y', [])
        
        # Handle multiple series
        if isinstance(y, dict):
            for label, y_values in y.items():
                ax.plot(x, y_values, label=label, **config.get('line_style', {}))
            ax.legend()
        else:
            ax.plot(x, y, **config.get('line_style', {}))
    
    def _render_scatter_plot(self, ax: Axes, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render scatter plot."""
        x = data.get('x', [])
        y = data.get('y', [])
        
        scatter_config = config.get('scatter_style', {})
        
        # Handle color mapping
        c = data.get('color', scatter_config.get('c', None))
        s = data.get('size', scatter_config.get('s', 50))
        
        ax.scatter(x, y, c=c, s=s, **{k: v for k, v in scatter_config.items() if k not in ['c', 's']})
    
    def _render_bar_plot(self, ax: Axes, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render bar plot."""
        x = data.get('x', [])
        y = data.get('y', [])
        
        orientation = config.get('orientation', 'vertical')
        bar_config = config.get('bar_style', {})
        
        if orientation == 'horizontal':
            ax.barh(x, y, **bar_config)
        else:
            ax.bar(x, y, **bar_config)
    
    def _render_histogram(self, ax: Axes, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render histogram."""
        values = data.get('values', [])
        bins = config.get('bins', 'auto')
        hist_config = config.get('hist_style', {})
        
        ax.hist(values, bins=bins, **hist_config)
    
    def _render_box_plot(self, ax: Axes, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render box plot."""
        values = data.get('values', [])
        box_config = config.get('box_style', {})
        
        if isinstance(values, dict):
            # Multiple box plots
            ax.boxplot(list(values.values()), labels=list(values.keys()), **box_config)
        else:
            ax.boxplot(values, **box_config)
    
    def _render_heatmap(self, ax: Axes, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render heatmap."""
        matrix = data.get('matrix', [])
        heatmap_config = config.get('heatmap_style', {})
        
        # Convert to numpy array if needed
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        
        im = ax.imshow(matrix, **heatmap_config)
        
        # Add colorbar if requested
        if config.get('colorbar', True):
            plt.colorbar(im, ax=ax)
    
    def _render_area_plot(self, ax: Axes, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render area plot."""
        x = data.get('x', [])
        y = data.get('y', [])
        area_config = config.get('area_style', {})
        
        if isinstance(y, dict):
            # Stacked area plot
            bottom = np.zeros(len(x))
            for label, y_values in y.items():
                ax.fill_between(x, bottom, bottom + y_values, label=label, **area_config)
                bottom += y_values
            ax.legend()
        else:
            ax.fill_between(x, y, **area_config)
    
    def _render_pie_chart(self, ax: Axes, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render pie chart."""
        values = data.get('values', [])
        labels = data.get('labels', [])
        pie_config = config.get('pie_style', {})
        
        ax.pie(values, labels=labels, **pie_config)
    
    def _render_violin_plot(self, ax: Axes, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render violin plot."""
        values = data.get('values', [])
        violin_config = config.get('violin_style', {})
        
        if isinstance(values, dict):
            ax.violinplot(list(values.values()), **violin_config)
            ax.set_xticks(range(1, len(values) + 1))
            ax.set_xticklabels(list(values.keys()))
        else:
            ax.violinplot(values, **violin_config)
    
    def _render_density_plot(self, ax: Axes, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Render density plot."""
        values = data.get('values', [])
        density_config = config.get('density_style', {})
        
        # Use pandas for density estimation if available
        try:
            import pandas as pd
            series = pd.Series(values)
            series.plot.density(ax=ax, **density_config)
        except ImportError:
            # Fallback to histogram with density=True
            ax.hist(values, density=True, alpha=0.7, **density_config)
    
    def _apply_plot_styling(self, ax: Axes, config: Dict[str, Any]) -> None:
        """Apply general plot styling."""
        # Set labels
        if 'xlabel' in config:
            ax.set_xlabel(config['xlabel'])
        if 'ylabel' in config:
            ax.set_ylabel(config['ylabel'])
        if 'title' in config:
            ax.set_title(config['title'])
        
        # Set limits
        if 'xlim' in config:
            ax.set_xlim(config['xlim'])
        if 'ylim' in config:
            ax.set_ylim(config['ylim'])
        
        # Grid
        if config.get('grid', False):
            ax.grid(True, alpha=config.get('grid_alpha', 0.3))
        
        # Tight layout (skip for multiple subplot rendering to avoid conflicts)
        if config.get('tight_layout', True) and not config.get('_skip_tight_layout', False):
            plt.tight_layout()
    
    def _save_to_file(self, fig: Figure, output_path: str, config: Dict[str, Any]) -> str:
        """Save figure to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get format from file extension or config
        format_type = config.get('format', path.suffix.lstrip('.').lower())
        if not format_type or format_type not in self.supported_formats_set:
            format_type = 'png'
        
        # Format-specific settings
        save_kwargs = self._get_format_kwargs(format_type, config)
        
        fig.savefig(output_path, format=format_type, **save_kwargs)
        logger.info(f"Plot saved to: {output_path}")
        return output_path
    
    def _save_to_bytes(self, fig: Figure, config: Dict[str, Any]) -> bytes:
        """Save figure to bytes."""
        format_type = config.get('format', 'png')
        if format_type not in self.supported_formats_set:
            format_type = 'png'
        
        save_kwargs = self._get_format_kwargs(format_type, config)
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format_type, **save_kwargs)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _get_format_kwargs(self, format_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get format-specific save arguments."""
        kwargs = {
            'dpi': config.get('dpi', self.default_dpi),
            'bbox_inches': config.get('bbox_inches', 'tight'),
            'facecolor': config.get('facecolor', 'white'),
            'edgecolor': config.get('edgecolor', 'none'),
        }
        
        # Format-specific optimizations
        if format_type == 'png':
            # PNG-specific settings supported by matplotlib
            kwargs.update({
                'transparent': config.get('transparent', False),
            })
        elif format_type == 'svg':
            kwargs.update({
                'transparent': True,
            })
        elif format_type == 'pdf':
            kwargs.update({
                'metadata': {
                    'Creator': 'Experiment Manager Visualization System',
                    'Producer': 'matplotlib',
                }
            })
        
        return kwargs
    
    def render_multiple(self, plot_specs: List[Dict[str, Any]], layout: Optional[Dict[str, Any]] = None) -> Union[str, bytes]:
        """
        Render multiple plots in a single figure.
        
        Args:
            plot_specs: List of plot specifications
            layout: Layout configuration (rows, cols, etc.)
            
        Returns:
            Rendered output
        """
        try:
            # Determine layout
            n_plots = len(plot_specs)
            if layout:
                rows = layout.get('rows', 1)
                cols = layout.get('cols', n_plots)
            else:
                # Auto-determine layout
                cols = min(3, n_plots)
                rows = (n_plots + cols - 1) // cols
            
            # Create figure with subplots
            figsize = layout.get('figsize', (5 * cols, 4 * rows)) if layout else (5 * cols, 4 * rows)
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            
            # Ensure axes is always a list
            if n_plots == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                # Handle both numpy arrays and lists
                if hasattr(axes, 'flatten'):
                    axes = axes.flatten()
                elif isinstance(axes, (list, tuple)):
                    axes = list(axes)
                else:
                    axes = [axes]
            else:
                # Handle both numpy arrays and lists for 2D case
                if hasattr(axes, 'flatten'):
                    axes = axes.flatten()
                elif isinstance(axes, (list, tuple)):
                    # Flatten nested list
                    flat_axes = []
                    for row in axes:
                        if isinstance(row, (list, tuple)):
                            flat_axes.extend(row)
                        else:
                            flat_axes.append(row)
                    axes = flat_axes
                else:
                    axes = [axes]
            
            # Apply theme
            if self._current_theme:
                self._apply_theme_to_matplotlib()
            
            # Render each plot
            for i, plot_spec in enumerate(plot_specs):
                if i < len(axes):
                    ax = axes[i]
                    plot_type = plot_spec.get('type', 'line')
                    data = plot_spec.get('data', {})
                    config = plot_spec.get('config', {}).copy()
                    
                    # Skip tight_layout in individual subplot styling
                    config['_skip_tight_layout'] = True
                    
                    self._render_plot_type(ax, plot_type, data, config)
                    self._apply_plot_styling(ax, config)
            
            # Hide unused subplots
            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Handle output
            output_config = layout.get('output', {}) if layout else {}
            output_path = output_config.get('path')
            
            if output_path:
                return self._save_to_file(fig, output_path, output_config)
            else:
                return self._save_to_bytes(fig, output_config)
                
        except Exception as e:
            logger.error(f"Error rendering multiple plots: {e}")
            raise
        finally:
            plt.close('all')
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get renderer capabilities."""
        return {
            'name': self.name,
            'plot_types': self.supported_plot_types,
            'output_formats': list(self.supported_formats_set),
            'features': [
                'themes',
                'multiple_plots',
                'high_quality_output',
                'custom_styling',
                'interactive_elements',
                'statistical_plots'
            ],
            'version': '1.0.0'
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        plt.close('all')
        self._figure_cache.clear()
        logger.debug("MatplotlibRendererPlugin cleaned up") 