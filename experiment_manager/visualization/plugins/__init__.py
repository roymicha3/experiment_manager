"""
Plugin interfaces and base classes for the visualization system.

This package provides abstract base classes and contracts that define
the interface for different types of visualization plugins.
"""

from .base import BasePlugin
from .plot_plugin import PlotPlugin
from .renderer_plugin import RendererPlugin
from .export_plugin import ExportPlugin
from .data_processor_plugin import DataProcessorPlugin
from .theme_plugin import ThemePlugin

__all__ = [
    "BasePlugin",
    "PlotPlugin", 
    "RendererPlugin",
    "ExportPlugin",
    "DataProcessorPlugin",
    "ThemePlugin",
] 