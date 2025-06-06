"""
Plugin interfaces and base classes for the visualization system.

This package provides abstract base classes and contracts that define
the interface for different types of visualization plugins.
"""

from experiment_manager.visualization.plugins.base import BasePlugin
from experiment_manager.visualization.plugins.plot_plugin import PlotPlugin
from experiment_manager.visualization.plugins.renderer_plugin import RendererPlugin
from experiment_manager.visualization.plugins.export_plugin import ExportPlugin
from experiment_manager.visualization.plugins.data_processor_plugin import DataProcessorPlugin
from experiment_manager.visualization.plugins.theme_plugin import ThemePlugin

__all__ = [
    "BasePlugin",
    "PlotPlugin", 
    "RendererPlugin",
    "ExportPlugin",
    "DataProcessorPlugin",
    "ThemePlugin",
] 