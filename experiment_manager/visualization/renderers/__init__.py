"""
Visualization renderer plugins.

This module contains implementations of various rendering backends
for converting plot objects to different output formats.

Available renderers:
- MatplotlibRendererPlugin: Matplotlib-based rendering with static and basic interactive plots
"""

from .matplotlib_renderer import MatplotlibRendererPlugin

__all__ = [
    'MatplotlibRendererPlugin',
] 