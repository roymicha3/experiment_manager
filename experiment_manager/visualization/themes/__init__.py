"""
Theme Registry and Management System

This module provides centralized theme management for the visualization system,
including built-in themes, custom theme creation, inheritance, and runtime switching.
"""

from experiment_manager.visualization.themes.registry import ThemeRegistry
from experiment_manager.visualization.themes.builtin_themes import (
    DefaultThemePlugin,
    DarkThemePlugin, 
    PublicationThemePlugin
)

__all__ = [
    "ThemeRegistry",
    "DefaultThemePlugin",
    "DarkThemePlugin",
    "PublicationThemePlugin",
] 