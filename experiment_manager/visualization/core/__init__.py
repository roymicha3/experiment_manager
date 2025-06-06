"""
Core components for the visualization system.

This module contains the foundational components that enable the plugin architecture:
    - PluginRegistry: Central plugin management and discovery
    - EventBus: Event-driven communication between components
    - ConfigManager: Configuration management with validation
"""

from .plugin_registry import PluginRegistry
from .event_bus import EventBus, Event, EventType, EventPriority, EventFilter
from .config_manager import ConfigManager, VisualizationConfig, ConfigMetadata, ConfigFormat

__all__ = [
    "PluginRegistry",
    "EventBus",
    "Event",
    "EventType", 
    "EventPriority",
    "EventFilter",
    "ConfigManager",
    "VisualizationConfig",
    "ConfigMetadata",
    "ConfigFormat",
] 