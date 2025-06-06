"""
Base plugin interface for the visualization system.

This module re-exports the BasePlugin class from the core plugin registry
and provides additional base functionality for plugin development.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Import the BasePlugin from the core registry
from ..core.plugin_registry import BasePlugin as CoreBasePlugin, PluginType


class BasePlugin(CoreBasePlugin):
    """
    Enhanced base plugin class for visualization plugins.
    
    This extends the core BasePlugin with additional functionality
    specific to visualization plugins.
    """
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin information as a dictionary.
        
        Returns:
            Dictionary containing plugin metadata
        """
        return {
            "name": self.plugin_name,
            "type": self.plugin_type.value,
            "version": self.plugin_version,
            "description": self.plugin_description,
            "dependencies": self.plugin_dependencies,
            "capabilities": self.supported_capabilities,
            "config_schema": self.config_schema,
        }
    
    def is_compatible_with(self, other_plugin: 'BasePlugin') -> bool:
        """
        Check if this plugin is compatible with another plugin.
        
        Args:
            other_plugin: Another plugin to check compatibility with
            
        Returns:
            True if plugins are compatible, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True
    
    def get_required_config_keys(self) -> List[str]:
        """
        Get list of required configuration keys.
        
        Returns:
            List of configuration keys that must be provided
        """
        # Default implementation extracts from schema if available
        if self.config_schema and "required" in self.config_schema:
            return self.config_schema["required"]
        return []
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the plugin.
        
        Returns:
            Dictionary with default configuration values
        """
        # Default implementation returns empty dict
        # Subclasses should override to provide sensible defaults
        return {}


# Re-export for convenience
__all__ = ["BasePlugin", "PluginType"] 