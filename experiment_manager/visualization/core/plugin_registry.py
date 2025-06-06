"""
Plugin Registry System for Visualization Components

This module provides centralized plugin management, discovery, and lifecycle
management for the visualization system. It implements a registry pattern
that allows dynamic loading and registration of plugins.
"""

import logging
import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Enumeration of supported plugin types."""
    PLOT = "plot"
    RENDERER = "renderer" 
    EXPORTER = "exporter"
    DATA_PROCESSOR = "data_processor"
    THEME = "theme"


@dataclass
class PluginInfo:
    """Information about a registered plugin."""
    name: str
    plugin_type: PluginType
    plugin_class: Type
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    is_enabled: bool = True
    
    @property
    def full_name(self) -> str:
        """Get the full plugin name including type."""
        return f"{self.plugin_type.value}.{self.name}"


class BasePlugin(ABC):
    """
    Abstract base class for all visualization plugins.
    
    All plugins must inherit from this class and implement the required methods.
    The plugin system uses this interface to manage plugin lifecycle and capabilities.
    """
    
    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """The type of plugin (plot, renderer, exporter, etc.)."""
        pass
        
    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Unique name for the plugin within its type."""
        pass
    
    @property
    def plugin_version(self) -> str:
        """Version of the plugin."""
        return "1.0.0"
    
    @property
    def plugin_description(self) -> str:
        """Description of the plugin functionality."""
        return ""
    
    @property
    def plugin_dependencies(self) -> List[str]:
        """List of plugin names this plugin depends on."""
        return []
        
    @property
    def supported_capabilities(self) -> List[str]:
        """List of capabilities this plugin supports."""
        return []
    
    @property
    def config_schema(self) -> Optional[Dict[str, Any]]:
        """JSON schema for plugin configuration."""
        return None
        
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin-specific configuration dictionary
            
        Raises:
            PluginInitializationError: If initialization fails
        """
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup plugin resources.
        
        Called when the plugin is being unloaded or the system is shutting down.
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # Default implementation always returns True
        # Subclasses can override for custom validation
        return True


class PluginRegistryError(Exception):
    """Base exception for plugin registry errors."""
    pass


class PluginNotFoundError(PluginRegistryError):
    """Raised when a requested plugin is not found."""
    pass


class PluginDependencyError(PluginRegistryError):
    """Raised when plugin dependencies cannot be resolved."""
    pass


class PluginInitializationError(PluginRegistryError):
    """Raised when plugin initialization fails."""
    pass


class PluginRegistry:
    """
    Central registry for managing visualization plugins.
    
    The registry provides:
    - Plugin discovery from modules and directories
    - Plugin registration and validation
    - Dependency resolution
    - Plugin lifecycle management
    - Plugin querying and filtering
    
    Example:
        ```python
        registry = PluginRegistry()
        
        # Discover plugins from package
        registry.discover_plugins("experiment_manager.visualization.plugins")
        
        # Register a plugin manually
        registry.register_plugin(MyCustomPlugin)
        
        # Get all plot plugins
        plot_plugins = registry.get_plugins_by_type(PluginType.PLOT)
        
        # Create plugin instance
        plugin = registry.create_plugin("plot.training_curves", config={})
        ```
    """
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, PluginInfo] = {}
        self._instances: Dict[str, BasePlugin] = {}
        self._discovery_callbacks: List[Callable[[PluginInfo], None]] = []
        self._initialization_order: List[str] = []
        
    def register_plugin(self, plugin_class: Type[BasePlugin], 
                       override: bool = False) -> None:
        """
        Register a plugin class with the registry.
        
        Args:
            plugin_class: Plugin class to register
            override: Whether to override existing plugin with same name
            
        Raises:
            PluginRegistryError: If plugin is invalid or already exists
        """
        if not self._is_valid_plugin_class(plugin_class):
            raise PluginRegistryError(
                f"Invalid plugin class: {plugin_class}. "
                "Must inherit from BasePlugin and implement required methods."
            )
        
        # Create temporary instance to get plugin info
        try:
            temp_instance = plugin_class()
            plugin_info = PluginInfo(
                name=temp_instance.plugin_name,
                plugin_type=temp_instance.plugin_type,
                plugin_class=plugin_class,
                version=temp_instance.plugin_version,
                description=temp_instance.plugin_description,
                dependencies=temp_instance.plugin_dependencies,
                capabilities=temp_instance.supported_capabilities,
                config_schema=temp_instance.config_schema
            )
        except Exception as e:
            raise PluginRegistryError(
                f"Failed to inspect plugin {plugin_class}: {e}"
            )
        
        full_name = plugin_info.full_name
        
        if full_name in self._plugins and not override:
            raise PluginRegistryError(
                f"Plugin '{full_name}' is already registered. "
                "Use override=True to replace."
            )
        
        self._plugins[full_name] = plugin_info
        logger.info(f"Registered plugin: {full_name} v{plugin_info.version}")
        
        # Notify discovery callbacks
        for callback in self._discovery_callbacks:
            callback(plugin_info)
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """
        Unregister a plugin and cleanup its instance.
        
        Args:
            plugin_name: Full name of plugin to unregister
            
        Raises:
            PluginNotFoundError: If plugin is not registered
        """
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{plugin_name}' not found")
        
        # Cleanup instance if it exists
        if plugin_name in self._instances:
            try:
                self._instances[plugin_name].cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up plugin {plugin_name}: {e}")
            del self._instances[plugin_name]
        
        del self._plugins[plugin_name]
        logger.info(f"Unregistered plugin: {plugin_name}")
    
    def discover_plugins(self, package_name: str, recursive: bool = True) -> int:
        """
        Discover and register plugins from a package.
        
        Args:
            package_name: Name of package to search for plugins
            recursive: Whether to search subpackages recursively
            
        Returns:
            Number of plugins discovered and registered
        """
        discovered_count = 0
        
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.warning(f"Could not import package {package_name}: {e}")
            return discovered_count
        
        # Get package path
        if hasattr(package, '__path__'):
            package_paths = package.__path__
        else:
            logger.warning(f"Package {package_name} has no __path__ attribute")
            return discovered_count
        
        # Walk through modules in package
        for importer, modname, ispkg in pkgutil.walk_packages(
            package_paths, 
            prefix=f"{package_name}.",
            onerror=lambda x: None
        ):
            if not recursive and '.' in modname.replace(f"{package_name}.", ""):
                continue
                
            try:
                module = importlib.import_module(modname)
                discovered_count += self._scan_module_for_plugins(module)
            except Exception as e:
                logger.warning(f"Error importing module {modname}: {e}")
        
        logger.info(f"Discovered {discovered_count} plugins from {package_name}")
        return discovered_count
    
    def discover_plugins_from_directory(self, directory: Path, 
                                       pattern: str = "*.py") -> int:
        """
        Discover plugins from Python files in a directory.
        
        Args:
            directory: Directory to search for plugin files
            pattern: File pattern to match (default: "*.py")
            
        Returns:
            Number of plugins discovered and registered
        """
        discovered_count = 0
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return discovered_count
        
        for plugin_file in directory.glob(pattern):
            if plugin_file.name.startswith('__'):
                continue
                
            try:
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem, plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                discovered_count += self._scan_module_for_plugins(module)
            except Exception as e:
                logger.warning(f"Error loading plugin file {plugin_file}: {e}")
        
        logger.info(f"Discovered {discovered_count} plugins from {directory}")
        return discovered_count
    
    def get_plugin_info(self, plugin_name: str) -> PluginInfo:
        """
        Get information about a registered plugin.
        
        Args:
            plugin_name: Full name of the plugin
            
        Returns:
            PluginInfo object
            
        Raises:
            PluginNotFoundError: If plugin is not registered
        """
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{plugin_name}' not found")
        
        return self._plugins[plugin_name]
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """
        Get all plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to retrieve
            
        Returns:
            List of PluginInfo objects
        """
        return [
            info for info in self._plugins.values() 
            if info.plugin_type == plugin_type and info.is_enabled
        ]
    
    def get_plugins_with_capability(self, capability: str) -> List[PluginInfo]:
        """
        Get all plugins that support a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of PluginInfo objects
        """
        return [
            info for info in self._plugins.values()
            if capability in info.capabilities and info.is_enabled
        ]
    
    def list_all_plugins(self) -> List[PluginInfo]:
        """
        Get list of all registered plugins.
        
        Returns:
            List of all PluginInfo objects
        """
        return list(self._plugins.values())
    
    def create_plugin(self, plugin_name: str, 
                     config: Optional[Dict[str, Any]] = None) -> BasePlugin:
        """
        Create and initialize a plugin instance.
        
        Args:
            plugin_name: Full name of the plugin
            config: Configuration for the plugin
            
        Returns:
            Initialized plugin instance
            
        Raises:
            PluginNotFoundError: If plugin is not registered
            PluginInitializationError: If plugin initialization fails
        """
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{plugin_name}' not found")
        
        plugin_info = self._plugins[plugin_name]
        
        if not plugin_info.is_enabled:
            raise PluginInitializationError(
                f"Plugin '{plugin_name}' is disabled"
            )
        
        # Check if instance already exists
        if plugin_name in self._instances:
            return self._instances[plugin_name]
        
        # Validate dependencies
        self._validate_dependencies(plugin_info)
        
        # Create instance
        try:
            instance = plugin_info.plugin_class()
            
            # Validate configuration
            config = config or {}
            if not instance.validate_config(config):
                raise PluginInitializationError(
                    f"Invalid configuration for plugin '{plugin_name}'"
                )
            
            # Initialize plugin
            instance.initialize(config)
            
            self._instances[plugin_name] = instance
            logger.info(f"Created plugin instance: {plugin_name}")
            
            return instance
            
        except Exception as e:
            raise PluginInitializationError(
                f"Failed to create plugin '{plugin_name}': {e}"
            ) from e
    
    def enable_plugin(self, plugin_name: str) -> None:
        """
        Enable a plugin.
        
        Args:
            plugin_name: Full name of the plugin
            
        Raises:
            PluginNotFoundError: If plugin is not registered
        """
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{plugin_name}' not found")
        
        self._plugins[plugin_name].is_enabled = True
        logger.info(f"Enabled plugin: {plugin_name}")
    
    def disable_plugin(self, plugin_name: str) -> None:
        """
        Disable a plugin and cleanup its instance.
        
        Args:
            plugin_name: Full name of the plugin
            
        Raises:
            PluginNotFoundError: If plugin is not registered
        """
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{plugin_name}' not found")
        
        self._plugins[plugin_name].is_enabled = False
        
        # Cleanup instance if it exists
        if plugin_name in self._instances:
            try:
                self._instances[plugin_name].cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up plugin {plugin_name}: {e}")
            del self._instances[plugin_name]
        
        logger.info(f"Disabled plugin: {plugin_name}")
    
    def add_discovery_callback(self, callback: Callable[[PluginInfo], None]) -> None:
        """
        Add a callback to be called when plugins are discovered.
        
        Args:
            callback: Function to call with PluginInfo when plugin is registered
        """
        self._discovery_callbacks.append(callback)
    
    def cleanup_all(self) -> None:
        """
        Cleanup all plugin instances.
        
        This should be called when shutting down the visualization system.
        """
        for plugin_name, instance in self._instances.items():
            try:
                instance.cleanup()
                logger.info(f"Cleaned up plugin: {plugin_name}")
            except Exception as e:
                logger.warning(f"Error cleaning up plugin {plugin_name}: {e}")
        
        self._instances.clear()
        logger.info("All plugins cleaned up")
    
    def _is_valid_plugin_class(self, plugin_class: Type) -> bool:
        """
        Validate that a class is a proper plugin class.
        
        Args:
            plugin_class: Class to validate
            
        Returns:
            True if valid plugin class, False otherwise
        """
        if not inspect.isclass(plugin_class):
            return False
        
        if not issubclass(plugin_class, BasePlugin):
            return False
        
        # Check if class is abstract (has unimplemented abstract methods)
        if getattr(plugin_class, '__abstractmethods__', None):
            return False
        
        return True
    
    def _scan_module_for_plugins(self, module) -> int:
        """
        Scan a module for plugin classes and register them.
        
        Args:
            module: Module to scan
            
        Returns:
            Number of plugins found and registered
        """
        discovered_count = 0
        
        for name in dir(module):
            obj = getattr(module, name)
            
            if self._is_valid_plugin_class(obj):
                try:
                    self.register_plugin(obj)
                    discovered_count += 1
                except PluginRegistryError as e:
                    logger.warning(f"Failed to register plugin {name}: {e}")
        
        return discovered_count
    
    def _validate_dependencies(self, plugin_info: PluginInfo) -> None:
        """
        Validate that plugin dependencies are satisfied.
        
        Args:
            plugin_info: Plugin to validate dependencies for
            
        Raises:
            PluginDependencyError: If dependencies are not satisfied
        """
        for dependency in plugin_info.dependencies:
            if dependency not in self._plugins:
                raise PluginDependencyError(
                    f"Plugin '{plugin_info.full_name}' depends on "
                    f"'{dependency}' which is not registered"
                )
            
            if not self._plugins[dependency].is_enabled:
                raise PluginDependencyError(
                    f"Plugin '{plugin_info.full_name}' depends on "
                    f"'{dependency}' which is disabled"
                ) 