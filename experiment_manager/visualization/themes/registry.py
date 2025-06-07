"""
Theme Registry Implementation

Centralized management system for visual themes and styling in the visualization system.
Provides theme discovery, inheritance, composition, runtime switching, and custom theme support.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, field
from datetime import datetime
import weakref

from experiment_manager.visualization.core.plugin_registry import PluginRegistry, PluginType
from experiment_manager.visualization.core.event_bus import EventBus, Event, EventType, EventPriority
from experiment_manager.visualization.core.config_manager import ConfigManager
from experiment_manager.visualization.plugins.theme_plugin import (
    ThemePlugin, ThemeConfig, ColorPalette
)

logger = logging.getLogger(__name__)


@dataclass
class ThemeInfo:
    """Metadata information about a registered theme."""
    name: str
    plugin_name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    is_builtin: bool = True
    is_custom: bool = False
    parent_theme: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "plugin_name": self.plugin_name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "is_builtin": self.is_builtin,
            "is_custom": self.is_custom,
            "parent_theme": self.parent_theme
        }


@dataclass
class ThemePreview:
    """Preview information for a theme including sample colors and styling."""
    theme_name: str
    primary_colors: List[str]
    background_color: str
    text_color: str
    accent_color: str
    sample_plot_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "theme_name": self.theme_name,
            "primary_colors": self.primary_colors,
            "background_color": self.background_color,
            "text_color": self.text_color,
            "accent_color": self.accent_color,
            "sample_plot_config": self.sample_plot_config
        }


class ThemeRegistryError(Exception):
    """Base exception for theme registry errors."""
    pass


class ThemeNotFoundError(ThemeRegistryError):
    """Raised when a requested theme is not found."""
    pass


class ThemeValidationError(ThemeRegistryError):
    """Raised when theme validation fails."""
    pass


class ThemeRegistry:
    """
    Centralized theme registry and management system.
    
    Manages theme plugins, provides theme discovery, inheritance, composition,
    runtime switching, and custom theme creation capabilities.
    """
    
    def __init__(self,
                 plugin_registry: Optional[PluginRegistry] = None,
                 event_bus: Optional[EventBus] = None,
                 config_manager: Optional[ConfigManager] = None,
                 custom_themes_path: Optional[Path] = None):
        """
        Initialize theme registry.
        
        Args:
            plugin_registry: Plugin registry for theme plugin management
            event_bus: Event bus for theme change notifications
            config_manager: Configuration manager for theme settings
            custom_themes_path: Path to custom themes directory
        """
        self.plugin_registry = plugin_registry or PluginRegistry()
        self.event_bus = event_bus or EventBus()
        self.config_manager = config_manager or ConfigManager()
        
        # Theme storage
        self._themes: Dict[str, ThemeConfig] = {}
        self._theme_info: Dict[str, ThemeInfo] = {}
        self._theme_plugins: Dict[str, ThemePlugin] = {}
        self._palettes: Dict[str, ColorPalette] = {}
        
        # Current theme state
        self._current_theme: Optional[str] = None
        self._theme_stack: List[str] = []  # For theme inheritance
        
        # Custom themes
        self.custom_themes_path = custom_themes_path or Path("themes/custom")
        self._custom_themes: Dict[str, ThemeConfig] = {}
        
        # Event subscriptions
        self._subscriptions: List[str] = []
        
        # Initialize
        self._setup_event_handlers()
        self._discover_themes()
        self._load_custom_themes()
        self._set_default_theme()
        
        logger.info("ThemeRegistry initialized with theme discovery")
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for theme-related events."""
        # Subscribe to plugin registration events
        subscription_id = self.event_bus.subscribe(
            self,
            self._handle_plugin_event,
            weak_ref=True
        )
        self._subscriptions.append(subscription_id)
        
        logger.debug("Theme registry event handlers configured")
    
    def _handle_plugin_event(self, event: Event) -> None:
        """Handle plugin registration/unregistration events."""
        if event.event_type == EventType.PLUGIN_REGISTERED:
            plugin = event.data.get("plugin")
            if isinstance(plugin, ThemePlugin):
                self._register_theme_plugin(plugin)
        
        elif event.event_type == EventType.PLUGIN_UNREGISTERED:
            plugin_name = event.data.get("plugin_name", "")
            if plugin_name in self._theme_plugins:
                self._unregister_theme_plugin(plugin_name)
    
    def _discover_themes(self) -> None:
        """Discover and register theme plugins."""
        try:
            # Get theme plugin info from plugin registry
            theme_plugin_infos = self.plugin_registry.get_plugins_by_type(PluginType.THEME)
            
            for plugin_info in theme_plugin_infos:
                # Create plugin instance
                plugin_instance = self.plugin_registry.create_plugin(plugin_info.full_name, {})
                if isinstance(plugin_instance, ThemePlugin):
                    self._register_theme_plugin(plugin_instance)
            
            logger.info(f"Discovered {len(theme_plugin_infos)} theme plugins")
            
        except Exception as e:
            logger.error(f"Failed to discover themes: {e}")
    
    def _register_theme_plugin(self, plugin: ThemePlugin) -> None:
        """Register a theme plugin and its themes."""
        try:
            plugin_name = plugin.plugin_name
            self._theme_plugins[plugin_name] = plugin
            
            # Register all themes from the plugin
            for theme_name in plugin.available_themes:
                theme_config = plugin.get_theme(theme_name)
                self._themes[theme_name] = theme_config
                
                # Create theme info
                theme_info = ThemeInfo(
                    name=theme_name,
                    plugin_name=plugin_name,
                    description=theme_config.metadata.get("description", ""),
                    version=theme_config.metadata.get("version", "1.0.0"),
                    author=theme_config.metadata.get("author", ""),
                    tags=set(theme_config.metadata.get("tags", [])),
                    is_builtin=True,
                    is_custom=False
                )
                self._theme_info[theme_name] = theme_info
            
            # Register all palettes from the plugin
            for palette_name in plugin.available_palettes:
                palette = plugin.get_palette(palette_name)
                self._palettes[palette_name] = palette
            
            logger.info(f"Registered theme plugin '{plugin_name}' with {len(plugin.available_themes)} themes")
            
        except Exception as e:
            logger.error(f"Failed to register theme plugin '{plugin.plugin_name}': {e}")
    
    def _unregister_theme_plugin(self, plugin_name: str) -> None:
        """Unregister a theme plugin and its themes."""
        if plugin_name not in self._theme_plugins:
            return
        
        plugin = self._theme_plugins[plugin_name]
        
        # Remove themes
        for theme_name in plugin.available_themes:
            self._themes.pop(theme_name, None)
            self._theme_info.pop(theme_name, None)
        
        # Remove palettes
        for palette_name in plugin.available_palettes:
            self._palettes.pop(palette_name, None)
        
        # Remove plugin
        del self._theme_plugins[plugin_name]
        
        logger.info(f"Unregistered theme plugin '{plugin_name}'")
    
    def _load_custom_themes(self) -> None:
        """Load custom themes from the custom themes directory."""
        if not self.custom_themes_path.exists():
            return
        
        try:
            for theme_file in self.custom_themes_path.glob("*.yaml"):
                self._load_custom_theme_file(theme_file)
            
            for theme_file in self.custom_themes_path.glob("*.json"):
                self._load_custom_theme_file(theme_file)
            
            logger.info(f"Loaded {len(self._custom_themes)} custom themes")
            
        except Exception as e:
            logger.error(f"Failed to load custom themes: {e}")
    
    def _load_custom_theme_file(self, theme_file: Path) -> None:
        """Load a custom theme from a file."""
        try:
            with open(theme_file, 'r') as f:
                if theme_file.suffix == '.yaml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            theme_name = data.get("name", theme_file.stem)
            theme_config = self._create_theme_from_dict(data)
            
            self._custom_themes[theme_name] = theme_config
            self._themes[theme_name] = theme_config
            
            # Create theme info for custom theme
            theme_info = ThemeInfo(
                name=theme_name,
                plugin_name="custom",
                description=data.get("description", ""),
                version=data.get("version", "1.0.0"),
                author=data.get("author", ""),
                tags=set(data.get("tags", [])),
                is_builtin=False,
                is_custom=True,
                parent_theme=data.get("parent_theme")
            )
            self._theme_info[theme_name] = theme_info
            
            logger.debug(f"Loaded custom theme '{theme_name}' from {theme_file}")
            
        except Exception as e:
            logger.error(f"Failed to load custom theme from {theme_file}: {e}")
    
    def _create_theme_from_dict(self, data: Dict[str, Any]) -> ThemeConfig:
        """Create a ThemeConfig from dictionary data."""
        return ThemeConfig(
            name=data.get("name", "unnamed"),
            colors=data.get("colors", {}),
            fonts=data.get("fonts", {}),
            spacing=data.get("spacing", {}),
            borders=data.get("borders", {}),
            backgrounds=data.get("backgrounds", {}),
            metadata=data.get("metadata", {})
        )
    
    def _set_default_theme(self) -> None:
        """Set the default theme from configuration."""
        default_theme = self.config_manager.get("themes.default_theme", "default")
        if default_theme in self._themes:
            self.set_current_theme(default_theme)
        elif self._themes:
            # Use first available theme
            first_theme = next(iter(self._themes.keys()))
            self.set_current_theme(first_theme)
            logger.warning(f"Default theme '{default_theme}' not found, using '{first_theme}'")
    
    # Public API methods
    
    def get_available_themes(self) -> List[str]:
        """Get list of all available theme names."""
        return list(self._themes.keys())
    
    def get_theme_info(self, theme_name: str) -> ThemeInfo:
        """Get metadata information about a theme."""
        if theme_name not in self._theme_info:
            raise ThemeNotFoundError(f"Theme '{theme_name}' not found")
        return self._theme_info[theme_name]
    
    def get_theme(self, theme_name: str) -> ThemeConfig:
        """Get theme configuration by name."""
        if theme_name not in self._themes:
            raise ThemeNotFoundError(f"Theme '{theme_name}' not found")
        return self._themes[theme_name]
    
    def has_theme(self, theme_name: str) -> bool:
        """Check if a theme exists."""
        return theme_name in self._themes
    
    def get_current_theme(self) -> Optional[str]:
        """Get the name of the currently active theme."""
        return self._current_theme
    
    def get_current_theme_config(self) -> Optional[ThemeConfig]:
        """Get the configuration of the currently active theme."""
        if self._current_theme:
            return self.get_theme(self._current_theme)
        return None
    
    def set_current_theme(self, theme_name: str) -> None:
        """
        Set the current active theme.
        
        Args:
            theme_name: Name of the theme to activate
            
        Raises:
            ThemeNotFoundError: If theme doesn't exist
        """
        if theme_name not in self._themes:
            raise ThemeNotFoundError(f"Theme '{theme_name}' not found")
        
        old_theme = self._current_theme
        self._current_theme = theme_name
        
        # Publish theme change event
        self.event_bus.publish(Event(
            event_type=EventType.THEME_CHANGED,
            source="ThemeRegistry",
            data={
                "old_theme": old_theme,
                "new_theme": theme_name,
                "theme_config": self._themes[theme_name].to_dict()
            },
            priority=EventPriority.NORMAL
        ), async_mode=False)
        
        logger.info(f"Theme changed from '{old_theme}' to '{theme_name}'")
    
    def get_theme_preview(self, theme_name: str) -> ThemePreview:
        """Get a preview of a theme's visual appearance."""
        theme_config = self.get_theme(theme_name)
        
        # Extract key colors for preview
        colors = theme_config.colors
        primary_colors = []
        
        # Try to extract common color keys
        for key in ["primary", "secondary", "accent", "success", "warning", "error"]:
            if key in colors:
                primary_colors.append(colors[key])
        
        # If no common keys, use first few colors
        if not primary_colors:
            primary_colors = list(colors.values())[:6]
        
        return ThemePreview(
            theme_name=theme_name,
            primary_colors=primary_colors[:6],  # Limit to 6 colors
            background_color=colors.get("background", "#ffffff"),
            text_color=colors.get("text", "#000000"),
            accent_color=colors.get("accent", colors.get("primary", "#0066cc")),
            sample_plot_config={
                "line_color": colors.get("primary", "#0066cc"),
                "fill_color": colors.get("secondary", "#66ccff"),
                "grid_color": colors.get("grid", "#eeeeee"),
                "font_family": theme_config.get_font("default", "Arial"),
                "font_size": theme_config.get_spacing("font_size", 12)
            }
        )
    
    def create_custom_theme(self,
                           name: str,
                           base_theme: Optional[str] = None,
                           modifications: Optional[Dict[str, Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> ThemeConfig:
        """
        Create a custom theme based on an existing theme.
        
        Args:
            name: Name for the new theme
            base_theme: Name of base theme to inherit from
            modifications: Modifications to apply to base theme
            metadata: Additional metadata for the theme
            
        Returns:
            Created ThemeConfig
        """
        modifications = modifications or {}
        metadata = metadata or {}
        
        # Start with base theme or empty config
        if base_theme:
            base_config = self.get_theme(base_theme)
            theme_config = ThemeConfig(
                name=name,
                colors=base_config.colors.copy(),
                fonts=base_config.fonts.copy(),
                spacing=base_config.spacing.copy(),
                borders=base_config.borders.copy(),
                backgrounds=base_config.backgrounds.copy(),
                metadata=metadata
            )
        else:
            theme_config = ThemeConfig(
                name=name,
                colors={},
                fonts={},
                spacing={},
                borders={},
                backgrounds={},
                metadata=metadata
            )
        
        # Apply modifications
        for key, value in modifications.items():
            if key == "colors":
                theme_config.colors.update(value)
            elif key == "fonts":
                theme_config.fonts.update(value)
            elif key == "spacing":
                theme_config.spacing.update(value)
            elif key == "borders":
                theme_config.borders.update(value)
            elif key == "backgrounds":
                theme_config.backgrounds.update(value)
            elif key == "metadata":
                theme_config.metadata.update(value)
        
        # Register the custom theme
        self._custom_themes[name] = theme_config
        self._themes[name] = theme_config
        
        # Create theme info
        theme_info = ThemeInfo(
            name=name,
            plugin_name="custom",
            description=metadata.get("description", ""),
            version=metadata.get("version", "1.0.0"),
            author=metadata.get("author", ""),
            tags=set(metadata.get("tags", [])),
            is_builtin=False,
            is_custom=True,
            parent_theme=base_theme
        )
        self._theme_info[name] = theme_info
        
        logger.info(f"Created custom theme '{name}' based on '{base_theme}'")
        return theme_config
    
    def save_custom_theme(self, theme_name: str, output_path: Optional[Path] = None) -> Path:
        """
        Save a custom theme to file.
        
        Args:
            theme_name: Name of theme to save
            output_path: Optional output path (default: custom themes directory)
            
        Returns:
            Path where theme was saved
        """
        if theme_name not in self._custom_themes:
            raise ThemeNotFoundError(f"Custom theme '{theme_name}' not found")
        
        theme_config = self._custom_themes[theme_name]
        theme_info = self._theme_info[theme_name]
        
        if output_path is None:
            self.custom_themes_path.mkdir(parents=True, exist_ok=True)
            output_path = self.custom_themes_path / f"{theme_name}.yaml"
        
        # Prepare data for saving
        data = {
            **theme_config.to_dict(),
            **theme_info.to_dict(),
        }
        
        # Save as YAML
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved custom theme '{theme_name}' to {output_path}")
        return output_path
    
    def delete_custom_theme(self, theme_name: str) -> None:
        """Delete a custom theme."""
        if theme_name not in self._custom_themes:
            raise ThemeNotFoundError(f"Custom theme '{theme_name}' not found")
        
        # Remove from all registries
        del self._custom_themes[theme_name]
        del self._themes[theme_name]
        del self._theme_info[theme_name]
        
        # If it's the current theme, switch to default
        if self._current_theme == theme_name:
            default_theme = self.config_manager.get("themes.default_theme", "default")
            if default_theme in self._themes:
                self.set_current_theme(default_theme)
            elif self._themes:
                self.set_current_theme(next(iter(self._themes.keys())))
        
        logger.info(f"Deleted custom theme '{theme_name}'")
    
    def get_available_palettes(self) -> List[str]:
        """Get list of all available color palette names."""
        return list(self._palettes.keys())
    
    def get_palette(self, palette_name: str) -> ColorPalette:
        """Get color palette by name."""
        if palette_name not in self._palettes:
            raise ThemeNotFoundError(f"Palette '{palette_name}' not found")
        return self._palettes[palette_name]
    
    def has_palette(self, palette_name: str) -> bool:
        """Check if a palette exists."""
        return palette_name in self._palettes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_themes": len(self._themes),
            "builtin_themes": len([t for t in self._theme_info.values() if t.is_builtin]),
            "custom_themes": len([t for t in self._theme_info.values() if t.is_custom]),
            "total_palettes": len(self._palettes),
            "theme_plugins": len(self._theme_plugins),
            "current_theme": self._current_theme,
            "available_themes": list(self._themes.keys()),
            "available_palettes": list(self._palettes.keys())
        }
    
    def cleanup(self) -> None:
        """Clean up resources and unsubscribe from events."""
        # Unsubscribe from events
        for subscription_id in self._subscriptions:
            self.event_bus.unsubscribe(subscription_id)
        self._subscriptions.clear()
        
        # Clear themes
        self._themes.clear()
        self._theme_info.clear()
        self._theme_plugins.clear()
        self._palettes.clear()
        self._custom_themes.clear()
        
        logger.info("ThemeRegistry cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ThemeRegistry(themes={len(self._themes)}, "
                f"palettes={len(self._palettes)}, "
                f"current='{self._current_theme}')") 