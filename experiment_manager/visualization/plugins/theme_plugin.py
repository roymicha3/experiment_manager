"""
Theme plugin interface for managing visual themes and styling.

This module defines the abstract interface that all theme plugins must implement.
Theme plugins are responsible for providing consistent visual styling, color schemes,
fonts, and other aesthetic elements across visualizations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json

from .base import BasePlugin, PluginType


class ColorPalette:
    """
    Container for color palette information.
    
    This class encapsulates a set of colors with metadata about
    their intended usage and accessibility properties.
    """
    
    def __init__(self,
                 name: str,
                 colors: List[str],
                 description: str = "",
                 color_type: str = "categorical",
                 accessibility_info: Optional[Dict[str, Any]] = None):
        """
        Initialize color palette.
        
        Args:
            name: Name of the color palette
            colors: List of color values (hex, rgb, named colors)
            description: Description of the palette
            color_type: Type of palette ('categorical', 'sequential', 'diverging')
            accessibility_info: Information about accessibility features
        """
        self.name = name
        self.colors = colors
        self.description = description
        self.color_type = color_type
        self.accessibility_info = accessibility_info or {}
        
    @property
    def size(self) -> int:
        """Get number of colors in palette."""
        return len(self.colors)
    
    def get_color(self, index: int, wrap: bool = True) -> str:
        """
        Get color by index.
        
        Args:
            index: Color index
            wrap: Whether to wrap around if index exceeds palette size
            
        Returns:
            Color value as string
        """
        if wrap:
            return self.colors[index % len(self.colors)]
        else:
            return self.colors[index]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert palette to dictionary representation."""
        return {
            "name": self.name,
            "colors": self.colors,
            "description": self.description,
            "color_type": self.color_type,
            "accessibility_info": self.accessibility_info,
            "size": self.size,
        }


class ThemeConfig:
    """
    Configuration for a visual theme.
    
    This class encapsulates all styling configuration for a theme,
    including colors, fonts, spacing, and other visual elements.
    """
    
    def __init__(self,
                 name: str,
                 colors: Dict[str, Any],
                 fonts: Optional[Dict[str, Any]] = None,
                 spacing: Optional[Dict[str, Any]] = None,
                 borders: Optional[Dict[str, Any]] = None,
                 backgrounds: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize theme configuration.
        
        Args:
            name: Theme name
            colors: Color configuration
            fonts: Font configuration
            spacing: Spacing configuration
            borders: Border configuration
            backgrounds: Background configuration
            metadata: Additional theme metadata
        """
        self.name = name
        self.colors = colors
        self.fonts = fonts or {}
        self.spacing = spacing or {}
        self.borders = borders or {}
        self.backgrounds = backgrounds or {}
        self.metadata = metadata or {}
    
    def get_color(self, key: str, default: str = "#000000") -> str:
        """Get color value by key."""
        return self.colors.get(key, default)
    
    def get_font(self, key: str, default: str = "Arial") -> str:
        """Get font value by key."""
        return self.fonts.get(key, default)
    
    def get_spacing(self, key: str, default: int = 10) -> int:
        """Get spacing value by key."""
        return self.spacing.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert theme config to dictionary."""
        return {
            "name": self.name,
            "colors": self.colors,
            "fonts": self.fonts,
            "spacing": self.spacing,
            "borders": self.borders,
            "backgrounds": self.backgrounds,
            "metadata": self.metadata,
        }
    
    def merge_with(self, other: 'ThemeConfig') -> 'ThemeConfig':
        """
        Merge this theme with another theme.
        
        Args:
            other: Theme to merge with
            
        Returns:
            New merged theme configuration
        """
        merged = ThemeConfig(
            name=f"{self.name}+{other.name}",
            colors={**self.colors, **other.colors},
            fonts={**self.fonts, **other.fonts},
            spacing={**self.spacing, **other.spacing},
            borders={**self.borders, **other.borders},
            backgrounds={**self.backgrounds, **other.backgrounds},
            metadata={**self.metadata, **other.metadata},
        )
        return merged


class ThemePlugin(BasePlugin):
    """
    Abstract base class for theme plugins.
    
    Theme plugins provide visual styling configuration for visualizations,
    including color schemes, fonts, spacing, and other aesthetic elements.
    Each plugin can define one or more themes and color palettes.
    """
    
    @property
    def plugin_type(self) -> PluginType:
        """Theme plugins always return THEME type."""
        return PluginType.THEME
    
    @property
    @abstractmethod
    def available_themes(self) -> List[str]:
        """
        List of theme names provided by this plugin.
        
        Returns:
            List of theme name identifiers
        """
        pass
    
    @property
    @abstractmethod
    def available_palettes(self) -> List[str]:
        """
        List of color palette names provided by this plugin.
        
        Returns:
            List of palette name identifiers
        """
        pass
    
    @property
    def default_theme(self) -> str:
        """
        Default theme name for this plugin.
        
        Returns:
            Default theme identifier
        """
        return self.available_themes[0] if self.available_themes else "default"
    
    @property
    def default_palette(self) -> str:
        """
        Default color palette name for this plugin.
        
        Returns:
            Default palette identifier
        """
        return self.available_palettes[0] if self.available_palettes else "default"
    
    @abstractmethod
    def get_theme(self, theme_name: str) -> ThemeConfig:
        """
        Get theme configuration by name.
        
        Args:
            theme_name: Name of the theme to retrieve
            
        Returns:
            ThemeConfig object for the specified theme
            
        Raises:
            ValueError: If theme name is not available
        """
        pass
    
    @abstractmethod
    def get_palette(self, palette_name: str) -> ColorPalette:
        """
        Get color palette by name.
        
        Args:
            palette_name: Name of the palette to retrieve
            
        Returns:
            ColorPalette object for the specified palette
            
        Raises:
            ValueError: If palette name is not available
        """
        pass
    
    def has_theme(self, theme_name: str) -> bool:
        """
        Check if theme is available in this plugin.
        
        Args:
            theme_name: Theme name to check
            
        Returns:
            True if theme is available
        """
        return theme_name in self.available_themes
    
    def has_palette(self, palette_name: str) -> bool:
        """
        Check if palette is available in this plugin.
        
        Args:
            palette_name: Palette name to check
            
        Returns:
            True if palette is available
        """
        return palette_name in self.available_palettes
    
    def get_theme_preview(self, theme_name: str) -> Dict[str, Any]:
        """
        Get preview information for a theme.
        
        Args:
            theme_name: Theme name to preview
            
        Returns:
            Dictionary with preview information
        """
        if not self.has_theme(theme_name):
            raise ValueError(f"Theme '{theme_name}' not available")
            
        theme = self.get_theme(theme_name)
        return {
            "name": theme.name,
            "primary_colors": list(theme.colors.values())[:6],  # First 6 colors
            "background": theme.backgrounds.get("primary", "#ffffff"),
            "text_color": theme.colors.get("text", "#000000"),
            "accent_color": theme.colors.get("accent", "#007bff"),
        }
    
    def get_palette_preview(self, palette_name: str) -> Dict[str, Any]:
        """
        Get preview information for a color palette.
        
        Args:
            palette_name: Palette name to preview
            
        Returns:
            Dictionary with preview information
        """
        if not self.has_palette(palette_name):
            raise ValueError(f"Palette '{palette_name}' not available")
            
        palette = self.get_palette(palette_name)
        return {
            "name": palette.name,
            "colors": palette.colors,
            "color_type": palette.color_type,
            "size": palette.size,
            "description": palette.description,
        }
    
    def create_custom_theme(self,
                          base_theme: str,
                          modifications: Dict[str, Any],
                          name: Optional[str] = None) -> ThemeConfig:
        """
        Create a custom theme based on an existing theme.
        
        Args:
            base_theme: Name of base theme to modify
            modifications: Dictionary of modifications to apply
            name: Optional name for the custom theme
            
        Returns:
            Custom ThemeConfig object
        """
        if not self.has_theme(base_theme):
            raise ValueError(f"Base theme '{base_theme}' not available")
            
        base = self.get_theme(base_theme)
        custom_name = name or f"{base_theme}_custom"
        
        # Apply modifications
        custom_colors = {**base.colors}
        custom_fonts = {**base.fonts}
        custom_spacing = {**base.spacing}
        custom_borders = {**base.borders}
        custom_backgrounds = {**base.backgrounds}
        
        if "colors" in modifications:
            custom_colors.update(modifications["colors"])
        if "fonts" in modifications:
            custom_fonts.update(modifications["fonts"])
        if "spacing" in modifications:
            custom_spacing.update(modifications["spacing"])
        if "borders" in modifications:
            custom_borders.update(modifications["borders"])
        if "backgrounds" in modifications:
            custom_backgrounds.update(modifications["backgrounds"])
        
        return ThemeConfig(
            name=custom_name,
            colors=custom_colors,
            fonts=custom_fonts,
            spacing=custom_spacing,
            borders=custom_borders,
            backgrounds=custom_backgrounds,
            metadata={**base.metadata, "custom": True, "base_theme": base_theme}
        )
    
    def export_theme(self, theme_name: str, output_path: Path) -> None:
        """
        Export theme configuration to file.
        
        Args:
            theme_name: Theme name to export
            output_path: Path where to save the theme file
        """
        if not self.has_theme(theme_name):
            raise ValueError(f"Theme '{theme_name}' not available")
            
        theme = self.get_theme(theme_name)
        with open(output_path, 'w') as f:
            json.dump(theme.to_dict(), f, indent=2)
    
    def export_palette(self, palette_name: str, output_path: Path) -> None:
        """
        Export color palette to file.
        
        Args:
            palette_name: Palette name to export
            output_path: Path where to save the palette file
        """
        if not self.has_palette(palette_name):
            raise ValueError(f"Palette '{palette_name}' not available")
            
        palette = self.get_palette(palette_name)
        with open(output_path, 'w') as f:
            json.dump(palette.to_dict(), f, indent=2)
    
    def validate_theme_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate theme configuration structure.
        
        Args:
            config: Theme configuration to validate
            
        Returns:
            True if configuration is valid
        """
        required_keys = ["name", "colors"]
        return all(key in config for key in required_keys)
    
    def get_accessibility_info(self, theme_name: str) -> Dict[str, Any]:
        """
        Get accessibility information for a theme.
        
        Args:
            theme_name: Theme name to check
            
        Returns:
            Dictionary with accessibility information
        """
        if not self.has_theme(theme_name):
            raise ValueError(f"Theme '{theme_name}' not available")
            
        # Default implementation returns basic info
        # Subclasses can override to provide detailed accessibility analysis
        return {
            "theme": theme_name,
            "wcag_compliant": "unknown",
            "contrast_ratios": {},
            "colorblind_friendly": "unknown",
        } 