"""
Built-in Theme Plugins

Provides the default set of themes for the visualization system including
default, dark, and publication-ready themes with comprehensive styling.
"""

from typing import Dict, List, Any
from experiment_manager.visualization.plugins.theme_plugin import (
    ThemePlugin, ThemeConfig, ColorPalette
)


class DefaultThemePlugin(ThemePlugin):
    """Default theme plugin providing standard light theme."""
    
    @property
    def plugin_name(self) -> str:
        """Plugin name identifier."""
        return "default_themes"
    
    @property
    def plugin_version(self) -> str:
        """Plugin version."""
        return "1.0.0"
    
    @property
    def plugin_description(self) -> str:
        """Plugin description."""
        return "Default light theme with standard colors and typography"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        # No special initialization needed for built-in themes
        pass
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # No resources to clean up for built-in themes
        pass
    
    @property
    def available_themes(self) -> List[str]:
        """Available theme names."""
        return ["default"]
    
    @property
    def available_palettes(self) -> List[str]:
        """Available color palette names."""
        return ["default", "categorical", "sequential", "diverging"]
    
    def get_theme(self, theme_name: str) -> ThemeConfig:
        """Get theme configuration by name."""
        if theme_name == "default":
            return self._create_default_theme()
        else:
            raise ValueError(f"Theme '{theme_name}' not available")
    
    def get_palette(self, palette_name: str) -> ColorPalette:
        """Get color palette by name."""
        palettes = {
            "default": ColorPalette(
                name="default",
                colors=[
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
                ],
                description="Default matplotlib color cycle",
                color_type="categorical"
            ),
            "categorical": ColorPalette(
                name="categorical",
                colors=[
                    "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
                    "#34495e", "#e67e22", "#95a5a6", "#1abc9c", "#f1c40f"
                ],
                description="Vibrant categorical colors",
                color_type="categorical"
            ),
            "sequential": ColorPalette(
                name="sequential",
                colors=[
                    "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
                    "#4292c6", "#2171b5", "#08519c", "#08306b"
                ],
                description="Blue sequential palette",
                color_type="sequential"
            ),
            "diverging": ColorPalette(
                name="diverging",
                colors=[
                    "#8e0152", "#c51b7d", "#de77ae", "#f1b6da", "#fde0ef",
                    "#f7f7f7", "#e6f5d0", "#b8e186", "#7fbc41", "#4d9221", "#276419"
                ],
                description="Purple-green diverging palette",
                color_type="diverging"
            )
        }
        
        if palette_name not in palettes:
            raise ValueError(f"Palette '{palette_name}' not available")
        
        return palettes[palette_name]
    
    def _create_default_theme(self) -> ThemeConfig:
        """Create the default light theme configuration."""
        return ThemeConfig(
            name="default",
            colors={
                # Primary colors
                "primary": "#1f77b4",
                "secondary": "#ff7f0e", 
                "accent": "#2ca02c",
                "success": "#2ecc71",
                "warning": "#f39c12",
                "error": "#e74c3c",
                "info": "#3498db",
                
                # Background colors
                "background": "#ffffff",
                "surface": "#f8f9fa",
                "card": "#ffffff",
                "overlay": "rgba(0, 0, 0, 0.1)",
                
                # Text colors
                "text": "#212529",
                "text_secondary": "#6c757d",
                "text_muted": "#adb5bd",
                "text_inverse": "#ffffff",
                
                # Border and grid colors
                "border": "#dee2e6",
                "grid": "#e9ecef",
                "grid_minor": "#f8f9fa",
                "axis": "#495057",
                
                # Interactive colors
                "hover": "#e3f2fd",
                "active": "#bbdefb",
                "focus": "#2196f3",
                "disabled": "#e9ecef",
                
                # Chart specific colors
                "line_default": "#1f77b4",
                "fill_default": "rgba(31, 119, 180, 0.3)",
                "marker_default": "#1f77b4",
                "shadow": "rgba(0, 0, 0, 0.15)",
            },
            fonts={
                "default": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                "heading": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                "monospace": "'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace",
                "title_size": "16px",
                "subtitle_size": "14px",
                "body_size": "12px",
                "caption_size": "10px",
                "weight_normal": "400",
                "weight_medium": "500",
                "weight_bold": "700",
            },
            spacing={
                "base": 8,
                "xs": 4,
                "sm": 8,
                "md": 16,
                "lg": 24,
                "xl": 32,
                "xxl": 48,
                "padding": 16,
                "margin": 16,
                "gap": 8,
                "border_radius": 4,
                "line_height": 1.5,
            },
            borders={
                "width": 1,
                "style": "solid",
                "radius": 4,
                "radius_small": 2,
                "radius_large": 8,
            },
            backgrounds={
                "opacity_light": 0.1,
                "opacity_medium": 0.3,
                "opacity_heavy": 0.7,
                "gradient_start": "#ffffff",
                "gradient_end": "#f8f9fa",
            },
            metadata={
                "description": "Clean, modern light theme suitable for most use cases",
                "version": "1.0.0",
                "author": "Experiment Manager",
                "accessibility": {
                    "contrast_ratio": "AA",
                    "color_blind_safe": True,
                },
                "tags": ["light", "default", "clean", "modern"]
            }
        )


class DarkThemePlugin(ThemePlugin):
    """Dark theme plugin providing modern dark theme."""
    
    @property
    def plugin_name(self) -> str:
        """Plugin name identifier."""
        return "dark_themes"
    
    @property
    def plugin_version(self) -> str:
        """Plugin version."""
        return "1.0.0"
    
    @property
    def plugin_description(self) -> str:
        """Plugin description."""
        return "Modern dark theme with Material Design colors"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        # No special initialization needed for built-in themes
        pass
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # No resources to clean up for built-in themes
        pass
    
    @property
    def available_themes(self) -> List[str]:
        """Available theme names."""
        return ["dark"]
    
    @property
    def available_palettes(self) -> List[str]:
        """Available color palette names."""
        return ["dark", "neon", "cyberpunk"]
    
    def get_theme(self, theme_name: str) -> ThemeConfig:
        """Get theme configuration by name."""
        if theme_name == "dark":
            return self._create_dark_theme()
        else:
            raise ValueError(f"Theme '{theme_name}' not available")
    
    def get_palette(self, palette_name: str) -> ColorPalette:
        """Get color palette by name."""
        palettes = {
            "dark": ColorPalette(
                name="dark",
                colors=[
                    "#bb86fc", "#03dac6", "#cf6679", "#ffc107", "#4fc3f7",
                    "#81c784", "#ffb74d", "#f48fb1", "#9575cd", "#64b5f6"
                ],
                description="Material Design dark theme colors",
                color_type="categorical"
            ),
            "neon": ColorPalette(
                name="neon",
                colors=[
                    "#00ffff", "#ff1493", "#00ff00", "#ff4500", "#ffd700",
                    "#ff69b4", "#00bfff", "#adff2f", "#ff6347", "#da70d6"
                ],
                description="Bright neon colors for dark backgrounds",
                color_type="categorical"
            ),
            "cyberpunk": ColorPalette(
                name="cyberpunk",
                colors=[
                    "#ff2a6d", "#05d9e8", "#01012b", "#d1f7ff", "#005678",
                    "#01012b", "#ffeedd", "#ff6b35", "#f7931e", "#c05746"
                ],
                description="Cyberpunk-inspired color scheme",
                color_type="categorical"
            )
        }
        
        if palette_name not in palettes:
            raise ValueError(f"Palette '{palette_name}' not available")
        
        return palettes[palette_name]
    
    def _create_dark_theme(self) -> ThemeConfig:
        """Create the dark theme configuration."""
        return ThemeConfig(
            name="dark",
            colors={
                # Primary colors (Material Design dark theme)
                "primary": "#bb86fc",
                "secondary": "#03dac6",
                "accent": "#cf6679",
                "success": "#4caf50",
                "warning": "#ff9800",
                "error": "#f44336",
                "info": "#2196f3",
                
                # Background colors
                "background": "#121212",
                "surface": "#1e1e1e",
                "card": "#2d2d2d",
                "overlay": "rgba(255, 255, 255, 0.1)",
                
                # Text colors
                "text": "#ffffff",
                "text_secondary": "#b3b3b3",
                "text_muted": "#808080",
                "text_inverse": "#000000",
                
                # Border and grid colors
                "border": "#404040",
                "grid": "#333333",
                "grid_minor": "#2a2a2a",
                "axis": "#cccccc",
                
                # Interactive colors
                "hover": "#3d3d3d",
                "active": "#555555",
                "focus": "#bb86fc",
                "disabled": "#404040",
                
                # Chart specific colors
                "line_default": "#bb86fc",
                "fill_default": "rgba(187, 134, 252, 0.3)",
                "marker_default": "#bb86fc",
                "shadow": "rgba(255, 255, 255, 0.1)",
            },
            fonts={
                "default": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                "heading": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                "monospace": "'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace",
                "title_size": "16px",
                "subtitle_size": "14px", 
                "body_size": "12px",
                "caption_size": "10px",
                "weight_normal": "400",
                "weight_medium": "500",
                "weight_bold": "700",
            },
            spacing={
                "base": 8,
                "xs": 4,
                "sm": 8,
                "md": 16,
                "lg": 24,
                "xl": 32,
                "xxl": 48,
                "padding": 16,
                "margin": 16,
                "gap": 8,
                "border_radius": 4,
                "line_height": 1.5,
            },
            borders={
                "width": 1,
                "style": "solid",
                "radius": 4,
                "radius_small": 2,
                "radius_large": 8,
            },
            backgrounds={
                "opacity_light": 0.1,
                "opacity_medium": 0.3,
                "opacity_heavy": 0.7,
                "gradient_start": "#121212",
                "gradient_end": "#1e1e1e",
            },
            metadata={
                "description": "Modern dark theme with Material Design colors",
                "version": "1.0.0",
                "author": "Experiment Manager",
                "accessibility": {
                    "contrast_ratio": "AA",
                    "color_blind_safe": True,
                },
                "tags": ["dark", "modern", "material", "night"]
            }
        )


class PublicationThemePlugin(ThemePlugin):
    """Publication-ready theme plugin for academic and professional use."""
    
    @property
    def plugin_name(self) -> str:
        """Plugin name identifier."""
        return "publication_themes"
    
    @property
    def plugin_version(self) -> str:
        """Plugin version."""
        return "1.0.0"
    
    @property
    def plugin_description(self) -> str:
        """Plugin description."""
        return "Clean, professional theme optimized for academic publications"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        # No special initialization needed for built-in themes
        pass
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # No resources to clean up for built-in themes
        pass
    
    @property
    def available_themes(self) -> List[str]:
        """Available theme names."""
        return ["publication"]
    
    @property
    def available_palettes(self) -> List[str]:
        """Available color palette names."""
        return ["publication", "grayscale", "colorblind_safe"]
    
    def get_theme(self, theme_name: str) -> ThemeConfig:
        """Get theme configuration by name."""
        if theme_name == "publication":
            return self._create_publication_theme()
        else:
            raise ValueError(f"Theme '{theme_name}' not available")
    
    def get_palette(self, palette_name: str) -> ColorPalette:
        """Get color palette by name."""
        palettes = {
            "publication": ColorPalette(
                name="publication",
                colors=[
                    "#000000", "#d55e00", "#0173b2", "#de8f05", "#cc78bc",
                    "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9"
                ],
                description="High-contrast colors suitable for publication",
                color_type="categorical",
                accessibility_info={
                    "colorblind_safe": True,
                    "print_safe": True,
                    "high_contrast": True
                }
            ),
            "grayscale": ColorPalette(
                name="grayscale",
                colors=[
                    "#000000", "#333333", "#666666", "#999999", "#cccccc",
                    "#e6e6e6", "#f2f2f2", "#f8f8f8", "#fdfdfd", "#ffffff"
                ],
                description="Grayscale palette for print publications",
                color_type="sequential",
                accessibility_info={
                    "colorblind_safe": True,
                    "print_safe": True
                }
            ),
            "colorblind_safe": ColorPalette(
                name="colorblind_safe",
                colors=[
                    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
                    "#e6ab02", "#a6761d", "#666666", "#000000", "#ffffff"
                ],
                description="Colorblind-friendly palette (Brewer Set1)",
                color_type="categorical",
                accessibility_info={
                    "colorblind_safe": True,
                    "deuteranopia": True,
                    "protanopia": True,
                    "tritanopia": True
                }
            )
        }
        
        if palette_name not in palettes:
            raise ValueError(f"Palette '{palette_name}' not available")
        
        return palettes[palette_name]
    
    def _create_publication_theme(self) -> ThemeConfig:
        """Create the publication-ready theme configuration."""
        return ThemeConfig(
            name="publication",
            colors={
                # Primary colors (high contrast, print-safe)
                "primary": "#000000",
                "secondary": "#333333",
                "accent": "#d55e00",
                "success": "#0173b2",
                "warning": "#de8f05",
                "error": "#cc78bc",
                "info": "#0173b2",
                
                # Background colors (clean, minimal)
                "background": "#ffffff",
                "surface": "#ffffff",
                "card": "#ffffff",
                "overlay": "rgba(0, 0, 0, 0.05)",
                
                # Text colors (high contrast)
                "text": "#000000",
                "text_secondary": "#333333",
                "text_muted": "#666666",
                "text_inverse": "#ffffff",
                
                # Border and grid colors (subtle)
                "border": "#cccccc",
                "grid": "#e6e6e6",
                "grid_minor": "#f2f2f2",
                "axis": "#000000",
                
                # Interactive colors (minimal)
                "hover": "#f8f8f8",
                "active": "#e6e6e6",
                "focus": "#0173b2",
                "disabled": "#cccccc",
                
                # Chart specific colors
                "line_default": "#000000",
                "fill_default": "rgba(0, 0, 0, 0.1)",
                "marker_default": "#000000",
                "shadow": "rgba(0, 0, 0, 0.1)",
            },
            fonts={
                # Professional serif fonts for publications
                "default": "Times, 'Times New Roman', 'Liberation Serif', serif",
                "heading": "Times, 'Times New Roman', 'Liberation Serif', serif",
                "monospace": "'Courier New', Courier, 'Liberation Mono', monospace",
                "title_size": "16px",
                "subtitle_size": "14px",
                "body_size": "11px",  # Smaller for print
                "caption_size": "9px",
                "weight_normal": "400",
                "weight_medium": "500",
                "weight_bold": "700",
            },
            spacing={
                # Tighter spacing for publication layouts
                "base": 6,
                "xs": 3,
                "sm": 6,
                "md": 12,
                "lg": 18,
                "xl": 24,
                "xxl": 36,
                "padding": 12,
                "margin": 12,
                "gap": 6,
                "border_radius": 0,  # No rounded corners for publications
                "line_height": 1.4,  # Optimized for reading
            },
            borders={
                "width": 1,
                "style": "solid",
                "radius": 0,  # Square edges for formal look
                "radius_small": 0,
                "radius_large": 0,
            },
            backgrounds={
                "opacity_light": 0.05,
                "opacity_medium": 0.15,
                "opacity_heavy": 0.3,
                "gradient_start": "#ffffff",
                "gradient_end": "#ffffff",  # No gradients for publications
            },
            metadata={
                "description": "Clean, professional theme optimized for academic publications and formal presentations",
                "version": "1.0.0",
                "author": "Experiment Manager",
                "accessibility": {
                    "contrast_ratio": "AAA",  # Highest contrast
                    "color_blind_safe": True,
                    "print_safe": True,
                    "high_contrast": True,
                },
                "tags": ["publication", "academic", "formal", "print", "high-contrast"]
            }
        ) 