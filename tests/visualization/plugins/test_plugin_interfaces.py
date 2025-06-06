"""
Tests for visualization plugin interfaces.

This module tests the abstract base plugin interfaces to ensure they
provide correct contracts and can be properly implemented.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from experiment_manager.visualization.plugins import (
    BasePlugin, 
    PlotPlugin, 
    RendererPlugin, 
    ExportPlugin,
    DataProcessorPlugin,
    ThemePlugin
)
from experiment_manager.visualization.plugins.plot_plugin import PlotData, PlotResult
from experiment_manager.visualization.plugins.renderer_plugin import RenderContext, RenderResult
from experiment_manager.visualization.plugins.export_plugin import ExportData, ExportOptions, ExportResult
from experiment_manager.visualization.plugins.data_processor_plugin import ProcessingContext, ProcessingResult
from experiment_manager.visualization.plugins.theme_plugin import ColorPalette, ThemeConfig
from experiment_manager.visualization.core.plugin_registry import PluginType


class TestBasicPlugin(BasePlugin):
    """Concrete implementation of BasePlugin for testing."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.PLOT
    
    @property 
    def plugin_name(self) -> str:
        return "test_basic"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def cleanup(self) -> None:
        pass


class TestPlotPlugin(PlotPlugin):
    """Concrete implementation of PlotPlugin for testing."""
    
    @property
    def plugin_name(self) -> str:
        return "test_plot"
    
    @property
    def supported_data_types(self) -> List[str]:
        return ["timeseries", "scalar"]
    
    @property
    def plot_dimensions(self) -> str:
        return "2D"
    
    def can_handle_data(self, data: PlotData) -> bool:
        return True
    
    def generate_plot(self, data: PlotData, config: Optional[Dict[str, Any]] = None) -> PlotResult:
        return PlotResult(plot_object="mock_plot", success=True)
    
    def initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def cleanup(self) -> None:
        pass


class TestRendererPlugin(RendererPlugin):
    """Concrete implementation of RendererPlugin for testing."""
    
    @property
    def plugin_name(self) -> str:
        return "test_renderer"
    
    @property
    def supported_formats(self) -> List[str]:
        return ["png", "svg"]
    
    @property
    def supported_plot_types(self) -> List[str]:
        return ["matplotlib"]
    
    def can_render(self, plot_object: Any, output_format: str) -> bool:
        return output_format in self.supported_formats
    
    def render(self, plot_object: Any, context: RenderContext, config: Optional[Dict[str, Any]] = None) -> RenderResult:
        return RenderResult(data=b"mock_data", context=context, success=True)
    
    def initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def cleanup(self) -> None:
        pass


class TestExportPlugin(ExportPlugin):
    """Concrete implementation of ExportPlugin for testing."""
    
    @property
    def plugin_name(self) -> str:
        return "test_export"
    
    @property
    def supported_formats(self) -> List[str]:
        return ["json", "csv"]
    
    @property
    def supported_data_types(self) -> List[str]:
        return ["plot", "dataset"]
    
    def can_export(self, data: ExportData, export_format: str) -> bool:
        return export_format in self.supported_formats
    
    def export(self, data, options: ExportOptions, config: Optional[Dict[str, Any]] = None) -> ExportResult:
        return ExportResult(success=True, exported_items=1, export_format="json")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def cleanup(self) -> None:
        pass


class TestDataProcessorPlugin(DataProcessorPlugin):
    """Concrete implementation of DataProcessorPlugin for testing."""
    
    @property
    def plugin_name(self) -> str:
        return "test_processor"
    
    @property
    def supported_operations(self) -> List[str]:
        return ["filter", "aggregate"]
    
    @property
    def supported_input_types(self) -> List[str]:
        return ["dataframe", "array"]
    
    @property
    def supported_output_types(self) -> List[str]:
        return ["dataframe", "array"]
    
    def can_process(self, data: Any, operation: str, context: Optional[ProcessingContext] = None) -> bool:
        return operation in self.supported_operations
    
    def process(self, data: Any, operation: str, context: Optional[ProcessingContext] = None, config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        return ProcessingResult(data=data, success=True)
    
    def initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def cleanup(self) -> None:
        pass


class TestThemePlugin(ThemePlugin):
    """Concrete implementation of ThemePlugin for testing."""
    
    @property
    def plugin_name(self) -> str:
        return "test_theme"
    
    @property
    def available_themes(self) -> List[str]:
        return ["default", "dark"]
    
    @property
    def available_palettes(self) -> List[str]:
        return ["categorical", "sequential"]
    
    def get_theme(self, theme_name: str) -> ThemeConfig:
        return ThemeConfig(
            name=theme_name,
            colors={"primary": "#000000", "secondary": "#ffffff"}
        )
    
    def get_palette(self, palette_name: str) -> ColorPalette:
        return ColorPalette(
            name=palette_name,
            colors=["#ff0000", "#00ff00", "#0000ff"],
            color_type="categorical"
        )
    
    def initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def cleanup(self) -> None:
        pass


class TestBasePlugin:
    """Test BasePlugin interface."""
    
    def test_base_plugin_creation(self):
        """Test creating a basic plugin."""
        plugin = TestBasicPlugin()
        assert plugin.plugin_type == PluginType.PLOT
        assert plugin.plugin_name == "test_basic"
        assert plugin.plugin_version == "1.0.0"
        assert plugin.plugin_dependencies == []
        assert plugin.supported_capabilities == []
    
    def test_base_plugin_info(self):
        """Test getting plugin info."""
        plugin = TestBasicPlugin()
        info = plugin.get_info()
        assert info["name"] == "test_basic"
        assert info["type"] == "plot"
        assert info["version"] == "1.0.0"
    
    def test_base_plugin_compatibility(self):
        """Test plugin compatibility check."""
        plugin1 = TestBasicPlugin()
        plugin2 = TestBasicPlugin()
        assert plugin1.is_compatible_with(plugin2)
    
    def test_default_config(self):
        """Test default configuration."""
        plugin = TestBasicPlugin()
        assert plugin.get_default_config() == {}
        assert plugin.get_required_config_keys() == []


class TestPlotPluginInterface:
    """Test PlotPlugin interface."""
    
    def test_plot_plugin_creation(self):
        """Test creating a plot plugin."""
        plugin = TestPlotPlugin()
        assert plugin.plugin_type == PluginType.PLOT
        assert plugin.supported_data_types == ["timeseries", "scalar"]
        assert plugin.plot_dimensions == "2D"
    
    def test_plot_data_container(self):
        """Test PlotData container."""
        data = np.array([1, 2, 3, 4])
        plot_data = PlotData(data, metadata={"source": "test"})
        assert plot_data.shape == (4,)
        assert plot_data.get_metadata("source") == "test"
        assert plot_data.get_metadata("missing", "default") == "default"
    
    def test_plot_result_container(self):
        """Test PlotResult container."""
        result = PlotResult("mock_plot", metadata={"format": "png"})
        assert result.success is True
        assert result.plot_object == "mock_plot"
        assert result.get_metadata("format") == "png"
    
    def test_plot_generation(self):
        """Test plot generation."""
        plugin = TestPlotPlugin()
        data = PlotData(np.array([1, 2, 3]))
        assert plugin.can_handle_data(data)
        
        result = plugin.generate_plot(data)
        assert result.success
        assert result.plot_object == "mock_plot"
    
    def test_data_requirements(self):
        """Test data requirements."""
        plugin = TestPlotPlugin()
        requirements = plugin.get_data_requirements()
        assert "supported_data_types" in requirements
        assert "plot_dimensions" in requirements


class TestRendererPluginInterface:
    """Test RendererPlugin interface."""
    
    def test_renderer_plugin_creation(self):
        """Test creating a renderer plugin."""
        plugin = TestRendererPlugin()
        assert plugin.plugin_type == PluginType.RENDERER
        assert plugin.supported_formats == ["png", "svg"]
        assert plugin.default_format == "png"
    
    def test_render_context(self):
        """Test RenderContext."""
        context = RenderContext("png", dpi=300, quality=90)
        assert context.output_format == "png"
        assert context.dpi == 300
        assert context.get_mime_type() == "image/png"
    
    def test_render_result(self):
        """Test RenderResult."""
        context = RenderContext("png")
        result = RenderResult(b"image_data", context)
        assert result.success
        assert result.size > 0
        assert isinstance(result.data, bytes)
    
    def test_rendering(self):
        """Test rendering functionality."""
        plugin = TestRendererPlugin()
        context = RenderContext("png")
        assert plugin.can_render("mock_plot", "png")
        
        result = plugin.render("mock_plot", context)
        assert result.success
        assert result.context == context
    
    def test_format_info(self):
        """Test format information."""
        plugin = TestRendererPlugin()
        info = plugin.get_format_info("png")
        assert info["format"] == "png"
        assert info["mime_type"] == "image/png"


class TestExportPluginInterface:
    """Test ExportPlugin interface."""
    
    def test_export_plugin_creation(self):
        """Test creating an export plugin."""
        plugin = TestExportPlugin()
        assert plugin.plugin_type == PluginType.EXPORTER
        assert plugin.supported_formats == ["json", "csv"]
        assert plugin.supported_data_types == ["plot", "dataset"]
    
    def test_export_data_container(self):
        """Test ExportData container."""
        data = ExportData({"key": "value"}, "config")
        assert data.data_type == "config"
        assert data.get_metadata("missing", "default") == "default"
        data.add_metadata("test", "value")
        assert data.get_metadata("test") == "value"
    
    def test_export_options(self):
        """Test ExportOptions."""
        options = ExportOptions(
            output_path=Path("test.json"),
            include_metadata=True,
            overwrite=True
        )
        assert options.output_path == Path("test.json")
        assert options.include_metadata is True
        assert options.overwrite is True
    
    def test_export_result(self):
        """Test ExportResult."""
        result = ExportResult(
            success=True,
            exported_items=5,
            export_format="json"
        )
        assert result.success
        assert result.exported_items == 5
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
    
    def test_export_functionality(self):
        """Test export functionality."""
        plugin = TestExportPlugin()
        data = ExportData({"test": "data"}, "config")
        options = ExportOptions()
        
        assert plugin.can_export(data, "json")
        result = plugin.export(data, options)
        assert result.success


class TestDataProcessorPluginInterface:
    """Test DataProcessorPlugin interface."""
    
    def test_processor_plugin_creation(self):
        """Test creating a data processor plugin."""
        plugin = TestDataProcessorPlugin()
        assert plugin.plugin_type == PluginType.DATA_PROCESSOR
        assert plugin.supported_operations == ["filter", "aggregate"]
        assert plugin.supports_streaming is False
    
    def test_processing_context(self):
        """Test ProcessingContext."""
        context = ProcessingContext(
            "filter",
            constraints={"max_rows": 1000}
        )
        assert context.operation_type == "filter"
        assert context.get_constraint("max_rows") == 1000
        assert context.get_constraint("missing", "default") == "default"
    
    def test_processing_result(self):
        """Test ProcessingResult."""
        result = ProcessingResult(
            data=[1, 2, 3],
            processing_stats={"rows_processed": 100}
        )
        assert result.success
        assert result.shape == (3,)
        assert result.get_stat("rows_processed") == 100
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
    
    def test_processing_functionality(self):
        """Test processing functionality."""
        plugin = TestDataProcessorPlugin()
        data = [1, 2, 3, 4, 5]
        
        assert plugin.can_process(data, "filter")
        result = plugin.process(data, "filter")
        assert result.success
        assert result.data == data
    
    def test_operation_info(self):
        """Test operation information."""
        plugin = TestDataProcessorPlugin()
        info = plugin.get_operation_info("filter")
        assert info["operation"] == "filter"
        assert "input_types" in info


class TestThemePluginInterface:
    """Test ThemePlugin interface."""
    
    def test_theme_plugin_creation(self):
        """Test creating a theme plugin."""
        plugin = TestThemePlugin()
        assert plugin.plugin_type == PluginType.THEME
        assert plugin.available_themes == ["default", "dark"]
        assert plugin.default_theme == "default"
    
    def test_color_palette(self):
        """Test ColorPalette."""
        palette = ColorPalette(
            "test",
            ["#ff0000", "#00ff00", "#0000ff"],
            color_type="categorical"
        )
        assert palette.size == 3
        assert palette.get_color(0) == "#ff0000"
        assert palette.get_color(5, wrap=True) == "#0000ff"  # wraps around (5 % 3 = 2)
    
    def test_theme_config(self):
        """Test ThemeConfig."""
        config = ThemeConfig(
            "test_theme",
            colors={"primary": "#000000"},
            fonts={"body": "Arial"}
        )
        assert config.name == "test_theme"
        assert config.get_color("primary") == "#000000"
        assert config.get_font("body") == "Arial"
    
    def test_theme_functionality(self):
        """Test theme functionality."""
        plugin = TestThemePlugin()
        
        assert plugin.has_theme("default")
        assert plugin.has_palette("categorical")
        
        theme = plugin.get_theme("default")
        assert isinstance(theme, ThemeConfig)
        
        palette = plugin.get_palette("categorical")
        assert isinstance(palette, ColorPalette)
    
    def test_theme_preview(self):
        """Test theme preview."""
        plugin = TestThemePlugin()
        preview = plugin.get_theme_preview("default")
        assert "name" in preview
        assert "primary_colors" in preview
        
        palette_preview = plugin.get_palette_preview("categorical")
        assert "name" in palette_preview
        assert "colors" in palette_preview


if __name__ == "__main__":
    pytest.main([__file__]) 