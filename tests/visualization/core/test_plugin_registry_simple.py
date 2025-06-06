"""
Simple functional tests for the Plugin Registry System.

Tests the actual implemented API rather than the hypothetical complex one.
"""

import pytest
from typing import Dict, Any

from experiment_manager.visualization.core.plugin_registry import (
    PluginRegistry, PluginType, BasePlugin, PluginRegistryError, PluginNotFoundError
)


class TestPlotPlugin(BasePlugin):
    """Test plot plugin implementation."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.PLOT
    
    @property
    def plugin_name(self) -> str:
        return "test_plot"
    
    @property
    def plugin_version(self) -> str:
        return "1.0.0"
    
    @property
    def plugin_description(self) -> str:
        return "Test plot plugin"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self.initialized = True
        self.config = config
    
    def cleanup(self) -> None:
        """Cleanup the plugin."""
        self.cleaned_up = True


class TestRendererPlugin(BasePlugin):
    """Test renderer plugin implementation."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.RENDERER
    
    @property
    def plugin_name(self) -> str:
        return "test_renderer"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup the plugin."""
        pass


class BadPlugin:
    """Plugin that doesn't inherit from BasePlugin."""
    pass


class TestPluginRegistrySimple:
    """Simple test cases for the PluginRegistry."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = PluginRegistry()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.registry.cleanup_all()
    
    def test_plugin_registration_success(self):
        """Test successful plugin registration."""
        # Register plugin
        self.registry.register_plugin(TestPlotPlugin)
        
        # Check it's registered
        plugins = self.registry.list_all_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "test_plot"
        assert plugins[0].plugin_type == PluginType.PLOT
    
    def test_plugin_registration_duplicate(self):
        """Test duplicate registration fails."""
        # Register first plugin
        self.registry.register_plugin(TestPlotPlugin)
        
        # Try to register same plugin again
        with pytest.raises(PluginRegistryError, match="already registered"):
            self.registry.register_plugin(TestPlotPlugin)
    
    def test_plugin_registration_with_override(self):
        """Test duplicate registration with override succeeds."""
        # Register first plugin
        self.registry.register_plugin(TestPlotPlugin)
        
        # Register same plugin with override
        self.registry.register_plugin(TestPlotPlugin, override=True)
        
        # Should still have only one plugin
        plugins = self.registry.list_all_plugins()
        assert len(plugins) == 1
    
    def test_plugin_registration_invalid_class(self):
        """Test registration with invalid class fails."""
        with pytest.raises(PluginRegistryError, match="Invalid plugin class"):
            self.registry.register_plugin(BadPlugin)
    
    def test_plugin_creation(self):
        """Test plugin instance creation."""
        # Register plugin
        self.registry.register_plugin(TestPlotPlugin)
        
        # Create instance
        plugin = self.registry.create_plugin("plot.test_plot")
        
        assert plugin is not None
        assert isinstance(plugin, TestPlotPlugin)
        assert hasattr(plugin, 'initialized')
    
    def test_plugin_creation_with_config(self):
        """Test plugin creation with configuration."""
        # Register plugin
        self.registry.register_plugin(TestPlotPlugin)
        
        # Create instance with config
        config = {"setting1": "value1", "setting2": 42}
        plugin = self.registry.create_plugin("plot.test_plot", config)
        
        assert plugin.config == config
    
    def test_plugin_creation_nonexistent(self):
        """Test creation of non-existent plugin fails."""
        with pytest.raises(PluginNotFoundError):
            self.registry.create_plugin("nonexistent.plugin")
    
    def test_multiple_plugin_types(self):
        """Test registering multiple plugin types."""
        # Register different types
        self.registry.register_plugin(TestPlotPlugin)
        self.registry.register_plugin(TestRendererPlugin)
        
        # Check both are registered
        all_plugins = self.registry.list_all_plugins()
        assert len(all_plugins) == 2
        
        # Check by type
        plot_plugins = self.registry.get_plugins_by_type(PluginType.PLOT)
        renderer_plugins = self.registry.get_plugins_by_type(PluginType.RENDERER)
        
        assert len(plot_plugins) == 1
        assert len(renderer_plugins) == 1
        assert plot_plugins[0].name == "test_plot"
        assert renderer_plugins[0].name == "test_renderer"
    
    def test_plugin_info_retrieval(self):
        """Test getting plugin information."""
        # Register plugin
        self.registry.register_plugin(TestPlotPlugin)
        
        # Get plugin info
        info = self.registry.get_plugin_info("plot.test_plot")
        
        assert info.name == "test_plot"
        assert info.plugin_type == PluginType.PLOT
        assert info.version == "1.0.0"
        assert info.description == "Test plot plugin"
        assert info.is_enabled is True
    
    def test_plugin_info_nonexistent(self):
        """Test getting info for non-existent plugin fails."""
        with pytest.raises(PluginNotFoundError):
            self.registry.get_plugin_info("nonexistent.plugin")
    
    def test_plugin_unregistration(self):
        """Test plugin unregistration."""
        # Register plugin
        self.registry.register_plugin(TestPlotPlugin)
        
        # Verify it's registered
        assert len(self.registry.list_all_plugins()) == 1
        
        # Unregister
        self.registry.unregister_plugin("plot.test_plot")
        
        # Verify it's gone
        assert len(self.registry.list_all_plugins()) == 0
    
    def test_plugin_unregistration_nonexistent(self):
        """Test unregistration of non-existent plugin fails."""
        with pytest.raises(PluginNotFoundError):
            self.registry.unregister_plugin("nonexistent.plugin")
    
    def test_plugin_enable_disable(self):
        """Test enabling and disabling plugins."""
        # Register plugin
        self.registry.register_plugin(TestPlotPlugin)
        
        # Should be enabled by default
        info = self.registry.get_plugin_info("plot.test_plot")
        assert info.is_enabled is True
        
        # Disable plugin
        self.registry.disable_plugin("plot.test_plot")
        info = self.registry.get_plugin_info("plot.test_plot")
        assert info.is_enabled is False
        
        # Enable plugin
        self.registry.enable_plugin("plot.test_plot")
        info = self.registry.get_plugin_info("plot.test_plot")
        assert info.is_enabled is True
    
    def test_cleanup_all(self):
        """Test cleanup of all plugins."""
        # Register and create some plugins
        self.registry.register_plugin(TestPlotPlugin)
        self.registry.register_plugin(TestRendererPlugin)
        
        # Create instances
        plugin1 = self.registry.create_plugin("plot.test_plot")
        plugin2 = self.registry.create_plugin("renderer.test_renderer")
        
        # Cleanup all
        self.registry.cleanup_all()
        
        # Check that cleanup was called on instances
        assert hasattr(plugin1, 'cleaned_up')


if __name__ == "__main__":
    pytest.main([__file__]) 