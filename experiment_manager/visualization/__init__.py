"""
Experiment Manager Visualization Module

A modular, extensible visualization system built around a plugin architecture.
Provides rich, interactive visualizations for experimental data, training metrics,
and analytics results.

Key Components:
    - Plugin Registry: Central management of visualization plugins
    - Event Bus: Loose coupling between components
    - Data Pipeline: Chainable data processing with caching
    - Renderers: Multiple backend support (Matplotlib, Plotly, Bokeh)
    - Plot Types: Extensible plot type plugins
    - Dashboards: Interactive multi-plot dashboards

Example:
    ```python
    from experiment_manager.visualization import VisualizerManager
    
    # Create visualizer with default configuration
    visualizer = VisualizerManager.from_config()
    
    # Create a training curves plot
    plot = visualizer.create_plot(
        plot_type="training_curves",
        data_spec=DataSpec(trial_run_ids=[1, 2, 3])
    )
    
    # Render and save
    plot.render(engine="matplotlib")
    plot.save("training_curves.png")
    ```
"""

from experiment_manager.visualization.core.plugin_registry import PluginRegistry, PluginType
from experiment_manager.visualization.core.event_bus import EventBus, Event, EventType, EventPriority, EventFilter
from experiment_manager.visualization.core.config_manager import ConfigManager, VisualizationConfig, ConfigFormat

# Main visualizer manager
from experiment_manager.visualization.visualizer_manager import (
    VisualizerManager,
    DataSpec,
    PlotConfig,
    RenderOptions,
    Plot,
    RenderResult,
    VisualizationError,
    PlotCreationError,
    RenderingError,
    PluginError,
)

# Data processing pipeline
from experiment_manager.visualization.data import (
    DataPipeline,
    DataProcessor,
    ProcessingContext,
    ProcessingResult,
    CacheStrategy,
    PipelineExecutionError,
    ProcessorValidationError,
    ProcessorRegistrationError,
    # Built-in processors
    FilterProcessor,
    AggregationProcessor,
    NormalizationProcessor,
    get_builtin_processors,
    register_builtin_processors,
)

# Plugin interfaces
from experiment_manager.visualization.plugins import (
    BasePlugin,
    PlotPlugin,
    RendererPlugin,
    ExportPlugin,
    DataProcessorPlugin,
    ThemePlugin,
)

# Theme registry and management
from experiment_manager.visualization.themes import (
    ThemeRegistry,
    DefaultThemePlugin,
    DarkThemePlugin,
    PublicationThemePlugin,
)

# Renderer plugins
from experiment_manager.visualization.renderers import (
    MatplotlibRendererPlugin,
)

__version__ = "0.1.0"
__all__ = [
    # Main visualizer manager
    "VisualizerManager",
    "DataSpec",
    "PlotConfig", 
    "RenderOptions",
    "Plot",
    "RenderResult",
    "VisualizationError",
    "PlotCreationError",
    "RenderingError",
    "PluginError",
    # Core components
    "PluginRegistry",
    "PluginType",
    "EventBus",
    "Event",
    "EventType",
    "EventPriority", 
    "EventFilter",
    "ConfigManager",
    "VisualizationConfig",
    "ConfigFormat",
    # Data processing pipeline
    "DataPipeline",
    "DataProcessor",
    "ProcessingContext",
    "ProcessingResult",
    "CacheStrategy",
    "PipelineExecutionError",
    "ProcessorValidationError",
    "ProcessorRegistrationError",
    # Built-in processors
    "FilterProcessor",
    "AggregationProcessor",
    "NormalizationProcessor",
    "get_builtin_processors",
    "register_builtin_processors",
    # Plugin interfaces
    "BasePlugin",
    "PlotPlugin",
    "RendererPlugin",
    "ExportPlugin",
    "DataProcessorPlugin",
    "ThemePlugin",
    # Theme registry and management
    "ThemeRegistry",
    "DefaultThemePlugin",
    "DarkThemePlugin",
    "PublicationThemePlugin",
    # Renderer plugins
    "MatplotlibRendererPlugin",
] 