"""
Visualizer Manager - Central Orchestrator for Visualization System

This module provides the main VisualizerManager class that coordinates all
visualization components including plugins, renderers, data processing,
and configuration management. It serves as the primary entry point for
creating and managing visualizations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from omegaconf import DictConfig, OmegaConf

from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.visualization.core.plugin_registry import (
    PluginRegistry, PluginType, PluginInfo, BasePlugin
)
from experiment_manager.visualization.core.event_bus import (
    EventBus, Event, EventType, EventPriority
)
from experiment_manager.visualization.core.config_manager import (
    ConfigManager, VisualizationConfig
)
from experiment_manager.visualization.data import (
    DataPipeline, DataProcessor, ProcessingContext, ProcessingResult,
    CacheStrategy, PerformanceLevel
)
from experiment_manager.visualization.plugins import (
    PlotPlugin, RendererPlugin, ExportPlugin, ThemePlugin
)

logger = logging.getLogger(__name__)


@dataclass
class DataSpec:
    """
    Specification for data to be visualized.
    
    This class defines what data should be retrieved and processed
    for visualization, including filtering, aggregation, and metadata.
    """
    experiment_ids: Optional[List[int]] = None
    trial_ids: Optional[List[int]] = None
    trial_run_ids: Optional[List[int]] = None
    metric_types: Optional[List[str]] = None
    time_range: Optional[tuple] = None  # (start_datetime, end_datetime)
    filters: Optional[Dict[str, Any]] = None
    aggregation: Optional[str] = None
    processing_hints: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and normalize data specification."""
        if self.filters is None:
            self.filters = {}
        if self.processing_hints is None:
            self.processing_hints = {}
    
    def validate(self) -> List[str]:
        """
        Validate data specification.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # At least one data identifier must be provided
        if not any([
            self.experiment_ids,
            self.trial_ids, 
            self.trial_run_ids
        ]):
            errors.append("At least one of experiment_ids, trial_ids, or trial_run_ids must be provided")
        
        # Check for empty lists
        for field_name, field_value in [
            ("experiment_ids", self.experiment_ids),
            ("trial_ids", self.trial_ids),
            ("trial_run_ids", self.trial_run_ids),
            ("metric_types", self.metric_types)
        ]:
            if field_value is not None and len(field_value) == 0:
                errors.append(f"{field_name} cannot be empty if provided")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'experiment_ids': self.experiment_ids,
            'trial_ids': self.trial_ids,
            'trial_run_ids': self.trial_run_ids,
            'metric_types': self.metric_types,
            'time_range': self.time_range,
            'filters': self.filters,
            'aggregation': self.aggregation,
            'processing_hints': self.processing_hints
        }


@dataclass
class PlotConfig:
    """Configuration for plot creation and rendering."""
    theme: str = "default"
    style_overrides: Dict[str, Any] = field(default_factory=dict)
    export_settings: Dict[str, Any] = field(default_factory=dict)
    interactive: bool = False
    title: Optional[str] = None
    subtitle: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'theme': self.theme,
            'style_overrides': self.style_overrides,
            'export_settings': self.export_settings,
            'interactive': self.interactive,
            'title': self.title,
            'subtitle': self.subtitle
        }


@dataclass
class RenderOptions:
    """Options for plot rendering."""
    renderer: str = "matplotlib"
    output_format: str = "png"
    output_path: Optional[Path] = None
    dpi: int = 300
    quality: int = 90
    size: tuple = (10, 6)  # (width, height) in inches
    transparent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'renderer': self.renderer,
            'output_format': self.output_format,
            'output_path': str(self.output_path) if self.output_path else None,
            'dpi': self.dpi,
            'quality': self.quality,
            'size': self.size,
            'transparent': self.transparent
        }


class Plot:
    """
    Represents a generated plot with metadata and rendering capabilities.
    
    This class encapsulates a plot object along with its configuration,
    data specification, and provides methods for rendering and exporting.
    """
    
    def __init__(self, 
                 plot_object: Any,
                 plot_type: str,
                 data_spec: DataSpec,
                 config: PlotConfig,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Plot.
        
        Args:
            plot_object: The actual plot object (e.g., matplotlib figure)
            plot_type: Type of plot (e.g., "training_curves", "scatter")
            data_spec: Data specification used to create this plot
            config: Configuration used to create this plot
            metadata: Additional metadata about the plot
        """
        self.plot_object = plot_object
        self.plot_type = plot_type
        self.data_spec = data_spec
        self.config = config
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata key-value pair."""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plot information to dictionary (excluding plot_object)."""
        return {
            'plot_type': self.plot_type,
            'data_spec': self.data_spec.to_dict(),
            'config': self.config.to_dict(),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class RenderResult:
    """
    Result of a plot rendering operation.
    
    Contains the rendered data, metadata, and any warnings or errors
    that occurred during rendering.
    """
    
    def __init__(self,
                 data: Union[bytes, str],
                 format: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 success: bool = True,
                 warnings: Optional[List[str]] = None,
                 error_message: Optional[str] = None):
        """
        Initialize RenderResult.
        
        Args:
            data: Rendered data (bytes for binary formats, str for text)
            format: Output format (e.g., 'png', 'svg', 'html')
            metadata: Rendering metadata
            success: Whether rendering was successful
            warnings: List of warning messages
            error_message: Error message if rendering failed
        """
        self.data = data
        self.format = format
        self.metadata = metadata or {}
        self.success = success
        self.warnings = warnings or []
        self.error_message = error_message
        self.rendered_at = datetime.now()
    
    @property
    def size(self) -> int:
        """Get size of rendered data in bytes."""
        if isinstance(self.data, bytes):
            return len(self.data)
        elif isinstance(self.data, str):
            return len(self.data.encode('utf-8'))
        return 0
    
    def save_to_file(self, path: Path) -> None:
        """
        Save rendered data to file.
        
        Args:
            path: Path where to save the file
        """
        if isinstance(self.data, bytes):
            with open(path, 'wb') as f:
                f.write(self.data)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.data)


class VisualizationError(Exception):
    """Base exception for visualization errors."""
    pass


class PlotCreationError(VisualizationError):
    """Raised when plot creation fails."""
    pass


class RenderingError(VisualizationError):
    """Raised when plot rendering fails."""
    pass


class PluginError(VisualizationError):
    """Raised when plugin operations fail."""
    pass


@YAMLSerializable.register("VisualizerManager")
class VisualizerManager(YAMLSerializable):
    """
    Central manager coordinating all visualization components.
    
    The VisualizerManager serves as the main entry point for the visualization
    system. It coordinates plugins, manages data processing pipelines, handles
    configuration, and provides high-level APIs for creating and rendering plots.
    
    Key responsibilities:
    - Plugin registration and management
    - Plot creation and rendering
    - Data pipeline coordination
    - Configuration management
    - Event coordination
    - Resource cleanup
    
    Example:
        ```python
        # Initialize from default configuration
        visualizer = VisualizerManager.from_config()
        
        # Create a training curves plot
        plot = visualizer.create_plot(
            plot_type="training_curves",
            data_spec=DataSpec(trial_run_ids=[1, 2, 3])
        )
        
        # Render and save
        result = visualizer.render_plot(plot, RenderOptions(
            renderer="matplotlib",
            output_format="png",
            output_path=Path("training_curves.png")
        ))
        ```
    """
    
    def __init__(self, config: Optional[Union[VisualizationConfig, DictConfig, Dict[str, Any]]] = None):
        """
        Initialize VisualizerManager.
        
        Args:
            config: Visualization configuration (VisualizationConfig, DictConfig, or dict)
        """
        super().__init__()
        
        # Initialize configuration
        self.config_manager = ConfigManager()
        if config is not None:
            if isinstance(config, VisualizationConfig):
                self.config_manager._config = config
            elif isinstance(config, (DictConfig, dict)):
                self.config_manager.load_from_dict(dict(config))
            else:
                raise ValueError(f"Invalid config type: {type(config)}")
        else:
            # Use default configuration
            self.config_manager._config = VisualizationConfig()
        
        # Initialize core components
        self.event_bus = EventBus(
            max_history=self.config_manager.get("events.max_history", 1000),
            thread_pool_size=self.config_manager.get("events.thread_pool_size", 4),
            enable_debugging=self.config_manager.get("events.enable_debugging", False)
        )
        
        self.plugin_registry = PluginRegistry()
        # Convert config values to proper enums
        cache_strategy_str = self.config_manager.get("data.cache_strategy", "memory")
        cache_strategy = getattr(CacheStrategy, cache_strategy_str.upper(), CacheStrategy.MEMORY)
        
        performance_level_str = self.config_manager.get("data.performance_level", "basic")
        performance_level = getattr(PerformanceLevel, performance_level_str.upper(), PerformanceLevel.BASIC)
        
        self.data_pipeline = DataPipeline(
            pipeline_id=f"visualizer_pipeline_{id(self)}",
            event_bus=self.event_bus,
            cache_strategy=cache_strategy,
            max_parallel_processors=self.config_manager.get("data.max_parallel_processors", 4),
            enable_rollback=self.config_manager.get("data.enable_rollback", True),
            performance_level=performance_level
        )
        
        # State tracking
        self._initialized = False
        self._cleanup_callbacks = []
        self._active_plots = []
        
        # Subscribe to events
        self._setup_event_handlers()
        
        logger.info("VisualizerManager initialized")
    
    def initialize(self) -> None:
        """
        Initialize the visualization system.
        
        This method performs the complete initialization sequence:
        - Plugin discovery and registration
        - Data pipeline setup
        - Event system activation
        - Resource allocation
        """
        if self._initialized:
            logger.warning("VisualizerManager already initialized")
            return
        
        try:
            logger.info("Initializing VisualizerManager")
            
            # Publish initialization start event
            self.event_bus.publish(Event(
                event_type=EventType.SYSTEM_STARTUP,
                source="VisualizerManager",
                data={"component": "VisualizerManager"},
                priority=EventPriority.HIGH
            ), async_mode=False)
            
            # Discover and register plugins
            self._discover_plugins()
            
            # Initialize data pipeline with built-in processors
            self._setup_data_pipeline()
            
            # Mark as initialized
            self._initialized = True
            
            logger.info("VisualizerManager initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize VisualizerManager: {e}")
            raise VisualizationError(f"Initialization failed: {e}") from e
    
    def create_plot(self,
                   plot_type: str,
                   data_spec: DataSpec,
                   config: Optional[Union[PlotConfig, Dict[str, Any]]] = None) -> Plot:
        """
        Create a plot using registered plugins.
        
        Args:
            plot_type: Type of plot to create (e.g., "training_curves", "scatter")
            data_spec: Specification of data to visualize
            config: Plot configuration (PlotConfig or dict)
            
        Returns:
            Plot: Created plot object
            
        Raises:
            PlotCreationError: If plot creation fails
            PluginError: If required plugin is not found
        """
        if not self._initialized:
            self.initialize()
        
        # Validate inputs
        validation_errors = data_spec.validate()
        if validation_errors:
            raise PlotCreationError(f"Invalid data specification: {', '.join(validation_errors)}")
        
        # Normalize config
        if config is None:
            config = PlotConfig()
        elif isinstance(config, dict):
            config = PlotConfig(**config)
        
        logger.info(f"Creating plot of type '{plot_type}'")
        
        try:
            # Publish plot creation start event
            self.event_bus.publish(Event(
                event_type=EventType.PLOT_CREATING,
                source="VisualizerManager",
                data={
                    "plot_type": plot_type,
                    "data_spec": data_spec.to_dict(),
                    "config": config.to_dict()
                }
            ), async_mode=False)
            
            # Find appropriate plot plugin
            plot_plugin = self._get_plot_plugin(plot_type)
            
            # Process data through pipeline
            processed_data = self._process_data(data_spec, plot_type)
            
            # Generate plot using plugin
            plot_result = plot_plugin.generate_plot(processed_data, config.to_dict())
            
            if not plot_result.success:
                raise PlotCreationError(f"Plot generation failed: {plot_result.error_message}")
            
            # Create Plot object
            plot = Plot(
                plot_object=plot_result.plot_object,
                plot_type=plot_type,
                data_spec=data_spec,
                config=config,
                metadata=plot_result.metadata
            )
            
            # Track active plot
            self._active_plots.append(plot)
            
            # Publish plot creation complete event
            self.event_bus.publish(Event(
                event_type=EventType.PLOT_CREATED,
                source="VisualizerManager",
                data={
                    "plot_type": plot_type,
                    "plot_id": id(plot),
                    "metadata": plot.metadata
                }
            ), async_mode=False)
            
            logger.info(f"Successfully created plot of type '{plot_type}'")
            return plot
            
        except Exception as e:
            logger.error(f"Failed to create plot '{plot_type}': {e}")
            
            # Publish error event
            self.event_bus.publish(Event(
                event_type=EventType.PLOT_ERROR,
                source="VisualizerManager",
                data={
                    "plot_type": plot_type,
                    "error": str(e)
                },
                priority=EventPriority.HIGH
            ), async_mode=False)
            
            raise PlotCreationError(f"Failed to create plot '{plot_type}': {e}") from e
    
    def render_plot(self,
                   plot: Plot,
                   options: Optional[Union[RenderOptions, Dict[str, Any]]] = None) -> RenderResult:
        """
        Render plot using specified renderer.
        
        Args:
            plot: Plot to render
            options: Rendering options (RenderOptions or dict)
            
        Returns:
            RenderResult: Rendered plot data and metadata
            
        Raises:
            RenderingError: If rendering fails
            PluginError: If required renderer is not found
        """
        if not self._initialized:
            self.initialize()
        
        # Normalize options
        if options is None:
            options = RenderOptions()
        elif isinstance(options, dict):
            options = RenderOptions(**options)
        
        logger.info(f"Rendering plot with '{options.renderer}' renderer")
        
        try:
            # Publish rendering start event
            self.event_bus.publish(Event(
                event_type=EventType.PLOT_RENDERING,
                source="VisualizerManager",
                data={
                    "plot_type": plot.plot_type,
                    "plot_id": id(plot),
                    "renderer": options.renderer,
                    "format": options.output_format
                }
            ), async_mode=False)
            
            # Find appropriate renderer plugin
            renderer_plugin = self._get_renderer_plugin(options.renderer)
            
            # Check if renderer can handle this plot and format
            if not renderer_plugin.can_render(plot.plot_object, options.output_format):
                raise RenderingError(
                    f"Renderer '{options.renderer}' cannot render plot type "
                    f"'{plot.plot_type}' to format '{options.output_format}'"
                )
            
            # Create render context
            from experiment_manager.visualization.plugins.renderer_plugin import RenderContext
            render_context = RenderContext(
                output_format=options.output_format,
                output_path=options.output_path,
                dpi=options.dpi,
                quality=options.quality,
                interactive=getattr(options, 'interactive', False),
                embed_data=getattr(options, 'embed_data', False)
            )
            
            # Render plot
            render_result = renderer_plugin.render(
                plot.plot_object,
                render_context,
                config=options.to_dict()
            )
            
            if not render_result.success:
                raise RenderingError(f"Rendering failed: {render_result.error_message}")
            
            # Create RenderResult
            result = RenderResult(
                data=render_result.data,
                format=options.output_format,
                metadata=render_result.metadata,
                success=True
            )
            
            # Save to file if path specified
            if options.output_path:
                result.save_to_file(options.output_path)
                logger.info(f"Plot saved to {options.output_path}")
            
            # Publish rendering complete event
            self.event_bus.publish(Event(
                event_type=EventType.PLOT_RENDERED,
                source="VisualizerManager",
                data={
                    "plot_type": plot.plot_type,
                    "plot_id": id(plot),
                    "renderer": options.renderer,
                    "format": options.output_format,
                    "size": result.size
                }
            ), async_mode=False)
            
            logger.info(f"Successfully rendered plot with '{options.renderer}' renderer")
            return result
            
        except Exception as e:
            logger.error(f"Failed to render plot: {e}")
            
            # Publish error event
            self.event_bus.publish(Event(
                event_type=EventType.PLOT_ERROR,
                source="VisualizerManager",
                data={
                    "plot_type": plot.plot_type,
                    "plot_id": id(plot),
                    "renderer": options.renderer,
                    "error": str(e)
                },
                priority=EventPriority.HIGH
            ), async_mode=False)
            
            raise RenderingError(f"Failed to render plot: {e}") from e
    
    def get_available_plot_types(self) -> List[str]:
        """
        Get list of available plot types.
        
        Returns:
            List of plot type names that can be created
        """
        if not self._initialized:
            self.initialize()
        
        plot_plugins = self.plugin_registry.get_plugins_by_type(PluginType.PLOT)
        return [plugin.name for plugin in plot_plugins if plugin.is_enabled]
    
    def get_available_renderers(self) -> List[str]:
        """
        Get list of available renderers.
        
        Returns:
            List of renderer names that can be used
        """
        if not self._initialized:
            self.initialize()
        
        renderer_plugins = self.plugin_registry.get_plugins_by_type(PluginType.RENDERER)
        return [plugin.name for plugin in renderer_plugins if plugin.is_enabled]
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """
        Get information about a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            PluginInfo if found, None otherwise
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self.plugin_registry.get_plugin_info(plugin_name)
        except Exception:
            return None
    
    def shutdown(self) -> None:
        """
        Shutdown the visualization system and cleanup resources.
        
        This method performs graceful shutdown:
        - Cleanup active plots
        - Shutdown plugins
        - Stop event bus
        - Free resources
        """
        if not self._initialized:
            return
        
        logger.info("Shutting down VisualizerManager")
        
        try:
            # Publish shutdown event
            self.event_bus.publish(Event(
                event_type=EventType.SYSTEM_SHUTDOWN,
                source="VisualizerManager",
                priority=EventPriority.CRITICAL
            ), async_mode=False)
            
            # Cleanup active plots
            self._active_plots.clear()
            
            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")
            
            # Cleanup components
            self.plugin_registry.cleanup_all()
            self.event_bus.shutdown()
            self.config_manager.shutdown()
            
            self._initialized = False
            
            logger.info("VisualizerManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def add_cleanup_callback(self, callback) -> None:
        """
        Add callback to be executed during shutdown.
        
        Args:
            callback: Function to call during cleanup
        """
        self._cleanup_callbacks.append(callback)
    
    @classmethod
    def from_config(cls, config: Optional[Union[str, Path, DictConfig, Dict[str, Any]]] = None) -> 'VisualizerManager':
        """
        Create VisualizerManager from configuration.
        
        Args:
            config: Configuration source - file path, DictConfig, or dict
                   If None, uses default configuration
            
        Returns:
            Configured VisualizerManager instance
        """
        if config is None:
            return cls()
        
        if isinstance(config, (str, Path)):
            # Load from file
            config_manager = ConfigManager()
            config_manager.load_from_file(config)
            return cls(config_manager.get_config())
        
        elif isinstance(config, (DictConfig, dict)):
            return cls(config)
        
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
    
    # Private methods
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for internal events."""
        # Subscribe to plugin events
        self.event_bus.subscribe(
            self,
            self._handle_plugin_event,
            event_filter=None  # Handle all events for now
        )
    
    def _handle_plugin_event(self, event: Event) -> None:
        """Handle plugin-related events."""
        if event.event_name == EventType.PLUGIN_ERROR.value:
            logger.warning(f"Plugin error: {event.data}")
        elif event.event_name == EventType.PLUGIN_REGISTERED.value:
            logger.info(f"Plugin registered: {event.data.get('plugin_name')}")
    
    def _discover_plugins(self) -> None:
        """Discover and register plugins from configured paths."""
        discovery_paths = self.config_manager.get("plugins.discovery_paths", [
            "experiment_manager.visualization.plugins"
        ])
        
        for path in discovery_paths:
            try:
                count = self.plugin_registry.discover_plugins(path)
                logger.info(f"Discovered {count} plugins from {path}")
            except Exception as e:
                logger.warning(f"Failed to discover plugins from {path}: {e}")
    
    def _setup_data_pipeline(self) -> None:
        """Setup data pipeline with built-in processors."""
        try:
            from experiment_manager.visualization.data import register_builtin_processors
            register_builtin_processors(self.data_pipeline)
            logger.info("Data pipeline initialized with built-in processors")
        except Exception as e:
            logger.warning(f"Failed to setup data pipeline: {e}")
    
    def _get_plot_plugin(self, plot_type: str) -> PlotPlugin:
        """
        Get plot plugin for specified type.
        
        Args:
            plot_type: Type of plot to create
            
        Returns:
            PlotPlugin instance
            
        Raises:
            PluginError: If plugin not found or cannot be created
        """
        plugin_name = f"plot.{plot_type}"
        try:
            plugin = self.plugin_registry.create_plugin(plugin_name)
            if not isinstance(plugin, PlotPlugin):
                raise PluginError(f"Plugin '{plugin_name}' is not a valid PlotPlugin")
            return plugin
        except Exception as e:
            raise PluginError(f"Cannot create plot plugin '{plugin_name}': {e}") from e
    
    def _get_renderer_plugin(self, renderer_name: str) -> RendererPlugin:
        """
        Get renderer plugin for specified renderer.
        
        Args:
            renderer_name: Name of renderer to use
            
        Returns:
            RendererPlugin instance
            
        Raises:
            PluginError: If plugin not found or cannot be created
        """
        plugin_name = f"renderer.{renderer_name}"
        try:
            plugin = self.plugin_registry.create_plugin(plugin_name)
            if not isinstance(plugin, RendererPlugin):
                raise PluginError(f"Plugin '{plugin_name}' is not a valid RendererPlugin")
            return plugin
        except Exception as e:
            raise PluginError(f"Cannot create renderer plugin '{plugin_name}': {e}") from e
    
    def _process_data(self, data_spec: DataSpec, plot_type: str) -> Any:
        """
        Process data through the data pipeline.
        
        Args:
            data_spec: Data specification
            plot_type: Type of plot being created
            
        Returns:
            Processed data suitable for plot creation
        """
        # For now, return a placeholder data structure
        # In a full implementation, this would:
        # 1. Query data based on data_spec
        # 2. Process through data pipeline
        # 3. Return processed data in format expected by plot plugins
        
        logger.info(f"Processing data for plot type '{plot_type}'")
        
        # Placeholder implementation
        from experiment_manager.visualization.plugins.plot_plugin import PlotData
        import pandas as pd
        
        # Create sample data based on data_spec
        data_dict = {
            'data_spec': data_spec.to_dict(),
            'plot_type': plot_type,
            'processed_at': datetime.now().isoformat()
        }
        
        # Convert to DataFrame for compatibility
        df = pd.DataFrame([data_dict])
        
        return PlotData(
            data=df,
            metadata={
                'data_spec': data_spec.to_dict(),
                'plot_type': plot_type,
                'processing_timestamp': datetime.now().isoformat()
            }
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def __repr__(self) -> str:
        """String representation."""
        plot_types = len(self.get_available_plot_types()) if self._initialized else 0
        renderers = len(self.get_available_renderers()) if self._initialized else 0
        
        return (
            f"VisualizerManager("
            f"initialized={self._initialized}, "
            f"plot_types={plot_types}, "
            f"renderers={renderers}, "
            f"active_plots={len(self._active_plots)}"
            f")"
        ) 