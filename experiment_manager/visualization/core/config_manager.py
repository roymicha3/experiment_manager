"""
Configuration Management System for Visualization Components

This module provides centralized configuration management with validation,
environment overrides, hot-reloading, and integration with the event system.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydantic import BaseModel, ValidationError, Field, field_validator
from enum import Enum

from .event_bus import EventBus, Event, EventType, EventPriority

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    YAML = "yaml"
    JSON = "json"
    YML = "yml"


@dataclass
class ConfigMetadata:
    """Metadata about a configuration."""
    source_file: Optional[Path] = None
    format: Optional[ConfigFormat] = None
    loaded_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    version: str = "1.0.0"
    environment: str = "default"


class VisualizationConfig(BaseModel):
    """
    Main configuration schema for the visualization system.
    
    This Pydantic model defines the complete configuration structure
    with validation rules and default values.
    """
    
    # Core system settings
    system: Dict[str, Any] = Field(default_factory=lambda: {
        "debug": False,
        "log_level": "INFO",
        "max_workers": 4,
        "enable_hot_reload": True,
        "cache_enabled": True,
        "performance_monitoring": False
    })
    
    # Plugin system configuration
    plugins: Dict[str, Any] = Field(default_factory=lambda: {
        "discovery_paths": ["experiment_manager.visualization.plugins"],
        "auto_discover": True,
        "plugin_timeout": 30,
        "enable_plugin_validation": True,
        "disabled_plugins": []
    })
    
    # Event bus configuration
    events: Dict[str, Any] = Field(default_factory=lambda: {
        "max_history": 1000,
        "thread_pool_size": 4,
        "enable_debugging": False,
        "enable_error_recovery": True,
        "priority_processing": True
    })
    
    # Data pipeline configuration
    data: Dict[str, Any] = Field(default_factory=lambda: {
        "cache_size_mb": 512,
        "cache_ttl_seconds": 3600,
        "chunk_size": 10000,
        "lazy_loading": True,
        "compression_enabled": True,
        "parallel_processing": True
    })
    
    # Rendering configuration
    rendering: Dict[str, Any] = Field(default_factory=lambda: {
        "default_engine": "matplotlib",
        "dpi": 300,
        "figure_size": [10, 6],
        "color_palette": "viridis",
        "font_family": "sans-serif",
        "font_size": 12,
        "enable_interactive": True,
        "render_timeout": 60
    })
    
    # Theme configuration
    themes: Dict[str, Any] = Field(default_factory=lambda: {
        "default_theme": "default",
        "custom_themes_path": "themes",
        "allow_theme_override": True,
        "theme_validation": True
    })
    
    # Export configuration
    export: Dict[str, Any] = Field(default_factory=lambda: {
        "default_format": "png",
        "output_directory": "outputs",
        "include_metadata": True,
        "compress_exports": False,
        "export_timeout": 120
    })
    
    # Dashboard configuration
    dashboard: Dict[str, Any] = Field(default_factory=lambda: {
        "auto_layout": True,
        "grid_size": [12, 8],
        "enable_widgets": True,
        "auto_refresh": False,
        "refresh_interval": 30
    })
    
    # Performance configuration
    performance: Dict[str, Any] = Field(default_factory=lambda: {
        "max_data_points": 100000,
        "downsample_threshold": 50000,
        "memory_limit_mb": 2048,
        "gc_threshold": 0.8,
        "async_rendering": True
    })
    
    # Analytics integration
    analytics: Dict[str, Any] = Field(default_factory=lambda: {
        "database_connection": None,
        "query_timeout": 300,
        "result_caching": True,
        "batch_size": 1000
    })
    
    @field_validator('system')
    @classmethod
    def validate_system_config(cls, v):
        """Validate system configuration."""
        if 'log_level' in v:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if v['log_level'] not in valid_levels:
                raise ValueError(f"log_level must be one of {valid_levels}")
        
        if 'max_workers' in v and v['max_workers'] < 1:
            raise ValueError("max_workers must be at least 1")
        
        return v
    
    @field_validator('rendering')
    @classmethod
    def validate_rendering_config(cls, v):
        """Validate rendering configuration."""
        if 'dpi' in v and v['dpi'] < 50:
            raise ValueError("DPI must be at least 50")
        
        if 'figure_size' in v:
            if not isinstance(v['figure_size'], list) or len(v['figure_size']) != 2:
                raise ValueError("figure_size must be a list of two numbers [width, height]")
        
        return v
    
    @field_validator('performance')
    @classmethod
    def validate_performance_config(cls, v):
        """Validate performance configuration."""
        if 'max_data_points' in v and v['max_data_points'] < 100:
            raise ValueError("max_data_points must be at least 100")
        
        if 'memory_limit_mb' in v and v['memory_limit_mb'] < 64:
            raise ValueError("memory_limit_mb must be at least 64")
        
        return v


class ConfigFileWatcher(FileSystemEventHandler):
    """File system event handler for configuration hot-reloading."""
    
    def __init__(self, config_manager: 'ConfigManager'):
        """
        Initialize the file watcher.
        
        Args:
            config_manager: ConfigManager instance to notify of changes
        """
        super().__init__()
        self.config_manager = config_manager
        self._debounce_time = 1.0  # Seconds to wait before reloading
        self._last_reload = 0
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        current_time = time.time()
        if current_time - self._last_reload < self._debounce_time:
            return
        
        file_path = Path(event.src_path)
        
        # Check if this is a config file we're watching
        if file_path in self.config_manager._watched_files:
            logger.info(f"Configuration file changed: {file_path}")
            self._last_reload = current_time
            
            # Reload configuration in a separate thread to avoid blocking
            import threading
            reload_thread = threading.Thread(
                target=self.config_manager._reload_config_file,
                args=(file_path,),
                daemon=True
            )
            reload_thread.start()


class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass


class ConfigLoadError(ConfigurationError):
    """Raised when configuration loading fails."""
    pass


class ConfigManager:
    """
    Centralized configuration management system.
    
    The ConfigManager provides:
    - Loading from YAML/JSON files with validation
    - Environment variable overrides
    - Hot-reloading of configuration files
    - Event-driven change notifications
    - Thread-safe configuration access
    - Configuration schema validation with Pydantic
    
    Example:
        ```python
        # Initialize with event bus integration
        config_manager = ConfigManager(event_bus=event_bus)
        
        # Load configuration
        config_manager.load_from_file("config.yaml")
        
        # Access configuration
        debug_mode = config_manager.get("system.debug", False)
        
        # Set configuration with validation
        config_manager.set("rendering.dpi", 300)
        
        # Enable hot-reloading
        config_manager.enable_hot_reload()
        ```
    """
    
    def __init__(self, 
                 event_bus: Optional[EventBus] = None,
                 enable_env_override: bool = True,
                 env_prefix: str = "VIZ_"):
        """
        Initialize the configuration manager.
        
        Args:
            event_bus: EventBus instance for change notifications
            enable_env_override: Whether to enable environment variable overrides
            env_prefix: Prefix for environment variables (e.g., VIZ_SYSTEM_DEBUG)
        """
        self._config: VisualizationConfig = VisualizationConfig()
        self._metadata = ConfigMetadata()
        self._event_bus = event_bus
        self._enable_env_override = enable_env_override
        self._env_prefix = env_prefix
        
        # Hot-reloading support
        self._observer: Optional[Observer] = None
        self._file_watcher: Optional[ConfigFileWatcher] = None
        self._watched_files: set[Path] = set()
        self._hot_reload_enabled = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Change callbacks
        self._change_callbacks: List[Callable[[str, Any, Any], None]] = []
        
        # Load environment overrides
        if self._enable_env_override:
            self._apply_environment_overrides()
        
        logger.info("ConfigManager initialized")
    
    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to configuration file (YAML or JSON)
            
        Raises:
            ConfigLoadError: If file cannot be loaded
            ConfigValidationError: If configuration is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigLoadError(f"Configuration file not found: {file_path}")
        
        # Determine format from extension
        format_map = {
            '.yaml': ConfigFormat.YAML,
            '.yml': ConfigFormat.YML,
            '.json': ConfigFormat.JSON
        }
        
        file_format = format_map.get(file_path.suffix.lower())
        if not file_format:
            raise ConfigLoadError(f"Unsupported config format: {file_path.suffix}")
        
        logger.info(f"Loading configuration from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_format in [ConfigFormat.YAML, ConfigFormat.YML]:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Validate and create new config
            new_config = VisualizationConfig(**data)
            
            with self._lock:
                old_config_dict = self._config.model_dump()
                self._config = new_config
                self._metadata.source_file = file_path
                self._metadata.format = file_format
                self._metadata.loaded_at = datetime.now()
                self._metadata.modified_at = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # Apply environment overrides
                if self._enable_env_override:
                    self._apply_environment_overrides()
                
                # Add to watched files for hot-reloading
                self._watched_files.add(file_path)
            
            # Notify of configuration change
            self._notify_config_changed("config_loaded", old_config_dict, self._config.model_dump())
            
            logger.info(f"Configuration loaded successfully from {file_path}")
            
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"YAML parsing error: {e}")
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"JSON parsing error: {e}")
        except ValidationError as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Unexpected error loading config: {e}")
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from a dictionary.
        
        Args:
            config_dict: Configuration data
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        try:
            with self._lock:
                old_config_dict = self._config.model_dump()
                self._config = VisualizationConfig(**config_dict)
                self._metadata = ConfigMetadata()
                
                # Apply environment overrides
                if self._enable_env_override:
                    self._apply_environment_overrides()
            
            # Notify of configuration change
            self._notify_config_changed("config_loaded", old_config_dict, self._config.model_dump())
            
            logger.info("Configuration loaded from dictionary")
            
        except ValidationError as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "system.debug")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        with self._lock:
            config_dict = self._config.model_dump()
        
        # Navigate through nested keys
        keys = key.split('.')
        value = config_dict
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "system.debug")
            value: Value to set
            
        Raises:
            ConfigValidationError: If new configuration is invalid
        """
        with self._lock:
            # Get current config as dict
            config_dict = self._config.model_dump()
            old_value = self.get(key)
            
            # Set the new value
            keys = key.split('.')
            current = config_dict
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
            
            # Validate the new configuration
            try:
                new_config = VisualizationConfig(**config_dict)
                self._config = new_config
            except ValidationError as e:
                raise ConfigValidationError(f"Invalid configuration: {e}")
        
        # Notify of change
        self._notify_config_changed(key, old_value, value)
        
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def get_config(self) -> VisualizationConfig:
        """
        Get the complete configuration object.
        
        Returns:
            Current VisualizationConfig instance
        """
        with self._lock:
            return self._config.model_copy(deep=True)
    
    def get_metadata(self) -> ConfigMetadata:
        """
        Get configuration metadata.
        
        Returns:
            ConfigMetadata instance
        """
        with self._lock:
            return self._metadata
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        try:
            with self._lock:
                # Re-validate current config
                config_dict = self._config.model_dump()
                VisualizationConfig(**config_dict)
            return True
        except ValidationError as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
    
    def export_to_file(self, file_path: Union[str, Path], 
                      format: Optional[ConfigFormat] = None) -> None:
        """
        Export current configuration to a file.
        
        Args:
            file_path: Output file path
            format: Output format (auto-detected from extension if None)
        """
        file_path = Path(file_path)
        
        if format is None:
            # Auto-detect format from extension
            format_map = {
                '.yaml': ConfigFormat.YAML,
                '.yml': ConfigFormat.YML,
                '.json': ConfigFormat.JSON
            }
            format = format_map.get(file_path.suffix.lower(), ConfigFormat.YAML)
        
        with self._lock:
            config_dict = self._config.model_dump()
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format in [ConfigFormat.YAML, ConfigFormat.YML]:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration exported to: {file_path}")
    
    def enable_hot_reload(self) -> None:
        """Enable hot-reloading of configuration files."""
        if self._hot_reload_enabled:
            return
        
        if not self._watched_files:
            logger.warning("No configuration files to watch for hot-reloading")
            return
        
        self._file_watcher = ConfigFileWatcher(self)
        self._observer = Observer()
        
        # Watch directories containing config files
        watched_dirs = set()
        for file_path in self._watched_files:
            dir_path = file_path.parent
            if dir_path not in watched_dirs:
                self._observer.schedule(self._file_watcher, str(dir_path), recursive=False)
                watched_dirs.add(dir_path)
        
        self._observer.start()
        self._hot_reload_enabled = True
        
        logger.info("Hot-reloading enabled for configuration files")
    
    def disable_hot_reload(self) -> None:
        """Disable hot-reloading of configuration files."""
        if not self._hot_reload_enabled:
            return
        
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        
        self._file_watcher = None
        self._hot_reload_enabled = False
        
        logger.info("Hot-reloading disabled")
    
    def add_change_callback(self, callback: Callable[[str, Any, Any], None]) -> None:
        """
        Add a callback for configuration changes.
        
        Args:
            callback: Function called with (key, old_value, new_value)
        """
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[str, Any, Any], None]) -> None:
        """
        Remove a configuration change callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    def shutdown(self) -> None:
        """Shutdown the configuration manager and cleanup resources."""
        logger.info("Shutting down ConfigManager...")
        
        self.disable_hot_reload()
        
        with self._lock:
            self._change_callbacks.clear()
            self._watched_files.clear()
        
        logger.info("ConfigManager shutdown complete")
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        config_dict = self._config.model_dump()
        overrides_applied = 0
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(self._env_prefix):
                continue
            
            # Convert environment key to config key
            # VIZ_SYSTEM_DEBUG -> system.debug
            config_key = env_key[len(self._env_prefix):].lower().replace('_', '.')
            
            # Try to parse the value
            parsed_value = self._parse_env_value(env_value)
            
            # Set in config dict
            keys = config_key.split('.')
            current = config_dict
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            if isinstance(current, dict):
                current[keys[-1]] = parsed_value
                overrides_applied += 1
                logger.debug(f"Environment override: {config_key} = {parsed_value}")
        
        if overrides_applied > 0:
            try:
                self._config = VisualizationConfig(**config_dict)
                logger.info(f"Applied {overrides_applied} environment overrides")
            except ValidationError as e:
                logger.error(f"Environment overrides caused validation error: {e}")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try boolean first
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try JSON (for lists/dicts)
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Return as string
        return value
    
    def _reload_config_file(self, file_path: Path) -> None:
        """Reload configuration from a specific file."""
        try:
            logger.info(f"Reloading configuration from: {file_path}")
            self.load_from_file(file_path)
            
            # Notify via event bus
            if self._event_bus:
                event = Event(
                    event_type=EventType.CONFIG_CHANGED,
                    data={"file_path": str(file_path), "reload": True},
                    source="config_manager",
                    priority=EventPriority.HIGH
                )
                self._event_bus.publish(event)
            
        except Exception as e:
            logger.error(f"Failed to reload configuration from {file_path}: {e}")
            
            # Notify of error via event bus
            if self._event_bus:
                event = Event(
                    event_type=EventType.PLUGIN_ERROR,
                    data={
                        "component": "config_manager",
                        "error": str(e),
                        "file_path": str(file_path)
                    },
                    source="config_manager",
                    priority=EventPriority.HIGH
                )
                self._event_bus.publish(event)
    
    def _notify_config_changed(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify listeners of configuration changes."""
        # Call callbacks
        for callback in self._change_callbacks:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")
        
        # Publish event
        if self._event_bus:
            event = Event(
                event_type=EventType.CONFIG_CHANGED,
                data={
                    "key": key,
                    "old_value": old_value,
                    "new_value": new_value
                },
                source="config_manager",
                priority=EventPriority.NORMAL
            )
            self._event_bus.publish(event) 