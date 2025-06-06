import os
import torch
from enum import Enum
from datetime import datetime
from omegaconf import OmegaConf, DictConfig

from experiment_manager.common.factory import Factory
from experiment_manager.common.common import LOG_NAME
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.trackers.tracker_manager import TrackerManager
from experiment_manager.logger import FileLogger, ConsoleLogger, CompositeLogger, EmptyLogger


class ProductPaths(Enum):
    CONFIG_FILE = "env.yaml"
    
    LOG_DIR = "logs"
    ARTIFACT_DIR = "artifacts"
    CONFIG_DIR = "configs"
    ANALYTICS_DIR = "analytics"


class Environment(YAMLSerializable):
    """
    Environment class for managing the environment.
    All experiment outputs will be written to the workspace directory.
    
    The Environment manages both standard workspace directories (logs, artifacts, configs)
    and analytics workspace directories (analytics, reports, cache, analytics artifacts).
    """
    
    def __init__(self, 
                 workspace: str,
                 config: DictConfig,
                 factory: Factory = None,
                 verbose: bool = False, 
                 debug: bool = False,
                 args: DictConfig = None,
                 device: str = "cpu",
                 tracker_manager: TrackerManager = None):
        super().__init__()
        
        self.workspace = os.path.abspath(workspace)  # Convert to absolute path
        os.makedirs(self.workspace, exist_ok=True)
        
        self.config = config
        self.factory = factory
        self.verbose = verbose
        self.debug = debug
        self.args: DictConfig = args or DictConfig({})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_name = f"{LOG_NAME}-{timestamp}"
        
        if self.verbose:
            self.logger = CompositeLogger(name=self.log_name,
                                      log_dir=self.log_dir,
                                      debug=self.debug)
        else:
            self.logger = FileLogger(name=self.log_name,
                                     log_dir=self.log_dir,
                                     debug=self.debug)
        
        self.tracker_manager = tracker_manager or \
                TrackerManager.from_config(
                                    self.config,
                                    self.workspace)
        
        
        self.device = device
        self.logger.info(f"Using device: {self.device}")
        
        self.save()
    
        
    def set_workspace(self, new_workspace: str, inner: bool = False) -> None:
        """
        Set a new workspace and create required directories.
        """
        if inner:
            self.workspace = os.path.join(self.workspace, new_workspace)
        else:
            self.workspace = os.path.abspath(new_workspace)
        
        # Create required directories
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.analytics_dir, exist_ok=True)
        
        # Update the logger
        if not isinstance(self.logger, EmptyLogger) and not isinstance(self.logger, ConsoleLogger):
            self.logger.set_log_dir(self.log_dir)
        
        self.save()  # Save config after workspace change
    
    @property
    def log_dir(self):
        log_dir_path = os.path.join(self.workspace, ProductPaths.LOG_DIR.value)
        if not os.path.exists(log_dir_path):
            os.mkdir(log_dir_path)
        return log_dir_path
    
    @property
    def artifact_dir(self):
        artifact_dir_path = os.path.join(self.workspace, ProductPaths.ARTIFACT_DIR.value)
        if not os.path.exists(artifact_dir_path):
            os.mkdir(artifact_dir_path)
        return artifact_dir_path
    
    @property
    def config_dir(self):
        config_dir_path = os.path.join(self.workspace, ProductPaths.CONFIG_DIR.value)
        if not os.path.exists(config_dir_path):
            os.mkdir(config_dir_path)
        return config_dir_path
    
    @property
    def analytics_dir(self):
        """Analytics workspace directory for analysis results and configurations."""
        analytics_dir_path = os.path.join(self.workspace, ProductPaths.ANALYTICS_DIR.value)
        if not os.path.exists(analytics_dir_path):
            os.mkdir(analytics_dir_path)
        return analytics_dir_path
    
    @property
    def analytics_reports_dir(self):
        """Directory for analytics reports and analysis results."""
        reports_dir_path = os.path.join(self.analytics_dir, "reports")
        if not os.path.exists(reports_dir_path):
            os.mkdir(reports_dir_path)
        return reports_dir_path
    
    @property
    def analytics_cache_dir(self):
        """Directory for analytics caching to improve performance."""
        cache_dir_path = os.path.join(self.analytics_dir, "cache")
        if not os.path.exists(cache_dir_path):
            os.mkdir(cache_dir_path)
        return cache_dir_path
    
    @property
    def analytics_artifacts_dir(self):
        """Directory for analytics-specific artifacts (plots, data exports, etc.)."""
        artifacts_dir_path = os.path.join(self.analytics_dir, "artifacts")
        if not os.path.exists(artifacts_dir_path):
            os.mkdir(artifacts_dir_path)
        return artifacts_dir_path
    
    def create_analytics_artifact_path(self, filename: str) -> str:
        """
        Create a path for an analytics artifact within the analytics artifacts directory.
        
        Args:
            filename: Name of the artifact file
            
        Returns:
            Full path to the artifact location
        """
        return os.path.join(self.analytics_artifacts_dir, filename)
    
    def create_analytics_report_path(self, filename: str) -> str:
        """
        Create a path for an analytics report within the analytics reports directory.
        
        Args:
            filename: Name of the report file
            
        Returns:
            Full path to the report location
        """
        return os.path.join(self.analytics_reports_dir, filename)
    
    def get_analytics_workspace_info(self) -> dict:
        """
        Get information about the analytics workspace structure.
        
        Returns:
            Dictionary containing analytics workspace paths and status
        """
        return {
            "analytics_dir": self.analytics_dir,
            "reports_dir": self.analytics_reports_dir,
            "cache_dir": self.analytics_cache_dir,
            "artifacts_dir": self.analytics_artifacts_dir,
            "directories_exist": {
                "analytics": os.path.exists(self.analytics_dir),
                "reports": os.path.exists(self.analytics_reports_dir),
                "cache": os.path.exists(self.analytics_cache_dir),
                "artifacts": os.path.exists(self.analytics_artifacts_dir)
            }
        }
    
    def save(self) -> None:
        """Save environment configuration to file."""
        config_path = os.path.join(self.config_dir, ProductPaths.CONFIG_FILE.value)
        OmegaConf.save(self.config, config_path)
        self.logger.debug(f"Saved environment config to {config_path}")
    
    @classmethod
    def from_config(cls, config: DictConfig):
        """Create environment from configuration."""
        verbose = config.get("verbose", False)
        debug = config.get("debug", False)
        additional_args = config.get("args", None)
        
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = config.get("device", default_device)
        
        env = cls(
            workspace=config.workspace,
            config=config,
            verbose=verbose,
            debug=debug,
            device=device,
            args=additional_args)
        
        return env
    
    def copy(self):
        env = self.__class__(
            workspace=self.workspace,
            config=self.config,
            factory=self.factory,
            verbose=self.verbose,
            debug=self.debug,
            device=self.device,
            args=self.args)
    
        env.tracker_manager = self.tracker_manager
        return env
        
    def create_child(self, name: str, args: DictConfig = None) -> 'Environment':
        """
        Create a child environment with its own workspace.
        """
        child_workspace = os.path.join(self.workspace, name)
        child_env = self.__class__(
            workspace=child_workspace,
            config=self.config,
            factory=self.factory,
            verbose=self.verbose,
            debug=self.debug,
            tracker_manager=self.tracker_manager.create_child(child_workspace),
            device=self.device,
            args=self.args)
        
        if args:
            child_env.args = OmegaConf.merge(child_env.args, args)
        
        self.logger.debug(f"Created child environment '{name}' at {child_env.workspace}")
        return child_env