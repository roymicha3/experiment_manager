import os
from omegaconf import OmegaConf, DictConfig

from experiment_manager.common.factory import Factory
from experiment_manager.common.common import LOG_NAME, Level
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.trackers.tracker_manager import TrackerManager
from experiment_manager.trackers.tracker import Tracker
from experiment_manager.logger import FileLogger, ConsoleLogger, CompositeLogger, EmptyLogger


class Environment(YAMLSerializable):
    """
    Environment class for managing the environment.
    All experiment outputs will be written to the workspace directory.
    """
    CONFIG_FILE = "env.yaml"
    
    LOG_DIR = "logs"
    ARTIFACT_DIR = "artifacts"
    CONFIG_DIR = "configs"
    
    def __init__(self, 
        workspace: str,
                 config: DictConfig,
                 factory: Factory = None,
                 verbose: bool = False,
                 level: Level = Level.EXPERIMENT):
        super().__init__()
        self.workspace = os.path.abspath(workspace)  # Convert to absolute path
        self.config = config
        self.factory = factory
        self.verbose = verbose
        self.level = level
        
        self.logger = EmptyLogger()
        if self.verbose:
            self.logger = ConsoleLogger(name=LOG_NAME)

        self.tracker_manager = TrackerManager.from_config(
            self.config,
            self.artifact_dir
        )
        
    def setup_environment(self, verbose: bool = None) -> None:
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        if verbose is not None:
            self.verbose = verbose
            
        if self.verbose:
            self.logger = CompositeLogger(name=LOG_NAME,
                                         log_dir=self.log_dir)
        else:
            self.logger = FileLogger(name=LOG_NAME,
                                     log_dir=self.log_dir)
        
        self.save()  # Save config after setup
        
    def set_workspace(self, new_workspace: str, inner: bool = False) -> None:
        """Set a new workspace and create required directories."""
        if inner:
            self.workspace = os.path.join(self.workspace, new_workspace)
        else:
            self.workspace = os.path.abspath(new_workspace)
        
        # Create required directories
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Update the logger
        if not isinstance(self.logger, EmptyLogger) and not isinstance(self.logger, ConsoleLogger):
            self.logger.set_log_dir(self.log_dir)
        
        self.save()  # Save config after workspace change
    
    @property
    def log_dir(self):
        return os.path.join(self.workspace, Environment.LOG_DIR)
    
    @property
    def artifact_dir(self):
        return os.path.join(self.workspace, Environment.ARTIFACT_DIR)
    
    @property
    def config_dir(self):
        return os.path.join(self.workspace, Environment.CONFIG_DIR)
    
    def save(self) -> None:
        """Save environment configuration to file."""
        config_path = os.path.join(self.config_dir, Environment.CONFIG_FILE)
        OmegaConf.save(self.config, config_path)
        self.logger.debug(f"Saved environment config to {config_path}")
    
    @classmethod
    def from_config(cls, config: DictConfig):
        """Create environment from configuration."""
        verbose = config.get("verbose", False)
        env = cls(workspace=config.workspace, config=config, verbose=verbose, level=Level.EXPERIMENT)
        return env
    
    def copy(self):
        env = self.__class__(
            workspace=self.workspace,
            config=self.config,
            factory=self.factory,
            verbose=self.verbose,
            level=self.level
        )

        env.tracker_manager = self.tracker_manager
        return env
        
    def create_child(self, name: str) -> 'Environment':
        """
        Create a child environment with its own workspace.
        """
        # Calculate next level based on current level
        next_level = {
            Level.EXPERIMENT: Level.TRIAL,
            Level.TRIAL: Level.TRIAL_RUN,
            Level.TRIAL_RUN: Level.PIPELINE,
            Level.PIPELINE: Level.EPOCH,
            Level.EPOCH: Level.EPOCH
        }[self.level]
        
        child_env = self.__class__(
            workspace=os.path.join(self.workspace, name),
            config=self.config,
            factory=self.factory,
            verbose=self.verbose,
            level=next_level
        )
        child_env.setup_environment()
        self.logger.debug(f"Created child environment '{name}' at {child_env.workspace} with level {next_level}")
        return child_env