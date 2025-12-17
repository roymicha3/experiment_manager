import os
from omegaconf import OmegaConf, DictConfig

from experiment_manager.trial import Trial
from experiment_manager.common.factory_registry import FactoryRegistry
from experiment_manager.environment import Environment
from experiment_manager.common.common import Level, ConfigPaths
from experiment_manager.trackers.tracker_manager import TrackScope
from experiment_manager.common.yaml_utils import deep_merge_configs

"""
This is the main class that is responsible for the experiment.
In order to run it you should supply its configurations file in a given directory.
The configuration files are:
- env.yaml: the environment in which the experiment runs in, responsible for the workding dir, trackers and logs
- experiment.yaml: the configuration of the experiment - name, description and so on
- base.yaml: contains the objects that are used throughout the experiment - model, optimizer, dataset, pipeline etc.
- trials.yaml: the trials configuration file - each trial has its own configuration (very much like the base.yaml)
"""

class Experiment:
    
    def __init__(self, 
                 env_config: DictConfig,
                 experiment_conf: DictConfig,
                 base_config: DictConfig,
                 trials_config: DictConfig,
                 factory_registry: FactoryRegistry = None):
        """
        Initialize the experiment.
        Shouldnt be used by the user, please refer to the create function instead
        """
        
        # basic properties
        self.name             = experiment_conf.name
        self.desc             = experiment_conf.desc
        
        # environment
        self.env_config  = env_config
        self.env         = Environment.from_config(env_config)
        
        # Override factory registry if provided
        if factory_registry:
            self.env.factory_registry = factory_registry
        
        self.env.tracker_manager.on_create(Level.EXPERIMENT, self.name)

        self.env.logger.info(f"Creating experiment '{self.name}'")
        self.env.logger.info(f"Description: {self.desc}")
        
        # configurations of the experiment
        self.experiment_config = experiment_conf
        self.base_config       = base_config
        self.trials_config     = trials_config
        
        # helper function to setup the experiment
        self.setup_experiment()
        
    
    def setup_experiment(self) -> None:
        """
        Setup experiment by loading configurations.
        """
        self.env.logger.info("Setting up experiment configuration")
        
        self.experiment_config.settings = self.base_config
        
        # Save the configuration files
        OmegaConf.save(self.experiment_config, os.path.join(self.env.config_dir, ConfigPaths.CONFIG_FILE.value))
        OmegaConf.save(self.base_config, os.path.join(self.env.config_dir, ConfigPaths.BASE_CONFIG.value))
        OmegaConf.save(self.trials_config, os.path.join(self.env.config_dir, ConfigPaths.TRIALS_CONFIG.value))
        OmegaConf.save(self.env_config, os.path.join(self.env.config_dir, ConfigPaths.ENV_CONFIG.value))
        
        self.env.logger.info("Experiment setup complete")
    
    def run(self) -> None:
        """
        Run the experiment.
        """
        # TODO: initialize registry or load existing one
        with TrackScope(self.env.tracker_manager, level = Level.EXPERIMENT):
            for conf in self.trials_config:
                conf.settings = deep_merge_configs(self.experiment_config.settings, conf.settings)
                
                trial = Trial.from_config(conf, self.env)
                trial.run()
            
        
    @staticmethod
    def create(config_dir: str, factory_registry: FactoryRegistry = None, workdir: str = None) -> "Experiment":
        """
        Create a new environment from a configuration directory - you should always use this method to create a new experiment.
        
        Args:
            config_dir: Path to the configuration directory containing experiment YAML files
            factory_registry: Optional custom FactoryRegistry for overriding default factories
            workdir: Optional custom workspace directory
            
        Returns:
            Experiment instance
        """
        if not os.path.exists(config_dir):
            raise ValueError(f"Config directory {config_dir} does not exist.")
        
        if not Experiment.check_files_exist(config_dir):
            raise ValueError(f"Missing configuration files in directory {config_dir}.")
        
        env_config = OmegaConf.load(os.path.join(config_dir, ConfigPaths.ENV_CONFIG.value))
        if workdir:
            env_config.workspace = workdir
        
        experiment_config = OmegaConf.load(os.path.join(config_dir, ConfigPaths.CONFIG_FILE.value))
        base_config = OmegaConf.load(os.path.join(config_dir, ConfigPaths.BASE_CONFIG.value))
        trials_config = OmegaConf.load(os.path.join(config_dir, ConfigPaths.TRIALS_CONFIG.value))
        
        return Experiment(env_config, experiment_config, base_config, trials_config, factory_registry)
    
    @staticmethod
    def check_files_exist(config_dir: str) -> bool:
        """
        Check if all required configuration files exist.
        """
        required_files = [
            os.path.join(config_dir, path.value) for path in ConfigPaths]
        
        return all(os.path.exists(f) for f in required_files)
