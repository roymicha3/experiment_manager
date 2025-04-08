import os
from omegaconf import OmegaConf, DictConfig

from experiment_manager.common.common import Level
from experiment_manager.trial import Trial
from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable


class Experiment(YAMLSerializable):
    
    CONFIG_FILE = "experiment.yaml"
    BASE_CONFIG = "base.yaml"
    TRIALS_CONFIG = "trials.yaml"
    
    def __init__(self, 
                 name: str, 
                 id: int,
                 desc: str,
                 env: Environment,
                 config_dir_path: str = None):
        
        super().__init__()
        
        # basic properties
        self.name = name
        self.id = id
        self.desc = desc
        
        # environment
        self.env = env.create_child(self.name, root=True)
        self.env.tracker_manager.on_create(Level.EXPERIMENT, self.name)
        
        # TODO: might be a better idea to only receive the config_dir_path and build from there
        self.config_dir_path = config_dir_path if config_dir_path is not None else self.env.config_dir
        
        # configurations of the experiment
        self.config = None
        self.base_config = None
        self.trials_config = None
        
        if not self.check_files_exist():
            raise ValueError(f"Missing configuration files in directory {self.config_dir_path}.")
        
        # helper function to setup the experiment
        self.setup_experiment()
        
        self.env.logger.info(f"Creating experiment '{self.name}' (ID: {self.id})")
        self.env.logger.info(f"Description: {self.desc}")
    
    
    def check_files_exist(self) -> bool:
        """
        Check if all required configuration files exist.
        """
        required_files = [
            os.path.join(self.config_dir_path, self.CONFIG_FILE),
            os.path.join(self.config_dir_path, self.BASE_CONFIG),
            os.path.join(self.config_dir_path, self.TRIALS_CONFIG)
        ]
        return all(os.path.exists(f) for f in required_files)
        
        
    def setup_experiment(self) -> None:
        """
        Setup experiment by loading configurations.
        """
        self.env.logger.info("Setting up experiment configuration")
        
        # load configurations
        self.config = OmegaConf.load(os.path.join(self.config_dir_path, self.CONFIG_FILE))
        self.base_config = OmegaConf.load(os.path.join(self.config_dir_path, self.BASE_CONFIG))
        self.trials_config = OmegaConf.load(os.path.join(self.config_dir_path, self.TRIALS_CONFIG))
        
        # Save the configuration files
        OmegaConf.save(self.config, os.path.join(self.env.config_dir, self.CONFIG_FILE))
        OmegaConf.save(self.base_config, os.path.join(self.env.config_dir, self.BASE_CONFIG))
        OmegaConf.save(self.trials_config, os.path.join(self.env.config_dir, self.TRIALS_CONFIG))
        self.env.save()
        
        self.env.logger.info("Experiment setup complete")
    
    def run(self) -> None:
        """
        Run the experiment.
        """
        self.trials_env = self.env.create_child("trials")
        # TODO: initialize registry or load existing one
        for conf in self.trials_config:
            conf.settings = OmegaConf.merge(self.config.settings, conf.settings)
            
            trial = Trial.from_config(conf, self.trials_env)
            trial.run()
            
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        # Get config_dir_path from config or use env.config_dir as default
        config_dir_path = getattr(config, "config_dir_path", None)
        return cls(name=config.name, 
                 id=config.id,
                 desc=config.desc,
                 env=env,
                 config_dir_path=config_dir_path)
