import os
from omegaconf import OmegaConf, DictConfig

from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable


class Trial(YAMLSerializable):
    CONFIG_FILE = "trial.yaml"
    
    def __init__(self, name: str, id: int, repeat: int, config: DictConfig, env: Environment):
        super().__init__()
        
        # basic properties
        self.name = name
        self.id = id
        self.repeat = repeat
        
        # environment
        self.env = env
        self.env.set_workspace(self.name, inner=True)
        self.env.setup_environment()
        
        # configurations of the trial
        self.config = config
        
        # TODO: might be a better idea to only receive the config_dir_path and build from there
        self.config_dir_path = self.env.config_dir