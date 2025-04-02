import os
from omegaconf import OmegaConf, DictConfig

from common.serializable import YAMLSerializable


class Environment(YAMLSerializable):
    """
    Environment class for managing the environment.
    """
    CONFIG_FILE = "env.yaml"
    
    LOG_DIR = "logs"
    ARTIFACT_DIR = "artifacts"
    CONFIG_DIR = "configs"
    
    def __init__(self, workspace: str,
                 config: DictConfig):
        super().__init__()
        self.workspace = workspace
        self.config = config
        
        self.setup_environment()
        
    def setup_environment(self) -> None:
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
    
    @property
    def log_dir(self):
        return os.path.join(self.workspace, Environment.LOG_DIR)
    
    @property
    def artifact_dir(self):
        return os.path.join(self.workspace, Environment.ARTIFACT_DIR)
    
    @property
    def config_dir(self):
        return os.path.join(self.workspace, Environment.CONFIG_DIR)
    
    
    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(workspace=config.workspace,
                   config=config)
        
    def save(self, path: str = None):
        if path is None:
            path = self.config_dir
        OmegaConf.save(self.config, os.path.join(path, self.CONFIG_FILE))