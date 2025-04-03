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
        self.env = env.create_child(self.name)
        
        # configurations of the trial
        self.config = config
        
        OmegaConf.save(self.config, os.path.join(self.env.config_dir, self.CONFIG_FILE))
        
        self.env.logger.info(f"Trial '{self.name}' (ID: {self.id}) created")
    
    
    def run(self) -> None:
        self.env.logger.info(f"Running trial '{self.name}' (ID: {self.id})")
        for i in range(self.repeat):
            self.env.logger.info(f"Trial '{self.name}' (ID: {self.id}) repeat {i}")
            self.run_single()
            self.env.logger.info(f"Trial '{self.name}' (ID: {self.id}) repeat {i} completed")
            
    def run_single(self) -> None:
        trial_run_env = self.env.create_child(self.name)
        self.env.logger.info(f"Trial Run'{self.name}' (ID: {self.id}) running single")
        
        
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        # TODO: make config_dir_path optional
        return cls(name=config.name, 
                 id=config.id,
                 repeat=config.repeat,
                 config=config,
                 env=env)