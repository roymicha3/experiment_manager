import os
from omegaconf import OmegaConf, DictConfig

from experiment_manager.common.common import Level
from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.common.common import ConfigPaths


class Trial(YAMLSerializable):
    
    def __init__(self, name: str, repeat: int, config: DictConfig, env: Environment):
        super().__init__()
        
        # basic properties
        self.name = name
        self.repeat = repeat
        
        # environment
        self.env = env.create_child(self.name)
        self.env.tracker_manager.on_create(Level.TRIAL, self.name)
        
        # configurations of the trial
        self.config = config
        
        # save config
        OmegaConf.save(self.config, os.path.join(self.env.config_dir, ConfigPaths.CONFIG_FILE.value))
        
        self.env.logger.info(f"Trial '{self.name}' created")
    
    
    def run(self) -> None:
        self.env.logger.info(f"Running trial '{self.name}'")
        self.env.tracker_manager.on_start(level=Level.TRIAL)
        
        
        for i in range(self.repeat):
            self.env.logger.info(f"Trial '{self.name}' repeat {i}")
            self.run_single(i)
            self.env.logger.info(f"Trial '{self.name}' repeat {i} completed")
            
    def run_single(self, repeat: int) -> None:
        trial_run_env = self.env.create_child(f"{self.name}-{repeat}")
        trial_run_env.tracker_manager.on_create(Level.TRIAL_RUN)
        self.env.logger.info(f"Trial Run'{self.name}' (repeat: {repeat}) running single")
        
        trial_run_env.tracker_manager.on_start(Level.TRIAL_RUN)
        
        try:
            if self.env.factory is None:
                self.env.logger.error("Factory not initialized in environment")
                return
            
            pipeline = self.env.factory.create(
                name=self.config.pipeline.type,
                config=self.config,
                env=trial_run_env)
            
            pipeline.run(self.config)
        
        except Exception as e:
            self.env.logger.error(f"Error running trial: {e}")
            raise e
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        return cls(name=config.name,
                  repeat=config.repeat,
                  config=config.settings,
                  env=env)