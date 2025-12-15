import os
from omegaconf import OmegaConf, DictConfig

from experiment_manager.common.common import Level
from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.common.common import ConfigPaths
from experiment_manager.trackers.tracker_manager import TrackScope
from experiment_manager.common.factory_registry import FactoryType


class Trial(YAMLSerializable):
    
    def __init__(self, name: str,
                 repeat: int,
                 config: DictConfig,
                 env: Environment,
                 env_args: DictConfig = None) -> None:
        super().__init__()
        
        # basic properties
        self.name = name
        self.repeat = repeat
        
        # environment
        self.env = env.create_child(self.name, args=env_args)
        self.env.tracker_manager.on_create(Level.TRIAL, self.name)
        
        # configurations of the trial
        self.config = config
        
        # save config
        OmegaConf.save(self.config, os.path.join(self.env.config_dir, ConfigPaths.CONFIG_FILE.value))
        
        self.env.logger.info(f"Trial '{self.name}' created")
    
    
    def run(self) -> None:
        self.env.logger.info(f"Running trial '{self.name}'")
        with TrackScope(self.env.tracker_manager, level=Level.TRIAL, trial_name=self.name):
        
            for i in range(self.repeat):
                self.env.logger.info(f"Trial '{self.name}' repeat {i}")
                self.run_single(i)
                self.env.logger.info(f"Trial '{self.name}' repeat {i} completed")
        
    
    
    def run_single(self, repeat: int) -> None:
        trial_run_env = self.env.create_child(f"{self.name}-{repeat}")
        trial_run_env.tracker_manager.on_create(Level.TRIAL_RUN)
        self.env.logger.info(f"Trial Run'{self.name}' (repeat: {repeat}) running single")
        
        with TrackScope(trial_run_env.tracker_manager, level = Level.TRIAL_RUN):
        
            try:
                factory = self.env.factory_registry.get(FactoryType.PIPELINE)
                pipeline = factory.create(
                    name=self.config.pipeline.type,
                    config=self.config,
                    env=trial_run_env)
                
                pipeline.run(self.config)
            
            except Exception as e:
                self.env.logger.error(f"Error running trial: {e}")
                raise e
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        
        env_args = config.get("args", None)
        return cls(name=config.name,
                  repeat=config.repeat,
                  config=config.settings,
                  env=env,
                  env_args=env_args)