import os
from omegaconf import OmegaConf, DictConfig

from experiment_manager.common.common import Level
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
        self.env.tracker_manager.on_create(Level.TRIAL)
        
        # configurations of the trial
        self.config = config
        
        OmegaConf.save(self.config, os.path.join(self.env.config_dir, self.CONFIG_FILE))
        
        self.env.logger.info(f"Trial '{self.name}' (ID: {self.id}) created")
    
    
    def run(self) -> None:
        self.env.logger.info(f"Running trial '{self.name}' (ID: {self.id})")
        for i in range(self.repeat):
            self.env.logger.info(f"Trial '{self.name}' (ID: {self.id}) repeat {i}")
            self.run_single(i)
            self.env.logger.info(f"Trial '{self.name}' (ID: {self.id}) repeat {i} completed")
            
    def run_single(self, repeat: int) -> None:
        # TODO: id logic here is not correct
        trial_run_env = self.env.create_child(f"{self.name}-{repeat}")
        trial_run_env.tracker_manager.on_create(Level.TRIAL_RUN)
        self.env.logger.info(f"Trial Run'{self.name}' (repeat: {repeat}) running single")
        
        try:
            if self.env.factory is None:
                self.env.logger.error("Factory not initialized in environment")
                return
            
            pipeline = self.env.factory.create(
                name=self.config.pipeline.type,
                config=self.config,
                env=trial_run_env,
                id=self.id)
            
            pipeline.run(self.config)
        
        except Exception as e:
            self.env.logger.error(f"Error running trial {self.id}: {e}")
            raise e
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        return cls(name=config.name,
                  id=config.id,
                  repeat=config.repeat,
                  config=config,
                  env=env)