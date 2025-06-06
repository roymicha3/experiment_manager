from abc import ABC
from functools import wraps
from omegaconf import DictConfig
from typing import Dict, Any, List

from experiment_manager.common.common import Level
from experiment_manager.common.common import RunStatus
from experiment_manager.environment import Environment
from experiment_manager.pipelines.callbacks.callback import Callback

class Pipeline(ABC):
    
    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.callbacks: List[Callback] = []
        self.env.tracker_manager.on_create(level=Level.PIPELINE)
        
        self.run_metrics = {}
        self.epoch_metrics = {}
        self.run_status = False
        
        self.env.logger.info("Pipeline initialized")
        
    
    def register_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)
        self.env.logger.info(f"Registered callback: {callback.__class__.__name__}")
        
    
    def _on_run_start(self) -> None:
        self.env.logger.info("Starting pipeline execution")
        self.env.tracker_manager.on_start(level=Level.PIPELINE)
        for callback in self.callbacks:
            self.env.logger.debug(f"Executing on_start for {callback.__class__.__name__}")
            callback.on_start()
            
    def _on_epoch_start(self) -> None:
        self.env.tracker_manager.on_create(Level.EPOCH)
        self.env.tracker_manager.on_start(Level.EPOCH)
            
    
    def _on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        self.env.tracker_manager.on_end(Level.EPOCH)
        
        stop_flag = False
        for callback in self.callbacks:
            self.env.logger.debug(f"Executing on_epoch_end for {callback.__class__.__name__}")
            should_continue = callback.on_epoch_end(epoch_idx, metrics)
            if not should_continue:
                stop_flag = True
        return stop_flag
    
    def _on_run_end(self, metrics: Dict[str, Any]) -> None:
        self.env.logger.info("Pipeline execution completed")
        self.env.tracker_manager.on_end(Level.PIPELINE)
        for callback in self.callbacks:
            self.env.logger.debug(f"Executing on_end for {callback.__class__.__name__}")
            callback.on_end(metrics)
            
    def run(self, config: DictConfig) -> RunStatus:
        """
        Run the pipeline with the given configuration.
        """
        raise NotImplementedError("Pipeline.run() must be implemented by subclasses")
    
    def run_epoch(self, epoch_idx, model, *args, **Kwargs) -> RunStatus:
        """
        Run one epoch of training.
        """
        raise NotImplementedError("Pipeline.run_epoch() must be implemented by subclasses")
    
    
    @staticmethod
    def run_wrapper(run_function):
        """
        Makes sure that on_start and on_end are called for the pipeline.
        """
        @wraps(run_function)
        def wrapper(self, config: DictConfig):
            status = RunStatus.SKIPPED
            self._on_run_start()
            try:
                status = RunStatus.RUNNING
                status = run_function(self, config)
            
            except StopIteration as e:
                # Early stopping is not a failure - it's a successful early completion
                self.env.logger.info(f"Pipeline completed early: {e}")
                status = RunStatus.STOPPED
            
            except Exception as e:
                self.env.logger.error(f"Pipeline run failed: {e}")
                status = RunStatus.FAILED
            
            finally:
                # Always call on_end for proper cleanup, except for true failures
                if status != RunStatus.FAILED:
                    if status == RunStatus.STOPPED:
                        self.env.logger.info("Pipeline completed early via early stopping")
                    else:
                        self.env.logger.info("Pipeline run completed successfully")
                    self._on_run_end(self.run_metrics)
                
                return status
        
        return wrapper
    
    
    @staticmethod
    def epoch_wrapper(epoch_function):
        """
        Makes sure that on_epoch_start and on_epoch_end are called for the pipeline.
        """
        @wraps(epoch_function)
        def wrapper(self, epoch_idx, model, *args, **kwargs):
            self._on_epoch_start()
            
            try:
                status = epoch_function(self, epoch_idx, model, *args, **kwargs)
                self.env.tracker_manager.track_dict(self.epoch_metrics, epoch_idx)
            
            except Exception as e:
                self.env.logger.error(f"Pipeline epoch failed: {e}")
                status = RunStatus.FAILED
            
            finally:
                should_stop = self._on_epoch_end(epoch_idx, self.epoch_metrics)
                self.epoch_metrics.clear()
                
                if should_stop:
                    self.env.logger.info("Stopping pipeline execution")
                    raise StopIteration("Stopping pipeline execution")
                
                return status
                    
        return wrapper