from typing import List, Optional, Union
from experiment_manager.results.experiment_data import ExperimentDataSource
from experiment_manager.results.data_models import Experiment, Trial, TrialRun, MetricRecord, Artifact
from omegaconf import DictConfig
from experiment_manager.common.serializable import YAMLSerializable

# TODO: Implement this
@YAMLSerializable.register("FileSystemDataSource")
class FileSystemDataSource(ExperimentDataSource):
    def __init__(self, workspace_path: str):
        ExperimentDataSource.__init__(self)
        
        self.workspace_path = workspace_path
        
    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(
            workspace_path=config.workspace_path
        )

    def get_experiment(self) -> Experiment:
        pass

    def get_trials(self, experiment: Experiment) -> List[Trial]:
        pass
    
    def get_trial_runs(self, trial: Trial) -> List[TrialRun]:
        pass
    
    def get_metrics(self, trial_run: TrialRun) -> List[MetricRecord]:
        pass
    
    def get_artifacts(self, entity_level: str, entity: Experiment | Trial | TrialRun) -> List[Artifact]:
        pass
    