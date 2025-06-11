from typing import List, Optional, Union
from experiment_manager.results.experiment_data import ExperimentDataSource
from experiment_manager.results.data_models import Experiment, Trial, TrialRun, MetricRecord, Artifact

class FileSystemDataSource(ExperimentDataSource):
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path

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
    