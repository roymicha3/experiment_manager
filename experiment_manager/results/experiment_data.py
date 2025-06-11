# datasource/base.py

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import pandas as pd

from experiment_manager.results.data_models import Experiment, Trial, TrialRun, MetricRecord, Artifact
from experiment_manager.common.serializable import YAMLSerializable

class ExperimentDataSource(ABC, YAMLSerializable):
    
    @abstractmethod
    def get_experiment(self, experiment_id: Optional[Union[str, int]] = None) -> Experiment:
        """
        Fetch the experiment and all associated trials and runs.
        
        Args:
            experiment_id: Optional experiment ID (int) or title (str). 
                          If None, returns the first experiment.
        """
        pass

    @abstractmethod
    def get_trials(self, experiment: Experiment) -> List[Trial]:
        """List all trials for an experiment."""
        pass

    @abstractmethod
    def get_trial_runs(self, trial: Trial) -> List[TrialRun]:
        """List all runs for a trial."""
        pass

    @abstractmethod
    def get_metrics(self, trial_run: TrialRun) -> List[MetricRecord]:
        """Fetch all metrics for a run (across all epochs)."""
        pass

    @abstractmethod
    def get_artifacts(self, entity_level: str, entity: Experiment | Trial | TrialRun) -> List[Artifact]:
        """
        Fetch artifacts attached to experiment, trial, or run.
        entity_level: "experiment", "trial", "trial_run", etc.
        """
        pass

    # Convenience for DataFrames, if desired:
    @abstractmethod
    def metrics_dataframe(self, experiment: Experiment) -> 'pd.DataFrame':
        """
        Return a DataFrame with columns: ['trial', 'trial_run', 'epoch', 'metric', 'value']
        Useful for plugins that want to pivot or aggregate.
        """
        pass
