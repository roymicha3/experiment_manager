from omegaconf import DictConfig
import pandas as pd

from experiment_manager.results.extractors.extractor import Extractor
from experiment_manager.results.sources.datasource import ExperimentDataSource

from experiment_manager.common.serializable import YAMLSerializable

@YAMLSerializable.register("DataFrameExtractor")
class DataFrameExtractor(Extractor, YAMLSerializable):
    def __init__(self):
        YAMLSerializable.__init__(self)

    def extract(self, datasource: ExperimentDataSource):
        data = []
        experiment = datasource.get_experiment()
        
        trials = datasource.get_trials(experiment)

        for trial in trials:
            runs = datasource.get_trial_runs(trial)
            for trial_run in runs:
                for metric_record in trial_run.metrics:
                    for metric_name, metric_value in metric_record.metrics.items():
                        # Skip per-label metrics in the main DataFrame to keep it simple
                        if not metric_name.endswith('_per_label'):
                            data.append({
                                'experiment_id': experiment.id,
                                'experiment_name': experiment.name,
                                'trial_id': trial.id,
                                'trial_name': trial.name,
                                'trial_run_id': trial_run.id,
                                'trial_run_status': trial_run.status,
                                'epoch': metric_record.epoch,
                                'metric': metric_name,
                                'value': metric_value,
                                'is_custom': metric_record.is_custom
                            })
        
        return pd.DataFrame(data)

    @classmethod
    def from_config(cls, config: DictConfig):
        return cls()