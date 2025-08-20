from omegaconf import DictConfig
import pandas as pd

from experiment_manager.results.extractors.extractor import Extractor
from experiment_manager.results.sources.datasource import ExperimentDataSource

from experiment_manager.common.serializable import YAMLSerializable

@YAMLSerializable.register("DataFrameExtractor")
class DataFrameExtractor(Extractor, YAMLSerializable):
    def __init__(self, granularity=None, include_per_label=False):
        """
        Args:
            granularity: List of granularity levels to include.
                        Options: ['results', 'epoch', 'batch']
                        Default: ['results', 'epoch'] (backward compatible)
            include_per_label: Whether to include per-label metrics as separate rows
        """
        YAMLSerializable.__init__(self)
        self.granularity = granularity or ['results', 'epoch']
        self.include_per_label = include_per_label
        
        # Validate granularity options
        valid_levels = {'results', 'epoch', 'batch'}
        invalid = set(self.granularity) - valid_levels
        if invalid:
            raise ValueError(f"Invalid granularity levels: {invalid}")

    def extract(self, datasource: ExperimentDataSource):
        data = []
        experiment = datasource.get_experiment()
        trials = datasource.get_trials(experiment)

        for trial in trials:
            runs = datasource.get_trial_runs(trial)
            for trial_run in runs:
                
                # Extract metrics based on requested granularity
                if 'results' in self.granularity:
                    data.extend(self._extract_results_metrics(
                        datasource, experiment, trial, trial_run))
                
                if 'epoch' in self.granularity:
                    data.extend(self._extract_epoch_metrics(
                        datasource, experiment, trial, trial_run))
                
                if 'batch' in self.granularity:
                    data.extend(self._extract_batch_metrics(
                        datasource, experiment, trial, trial_run))
        
        return pd.DataFrame(data)

    def _should_include_metric(self, metric_name: str) -> bool:
        """Determine if a metric should be included based on settings."""
        # Skip per-label metrics unless explicitly requested
        if metric_name.endswith('_per_label') and not self.include_per_label:
            return False
        return True

    def _extract_results_metrics(self, datasource, experiment, trial, trial_run):
        """Extract final results metrics (granularity-agnostic from current get_metrics)."""
        results_data = []
        
        # Get all metrics and filter for results-only (epoch=None)
        metrics_records = datasource.get_metrics(trial_run)
        
        for record in metrics_records:
            # Only include metrics that are final results (no epoch)
            if record.epoch is not None:
                continue
                
            for metric_name, metric_value in record.metrics.items():
                if not self._should_include_metric(metric_name):
                    continue
                    
                results_data.append({
                    'experiment_id': experiment.id,
                    'experiment_name': experiment.name,
                    'trial_id': trial.id,
                    'trial_name': trial.name,
                    'trial_run_id': trial_run.id,
                    'trial_run_status': trial_run.status,
                    'granularity': 'results',
                    'epoch': None,
                    'batch': None,
                    'metric': metric_name,
                    'value': metric_value,
                    'is_custom': record.is_custom,
                    'timestamp': getattr(record, 'timestamp', None)
                })
        
        return results_data

    def _extract_epoch_metrics(self, datasource, experiment, trial, trial_run):
        """Extract epoch-level metrics."""
        epoch_data = []
        
        # Get all metrics and filter for epoch-level (epoch is not None, batch is None)
        metrics_records = datasource.get_metrics(trial_run)
        
        for record in metrics_records:
            # Only include metrics that are epoch-level (has epoch, no batch)
            if record.epoch is None or getattr(record, 'batch', None) is not None:
                continue
                
            for metric_name, metric_value in record.metrics.items():
                if not self._should_include_metric(metric_name):
                    continue
                    
                epoch_data.append({
                    'experiment_id': experiment.id,
                    'experiment_name': experiment.name,
                    'trial_id': trial.id,
                    'trial_name': trial.name,
                    'trial_run_id': trial_run.id,
                    'trial_run_status': trial_run.status,
                    'granularity': 'epoch',
                    'epoch': record.epoch,
                    'batch': None,
                    'metric': metric_name,
                    'value': metric_value,
                    'is_custom': record.is_custom,
                    'timestamp': getattr(record, 'timestamp', None)
                })
        
        return epoch_data

    def _extract_batch_metrics(self, datasource, experiment, trial, trial_run):
        """Extract batch-level metrics."""
        batch_data = []
        
        # Use the new get_batch_metrics method if available
        if hasattr(datasource, 'get_batch_metrics'):
            batch_metrics = datasource.get_batch_metrics(trial_run)
        else:
            # Fallback: no batch metrics available
            return batch_data
        
        for record in batch_metrics:
            for metric_name, metric_value in record.metrics.items():
                if not self._should_include_metric(metric_name):
                    continue
                    
                batch_data.append({
                    'experiment_id': experiment.id,
                    'experiment_name': experiment.name,
                    'trial_id': trial.id,
                    'trial_name': trial.name,
                    'trial_run_id': trial_run.id,
                    'trial_run_status': trial_run.status,
                    'granularity': 'batch',
                    'epoch': record.epoch,
                    'batch': getattr(record, 'batch', None),
                    'metric': metric_name,
                    'value': metric_value,
                    'is_custom': record.is_custom,
                    'timestamp': getattr(record, 'timestamp', None)
                })
        
        return batch_data

    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(
            granularity=config.get('granularity', ['results', 'epoch']),
            include_per_label=config.get('include_per_label', False)
        )