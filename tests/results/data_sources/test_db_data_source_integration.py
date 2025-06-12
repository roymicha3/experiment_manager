"""
Test DBDataSource integration with real experiment data.
"""
import os
import pytest
from pathlib import Path

from experiment_manager.results.sources.db_datasource import DBDataSource


class TestDBDataSourceIntegration:
    """Test DBDataSource integration with shared MNIST experiment data."""
    
    def test_real_experiment_structure(self, experiment_data):
        """Test that real experiment has expected structure."""
        experiment = experiment_data['experiment']
        
        assert experiment.id is not None
        assert experiment.name == "test_mnist_baseline"
        assert len(experiment.trials) == 3
        
        # Check trial names
        trial_names = {trial.name for trial in experiment.trials}
        assert trial_names == {"small_lr", "medium_lr", "large_lr"}
        
        # Each trial should have 2 runs
        for trial in experiment.trials:
            assert len(trial.runs) == 2
    
    def test_db_datasource_with_real_data(self, experiment_data):
        """Test DBDataSource operations with real experiment data."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            # Test get_experiment works
            experiment = source.get_experiment()
            assert experiment is not None
            
            # Test get_trials works
            trials = source.get_trials(experiment)
            assert len(trials) == 3
            
            # Test get_trial_runs works
            first_trial = trials[0]
            runs = source.get_trial_runs(first_trial)
            assert len(runs) == 2
    
    def test_real_experiment_metrics(self, experiment_data):
        """Test that real experiment has expected metrics structure."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            experiment = source.get_experiment()
            first_trial = experiment.trials[0]
            first_run = first_trial.runs[0]
            
            # Should have epoch metrics (3 epochs × multiple metrics)
            epoch_metrics = [m for m in first_run.metrics if m.epoch is not None]
            assert len(epoch_metrics) > 0
            
            # Should have final result metrics (epoch=None)
            final_metrics = [m for m in first_run.metrics if m.epoch is None]
            assert len(final_metrics) > 0
            
            # Check for expected final metric types (from metrics dict keys)
            final_metric_types = set()
            for metric_record in final_metrics:
                final_metric_types.update(metric_record.metrics.keys())
            
            assert "test_acc" in final_metric_types
            assert "test_loss" in final_metric_types
    
    def test_real_experiment_dataframe(self, experiment_data):
        """Test DataFrame creation with real experiment data."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            experiment = source.get_experiment()
            df = source.metrics_dataframe(experiment)
            
            # Should have data
            assert len(df) > 0
            
            # Check required columns
            required_columns = ['trial_run_id', 'metric', 'value']
            for col in required_columns:
                assert col in df.columns
            
            # Should have final test metrics
            test_metrics = df[df['metric'].isin(['test_acc', 'test_loss'])]
            assert len(test_metrics) > 0
    
    def test_trial_parameter_differences(self, experiment_data):
        """Test that different trials have different configurations."""
        experiment = experiment_data['experiment']
        
        # Each trial should have different names reflecting their configs
        trial_names = [trial.name for trial in experiment.trials]
        assert "small_lr" in trial_names
        assert "medium_lr" in trial_names  
        assert "large_lr" in trial_names
    
    def test_epoch_progression(self, experiment_data):
        """Test that epochs progress correctly in the data."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            experiment = source.get_experiment()
            first_run = experiment.trials[0].runs[0]
            
            # Get epoch metrics and check progression
            epoch_metrics = [m for m in first_run.metrics if m.epoch is not None]
            epochs = sorted(set(m.epoch for m in epoch_metrics))
            
            # Should have 3 epochs (0, 1, 2)
            assert epochs == [0, 1, 2]
    
    def test_workspace_artifacts(self, experiment_data):
        """Test that workspace and artifacts are properly structured."""
        temp_dir = experiment_data['temp_dir']
        
        # Check that workspace structure exists
        assert os.path.exists(temp_dir)
        
        # Database should exist in artifacts directory
        db_path = experiment_data['db_path']
        assert os.path.exists(db_path)
        assert "artifacts" in db_path 
