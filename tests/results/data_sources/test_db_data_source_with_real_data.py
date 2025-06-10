"""
Test DBDataSource with real experiment data generated from MNIST baseline.
"""
import pytest
from experiment_manager.results.sources.db_data_source import DBDataSource


class TestDBDataSourceWithRealData:
    """Test DBDataSource with shared real experiment data."""
    
    def test_get_experiment(self, experiment_data):
        """Test getting experiment data."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            experiment = source.get_experiment()
            
            # Verify experiment structure
            assert experiment is not None
            assert experiment.name == "test_mnist_baseline"
            assert experiment.trials is not None
            
            # Should have 3 trials (small_lr, medium_lr, large_lr)
            assert len(experiment.trials) == 3
            
            # Check trial names
            trial_names = {trial.name for trial in experiment.trials}
            expected_names = {"small_lr", "medium_lr", "large_lr"}
            assert trial_names == expected_names
    
    def test_trial_structure(self, experiment_data):
        """Test trial structure and runs."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            experiment = source.get_experiment()
            
            for trial in experiment.trials:
                # Each trial should have 2 repetitions
                assert len(trial.runs) == 2, f"Trial {trial.name} should have 2 runs"
                
                # Verify trial run structure
                for run in trial.runs:
                    assert run.id is not None
                    assert run.trial_id is not None
                    assert run.status is not None
                    assert run.num_epochs is not None
    
    def test_metrics_data(self, experiment_data):
        """Test metrics data from real experiment."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            experiment = source.get_experiment()
            
            # Get metrics from all trial runs
            all_metrics = []
            for trial in experiment.trials:
                for run in trial.runs:
                    all_metrics.extend(run.metrics)
            
            # Should have metrics from all trial runs
            assert len(all_metrics) > 0
            
            # Check for expected metric types - collect from metric records
            all_metric_names = set()
            for metric_record in all_metrics:
                all_metric_names.update(metric_record.metrics.keys())
            
            expected_metrics = {"train_loss", "val_loss", "train_acc", "val_acc", "test_acc"}
            
            # Should have at least some of these metrics
            assert len(all_metric_names.intersection(expected_metrics)) > 0
    
    def test_metrics_dataframe(self, experiment_data):
        """Test metrics DataFrame creation."""
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
            
            # Should have some metrics
            assert df['metric'].nunique() > 0
    
    def test_parameter_variations(self, experiment_data):
        """Test that different trials show different parameter effects."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            experiment = source.get_experiment()
            
            # Should have trials with different names (indicating different parameters)
            trial_names = [trial.name for trial in experiment.trials]
            assert "small_lr" in trial_names
            assert "medium_lr" in trial_names
            assert "large_lr" in trial_names
            
            # Each trial should have runs with metrics
            for trial in experiment.trials:
                assert len(trial.runs) > 0
                for run in trial.runs:
                    assert len(run.metrics) > 0
    
    def test_trial_specific_data(self, experiment_data):
        """Test accessing data for specific trials."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            experiment = source.get_experiment()
            
            # Find the small_lr trial
            small_lr_trial = None
            for trial in experiment.trials:
                if trial.name == "small_lr":
                    small_lr_trial = trial
                    break
            
            assert small_lr_trial is not None
            assert len(small_lr_trial.runs) == 2  # 2 repetitions
            
            # Check that runs have metrics
            for run in small_lr_trial.runs:
                assert len(run.metrics) > 0
    
    def test_database_integrity(self, experiment_data):
        """Test database integrity and relationships."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            experiment = source.get_experiment()
            
            # Verify IDs are consistent
            trial_ids = set()
            for trial in experiment.trials:
                assert trial.experiment_id == experiment.id
                trial_ids.add(trial.id)
                
                for run in trial.runs:
                    assert run.trial_id == trial.id
    
    def test_artifacts(self, experiment_data):
        """Test artifact handling."""
        db_path = experiment_data['db_path']
        
        with DBDataSource(db_path) as source:
            experiment = source.get_experiment()
            
            # Test getting artifacts at different levels
            exp_artifacts = source.get_artifacts("experiment", experiment)
            assert isinstance(exp_artifacts, list)  # Should return empty list if no artifacts
            
            # Test trial level artifacts
            first_trial = experiment.trials[0]
            trial_artifacts = source.get_artifacts("trial", first_trial)
            assert isinstance(trial_artifacts, list)
            
            # Test trial run level artifacts
            first_run = first_trial.runs[0]
            run_artifacts = source.get_artifacts("trial_run", first_run)
            assert isinstance(run_artifacts, list) 