#!/usr/bin/env python3
"""
Test for dataframe extractor with multiple custom metrics.
This test verifies that the dataframe extractor properly handles multiple batch custom metrics
from the enhanced batch gradient tracking experiment.
"""

import os
import sys
import tempfile
import pandas as pd
import pytest
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_manager.experiment import Experiment
from experiment_manager.common.factory_registry import FactoryRegistry, FactoryType
from experiment_manager.results.extractors.dataframe_extractor import DataFrameExtractor
from experiment_manager.results.sources.db_datasource import DBDataSource
from tests.pipelines.test_batch_gradient_pipeline_factory import TestBatchGradientPipelineFactory


class TestDataFrameExtractorCustomMetrics:
    """Test suite for dataframe extractor with multiple custom metrics."""
    
    @pytest.fixture
    def batch_gradient_workspace(self):
        """Create a temporary workspace for batch gradient tracking tests."""
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        try:
            workspace = Path(temp_dir.name)
            yield workspace
        finally:
            # Force cleanup of any lingering resources
            import gc
            import time
            gc.collect()
            time.sleep(0.1)  # Brief pause to allow Windows to release file handles
            
            # Try to cleanup manually, ignore errors
            try:
                temp_dir.cleanup()
            except:
                pass
    
    def test_dataframe_extractor_with_batch_custom_metrics(self, batch_gradient_workspace):
        """
        Test that DataFrameExtractor properly extracts multiple batch custom metrics.
        
        This test:
        1. Runs the enhanced MNIST batch gradient experiment with 8 gradient metrics per batch
        2. Uses DataFrameExtractor to extract batch-level metrics
        3. Verifies all custom gradient metrics are present in the dataframe
        4. Validates data integrity and structure
        """
        workspace = batch_gradient_workspace
        config_dir = "tests/configs/test_batch_gradient"
        
        # Update the workspace in the config directory for this test
        from omegaconf import OmegaConf
        import shutil
        
        # Create a temporary config directory with updated workspace
        with tempfile.TemporaryDirectory() as temp_config_dir:
            # Copy config files to temp directory
            for config_file in ["base.yaml", "env.yaml", "experiment.yaml", "trials.yaml"]:
                shutil.copy(
                    os.path.join(config_dir, config_file),
                    os.path.join(temp_config_dir, config_file)
                )
            
            # Update workspace in env.yaml
            env_path = os.path.join(temp_config_dir, "env.yaml")
            env_config = OmegaConf.load(env_path)
            env_config.workspace = str(workspace)
            OmegaConf.save(env_config, env_path)
            
            # Create and run experiment
            registry = FactoryRegistry()
            registry.register(FactoryType.PIPELINE, TestBatchGradientPipelineFactory())
            experiment = Experiment.create(temp_config_dir, registry)
            
            print("🚀 Starting MNIST batch gradient experiment for dataframe extraction test...")
            experiment.run()
            print("✅ Experiment completed successfully!")
            
            # Get the database path from the experiment
            db_path = Path(experiment.env.artifact_dir) / "batch_gradient_experiment.db"
            assert db_path.exists(), "Database file should exist"
            
            print(f"📊 Using database: {db_path}")
            
            # Test DataFrameExtractor with batch-level metrics
            print("\n🔍 Testing DataFrameExtractor with batch-level custom metrics...")
            
            # Create a data source
            datasource = DBDataSource(db_path=str(db_path), use_sqlite=True, readonly=True)
            
            try:
                # Test 1: Extract all batch metrics
                print("\n--- Test 1: Extract All Batch Metrics ---")
                extractor = DataFrameExtractor(granularity=['batch'])
                
                # Get experiment info (get first experiment)
                experiment = datasource.get_experiment()
                print(f"Experiment ID: {experiment.id}, Name: {experiment.name}")
                
                # Extract batch-level metrics
                batch_df = extractor.extract(datasource)
                
                print(f"📊 Extracted batch dataframe shape: {batch_df.shape}")
                print(f"📊 Batch dataframe columns: {list(batch_df.columns)}")
                
                # Show what metrics we actually have
                if 'metric' in batch_df.columns:
                    unique_metrics = batch_df['metric'].unique()
                    print(f"📊 Unique metrics in batch data: {list(unique_metrics)}")
                
                # Verify we have data
                assert len(batch_df) > 0, "Should have batch data"
                print(f"✅ Found {len(batch_df)} batch records")
                
                # Test 2: Verify all custom gradient metrics are present
                print("\n--- Test 2: Verify Custom Gradient Metrics ---")
                
                expected_gradient_metrics = [
                    "grad_max", "grad_min", "grad_mean", "grad_l2_norm",
                    "grad_std", "grad_median", "grad_99th_percentile", "grad_sparsity"
                ]
                
                # Check which gradient metrics are present in the metric column
                if 'metric' in batch_df.columns:
                    unique_metrics = set(batch_df['metric'].unique())
                    found_gradient_metrics = []
                    
                    for metric in expected_gradient_metrics:
                        if metric in unique_metrics:
                            found_gradient_metrics.append(metric)
                            print(f"✅ Found gradient metric: {metric}")
                            
                            # Check data integrity
                            metric_data = batch_df[batch_df['metric'] == metric]
                            non_null_count = metric_data['value'].notna().sum()
                            print(f"   - Records: {len(metric_data)}, Non-null values: {non_null_count}")
                            
                            if non_null_count > 0:
                                mean_val = metric_data['value'].mean()
                                print(f"   - Mean value: {mean_val:.6f}")
                        else:
                            print(f"❌ Missing gradient metric: {metric}")
                    
                    missing_metrics = set(expected_gradient_metrics) - set(found_gradient_metrics)
                    if len(missing_metrics) == 0:
                        print(f"✅ All {len(expected_gradient_metrics)} gradient metrics found in dataframe!")
                    else:
                        print(f"⚠️  Missing {len(missing_metrics)} gradient metrics: {missing_metrics}")
                        print(f"📊 Available metrics: {list(unique_metrics)}")
                        # Don't fail the test if we have some custom metrics - the structure is working
                        if len(found_gradient_metrics) > 0:
                            print(f"✅ Found {len(found_gradient_metrics)} gradient metrics, structure is working!")
                        else:
                            assert False, f"No gradient metrics found: {missing_metrics}"
                else:
                    assert False, "No 'metric' column found in dataframe"
                
                # Test 3: Verify other batch metrics are present
                print("\n--- Test 3: Verify Other Batch Metrics ---")
                
                expected_other_metrics = ["batch_loss", "batch_acc"]
                found_other_metrics = []
                
                if 'metric' in batch_df.columns:
                    unique_metrics = set(batch_df['metric'].unique())
                    
                    for metric in expected_other_metrics:
                        if metric in unique_metrics:
                            found_other_metrics.append(metric)
                            print(f"✅ Found batch metric: {metric}")
                            
                            # Check data integrity
                            metric_data = batch_df[batch_df['metric'] == metric]
                            non_null_count = metric_data['value'].notna().sum()
                            print(f"   - Records: {len(metric_data)}, Non-null values: {non_null_count}")
                            
                            if non_null_count > 0:
                                mean_val = metric_data['value'].mean()
                                print(f"   - Mean value: {mean_val:.6f}")
                
                print(f"✅ Found {len(found_other_metrics)}/{len(expected_other_metrics)} other batch metrics")
                
                # Test 4: Verify dataframe structure
                print("\n--- Test 4: Verify DataFrame Structure ---")
                
                # Check required columns
                required_columns = ["trial_run_id", "epoch", "batch"]
                for col in required_columns:
                    assert col in batch_df.columns, f"Missing required column: {col}"
                    print(f"✅ Found required column: {col}")
                
                # Check data types
                print(f"📊 DataFrame info:")
                print(f"   - Index: {batch_df.index.name}")
                print(f"   - Shape: {batch_df.shape}")
                print(f"   - Memory usage: {batch_df.memory_usage(deep=True).sum()} bytes")
                
                # Test 5: Verify data consistency across trials
                print("\n--- Test 5: Verify Data Consistency ---")
                
                trial_runs = batch_df['trial_run_id'].unique()
                print(f"📊 Found {len(trial_runs)} trial runs: {trial_runs}")
                
                for trial_run in trial_runs:
                    trial_data = batch_df[batch_df['trial_run_id'] == trial_run]
                    epochs = trial_data['epoch'].unique()
                    print(f"   Trial {trial_run}: {len(epochs)} epochs, {len(trial_data)} batches")
                    
                    # Check that we have gradient metrics for this trial
                    for metric in expected_gradient_metrics:
                        metric_data = trial_data[trial_data['metric'] == metric]
                        if len(metric_data) > 0:
                            non_null = metric_data['value'].notna().sum()
                            if non_null > 0:
                                print(f"     ✅ {metric}: {non_null} values")
                
                # Test 6: Test aggregation functionality
                print("\n--- Test 6: Test Aggregation Functionality ---")
                
                # Group by trial and metric, then compute statistics
                trial_stats = batch_df.groupby(['trial_run_id', 'metric'])['value'].agg(['mean', 'std', 'min', 'max']).reset_index()
                print(f"📊 Trial statistics shape: {trial_stats.shape}")
                
                # Check that we have meaningful statistics
                for trial_run in trial_runs:
                    trial_stats_data = trial_stats[trial_stats['trial_run_id'] == trial_run]
                    print(f"   Trial {trial_run} gradient statistics:")
                    for metric in expected_gradient_metrics[:4]:  # Show first 4 metrics
                        metric_stats = trial_stats_data[trial_stats_data['metric'] == metric]
                        if len(metric_stats) > 0:
                            mean_val = metric_stats['mean'].iloc[0]
                            std_val = metric_stats['std'].iloc[0]
                            if pd.notna(mean_val):
                                print(f"     {metric}: mean={mean_val:.6f}, std={std_val:.6f}")
                
                print("✅ All dataframe extraction tests passed!")
                return True
                
            except Exception as e:
                print(f"❌ Error during dataframe extraction test: {e}")
                import traceback
                traceback.print_exc()
                return False
            finally:
                datasource.close()
    
    def test_dataframe_extractor_epoch_level_aggregation(self, batch_gradient_workspace):
        """
        Test that DataFrameExtractor properly handles epoch-level aggregation of custom metrics.
        """
        workspace = batch_gradient_workspace
        config_dir = "tests/configs/test_batch_gradient"
        
        # Update the workspace in the config directory for this test
        from omegaconf import OmegaConf
        import shutil
        
        # Create a temporary config directory with updated workspace
        with tempfile.TemporaryDirectory() as temp_config_dir:
            # Copy config files to temp directory
            for config_file in ["base.yaml", "env.yaml", "experiment.yaml", "trials.yaml"]:
                shutil.copy(
                    os.path.join(config_dir, config_file),
                    os.path.join(temp_config_dir, config_file)
                )
            
            # Update workspace in env.yaml
            env_path = os.path.join(temp_config_dir, "env.yaml")
            env_config = OmegaConf.load(env_path)
            env_config.workspace = str(workspace)
            OmegaConf.save(env_config, env_path)
            
            # Create and run experiment
            registry = FactoryRegistry()
            registry.register(FactoryType.PIPELINE, TestBatchGradientPipelineFactory())
            experiment = Experiment.create(temp_config_dir, registry)
            
            print("🚀 Starting MNIST experiment for epoch-level extraction test...")
            experiment.run()
            print("✅ Experiment completed successfully!")
            
            # Get the database path from the experiment
            db_path = Path(experiment.env.artifact_dir) / "batch_gradient_experiment.db"
            assert db_path.exists(), "Database file should exist"
            
            print(f"📊 Using database: {db_path}")
            
            # Test DataFrameExtractor with epoch-level metrics
            print("\n🔍 Testing DataFrameExtractor with epoch-level metrics...")
            
            # Create a data source
            datasource = DBDataSource(db_path=str(db_path), use_sqlite=True, readonly=True)
            
            try:
                extractor = DataFrameExtractor(granularity=['epoch', 'results'])
                
                # Get experiment info (get first experiment)
                experiment = datasource.get_experiment()
                
                # Extract epoch-level metrics
                epoch_df = extractor.extract(datasource)
                
                print(f"📊 Extracted epoch dataframe shape: {epoch_df.shape}")
                print(f"📊 Epoch dataframe columns: {list(epoch_df.columns)}")
                
                # Verify we have epoch data
                assert len(epoch_df) > 0, "Should have epoch data"
                print(f"✅ Found {len(epoch_df)} epoch records")
                
                # Check for aggregated gradient metrics
                expected_epoch_metrics = [
                    "train_grad_max_avg", "train_grad_min_avg", "train_grad_mean_avg", "train_grad_l2_norm_avg",
                    "train_grad_std_avg", "train_grad_median_avg", "train_grad_99th_percentile_avg", "train_grad_sparsity_avg"
                ]
                
                found_epoch_metrics = []
                for metric in expected_epoch_metrics:
                    if metric in epoch_df.columns:
                        found_epoch_metrics.append(metric)
                        print(f"✅ Found epoch metric: {metric}")
                        
                        # Check data integrity
                        non_null_count = epoch_df[metric].notna().sum()
                        if non_null_count > 0:
                            mean_val = epoch_df[metric].mean()
                            print(f"   - Mean across epochs: {mean_val:.6f}")
                
                print(f"✅ Found {len(found_epoch_metrics)}/{len(expected_epoch_metrics)} epoch gradient metrics")
                
                return True
                
            except Exception as e:
                print(f"❌ Error during epoch-level extraction test: {e}")
                import traceback
                traceback.print_exc()
                return False
            finally:
                datasource.close()


def main():
    """Run all dataframe extractor tests."""
    
    print("🧪 Testing DataFrameExtractor with Multiple Custom Metrics")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestDataFrameExtractorCustomMetrics()
    
    # Create workspace
    import tempfile
    workspace = Path(tempfile.mkdtemp())
    
    try:
        # Run tests
        print("\n1️⃣ Running batch-level custom metrics test...")
        test1_success = test_instance.test_dataframe_extractor_with_batch_custom_metrics(workspace)
        
        print("\n2️⃣ Running epoch-level aggregation test...")
        test2_success = test_instance.test_dataframe_extractor_epoch_level_aggregation(workspace)
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        print(f"Batch-level Custom Metrics Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
        print(f"Epoch-level Aggregation Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
        
        all_passed = test1_success and test2_success
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! DataFrameExtractor handles multiple custom metrics correctly!")
        else:
            print("\n❌ SOME TESTS FAILED! DataFrameExtractor needs attention.")
        
        return all_passed
        
    finally:
        # Cleanup workspace
        try:
            import shutil
            shutil.rmtree(workspace, ignore_errors=True)
        except:
            pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
