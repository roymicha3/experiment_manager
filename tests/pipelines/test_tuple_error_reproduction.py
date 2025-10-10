#!/usr/bin/env python3
"""
Test to specifically reproduce the tuple error that was reported.

This test tries to reproduce the exact error:
"Pipeline batch failed: Got <class 'tuple'>, but numpy array or torch tensor are expected"

By running the same pipeline configuration that was failing.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_manager.experiment import Experiment
from tests.pipelines.test_batch_gradient_pipeline_factory import TestBatchGradientPipelineFactory


class TestTupleErrorReproduction:
    """Test suite to reproduce the specific tuple error."""
    
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
    
    def test_reproduce_exact_tuple_error(self, batch_gradient_workspace):
        """
        Test to reproduce the exact tuple error by running multiple experiments
        with different configurations to trigger the error.
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
            
            # Try multiple runs to catch intermittent errors
            tuple_errors = []
            
            for run_num in range(3):  # Run 3 times to catch intermittent errors
                print(f"\n🔄 Run {run_num + 1}/3 - Attempting to reproduce tuple error...")
                
                try:
                    # Create fresh experiment for each run
                    experiment = Experiment.create(temp_config_dir, TestBatchGradientPipelineFactory)
                    
                    # Capture any tuple-related errors
                    original_logger_error = experiment.env.logger.error
                    
                    def error_interceptor(message):
                        """Intercept error messages to catch tuple-related errors."""
                        if "tuple" in str(message).lower() and ("numpy array" in str(message) or "torch tensor" in str(message)):
                            tuple_errors.append(f"Run {run_num + 1}: {message}")
                            print(f"🎯 CAUGHT TUPLE ERROR IN RUN {run_num + 1}: {message}")
                        original_logger_error(message)
                    
                    # Replace the logger's error method temporarily
                    experiment.env.logger.error = error_interceptor
                    
                    # Run the experiment
                    experiment.run()
                    print(f"✅ Run {run_num + 1} completed successfully")
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"❌ Run {run_num + 1} failed: {error_message}")
                    
                    if "tuple" in error_message.lower() and ("numpy array" in error_message or "torch tensor" in error_message):
                        tuple_errors.append(f"Run {run_num + 1}: {error_message}")
                        print(f"🎯 CAUGHT TUPLE ERROR IN RUN {run_num + 1}: {error_message}")
                        print(f"   Error type: {type(e).__name__}")
                        import traceback
                        traceback.print_exc()
                    else:
                        print(f"   Different error type: {type(e).__name__}")
                
                finally:
                    # Restore original logger
                    if 'experiment' in locals():
                        experiment.env.logger.error = original_logger_error
            
            # Report results
            if tuple_errors:
                print(f"\n❌ FOUND {len(tuple_errors)} TUPLE ERRORS:")
                for i, error in enumerate(tuple_errors, 1):
                    print(f"   {i}. {error}")
                
                # This test should fail if we catch tuple errors
                assert False, f"Reproduced {len(tuple_errors)} tuple errors"
            else:
                print("\n✅ No tuple errors detected in any run!")
                return True
    
    def test_stress_test_batch_processing(self, batch_gradient_workspace):
        """
        Stress test the batch processing to try to trigger the tuple error
        under different conditions.
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
            
            print("🔥 Stress testing batch processing...")
            
            # Try different batch sizes and configurations
            test_configs = [
                {"batch_size": 32, "epochs": 1},
                {"batch_size": 64, "epochs": 1}, 
                {"batch_size": 128, "epochs": 1},
            ]
            
            tuple_errors = []
            
            for i, config in enumerate(test_configs):
                print(f"\n🧪 Test config {i + 1}: {config}")
                
                # Update the trials config (pipeline settings are in trials)
                trials_config = OmegaConf.load(os.path.join(temp_config_dir, "trials.yaml"))
                for trial in trials_config:
                    trial.settings.batch_size = config["batch_size"]
                    trial.settings.epochs = config["epochs"]
                OmegaConf.save(trials_config, os.path.join(temp_config_dir, "trials.yaml"))
                
                try:
                    experiment = Experiment.create(temp_config_dir, TestBatchGradientPipelineFactory)
                    
                    # Capture any tuple-related errors
                    original_logger_error = experiment.env.logger.error
                    
                    def error_interceptor(message):
                        """Intercept error messages to catch tuple-related errors."""
                        if "tuple" in str(message).lower() and ("numpy array" in str(message) or "torch tensor" in str(message)):
                            tuple_errors.append(f"Config {i + 1}: {message}")
                            print(f"🎯 CAUGHT TUPLE ERROR IN CONFIG {i + 1}: {message}")
                        original_logger_error(message)
                    
                    experiment.env.logger.error = error_interceptor
                    
                    # Run the experiment
                    experiment.run()
                    print(f"✅ Config {i + 1} completed successfully")
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"❌ Config {i + 1} failed: {error_message}")
                    
                    if "tuple" in error_message.lower() and ("numpy array" in error_message or "torch tensor" in error_message):
                        tuple_errors.append(f"Config {i + 1}: {error_message}")
                        print(f"🎯 CAUGHT TUPLE ERROR IN CONFIG {i + 1}: {error_message}")
                        print(f"   Error type: {type(e).__name__}")
                        import traceback
                        traceback.print_exc()
                
                finally:
                    if 'experiment' in locals():
                        experiment.env.logger.error = original_logger_error
            
            # Report results
            if tuple_errors:
                print(f"\n❌ FOUND {len(tuple_errors)} TUPLE ERRORS IN STRESS TEST:")
                for i, error in enumerate(tuple_errors, 1):
                    print(f"   {i}. {error}")
                
                # This test should fail if we catch tuple errors
                assert False, f"Reproduced {len(tuple_errors)} tuple errors in stress test"
            else:
                print("\n✅ No tuple errors detected in stress test!")
                return True
    
    def test_debug_specific_pipeline_components(self, batch_gradient_workspace):
        """
        Test specific pipeline components that might be causing the tuple error.
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
            
            print("🔍 Debugging specific pipeline components...")
            
            # Test individual components
            try:
                experiment = Experiment.create(temp_config_dir, TestBatchGradientPipelineFactory)
                
                # Test the pipeline creation
                print("📋 Testing pipeline creation...")
                # Get pipeline type from base config
                pipeline_type = experiment.base_config.pipeline.type
                pipeline = experiment.env.factory.create(
                    name=pipeline_type,
                    config=experiment.base_config,
                    env=experiment.env
                )
                print("✅ Pipeline created successfully")
                
                # Test data loading
                print("📊 Testing data loading...")
                train_loader, val_loader, test_loader = pipeline._create_real_mnist_data()
                print(f"✅ Data loaded - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
                
                # Test model creation
                print("🧠 Testing model creation...")
                model = pipeline._create_model()
                print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
                
                # Test optimizer creation
                print("⚙️ Testing optimizer creation...")
                optimizer = pipeline._create_optimizer(model)
                print(f"✅ Optimizer created: {type(optimizer).__name__}")
                
                # Test gradient computation
                print("📈 Testing gradient computation...")
                # Create dummy data for testing
                import torch
                dummy_data = torch.randn(32, 784)
                dummy_target = torch.randint(0, 10, (32,))
                
                # Forward pass
                output = model(dummy_data)
                loss = pipeline.criterion(output, dummy_target)
                
                # Backward pass
                loss.backward()
                
                # Test gradient stats computation
                grad_stats = pipeline._compute_gradient_stats(model)
                print(f"✅ Gradient stats computed: {list(grad_stats.keys())}")
                
                # Test custom metrics list creation
                print("📝 Testing custom metrics list creation...")
                batch_metrics_data = {
                    "batch_loss": float(loss.item()),
                    "batch_acc": 0.5,
                    **grad_stats
                }
                
                custom_metrics_list = []
                for metric_name, metric_value in batch_metrics_data.items():
                    custom_metrics_list.append((metric_name, metric_value))
                
                print(f"✅ Custom metrics list created with {len(custom_metrics_list)} items")
                
                # Test tracking
                print("📊 Testing metric tracking...")
                from experiment_manager.common.common import Metric
                experiment.env.tracker_manager.track(Metric.CUSTOM, custom_metrics_list, step=0)
                print("✅ Custom metrics tracked successfully")
                
                return True
                
            except Exception as e:
                error_message = str(e)
                print(f"❌ Component test failed: {error_message}")
                
                if "tuple" in error_message.lower() and ("numpy array" in error_message or "torch tensor" in error_message):
                    print(f"🎯 FOUND TUPLE ERROR IN COMPONENT TEST: {error_message}")
                    print(f"   Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    
                    # This test should fail to indicate we found the error
                    assert False, f"Reproduced tuple error in component test: {error_message}"
                else:
                    # Re-raise if it's a different error
                    raise e


def main():
    """Run all tuple error reproduction tests."""
    
    print("🧪 Testing Tuple Error Reproduction")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestTupleErrorReproduction()
    
    # Create workspace
    import tempfile
    workspace = Path(tempfile.mkdtemp())
    
    try:
        # Run tests
        print("\n1️⃣ Running exact tuple error reproduction test...")
        test1_success = test_instance.test_reproduce_exact_tuple_error(workspace)
        
        print("\n2️⃣ Running stress test for tuple error...")
        test2_success = test_instance.test_stress_test_batch_processing(workspace)
        
        print("\n3️⃣ Running component debug test...")
        test3_success = test_instance.test_debug_specific_pipeline_components(workspace)
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        print(f"Exact Error Reproduction: {'✅ PASSED' if test1_success else '❌ FAILED'}")
        print(f"Stress Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
        print(f"Component Debug: {'✅ PASSED' if test3_success else '❌ FAILED'}")
        
        all_passed = test1_success and test2_success and test3_success
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! No tuple errors reproduced.")
            print("   The error might be fixed or requires specific conditions to trigger.")
        else:
            print("\n❌ SOME TESTS FAILED! Tuple errors were reproduced.")
        
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
