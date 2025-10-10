#!/usr/bin/env python3
"""
Test to reproduce and catch the tuple error in batch processing.

This test is designed to reproduce the error:
"Got <class 'tuple'>, but numpy array or torch tensor are expected"

The error occurs in the batch processing pipeline when tuples are passed
where tensors or arrays are expected.
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


class TestBatchTupleError:
    """Test suite to reproduce and catch the tuple error in batch processing."""
    
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
    
    def test_reproduce_tuple_error_in_batch_processing(self, batch_gradient_workspace):
        """
        Test to reproduce the tuple error in batch processing.
        
        This test runs the MNIST batch gradient pipeline and catches any
        tuple-related errors that occur during batch processing.
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
            experiment = Experiment.create(temp_config_dir, TestBatchGradientPipelineFactory)
            
            print("🚀 Starting MNIST batch gradient experiment to catch tuple error...")
            
            # Capture any tuple-related errors
            tuple_errors = []
            original_logger_error = experiment.env.logger.error
            
            def error_interceptor(message):
                """Intercept error messages to catch tuple-related errors."""
                if "tuple" in str(message).lower() and ("numpy array" in str(message) or "torch tensor" in str(message)):
                    tuple_errors.append(str(message))
                    print(f"🔍 CAUGHT TUPLE ERROR: {message}")
                original_logger_error(message)
            
            # Replace the logger's error method temporarily
            experiment.env.logger.error = error_interceptor
            
            try:
                experiment.run()
                print("✅ Experiment completed successfully!")
                
                # Check if we caught any tuple errors
                if tuple_errors:
                    print(f"\n❌ FOUND {len(tuple_errors)} TUPLE ERRORS:")
                    for i, error in enumerate(tuple_errors, 1):
                        print(f"   {i}. {error}")
                    
                    # This test should fail if we catch tuple errors
                    assert False, f"Caught {len(tuple_errors)} tuple errors during batch processing"
                else:
                    print("✅ No tuple errors detected!")
                    return True
                    
            except Exception as e:
                error_message = str(e)
                print(f"❌ Experiment failed with error: {error_message}")
                
                # Check if this is the tuple error we're looking for
                if "tuple" in error_message.lower() and ("numpy array" in error_message or "torch tensor" in error_message):
                    print(f"🎯 FOUND THE TUPLE ERROR: {error_message}")
                    print(f"   Error type: {type(e).__name__}")
                    print(f"   Full traceback:")
                    import traceback
                    traceback.print_exc()
                    
                    # This test should fail to indicate we found the error
                    assert False, f"Reproduced tuple error: {error_message}"
                else:
                    # Re-raise if it's a different error
                    raise e
            finally:
                # Restore original logger
                experiment.env.logger.error = original_logger_error
    
    def test_isolate_tuple_error_in_custom_metrics(self, batch_gradient_workspace):
        """
        Test to isolate the tuple error specifically in custom metrics processing.
        
        This test focuses on the custom metrics tracking to see if the error
        occurs when processing the list of custom metrics.
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
            
            # Create experiment but don't run it yet
            experiment = Experiment.create(temp_config_dir, TestBatchGradientPipelineFactory)
            
            print("🔍 Testing custom metrics processing to isolate tuple error...")
            
            # Get the pipeline to test custom metrics directly
            from tests.pipelines.mnist_batch_gradient_pipeline import MNISTBatchGradientPipeline
            from experiment_manager.common.common import Metric
            
            # Create a minimal pipeline instance for testing
            pipeline = MNISTBatchGradientPipeline(experiment.env)
            
            # Test the custom metrics list creation
            test_metrics_data = {
                "batch_loss": 0.5,
                "batch_acc": 0.8,
                "grad_max": 0.1,
                "grad_min": 0.001,
                "grad_mean": 0.05,
                "grad_l2_norm": 1.0,
                "grad_std": 0.02,
                "grad_median": 0.04,
                "grad_99th_percentile": 0.09,
                "grad_sparsity": 0.1
            }
            
            # Create custom metrics list (this is what the pipeline does)
            custom_metrics_list = []
            for metric_name, metric_value in test_metrics_data.items():
                custom_metrics_list.append((metric_name, metric_value))
            
            print(f"📊 Created custom metrics list with {len(custom_metrics_list)} items")
            print(f"   Sample: {custom_metrics_list[:3]}")
            
            # Test if the list contains tuples as expected
            for i, (name, value) in enumerate(custom_metrics_list):
                assert isinstance(name, str), f"Metric name {i} should be string, got {type(name)}"
                assert isinstance(value, (int, float)), f"Metric value {i} should be numeric, got {type(value)}"
                print(f"   ✅ {name}: {value} ({type(value).__name__})")
            
            # Test tracking the custom metrics
            try:
                print("\n🔍 Testing tracker_manager.track with custom metrics list...")
                
                # This is where the error might occur
                experiment.env.tracker_manager.track(Metric.CUSTOM, custom_metrics_list, step=0)
                print("✅ Successfully tracked custom metrics list!")
                
            except Exception as e:
                error_message = str(e)
                print(f"❌ Error during custom metrics tracking: {error_message}")
                
                if "tuple" in error_message.lower() and ("numpy array" in error_message or "torch tensor" in error_message):
                    print(f"🎯 FOUND TUPLE ERROR IN CUSTOM METRICS: {error_message}")
                    print(f"   Error type: {type(e).__name__}")
                    print(f"   Full traceback:")
                    import traceback
                    traceback.print_exc()
                    
                    # This test should fail to indicate we found the error
                    assert False, f"Reproduced tuple error in custom metrics: {error_message}"
                else:
                    # Re-raise if it's a different error
                    raise e
            
            return True
    
    def test_debug_batch_wrapper_execution(self, batch_gradient_workspace):
        """
        Test to debug the batch wrapper execution and catch where the tuple error occurs.
        
        This test adds detailed logging to the batch wrapper to see exactly where
        the tuple error is happening.
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
            
            # Create experiment
            experiment = Experiment.create(temp_config_dir, TestBatchGradientPipelineFactory)
            
            print("🔍 Debugging batch wrapper execution...")
            
            # Add detailed logging to catch the error
            original_logger_error = experiment.env.logger.error
            original_logger_info = experiment.env.logger.info
            original_logger_debug = experiment.env.logger.debug
            
            def debug_logger(message):
                """Enhanced logger that captures more details."""
                print(f"🔍 LOG: {message}")
                if "tuple" in str(message).lower():
                    print(f"   🎯 TUPLE DETECTED IN LOG: {message}")
                original_logger_info(message)
            
            def error_logger(message):
                """Enhanced error logger."""
                print(f"❌ ERROR: {message}")
                if "tuple" in str(message).lower():
                    print(f"   🎯 TUPLE ERROR DETECTED: {message}")
                    import traceback
                    print("   📍 Current traceback:")
                    traceback.print_exc()
                original_logger_error(message)
            
            # Replace loggers temporarily
            experiment.env.logger.info = debug_logger
            experiment.env.logger.error = error_logger
            experiment.env.logger.debug = debug_logger
            
            try:
                print("🚀 Starting experiment with enhanced logging...")
                experiment.run()
                print("✅ Experiment completed successfully!")
                return True
                
            except Exception as e:
                error_message = str(e)
                print(f"❌ Experiment failed: {error_message}")
                
                if "tuple" in error_message.lower():
                    print(f"🎯 CONFIRMED TUPLE ERROR: {error_message}")
                    print(f"   Error type: {type(e).__name__}")
                    print(f"   Full traceback:")
                    import traceback
                    traceback.print_exc()
                    
                    # This test should fail to indicate we found the error
                    assert False, f"Confirmed tuple error: {error_message}"
                else:
                    raise e
            finally:
                # Restore original loggers
                experiment.env.logger.info = original_logger_info
                experiment.env.logger.error = original_logger_error
                experiment.env.logger.debug = original_logger_debug


def main():
    """Run all tuple error tests."""
    
    print("🧪 Testing for Tuple Error in Batch Processing")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestBatchTupleError()
    
    # Create workspace
    import tempfile
    workspace = Path(tempfile.mkdtemp())
    
    try:
        # Run tests
        print("\n1️⃣ Running batch processing tuple error test...")
        test1_success = test_instance.test_reproduce_tuple_error_in_batch_processing(workspace)
        
        print("\n2️⃣ Running custom metrics tuple error test...")
        test2_success = test_instance.test_isolate_tuple_error_in_custom_metrics(workspace)
        
        print("\n3️⃣ Running batch wrapper debug test...")
        test3_success = test_instance.test_debug_batch_wrapper_execution(workspace)
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        print(f"Batch Processing Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
        print(f"Custom Metrics Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
        print(f"Batch Wrapper Debug Test: {'✅ PASSED' if test3_success else '❌ FAILED'}")
        
        all_passed = test1_success and test2_success and test3_success
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! No tuple errors detected.")
        else:
            print("\n❌ SOME TESTS FAILED! Tuple errors were detected and need fixing.")
        
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
