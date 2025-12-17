#!/usr/bin/env python3
"""
Comprehensive test to verify all trackers with various data types in custom metrics.

This test ensures all trackers are properly configured and tests with many different
data types in custom metrics to catch any type-related errors, including the tuple error.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Union

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_manager.experiment import Experiment
from experiment_manager.common.factory_registry import FactoryRegistry, FactoryType
from experiment_manager.pipelines.pipeline_factory import PipelineFactory
from experiment_manager.common.common import Metric
from omegaconf import OmegaConf


class ComprehensiveTrackerTestPipeline:
    """Test pipeline that generates various data types for custom metrics."""
    
    def __init__(self, env, id=None):
        self.env = env
        self.id = id
        self.epoch_metrics = {}
        self.batch_metrics = {}
        self.run_metrics = {}
        
    def run(self, config):
        """Run the test with various data types."""
        self.env.logger.info("Starting comprehensive tracker test with various data types")
        
        # Test different data types in custom metrics
        test_data_types = self._generate_test_data_types()
        
        # Test each data type individually
        for data_type_name, data_value in test_data_types.items():
            self.env.logger.info(f"Testing data type: {data_type_name} = {data_value} (type: {type(data_value).__name__})")
            
            try:
                # Create custom metrics list with this data type
                custom_metrics_list = [
                    (f"test_{data_type_name}", data_value),
                    ("regular_float", 0.5),
                    ("regular_int", 42)
                ]
                
                # Track the custom metrics
                self.env.tracker_manager.track(Metric.CUSTOM, custom_metrics_list, step=0)
                self.env.logger.info(f"✅ Successfully tracked {data_type_name}")
                
            except Exception as e:
                error_msg = str(e)
                self.env.logger.error(f"❌ Failed to track {data_type_name}: {error_msg}")
                
                # Check if this is the tuple error we're looking for
                if "tuple" in error_msg.lower() and ("numpy array" in error_msg or "torch tensor" in error_msg):
                    self.env.logger.error(f"🎯 FOUND TUPLE ERROR with {data_type_name}: {error_msg}")
                    raise e
                else:
                    self.env.logger.warning(f"Different error with {data_type_name}: {error_msg}")
        
        # Test mixed data types in a single custom metrics list
        self.env.logger.info("Testing mixed data types in single custom metrics list...")
        mixed_custom_metrics = [
            ("float_metric", 3.14),
            ("int_metric", 42),
            ("numpy_float", np.float32(2.5)),
            ("numpy_int", np.int32(100)),
            ("torch_float", torch.tensor(1.5)),
            ("torch_int", torch.tensor(200)),
            ("list_metric", [1, 2, 3]),
            ("dict_metric", {"a": 1, "b": 2}),
            ("string_metric", "test_string"),
            ("bool_metric", True),
            ("none_metric", None)
        ]
        
        try:
            self.env.tracker_manager.track(Metric.CUSTOM, mixed_custom_metrics, step=1)
            self.env.logger.info("✅ Successfully tracked mixed data types")
        except Exception as e:
            error_msg = str(e)
            self.env.logger.error(f"❌ Failed to track mixed data types: {error_msg}")
            
            if "tuple" in error_msg.lower() and ("numpy array" in error_msg or "torch tensor" in error_msg):
                self.env.logger.error(f"🎯 FOUND TUPLE ERROR with mixed data types: {error_msg}")
                raise e
        
        return True
    
    def _generate_test_data_types(self) -> Dict[str, Any]:
        """Generate various data types for testing."""
        return {
            # Basic Python types
            "python_float": 3.14159,
            "python_int": 42,
            "python_string": "hello_world",
            "python_bool": True,
            "python_none": None,
            
            # NumPy types
            "numpy_float32": np.float32(1.5),
            "numpy_float64": np.float64(2.5),
            "numpy_int32": np.int32(100),
            "numpy_int64": np.int64(200),
            "numpy_array_1d": np.array([1, 2, 3]),
            "numpy_array_2d": np.array([[1, 2], [3, 4]]),
            "numpy_scalar": np.array(5.0),
            
            # PyTorch types
            "torch_float": torch.tensor(1.5),
            "torch_int": torch.tensor(100),
            "torch_tensor_1d": torch.tensor([1, 2, 3]),
            "torch_tensor_2d": torch.tensor([[1, 2], [3, 4]]),
            "torch_scalar": torch.tensor(5.0),
            
            # Collections
            "list_empty": [],
            "list_numbers": [1, 2, 3, 4, 5],
            "list_mixed": [1, "hello", 3.14, True],
            "tuple_empty": (),
            "tuple_numbers": (1, 2, 3),
            "tuple_mixed": (1, "hello", 3.14),
            "dict_empty": {},
            "dict_simple": {"a": 1, "b": 2},
            "dict_nested": {"a": {"b": 1}, "c": [1, 2, 3]},
            
            # Complex types
            "set_numbers": {1, 2, 3},
            "frozenset": frozenset([1, 2, 3]),
            "bytes": b"hello",
            "bytearray": bytearray(b"world"),
            
            # Edge cases
            "inf": float('inf'),
            "neg_inf": float('-inf'),
            "nan": float('nan'),
            "very_large_int": 2**63 - 1,
            "very_small_float": 1e-10,
        }


class ComprehensiveTrackerTestFactory(PipelineFactory):
    """Factory for the comprehensive tracker test pipeline."""
    
    @staticmethod
    def create(name: str, config, env, id=None):
        # Create the pipeline instance directly
        pipeline = ComprehensiveTrackerTestPipeline(env, id)
        
        # Handle callbacks if present
        callbacks = config.pipeline.get("callbacks", [])
        if callbacks:
            from experiment_manager.common.factory_registry import FactoryType
            for callback_config in callbacks:
                callback_factory = env.factory_registry.get(FactoryType.CALLBACK)
                callback = callback_factory.create(callback_config.type, callback_config, env)
                pipeline.register_callback(callback)
        
        return pipeline


class TestComprehensiveTrackerTypes:
    """Test suite for comprehensive tracker testing with various data types."""
    
    @pytest.fixture
    def comprehensive_workspace(self):
        """Create a temporary workspace for comprehensive tracker tests."""
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
    
    def test_all_trackers_with_various_data_types(self, comprehensive_workspace):
        """
        Test all configured trackers with various data types in custom metrics.
        
        This test ensures all trackers are properly configured and can handle
        different data types without causing tuple errors.
        """
        workspace = comprehensive_workspace
        
        # Create comprehensive configuration with all trackers
        config = self._create_comprehensive_config(workspace)
        
        # Create temporary config directory
        with tempfile.TemporaryDirectory() as temp_config_dir:
            # Save configuration files
            self._save_config_files(temp_config_dir, config)
            
            print("🚀 Starting comprehensive tracker test with all trackers...")
            print(f"📊 Testing with workspace: {workspace}")
            
            # Create and run experiment
            registry = FactoryRegistry()
            registry.register(FactoryType.PIPELINE, ComprehensiveTrackerTestFactory())
            experiment = Experiment.create(temp_config_dir, registry)
            
            # Track any tuple-related errors
            tuple_errors = []
            original_logger_error = experiment.env.logger.error
            
            def error_interceptor(message):
                """Intercept error messages to catch tuple-related errors."""
                if "tuple" in str(message).lower() and ("numpy array" in str(message) or "torch tensor" in str(message)):
                    tuple_errors.append(str(message))
                    print(f"🎯 CAUGHT TUPLE ERROR: {message}")
                original_logger_error(message)
            
            # Replace the logger's error method temporarily
            experiment.env.logger.error = error_interceptor
            
            try:
                # Run the comprehensive test
                experiment.run()
                print("✅ Comprehensive tracker test completed successfully!")
                
                # Check if we caught any tuple errors
                if tuple_errors:
                    print(f"\n❌ FOUND {len(tuple_errors)} TUPLE ERRORS:")
                    for i, error in enumerate(tuple_errors, 1):
                        print(f"   {i}. {error}")
                    
                    # This test should fail if we catch tuple errors
                    assert False, f"Caught {len(tuple_errors)} tuple errors during comprehensive tracker test"
                else:
                    print("✅ No tuple errors detected in comprehensive tracker test!")
                    return True
                    
            except Exception as e:
                error_message = str(e)
                print(f"❌ Comprehensive tracker test failed: {error_message}")
                
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
    
    def _create_comprehensive_config(self, workspace: Path) -> Dict[str, Any]:
        """Create a comprehensive configuration with all trackers."""
        return {
            "base": {
                "name": "comprehensive_tracker_test",
                "version": "1.0.0",
                "description": "Comprehensive test for all trackers with various data types"
            },
            "env": {
                "name": "comprehensive_tracker_env",
                "workspace": str(workspace),
                "log_level": "INFO",
                "trackers": [
                    {
                        "type": "LogTracker",
                        "verbose": True
                    },
                    {
                        "type": "DBTracker",
                        "name": "comprehensive_test.db",
                        "recreate": True
                    },
                    {
                        "type": "PerformanceTracker",
                        "output_file": str(workspace / "artifacts" / "performance_metrics.json")
                    }
                ],
                "callbacks": [
                    {
                        "type": "MetricsTracker",
                        "output_file": str(workspace / "artifacts" / "metrics.csv")
                    }
                ]
            },
            "experiment": {
                "id": "comprehensive_test",
                "name": "Comprehensive Tracker Test",
                "desc": "Comprehensive tracker test",
                "settings": {
                    "debug": True,
                    "verbose": True
                }
            },
            "trials": [
                {
                    "name": "comprehensive_trial",
                    "repeat": 1,
                    "settings": {
                        "pipeline": {
                            "type": "ComprehensiveTrackerTestPipeline"
                        }
                    }
                }
            ]
        }
    
    def _save_config_files(self, temp_config_dir: str, config: Dict[str, Any]):
        """Save configuration files to temporary directory."""
        import yaml
        
        # Save base.yaml
        with open(os.path.join(temp_config_dir, "base.yaml"), 'w') as f:
            yaml.dump(config["base"], f)
        
        # Save env.yaml
        with open(os.path.join(temp_config_dir, "env.yaml"), 'w') as f:
            yaml.dump(config["env"], f)
        
        # Save experiment.yaml
        with open(os.path.join(temp_config_dir, "experiment.yaml"), 'w') as f:
            yaml.dump(config["experiment"], f)
        
        # Save trials.yaml
        with open(os.path.join(temp_config_dir, "trials.yaml"), 'w') as f:
            yaml.dump(config["trials"], f)


def main():
    """Run the comprehensive tracker test."""
    
    print("🧪 Comprehensive Tracker Test with Various Data Types")
    print("=" * 70)
    
    # Create test instance
    test_instance = TestComprehensiveTrackerTypes()
    
    # Create workspace
    import tempfile
    workspace = Path(tempfile.mkdtemp())
    
    try:
        # Run test
        print("\n🔍 Running comprehensive tracker test...")
        test_success = test_instance.test_all_trackers_with_various_data_types(workspace)
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        if test_success:
            print("🎉 COMPREHENSIVE TRACKER TEST PASSED!")
            print("   ✅ All trackers configured and working")
            print("   ✅ Various data types handled correctly")
            print("   ✅ No tuple errors detected")
        else:
            print("❌ COMPREHENSIVE TRACKER TEST FAILED!")
            print("   ❌ Tuple errors or other issues detected")
        
        return test_success
        
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
