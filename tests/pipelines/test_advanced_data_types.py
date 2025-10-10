#!/usr/bin/env python3
"""
Advanced test for various data types in custom metrics following the specified pattern.

This test includes tensors, numpy arrays, and other complex data types
using the Metric.CUSTOM list format with batch_metrics structure.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Union

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_manager.experiment import Experiment
from experiment_manager.common.common import Metric, RunStatus
from omegaconf import OmegaConf


class AdvancedDataTypesPipeline:
    """Test pipeline that generates various advanced data types for custom metrics."""
    
    def __init__(self, env, id=None):
        self.env = env
        self.id = id
        self.epoch_metrics = {}
        self.batch_metrics = {}
        self.run_metrics = {}
        
        # Create a simple model for testing
        self.model = self._create_test_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_test_model(self):
        """Create a simple neural network for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(10, 5),
                    nn.ReLU(),
                    nn.Linear(5, 2)
                ])
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return SimpleModel()
    
    def run(self, config):
        """Run the test with various advanced data types."""
        self.env.logger.info("Starting advanced data types test with tensors and numpy arrays")
        
        # Test 1: Basic tensor operations
        self._test_basic_tensor_metrics()
        
        # Test 2: NumPy array operations
        self._test_numpy_array_metrics()
        
        # Test 3: Complex tensor operations
        self._test_complex_tensor_metrics()
        
        # Test 4: Mixed data types
        self._test_mixed_data_types()
        
        # Test 5: Edge cases with tensors
        self._test_tensor_edge_cases()
        
        return True
    
    def _test_basic_tensor_metrics(self):
        """Test basic tensor operations in custom metrics."""
        self.env.logger.info("Testing basic tensor metrics...")
        
        # Create dummy input and forward pass
        dummy_input = torch.randn(32, 10)
        output = self.model(dummy_input)
        loss = torch.nn.functional.mse_loss(output, torch.randn(32, 2))
        
        # Simulate gradients
        loss.backward()
        
        # Create batch metrics with tensor operations
        self.batch_metrics = {
            Metric.TRAIN_LOSS: torch.sum(loss).item(),
            Metric.NETWORK: self.model,
            Metric.CUSTOM: [
                ("gradient_min", self.model.layers[0].weight.grad.min().item()),
                ("gradient_max", self.model.layers[0].weight.grad.max().item()),
                ("gradient_mean", self.model.layers[0].weight.grad.mean().item()),
                ("gradient_std", self.model.layers[0].weight.grad.std().item()),
                ("gradient_l2_norm", self.model.layers[0].weight.grad.norm().item()),
                ("gradient_sum", self.model.layers[0].weight.grad.sum().item()),
                ("gradient_count", self.model.layers[0].weight.grad.numel()),
                ("gradient_shape", str(self.model.layers[0].weight.grad.shape)),
                ("loss_tensor_sum", loss.sum().item()),
                ("loss_tensor_mean", loss.mean().item())
            ]
        }
        
        # Track the metrics
        try:
            self.env.tracker_manager.track_dict(self.batch_metrics, step=0)
            self.env.logger.info("✅ Successfully tracked basic tensor metrics")
        except Exception as e:
            self.env.logger.error(f"❌ Failed to track basic tensor metrics: {e}")
            if "tuple" in str(e).lower() and ("numpy array" in str(e) or "torch tensor" in str(e)):
                self.env.logger.error(f"🎯 FOUND TUPLE ERROR in basic tensor metrics: {e}")
                raise e
    
    def _test_numpy_array_metrics(self):
        """Test NumPy array operations in custom metrics."""
        self.env.logger.info("Testing NumPy array metrics...")
        
        # Create various NumPy arrays
        arr_1d = np.random.randn(100)
        arr_2d = np.random.randn(10, 10)
        arr_3d = np.random.randn(5, 5, 5)
        
        # Create batch metrics with NumPy operations
        self.batch_metrics = {
            Metric.TRAIN_LOSS: float(np.sum(arr_1d)),
            Metric.CUSTOM: [
                ("numpy_1d_min", float(np.min(arr_1d))),
                ("numpy_1d_max", float(np.max(arr_1d))),
                ("numpy_1d_mean", float(np.mean(arr_1d))),
                ("numpy_1d_std", float(np.std(arr_1d))),
                ("numpy_1d_norm", float(np.linalg.norm(arr_1d))),
                ("numpy_2d_min", float(np.min(arr_2d))),
                ("numpy_2d_max", float(np.max(arr_2d))),
                ("numpy_2d_mean", float(np.mean(arr_2d))),
                ("numpy_2d_trace", float(np.trace(arr_2d))),
                ("numpy_2d_det", float(np.linalg.det(arr_2d))),
                ("numpy_3d_min", float(np.min(arr_3d))),
                ("numpy_3d_max", float(np.max(arr_3d))),
                ("numpy_3d_mean", float(np.mean(arr_3d))),
                ("numpy_3d_std", float(np.std(arr_3d))),
                ("numpy_3d_norm", float(np.linalg.norm(arr_3d.flatten())))
            ]
        }
        
        # Track the metrics
        try:
            self.env.tracker_manager.track_dict(self.batch_metrics, step=1)
            self.env.logger.info("✅ Successfully tracked NumPy array metrics")
        except Exception as e:
            self.env.logger.error(f"❌ Failed to track NumPy array metrics: {e}")
            if "tuple" in str(e).lower() and ("numpy array" in str(e) or "torch tensor" in str(e)):
                self.env.logger.error(f"🎯 FOUND TUPLE ERROR in NumPy array metrics: {e}")
                raise e
    
    def _test_complex_tensor_metrics(self):
        """Test complex tensor operations in custom metrics."""
        self.env.logger.info("Testing complex tensor metrics...")
        
        # Create complex tensors
        tensor_2d = torch.randn(20, 20)
        tensor_3d = torch.randn(5, 5, 5)
        tensor_4d = torch.randn(2, 3, 4, 5)
        
        # Complex operations
        eigenvals = torch.linalg.eigvals(tensor_2d)
        svd_u, svd_s, svd_v = torch.linalg.svd(tensor_2d)
        
        # Create batch metrics with complex tensor operations
        self.batch_metrics = {
            Metric.TRAIN_LOSS: float(tensor_2d.sum()),
            Metric.CUSTOM: [
                ("tensor_2d_min", float(tensor_2d.min())),
                ("tensor_2d_max", float(tensor_2d.max())),
                ("tensor_2d_mean", float(tensor_2d.mean())),
                ("tensor_2d_std", float(tensor_2d.std())),
                ("tensor_2d_norm", float(tensor_2d.norm())),
                ("tensor_2d_trace", float(tensor_2d.trace())),
                ("tensor_2d_det", float(tensor_2d.det())),
                ("tensor_2d_rank", float(torch.linalg.matrix_rank(tensor_2d))),
                ("tensor_3d_min", float(tensor_3d.min())),
                ("tensor_3d_max", float(tensor_3d.max())),
                ("tensor_3d_mean", float(tensor_3d.mean())),
                ("tensor_3d_std", float(tensor_3d.std())),
                ("tensor_3d_norm", float(tensor_3d.norm())),
                ("tensor_4d_min", float(tensor_4d.min())),
                ("tensor_4d_max", float(tensor_4d.max())),
                ("tensor_4d_mean", float(tensor_4d.mean())),
                ("tensor_4d_std", float(tensor_4d.std())),
                ("tensor_4d_norm", float(tensor_4d.norm())),
                ("eigenvals_min", float(eigenvals.real.min())),
                ("eigenvals_max", float(eigenvals.real.max())),
                ("svd_s_min", float(svd_s.min())),
                ("svd_s_max", float(svd_s.max())),
                ("svd_s_mean", float(svd_s.mean()))
            ]
        }
        
        # Track the metrics
        try:
            self.env.tracker_manager.track_dict(self.batch_metrics, step=2)
            self.env.logger.info("✅ Successfully tracked complex tensor metrics")
        except Exception as e:
            self.env.logger.error(f"❌ Failed to track complex tensor metrics: {e}")
            if "tuple" in str(e).lower() and ("numpy array" in str(e) or "torch tensor" in str(e)):
                self.env.logger.error(f"🎯 FOUND TUPLE ERROR in complex tensor metrics: {e}")
                raise e
    
    def _test_mixed_data_types(self):
        """Test mixed data types in custom metrics."""
        self.env.logger.info("Testing mixed data types...")
        
        # Create mixed data
        torch_tensor = torch.randn(10, 10)
        numpy_array = np.random.randn(10, 10)
        python_list = [1, 2, 3, 4, 5]
        python_dict = {"a": 1, "b": 2, "c": 3}
        
        # Convert between types
        torch_to_numpy = torch_tensor.numpy()
        numpy_to_torch = torch.from_numpy(numpy_array)
        
        # Create batch metrics with mixed data types
        self.batch_metrics = {
            Metric.TRAIN_LOSS: float(torch_tensor.sum()),
            Metric.CUSTOM: [
                ("torch_tensor_min", float(torch_tensor.min())),
                ("torch_tensor_max", float(torch_tensor.max())),
                ("torch_tensor_mean", float(torch_tensor.mean())),
                ("numpy_array_min", float(numpy_array.min())),
                ("numpy_array_max", float(numpy_array.max())),
                ("numpy_array_mean", float(numpy_array.mean())),
                ("python_list_sum", float(sum(python_list))),
                ("python_list_mean", float(sum(python_list) / len(python_list))),
                ("python_dict_sum", float(sum(python_dict.values()))),
                ("python_dict_count", float(len(python_dict))),
                ("torch_to_numpy_min", float(torch_to_numpy.min())),
                ("torch_to_numpy_max", float(torch_to_numpy.max())),
                ("numpy_to_torch_min", float(numpy_to_torch.min())),
                ("numpy_to_torch_max", float(numpy_to_torch.max())),
                ("mixed_operations", float(torch_tensor.sum() + numpy_array.sum())),
                ("type_conversion_test", float(torch.tensor(numpy_array.sum()).item()))
            ]
        }
        
        # Track the metrics
        try:
            self.env.tracker_manager.track_dict(self.batch_metrics, step=3)
            self.env.logger.info("✅ Successfully tracked mixed data types")
        except Exception as e:
            self.env.logger.error(f"❌ Failed to track mixed data types: {e}")
            if "tuple" in str(e).lower() and ("numpy array" in str(e) or "torch tensor" in str(e)):
                self.env.logger.error(f"🎯 FOUND TUPLE ERROR in mixed data types: {e}")
                raise e
    
    def _test_tensor_edge_cases(self):
        """Test tensor edge cases in custom metrics."""
        self.env.logger.info("Testing tensor edge cases...")
        
        # Edge case tensors
        zero_tensor = torch.zeros(10, 10)
        ones_tensor = torch.ones(10, 10)
        inf_tensor = torch.tensor([float('inf'), float('-inf'), float('nan')])
        large_tensor = torch.randn(1000, 1000)
        small_tensor = torch.randn(1, 1) * 1e-10
        
        # Create batch metrics with edge cases
        self.batch_metrics = {
            Metric.TRAIN_LOSS: float(zero_tensor.sum()),
            Metric.CUSTOM: [
                ("zero_tensor_sum", float(zero_tensor.sum())),
                ("zero_tensor_mean", float(zero_tensor.mean())),
                ("ones_tensor_sum", float(ones_tensor.sum())),
                ("ones_tensor_mean", float(ones_tensor.mean())),
                ("inf_tensor_count", float(len(inf_tensor))),
                ("inf_tensor_isfinite", float(torch.isfinite(inf_tensor).sum())),
                ("large_tensor_min", float(large_tensor.min())),
                ("large_tensor_max", float(large_tensor.max())),
                ("large_tensor_mean", float(large_tensor.mean())),
                ("large_tensor_std", float(large_tensor.std())),
                ("small_tensor_min", float(small_tensor.min())),
                ("small_tensor_max", float(small_tensor.max())),
                ("small_tensor_mean", float(small_tensor.mean())),
                ("small_tensor_std", float(small_tensor.std())),
                ("tensor_memory_usage", float(large_tensor.numel() * large_tensor.element_size())),
                ("tensor_device", str(large_tensor.device)),
                ("tensor_dtype", str(large_tensor.dtype)),
                ("tensor_shape", str(large_tensor.shape)),
                ("tensor_ndim", float(large_tensor.ndim)),
                ("tensor_size", float(large_tensor.size(0))),
                ("tensor_numel", float(large_tensor.numel()))
            ]
        }
        
        # Track the metrics
        try:
            self.env.tracker_manager.track_dict(self.batch_metrics, step=4)
            self.env.logger.info("✅ Successfully tracked tensor edge cases")
        except Exception as e:
            self.env.logger.error(f"❌ Failed to track tensor edge cases: {e}")
            if "tuple" in str(e).lower() and ("numpy array" in str(e) or "torch tensor" in str(e)):
                self.env.logger.error(f"🎯 FOUND TUPLE ERROR in tensor edge cases: {e}")
                raise e


class AdvancedDataTypesFactory:
    """Factory for the advanced data types test pipeline."""
    
    @staticmethod
    def create(name: str, config, env, id=None):
        return AdvancedDataTypesPipeline(env, id)


class TestAdvancedDataTypes:
    """Test suite for advanced data types with tensors and numpy arrays."""
    
    @pytest.fixture
    def advanced_workspace(self):
        """Create a temporary workspace for advanced data types tests."""
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
    
    def test_advanced_data_types_with_tensors(self, advanced_workspace):
        """
        Test advanced data types including tensors and numpy arrays.
        
        This test follows the specified pattern with batch_metrics and Metric.CUSTOM
        as a list of tuples, testing various tensor and numpy operations.
        """
        workspace = advanced_workspace
        
        # Create configuration
        config = self._create_advanced_config(workspace)
        
        # Create temporary config directory
        with tempfile.TemporaryDirectory() as temp_config_dir:
            # Save configuration files
            self._save_config_files(temp_config_dir, config)
            
            print("🚀 Starting advanced data types test with tensors and numpy arrays...")
            print(f"📊 Testing with workspace: {workspace}")
            
            # Create and run experiment
            experiment = Experiment.create(temp_config_dir, AdvancedDataTypesFactory)
            
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
                # Run the advanced test
                experiment.run()
                print("✅ Advanced data types test completed successfully!")
                
                # Check if we caught any tuple errors
                if tuple_errors:
                    print(f"\n❌ FOUND {len(tuple_errors)} TUPLE ERRORS:")
                    for i, error in enumerate(tuple_errors, 1):
                        print(f"   {i}. {error}")
                    
                    # This test should fail if we catch tuple errors
                    assert False, f"Caught {len(tuple_errors)} tuple errors during advanced data types test"
                else:
                    print("✅ No tuple errors detected in advanced data types test!")
                    return True
                    
            except Exception as e:
                error_message = str(e)
                print(f"❌ Advanced data types test failed: {error_message}")
                
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
    
    def _create_advanced_config(self, workspace: Path) -> Dict[str, Any]:
        """Create configuration for advanced data types test."""
        return {
            "base": {
                "name": "advanced_data_types_test",
                "version": "1.0.0",
                "description": "Advanced test for tensors and numpy arrays in custom metrics"
            },
            "env": {
                "name": "advanced_data_types_env",
                "workspace": str(workspace),
                "log_level": "INFO",
                "trackers": [
                    {
                        "type": "LogTracker",
                        "verbose": True
                    },
                    {
                        "type": "DBTracker",
                        "name": "advanced_data_types.db",
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
                "id": "advanced_data_types",
                "name": "Advanced Data Types Test",
                "desc": "Advanced test for tensors and numpy arrays",
                "settings": {
                    "debug": True,
                    "verbose": True
                }
            },
            "trials": [
                {
                    "name": "advanced_trial",
                    "repeat": 1,
                    "settings": {
                        "pipeline": {
                            "type": "AdvancedDataTypesPipeline"
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
    """Run the advanced data types test."""
    
    print("🧪 Advanced Data Types Test with Tensors and NumPy Arrays")
    print("=" * 70)
    
    # Create test instance
    test_instance = TestAdvancedDataTypes()
    
    # Create workspace
    import tempfile
    workspace = Path(tempfile.mkdtemp())
    
    try:
        # Run test
        print("\n🔍 Running advanced data types test...")
        test_success = test_instance.test_advanced_data_types_with_tensors(workspace)
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        if test_success:
            print("🎉 ADVANCED DATA TYPES TEST PASSED!")
            print("   ✅ All trackers configured and working")
            print("   ✅ Tensor and NumPy operations handled correctly")
            print("   ✅ No tuple errors detected")
        else:
            print("❌ ADVANCED DATA TYPES TEST FAILED!")
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
