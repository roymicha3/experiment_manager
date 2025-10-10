#!/usr/bin/env python3
"""
Simple test to reproduce the tuple error by running the exact same pipeline
that was reported to have the error.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_manager.experiment import Experiment
from tests.pipelines.test_batch_gradient_pipeline_factory import TestBatchGradientPipelineFactory


def test_simple_tuple_error_reproduction():
    """
    Simple test to reproduce the tuple error by running the batch gradient experiment.
    """
    print("🧪 Simple Tuple Error Reproduction Test")
    print("=" * 50)
    
    # Create workspace
    workspace = Path(tempfile.mkdtemp())
    config_dir = "tests/configs/test_batch_gradient"
    
    try:
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
            
            print("🚀 Starting MNIST batch gradient experiment...")
            
            # Create and run experiment
            experiment = Experiment.create(temp_config_dir, TestBatchGradientPipelineFactory)
            
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
                experiment.run()
                print("✅ Experiment completed successfully!")
                
                if tuple_errors:
                    print(f"\n❌ FOUND {len(tuple_errors)} TUPLE ERRORS:")
                    for i, error in enumerate(tuple_errors, 1):
                        print(f"   {i}. {error}")
                    return False
                else:
                    print("✅ No tuple errors detected!")
                    return True
                    
            except Exception as e:
                error_message = str(e)
                print(f"❌ Experiment failed: {error_message}")
                
                if "tuple" in error_message.lower() and ("numpy array" in error_message or "torch tensor" in error_message):
                    print(f"🎯 FOUND TUPLE ERROR: {error_message}")
                    print(f"   Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    return False
                else:
                    print(f"   Different error type: {type(e).__name__}")
                    return True
            finally:
                # Restore original logger
                experiment.env.logger.error = original_logger_error
                
    finally:
        # Cleanup workspace
        try:
            import shutil
            shutil.rmtree(workspace, ignore_errors=True)
        except:
            pass


if __name__ == "__main__":
    success = test_simple_tuple_error_reproduction()
    if success:
        print("\n🎉 Test passed - No tuple errors detected!")
        print("   The error might be fixed or requires specific conditions to trigger.")
    else:
        print("\n❌ Test failed - Tuple errors were detected!")
    
    exit(0 if success else 1)
