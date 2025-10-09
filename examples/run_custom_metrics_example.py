#!/usr/bin/env python3
"""
Run Custom Metrics Example

This script demonstrates how to run the custom metrics example with all trackers.
It shows the proper way to use custom batch metrics with multiple key-value pairs.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiment_manager.experiment import Experiment
from experiment_manager.common.factory import Factory
from examples.pipelines.custom_metrics_factory import CustomMetricsPipelineFactory


def main():
    """Run the custom metrics example."""
    print("=== Custom Metrics Example ===")
    print("This example demonstrates custom batch metrics with multiple key-value pairs.")
    print("It shows how to use the new list format for custom metrics that works with all trackers.")
    print()
    
    # Get the configuration directory
    config_dir = Path(__file__).parent / "configs" / "custom_metrics_test"
    
    if not config_dir.exists():
        print(f"❌ Configuration directory not found: {config_dir}")
        print("Please ensure the configuration files are in the correct location.")
        return False
    
    print(f"📁 Using configuration directory: {config_dir}")
    print()
    
    try:
        # Create the experiment
        print("🔧 Creating experiment...")
        experiment = Experiment.create(str(config_dir), CustomMetricsPipelineFactory)
        
        print("✅ Experiment created successfully")
        print(f"📊 Experiment name: {experiment.name}")
        print(f"📝 Description: {experiment.desc}")
        print()
        
        # Run the experiment
        print("🚀 Running experiment...")
        experiment.run()
        
        print("✅ Experiment completed successfully!")
        print()
        print("📈 Check the following for results:")
        print(f"   - Database: {experiment.env.workspace}/artifacts/experiment.db")
        print(f"   - MLflow: {experiment.env.workspace}/mlruns/")
        print(f"   - TensorBoard: {experiment.env.workspace}/tensorboard/")
        print(f"   - Logs: {experiment.env.workspace}/logs/")
        print()
        print("🎯 The custom metrics should now be visible in all trackers!")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
