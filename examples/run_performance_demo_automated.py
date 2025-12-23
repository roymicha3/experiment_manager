#!/usr/bin/env python3
"""
Automated Performance Monitoring Demo

This script runs the performance monitoring demo automatically without user interaction.
Perfect for quick testing and CI/CD pipelines.
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to sys.path to import experiment_manager
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_manager.experiment import Experiment
from experiment_manager.common.factory_registry import FactoryRegistry, FactoryType
from examples.pipelines.performance_demo_factory import PerformanceDemoFactory


def main():
    """Run the performance monitoring demonstration experiment automatically."""
    print("🚀 Automated Performance Monitoring Demo")
    print("=" * 50)
    
    # Create and configure experiment
    config_dir = os.path.join(os.path.dirname(__file__), "configs", "performance_demo")
    
    print(f"📁 Loading configuration from: {config_dir}")
    
    # Create custom factory registry
    registry = FactoryRegistry()
    registry.register(FactoryType.PIPELINE, PerformanceDemoFactory())
    
    experiment = Experiment.create(config_dir, registry)
    
    print(f"✅ Experiment: {experiment.name}")
    print(f"📂 Workspace: {experiment.env.workspace}")
    
    # Run the experiment
    print(f"\n🏃 Running experiment automatically...")
    
    start_time = time.time()
    try:
        result = experiment.run()
        duration = time.time() - start_time
        
        print(f"\n✅ Experiment completed!")
        print(f"⏱️  Duration: {duration:.1f}s")
        
        # Show basic results
        if isinstance(result, dict):
            print(f"📈 Results: {result.get('status', 'unknown')}")
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n❌ Experiment failed after {duration:.1f}s: {e}")
        return 1
    
    # Count generated files
    workspace_path = Path(experiment.env.workspace)
    if workspace_path.exists():
        file_count = sum(1 for _ in workspace_path.rglob("*.json"))
        print(f"📁 Generated {file_count} JSON files in workspace")
    
    print(f"🎉 Demo completed successfully!")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1) 