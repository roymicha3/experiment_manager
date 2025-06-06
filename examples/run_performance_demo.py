#!/usr/bin/env python3
"""
Performance Monitoring Demo Experiment

This script demonstrates comprehensive performance monitoring during ML workloads
using the Experiment Manager's PerformanceTracker and pipeline system.

Features demonstrated:
- Real-time CPU, memory, and GPU monitoring
- Performance alerts and bottleneck detection
- Integration with experiment lifecycle
- Multiple trial configurations
- Automated performance reporting and visualization
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to sys.path to import experiment_manager
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_manager.experiment import Experiment
from examples.pipelines.performance_demo_factory import PerformanceDemoFactory


def main():
    """Run the performance monitoring demonstration experiment."""
    print("🚀 Performance Monitoring Demo - Experiment Manager")
    print("=" * 60)
    
    # Create and configure experiment
    config_dir = os.path.join(os.path.dirname(__file__), "configs", "performance_demo")
    
    print(f"📁 Loading configuration from: {config_dir}")
    experiment = Experiment.create(config_dir, PerformanceDemoFactory)
    
    print(f"✅ Created experiment: {experiment.name}")
    print(f"📂 Workspace: {experiment.env.workspace}")
    
    # Show experiment details
    print(f"\n📋 Experiment Details:")
    print(f"   ID: {experiment.config.id}")
    print(f"   Description: {experiment.config.desc}")
    
    # Show pipeline configuration
    pipeline_config = experiment.config.pipeline
    print(f"\n⚙️  Pipeline Configuration:")
    print(f"   Type: {pipeline_config.type}")
    print(f"   Epochs: {pipeline_config.epochs}")
    print(f"   Batches per epoch: {pipeline_config.batches_per_epoch}")
    print(f"   Work duration: {pipeline_config.work_duration_seconds}s")
    print(f"   CPU intensity: {pipeline_config.cpu_intensity}")
    print(f"   Memory usage: {pipeline_config.memory_usage_mb}MB")
    
    # Show tracker configuration
    trackers = experiment.config.get('trackers', [])
    perf_tracker = next((t for t in trackers if t.type == "PerformanceTracker"), None)
    if perf_tracker:
        print(f"\n📊 Performance Monitoring Configuration:")
        print(f"   Monitoring interval: {perf_tracker.config.monitoring_interval}s")
        print(f"   CPU threshold: {perf_tracker.config.cpu_threshold}%")
        print(f"   Memory threshold: {perf_tracker.config.memory_threshold}%")
        print(f"   Alerts enabled: {perf_tracker.config.enable_alerts}")
        print(f"   Bottleneck detection: {perf_tracker.config.enable_bottleneck_detection}")
    
    # Show trials if configured
    if hasattr(experiment.config, 'trials') and experiment.config.trials:
        print(f"\n🧪 Configured Trials: {len(experiment.config.trials.trials)}")
        for i, trial in enumerate(experiment.config.trials.trials):
            print(f"   {i+1}. {trial.name}: {trial.desc}")
    
    input("\n⏸️  Press Enter to start the experiment...")
    
    # Run the experiment
    print("\n🏃 Running experiment...")
    print("💡 Monitor system performance in real-time!")
    print("⚠️  Performance alerts will be displayed if thresholds are exceeded")
    
    start_time = time.time()
    try:
        result = experiment.run()
        duration = time.time() - start_time
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"⏱️  Total duration: {duration:.1f} seconds")
        
        if isinstance(result, dict):
            print(f"📈 Final Results:")
            for key, value in result.items():
                print(f"   {key}: {value}")
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n❌ Experiment failed after {duration:.1f} seconds")
        print(f"Error: {e}")
        raise
    
    # Display workspace contents
    print(f"\n📁 Generated Files in {experiment.env.workspace}:")
    workspace_path = Path(experiment.env.workspace)
    if workspace_path.exists():
        for root, dirs, files in os.walk(workspace_path):
            level = root.replace(str(workspace_path), '').count(os.sep)
            indent = '  ' * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = '  ' * (level + 1)
            for file in sorted(files):
                file_path = Path(root) / file
                size = file_path.stat().st_size if file_path.exists() else 0
                size_str = f"({size:,} bytes)" if size > 0 else ""
                print(f"{subindent}{file} {size_str}")
    
    # Show performance summary
    print("\n📊 Performance Summary:")
    print("   Check the workspace for detailed performance reports:")
    print(f"   - performance_data.json: Raw performance measurements")
    print(f"   - performance_alerts.json: Resource alerts triggered")
    print(f"   - bottleneck_analysis.json: Bottleneck detection results")
    print(f"   - overall_performance_summary.json: Aggregated statistics")
    print(f"   - overall_performance_plot.png: Performance visualization")
    
    print(f"\n🎉 Performance monitoring demo completed!")
    print(f"💡 Explore the generated files to analyze system performance during the experiment")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 