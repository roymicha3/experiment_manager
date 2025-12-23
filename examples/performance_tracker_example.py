#!/usr/bin/env python3
"""
PerformanceTracker Usage Example

This example demonstrates how to use the PerformanceTracker to monitor
system performance during machine learning experiments.
"""

import os
import time
import tempfile
from pathlib import Path

from experiment_manager.trackers import PerformanceTracker
from experiment_manager.common import Level

def simulate_ml_experiment():
    """Simulate a machine learning experiment with performance monitoring."""
    print("🚀 Starting ML Experiment with Performance Monitoring")
    
    # Create workspace for this example
    workspace = Path(tempfile.mkdtemp()) / "ml_experiment"
    workspace.mkdir(exist_ok=True)
    
    # Initialize PerformanceTracker with production settings
    tracker = PerformanceTracker(
        workspace=str(workspace),
        monitoring_interval=1.0,              # Sample every 1 second (reduced from 0.5s)
        enable_alerts=True,                   # Enable resource alerts
        enable_bottleneck_detection=True,     # Enable bottleneck analysis
        cpu_threshold=85.0,                   # Alert at 85% CPU
        memory_threshold=85.0,                # Alert at 85% memory
        gpu_threshold=95.0,                   # Alert at 95% GPU
        history_size=500,                     # Keep 500 snapshots (reduced from 2000)
        lightweight_mode=False,               # Full monitoring mode
        test_mode=False                       # Enable actual monitoring
    )
    
    print(f"📊 Tracker initialized at: {tracker.workspace}")
    print(f"🖥️  GPU monitoring: {'Enabled' if tracker.gpu_count > 0 else 'Disabled'}")
    
    try:
        # Experiment lifecycle
        tracker.on_create(Level.EXPERIMENT, experiment_id="performance_demo")
        tracker.log_params({
            "model": "ResNet50",
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "Adam"
        })
        
        # Start experiment-level monitoring
        tracker.on_start(Level.EXPERIMENT)
        print("✅ Experiment started - monitoring active")
        
        # Simulate training iterations
        for epoch in range(3):
            print(f"\n📈 Epoch {epoch + 1}/3")
            
            # Start epoch monitoring
            tracker.on_create(Level.EPOCH, epoch_id=epoch)
            tracker.on_start(Level.EPOCH)
            
            # Simulate epoch work with varying CPU load
            for batch in range(5):
                print(f"  Batch {batch + 1}/5", end="")
                
                # Simulate CPU-intensive work
                start_time = time.time()
                while time.time() - start_time < 0.3:  # 300ms of work (reduced from 800ms)
                    _ = sum(i * i for i in range(500))  # Reduced computation load
                
                # Track custom metrics
                loss = 1.5 - (epoch * 0.3) - (batch * 0.05)
                accuracy = 0.6 + (epoch * 0.15) + (batch * 0.02)
                
                tracker.track("custom", ("loss", loss), step=epoch * 5 + batch)
                tracker.track("custom", ("accuracy", accuracy), step=epoch * 5 + batch)
                
                print(f" - Loss: {loss:.3f}, Acc: {accuracy:.3f}")
                
                # Simulate occasional checkpoint
                if batch == 2:
                    checkpoint_path = f"checkpoints/epoch_{epoch}_batch_{batch}.pth"
                    tracker.on_checkpoint(None, checkpoint_path, {
                        "loss": loss,
                        "accuracy": accuracy
                    })
                    print(f"    💾 Checkpoint saved: {checkpoint_path}")
            
            # End epoch
            tracker.on_end(Level.EPOCH)
            
            # Get epoch performance summary
            epoch_summary = tracker.get_performance_summary(Level.EPOCH)
            if epoch_summary:
                cpu_avg = epoch_summary.get('cpu', {}).get('avg', 0)
                memory_avg = epoch_summary.get('memory', {}).get('avg', 0)
                print(f"    📊 Epoch {epoch + 1} - Avg CPU: {cpu_avg:.1f}%, Avg Memory: {memory_avg:.1f}%")
                
                if 'gpu' in epoch_summary:
                    gpu_avg = epoch_summary['gpu'].get('avg_utilization', 0)
                    print(f"        GPU: {gpu_avg:.1f}%")
        
        # End experiment
        tracker.on_end(Level.EXPERIMENT)
        print("\n✅ Experiment completed")
        
        # Generate final performance report
        print("\n📋 Generating Performance Report...")
        final_summary = tracker.get_performance_summary()
        
        if final_summary:
            print(f"📊 Performance Summary:")
            print(f"  📈 Total measurements: {final_summary.get('measurement_count', 0)}")
            
            if 'time_range' in final_summary:
                duration = final_summary['time_range']['duration_minutes']
                print(f"  ⏱️  Duration: {duration:.1f} minutes")
            
            if 'cpu' in final_summary:
                cpu = final_summary['cpu']
                print(f"  🖥️  CPU - Avg: {cpu['avg']:.1f}%, Peak: {cpu['max']:.1f}%")
            
            if 'memory' in final_summary:
                memory = final_summary['memory']
                print(f"  💾 Memory - Avg: {memory['avg']:.1f}%, Peak: {memory['peak_gb']:.1f}GB")
            
            if 'gpu' in final_summary:
                gpu = final_summary['gpu']
                print(f"  🎮 GPU - Avg: {gpu['avg_utilization']:.1f}%, Peak Memory: {gpu['peak_memory_gb']:.1f}GB")
            
            # Alerts summary
            alerts = final_summary.get('alerts', {})
            total_alerts = alerts.get('total', 0)
            if total_alerts > 0:
                print(f"  ⚠️  Alerts: {total_alerts} ({alerts.get('critical', 0)} critical, {alerts.get('warnings', 0)} warnings)")
            else:
                print(f"  ✅ No performance alerts")
            
            # Bottleneck analysis
            if 'bottlenecks' in final_summary:
                bottleneck = final_summary['bottlenecks']['primary_bottleneck']
                print(f"  🎯 Primary bottleneck: {bottleneck}")
        
        # Show recent alerts
        if tracker.alerts:
            print(f"\n⚠️  Recent Performance Alerts:")
            for alert in tracker.alerts[-5:]:  # Last 5 alerts
                print(f"    {alert.level.upper()}: {alert.message}")
        
        # Show bottleneck recommendations
        if tracker.bottlenecks:
            latest_bottleneck = tracker.bottlenecks[-1]
            print(f"\n💡 Performance Recommendations:")
            for recommendation in latest_bottleneck.recommendations[:3]:
                print(f"    • {recommendation}")
        
        # Save all performance data
        tracker.save()
        print(f"\n💾 Performance data saved to: {tracker.workspace}")
        
        # List generated files
        workspace_path = Path(tracker.workspace)
        if workspace_path.exists():
            files = list(workspace_path.glob("*.json"))
            if files:
                print(f"📁 Generated files:")
                for file_path in sorted(files):
                    print(f"    • {file_path.name}")
    
    finally:
        # Cleanup
        tracker.stop_monitoring()
        print("\n🔚 Monitoring stopped")

def demonstrate_lightweight_mode():
    """Demonstrate lightweight mode for faster initialization."""
    print("\n🪶 Demonstrating Lightweight Mode")
    
    workspace = Path(tempfile.mkdtemp()) / "lightweight_demo"
    workspace.mkdir(exist_ok=True)
    
    # Configuration for lightweight mode
    config = {
        "monitoring_interval": 1.0,
        "lightweight_mode": True,     # Fast initialization
        "enable_alerts": True,
        "cpu_threshold": 90.0,
        "memory_threshold": 90.0,
        "test_mode": False
    }
    
    # Create tracker from config (typical usage)
    tracker = PerformanceTracker.from_config(config, str(workspace))
    print(f"✅ Lightweight tracker created quickly")
    
    # Capture basic performance info
    snapshot = tracker._capture_snapshot()
    print(f"📊 Current performance: CPU {snapshot.cpu_percent:.1f}%, Memory {snapshot.memory_percent:.1f}%")
    
    # Upgrade to full monitoring when needed
    tracker.enable_full_monitoring()
    print(f"⚡ Upgraded to full monitoring mode")
    
    tracker.stop_monitoring()

def show_config_integration():
    """Show how to integrate PerformanceTracker with YAML configuration."""
    print("\n📝 Configuration Integration Example")
    
    # Example YAML configuration
    yaml_config = """
trackers:
  - type: PerformanceTracker
    config:
      monitoring_interval: 2.0
      enable_alerts: true
      enable_bottleneck_detection: true
      cpu_threshold: 85.0
      memory_threshold: 80.0
      gpu_threshold: 95.0
      history_size: 1000
      lightweight_mode: true  # Fast startup for development
      test_mode: false
"""
    
    print("📄 Example YAML configuration:")
    print(yaml_config)
    
    # Simulate config loading
    config_dict = {
        "monitoring_interval": 2.0,
        "enable_alerts": True,
        "enable_bottleneck_detection": True,
        "cpu_threshold": 85.0,
        "memory_threshold": 80.0,
        "gpu_threshold": 95.0,
        "history_size": 1000,
        "lightweight_mode": True,
        "test_mode": False
    }
    
    workspace = Path(tempfile.mkdtemp()) / "config_demo"
    workspace.mkdir(exist_ok=True)
    
    tracker = PerformanceTracker.from_config(config_dict, str(workspace))
    print(f"✅ Tracker created from configuration")
    print(f"   Monitoring interval: {tracker.monitoring_interval}s")
    print(f"   CPU threshold: {tracker.cpu_threshold}%")
    print(f"   Lightweight mode: {tracker.lightweight_mode}")

if __name__ == "__main__":
    print("🎯 PerformanceTracker Usage Examples\n")
    
    # Run examples
    simulate_ml_experiment()
    demonstrate_lightweight_mode()
    show_config_integration()
    
    print("\n🎉 All examples completed!")
    print("\nKey Features Demonstrated:")
    print("✅ Real-time performance monitoring (CPU, Memory, GPU)")
    print("✅ Alert system for resource constraints")
    print("✅ Bottleneck detection and recommendations")
    print("✅ Hierarchical monitoring (Experiment → Epoch → Batch)")
    print("✅ Performance visualization and reporting")
    print("✅ Integration with experiment lifecycle")
    print("✅ Lightweight mode for fast initialization")
    print("✅ YAML configuration support")
    print("✅ Cross-platform compatibility") 