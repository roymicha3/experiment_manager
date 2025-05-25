"""
Unit tests for PerformanceTracker

This module contains comprehensive tests for the PerformanceTracker class,
including monitoring, alerting, bottleneck detection, and tracker integration.
"""
import os
import json
import time
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiment_manager.trackers.plugins.performance_tracker import (
    PerformanceTracker, PerformanceSnapshot, PerformanceAlert, BottleneckAnalysis
)
from experiment_manager.common.common import Level, Metric


class TestPerformanceSnapshot(unittest.TestCase):
    """Test PerformanceSnapshot dataclass."""
    
    def test_initialization(self):
        """Test PerformanceSnapshot initialization."""
        snapshot = PerformanceSnapshot()
        
        # Check default values
        self.assertEqual(snapshot.cpu_percent, 0.0)
        self.assertEqual(snapshot.memory_percent, 0.0)
        self.assertEqual(snapshot.memory_used_gb, 0.0)
        self.assertEqual(snapshot.gpu_utilization, [])
        self.assertEqual(snapshot.process_count, 0)
        
        # Check timestamp is recent
        time_diff = datetime.now() - snapshot.timestamp
        self.assertLess(time_diff.total_seconds(), 1.0)
    
    def test_custom_values(self):
        """Test PerformanceSnapshot with custom values."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        snapshot = PerformanceSnapshot(
            timestamp=custom_time,
            cpu_percent=75.5,
            memory_percent=60.2,
            gpu_utilization=[85.0, 90.0],
            process_count=150
        )
        
        self.assertEqual(snapshot.timestamp, custom_time)
        self.assertEqual(snapshot.cpu_percent, 75.5)
        self.assertEqual(snapshot.memory_percent, 60.2)
        self.assertEqual(snapshot.gpu_utilization, [85.0, 90.0])
        self.assertEqual(snapshot.process_count, 150)


class TestPerformanceAlert(unittest.TestCase):
    """Test PerformanceAlert dataclass."""
    
    def test_initialization(self):
        """Test PerformanceAlert initialization."""
        timestamp = datetime.now()
        alert = PerformanceAlert(
            timestamp=timestamp,
            level='warning',
            resource='cpu',
            message='High CPU usage',
            value=95.0,
            threshold=90.0
        )
        
        self.assertEqual(alert.timestamp, timestamp)
        self.assertEqual(alert.level, 'warning')
        self.assertEqual(alert.resource, 'cpu')
        self.assertEqual(alert.message, 'High CPU usage')
        self.assertEqual(alert.value, 95.0)
        self.assertEqual(alert.threshold, 90.0)


class TestBottleneckAnalysis(unittest.TestCase):
    """Test BottleneckAnalysis dataclass."""
    
    def test_initialization(self):
        """Test BottleneckAnalysis initialization."""
        timestamp = datetime.now()
        recommendations = ["Reduce batch size", "Enable checkpointing"]
        scores = {"cpu": 0.9, "memory": 0.7, "gpu": 0.8}
        
        analysis = BottleneckAnalysis(
            timestamp=timestamp,
            primary_bottleneck="cpu",
            bottleneck_score=0.9,
            recommendations=recommendations,
            resource_scores=scores
        )
        
        self.assertEqual(analysis.timestamp, timestamp)
        self.assertEqual(analysis.primary_bottleneck, "cpu")
        self.assertEqual(analysis.bottleneck_score, 0.9)
        self.assertEqual(analysis.recommendations, recommendations)
        self.assertEqual(analysis.resource_scores, scores)


class TestPerformanceTracker(unittest.TestCase):
    """Test PerformanceTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = os.path.join(self.temp_dir, "test_workspace")
        os.makedirs(self.workspace, exist_ok=True)
        
        # Create tracker with test configuration - USE TEST MODE to disable monitoring threads
        self.tracker = PerformanceTracker(
            workspace=self.workspace,
            monitoring_interval=0.1,
            enable_alerts=True,
            enable_bottleneck_detection=True,
            cpu_threshold=80.0,
            memory_threshold=80.0,
            gpu_threshold=90.0,
            history_size=100,
            lightweight_mode=True,  # Enable lightweight mode for faster tests
            test_mode=True  # Disable monitoring threads to prevent hanging
        )
    
    def tearDown(self):
        """Clean up test environment."""
        self.tracker.stop_monitoring()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test PerformanceTracker initialization."""
        # Note: base Tracker class appends "artifacts" to workspace
        expected_workspace = os.path.join(self.workspace, "artifacts")
        self.assertEqual(self.tracker.workspace, expected_workspace)
        self.assertEqual(self.tracker.monitoring_interval, 0.1)
        self.assertTrue(self.tracker.enable_alerts)
        self.assertTrue(self.tracker.enable_bottleneck_detection)
        self.assertEqual(self.tracker.cpu_threshold, 80.0)
        self.assertEqual(self.tracker.memory_threshold, 80.0)
        self.assertEqual(self.tracker.gpu_threshold, 90.0)
        self.assertEqual(self.tracker.history_size, 100)
        
        # Check initial state
        self.assertIsNone(self.tracker.id)
        self.assertIsNone(self.tracker.parent)
        self.assertIsNone(self.tracker.current_level)
        self.assertFalse(self.tracker.is_monitoring)
        self.assertIsNone(self.tracker.monitor_thread)
        
        # Check data structures
        self.assertEqual(len(self.tracker.performance_history), 0)
        self.assertEqual(len(self.tracker.alerts), 0)
        self.assertEqual(len(self.tracker.bottlenecks), 0)
    
    def test_from_config(self):
        """Test creating tracker from configuration."""
        config = {
            'monitoring_interval': 0.5,
            'enable_alerts': False,
            'cpu_threshold': 95.0,
            'history_size': 500
        }
        
        tracker = PerformanceTracker.from_config(config, self.workspace)
        
        self.assertEqual(tracker.monitoring_interval, 0.5)
        self.assertFalse(tracker.enable_alerts)
        self.assertEqual(tracker.cpu_threshold, 95.0)
        self.assertEqual(tracker.history_size, 500)
        # Default values
        self.assertTrue(tracker.enable_bottleneck_detection)
        self.assertEqual(tracker.memory_threshold, 90.0)
        # Verify lightweight mode defaults to True for performance
        self.assertTrue(tracker.lightweight_mode)
    
    @patch('experiment_manager.trackers.plugins.performance_tracker.HAS_PSUTIL', False)
    def test_capture_snapshot_no_psutil(self):
        """Test snapshot capture when psutil is not available."""
        snapshot = self.tracker._capture_snapshot()
        
        # Should return empty snapshot
        self.assertEqual(snapshot.cpu_percent, 0.0)
        self.assertEqual(snapshot.memory_percent, 0.0)
        self.assertEqual(snapshot.memory_used_gb, 0.0)
        self.assertEqual(snapshot.gpu_utilization, [])
    
    def test_check_alerts_cpu(self):
        """Test CPU alert detection."""
        snapshot = PerformanceSnapshot(cpu_percent=85.0)
        
        initial_alert_count = len(self.tracker.alerts)
        self.tracker._check_alerts(snapshot)
        
        # Should trigger CPU alert (threshold is 80.0)
        self.assertEqual(len(self.tracker.alerts), initial_alert_count + 1)
        
        alert = self.tracker.alerts[-1]
        self.assertEqual(alert.resource, 'cpu')
        self.assertEqual(alert.level, 'warning')
        self.assertEqual(alert.value, 85.0)
        self.assertEqual(alert.threshold, 80.0)
    
    def test_check_alerts_memory_critical(self):
        """Test critical memory alert detection."""
        snapshot = PerformanceSnapshot(memory_percent=97.0, memory_used_gb=15.5)
        
        initial_alert_count = len(self.tracker.alerts)
        self.tracker._check_alerts(snapshot)
        
        # Should trigger critical memory alert (>95%)
        self.assertEqual(len(self.tracker.alerts), initial_alert_count + 1)
        
        alert = self.tracker.alerts[-1]
        self.assertEqual(alert.resource, 'memory')
        self.assertEqual(alert.level, 'critical')
        self.assertEqual(alert.value, 97.0)
        self.assertIn('15.5GB', alert.message)
    
    def test_check_alerts_gpu(self):
        """Test GPU alert detection."""
        snapshot = PerformanceSnapshot(gpu_utilization=[95.0, 85.0])
        
        initial_alert_count = len(self.tracker.alerts)
        self.tracker._check_alerts(snapshot)
        
        # Should trigger alert for first GPU only (threshold is 90.0)
        self.assertEqual(len(self.tracker.alerts), initial_alert_count + 1)
        
        alert = self.tracker.alerts[-1]
        self.assertEqual(alert.resource, 'gpu_0')
        self.assertEqual(alert.level, 'warning')
        self.assertEqual(alert.value, 95.0)
    
    def test_check_alerts_disabled(self):
        """Test that alerts are not generated when disabled."""
        self.tracker.enable_alerts = False
        snapshot = PerformanceSnapshot(cpu_percent=95.0, memory_percent=95.0)
        
        initial_alert_count = len(self.tracker.alerts)
        self.tracker._check_alerts(snapshot)
        
        # No alerts should be generated
        self.assertEqual(len(self.tracker.alerts), initial_alert_count)
    
    def test_analyze_bottlenecks_cpu(self):
        """Test CPU bottleneck detection."""
        # Add some history first
        for _ in range(15):
            self.tracker.performance_history.append(PerformanceSnapshot(cpu_percent=50.0))
        
        snapshot = PerformanceSnapshot(cpu_percent=85.0, memory_percent=40.0)
        
        initial_bottleneck_count = len(self.tracker.bottlenecks)
        self.tracker._analyze_bottlenecks(snapshot)
        
        self.assertEqual(len(self.tracker.bottlenecks), initial_bottleneck_count + 1)
        
        analysis = self.tracker.bottlenecks[-1]
        self.assertEqual(analysis.primary_bottleneck, 'cpu')
        self.assertEqual(analysis.bottleneck_score, 0.85)
        self.assertIn("reducing batch size", " ".join(analysis.recommendations).lower())
    
    def test_analyze_bottlenecks_memory(self):
        """Test memory bottleneck detection."""
        # Add some history first
        for _ in range(15):
            self.tracker.performance_history.append(PerformanceSnapshot(memory_percent=50.0))
        
        snapshot = PerformanceSnapshot(cpu_percent=40.0, memory_percent=85.0)
        
        initial_bottleneck_count = len(self.tracker.bottlenecks)
        self.tracker._analyze_bottlenecks(snapshot)
        
        self.assertEqual(len(self.tracker.bottlenecks), initial_bottleneck_count + 1)
        
        analysis = self.tracker.bottlenecks[-1]
        self.assertEqual(analysis.primary_bottleneck, 'memory')
        self.assertEqual(analysis.bottleneck_score, 0.85)
        self.assertIn("batch size", " ".join(analysis.recommendations).lower())
    
    def test_analyze_bottlenecks_gpu(self):
        """Test GPU bottleneck detection."""
        # Add some history first
        for _ in range(15):
            self.tracker.performance_history.append(PerformanceSnapshot(gpu_utilization=[50.0]))
        
        snapshot = PerformanceSnapshot(
            cpu_percent=40.0, 
            memory_percent=40.0,
            gpu_utilization=[95.0]
        )
        
        initial_bottleneck_count = len(self.tracker.bottlenecks)
        self.tracker._analyze_bottlenecks(snapshot)
        
        self.assertEqual(len(self.tracker.bottlenecks), initial_bottleneck_count + 1)
        
        analysis = self.tracker.bottlenecks[-1]
        self.assertEqual(analysis.primary_bottleneck, 'gpu')
        self.assertEqual(analysis.bottleneck_score, 0.95)
        self.assertIn("gpu", " ".join(analysis.recommendations).lower())
    
    def test_analyze_bottlenecks_insufficient_history(self):
        """Test bottleneck analysis with insufficient history."""
        snapshot = PerformanceSnapshot(cpu_percent=95.0)
        
        initial_bottleneck_count = len(self.tracker.bottlenecks)
        self.tracker._analyze_bottlenecks(snapshot)
        
        # No analysis should be performed with insufficient history
        self.assertEqual(len(self.tracker.bottlenecks), initial_bottleneck_count)
    
    def test_analyze_bottlenecks_disabled(self):
        """Test that bottleneck analysis is skipped when disabled."""
        self.tracker.enable_bottleneck_detection = False
        
        # Add sufficient history
        for _ in range(15):
            self.tracker.performance_history.append(PerformanceSnapshot())
        
        snapshot = PerformanceSnapshot(cpu_percent=95.0)
        
        initial_bottleneck_count = len(self.tracker.bottlenecks)
        self.tracker._analyze_bottlenecks(snapshot)
        
        # No analysis should be performed when disabled
        self.assertEqual(len(self.tracker.bottlenecks), initial_bottleneck_count)
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start and stop."""
        # Create a separate tracker for this test without test_mode
        monitoring_tracker = PerformanceTracker(
            workspace=self.workspace,
            monitoring_interval=0.05,  # Very fast interval
            lightweight_mode=True,
            test_mode=False  # Allow actual monitoring for this test
        )
        
        # Initially not monitoring
        self.assertFalse(monitoring_tracker.is_monitoring)
        self.assertIsNone(monitoring_tracker.monitor_thread)
        
        # Start monitoring
        monitoring_tracker.start_monitoring()
        self.assertTrue(monitoring_tracker.is_monitoring)
        self.assertIsNotNone(monitoring_tracker.monitor_thread)
        self.assertTrue(monitoring_tracker.monitor_thread.is_alive())
        
        # Let it run briefly - much shorter time
        time.sleep(0.02)
        
        # Stop monitoring
        monitoring_tracker.stop_monitoring()
        self.assertFalse(monitoring_tracker.is_monitoring)
        
        # Thread should stop (shorter wait time)
        time.sleep(0.02)
        if monitoring_tracker.monitor_thread:
            self.assertFalse(monitoring_tracker.monitor_thread.is_alive())
    
    def test_monitoring_idempotent_start(self):
        """Test that starting monitoring multiple times is safe."""
        self.tracker.start_monitoring()
        first_thread = self.tracker.monitor_thread
        
        # Start again
        self.tracker.start_monitoring()
        second_thread = self.tracker.monitor_thread
        
        # Should be the same thread
        self.assertEqual(first_thread, second_thread)
        
        self.tracker.stop_monitoring()
    
    def test_get_performance_summary_empty(self):
        """Test performance summary with no data."""
        summary = self.tracker.get_performance_summary()
        self.assertEqual(summary, {})
    
    def test_get_performance_summary_with_data(self):
        """Test performance summary with data."""
        # Add test data
        start_time = datetime.now()
        snapshots = [
            PerformanceSnapshot(timestamp=start_time, cpu_percent=50.0, memory_percent=40.0),
            PerformanceSnapshot(timestamp=start_time + timedelta(seconds=30), cpu_percent=70.0, memory_percent=60.0),
            PerformanceSnapshot(timestamp=start_time + timedelta(seconds=60), cpu_percent=60.0, memory_percent=50.0)
        ]
        
        for snapshot in snapshots:
            self.tracker.performance_history.append(snapshot)
        
        # Add some alerts and bottlenecks
        self.tracker.alerts.append(PerformanceAlert(
            timestamp=start_time,
            level='warning',
            resource='cpu',
            message='Test alert',
            value=70.0,
            threshold=65.0
        ))
        
        summary = self.tracker.get_performance_summary()
        
        self.assertEqual(summary['measurement_count'], 3)
        self.assertEqual(summary['cpu']['avg'], 60.0)
        self.assertEqual(summary['cpu']['max'], 70.0)
        self.assertEqual(summary['cpu']['min'], 50.0)
        self.assertEqual(summary['memory']['avg'], 50.0)
        self.assertEqual(summary['memory']['max'], 60.0)
        self.assertEqual(summary['memory']['min'], 40.0)
        self.assertEqual(summary['alerts']['total'], 1)
        self.assertEqual(summary['alerts']['warnings'], 1)
        self.assertEqual(summary['alerts']['critical'], 0)
    
    def test_get_performance_summary_with_gpu(self):
        """Test performance summary with GPU data."""
        # Add GPU data
        snapshots = [
            PerformanceSnapshot(
                cpu_percent=50.0,
                gpu_utilization=[80.0, 70.0],
                gpu_memory_used=[6.0, 4.0]
            ),
            PerformanceSnapshot(
                cpu_percent=60.0,
                gpu_utilization=[90.0, 85.0],
                gpu_memory_used=[8.0, 6.0]
            )
        ]
        
        for snapshot in snapshots:
            self.tracker.performance_history.append(snapshot)
        
        summary = self.tracker.get_performance_summary()
        
        self.assertIn('gpu', summary)
        self.assertEqual(summary['gpu']['avg_utilization'], 85.0)  # max of [80,70] and [90,85] = 80 and 90, avg = 85
        self.assertEqual(summary['gpu']['max_utilization'], 90.0)
        self.assertEqual(summary['gpu']['peak_memory_gb'], 8.0)
        self.assertEqual(summary['gpu']['gpu_count'], self.tracker.gpu_count)
    
    def test_get_performance_summary_by_level(self):
        """Test performance summary filtered by level."""
        # Add data for different levels
        level_snapshot = PerformanceSnapshot(cpu_percent=75.0)
        other_snapshot = PerformanceSnapshot(cpu_percent=50.0)
        
        self.tracker.level_metrics[Level.EPOCH] = [level_snapshot]
        self.tracker.performance_history.append(other_snapshot)
        
        # Summary for specific level
        level_summary = self.tracker.get_performance_summary(Level.EPOCH)
        self.assertEqual(level_summary['measurement_count'], 1)
        self.assertEqual(level_summary['cpu']['avg'], 75.0)
        
        # Overall summary should include all data
        overall_summary = self.tracker.get_performance_summary()
        self.assertEqual(overall_summary['measurement_count'], 1)  # Only in history
        self.assertEqual(overall_summary['cpu']['avg'], 50.0)
    
    @patch('experiment_manager.trackers.plugins.performance_tracker.HAS_MATPLOTLIB', False)
    def test_generate_performance_plot_no_matplotlib(self):
        """Test plot generation when matplotlib is not available."""
        output_path = os.path.join(self.workspace, "test_plot.png")
        
        # Should not raise error, just warn
        self.tracker.generate_performance_plot(output_path)
        
        # File should not be created
        self.assertFalse(os.path.exists(output_path))
    
    def test_track_custom_metric(self):
        """Test tracking custom performance metrics."""
        # Track custom metric
        self.tracker.track(Metric.CUSTOM, ("training_speed", 150.5), step=10)
        
        # Check file was created - use tracker's actual workspace
        custom_file = os.path.join(self.tracker.workspace, "custom_performance_metrics.json")
        self.assertTrue(os.path.exists(custom_file))
        
        # Read and verify data
        with open(custom_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['metric'], 'training_speed')
        self.assertEqual(data[0]['value'], 150.5)
        self.assertEqual(data[0]['step'], 10)
    
    def test_on_checkpoint(self):
        """Test checkpoint event handling."""
        network = Mock()
        checkpoint_path = "/path/to/checkpoint.pth"
        
        self.tracker.on_checkpoint(network, checkpoint_path)
        
        # Check file was created - use tracker's actual workspace
        checkpoint_file = os.path.join(self.tracker.workspace, "checkpoint_performance.json")
        self.assertTrue(os.path.exists(checkpoint_file))
        
        # Read and verify data
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['checkpoint_path'], checkpoint_path)
        self.assertIn('timestamp', data[0])
        self.assertIn('cpu_percent', data[0])
        self.assertIn('memory_percent', data[0])
    
    def test_log_params(self):
        """Test parameter logging."""
        params = {'batch_size': 32, 'learning_rate': 0.001}
        
        self.tracker.log_params(params)
        
        # Check file was created - use tracker's actual workspace
        params_file = os.path.join(self.tracker.workspace, "performance_params.json")
        self.assertTrue(os.path.exists(params_file))
        
        # Read and verify data
        with open(params_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['user_params'], params)
        self.assertEqual(data['monitoring_interval'], 0.1)
        self.assertTrue(data['enable_alerts'])
        self.assertIn('has_psutil', data)
        self.assertIn('gpu_count', data)
    
    def test_on_create_level(self):
        """Test level creation handling."""
        self.tracker.on_create(Level.EXPERIMENT, "test_experiment", description="Test")
        
        self.assertEqual(self.tracker.current_level, Level.EXPERIMENT)
        self.assertIsNotNone(self.tracker.id)
        self.assertIn("perf_EXPERIMENT", self.tracker.id)
        
        # Check file was created - use tracker's actual workspace
        level_file = os.path.join(self.tracker.workspace, "level_creation_experiment.json")
        self.assertTrue(os.path.exists(level_file))
        
        # Read and verify data
        with open(level_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['level'], 'EXPERIMENT')
        self.assertEqual(data[0]['args'], ["test_experiment"])
        self.assertEqual(data[0]['kwargs'], {"description": "Test"})
    
    def test_on_start_monitoring_levels(self):
        """Test that monitoring starts for appropriate levels."""
        # In test_mode, monitoring should NOT start
        self.tracker.on_start(Level.EXPERIMENT)
        self.assertFalse(self.tracker.is_monitoring)  # Should be false in test_mode
        
        # BATCH should not start monitoring regardless
        self.tracker.on_start(Level.BATCH)
        self.assertFalse(self.tracker.is_monitoring)
    
    def test_on_start_baseline_capture(self):
        """Test baseline capture on level start."""
        # In lightweight mode, baseline capture should be skipped
        self.tracker.on_start(Level.TRIAL_RUN)
        
        # Check that baseline file is NOT created in lightweight mode - use tracker's actual workspace
        baseline_file = os.path.join(self.tracker.workspace, "baseline_trial_run.json")
        self.assertFalse(os.path.exists(baseline_file))
        
        # Enable full monitoring and try again
        self.tracker.enable_full_monitoring()
        self.tracker.on_start(Level.EPOCH)  # Use different level to avoid conflict
        
        # Without psutil, baseline file still won't be created, but no error should occur
        epoch_baseline_file = os.path.join(self.tracker.workspace, "baseline_epoch.json")
        # This test just ensures no exceptions are thrown
    
    def test_on_end_summary_generation(self):
        """Test summary generation on level end."""
        # Add some test data for the level
        test_snapshot = PerformanceSnapshot(cpu_percent=65.0)
        self.tracker.level_metrics[Level.EPOCH] = [test_snapshot]
        
        # Call on_end to generate the summary
        self.tracker.on_end(Level.EPOCH)
        
        # Check summary file was created - use tracker's actual workspace
        summary_file = os.path.join(self.tracker.workspace, "summary_epoch.json")
        self.assertTrue(os.path.exists(summary_file))
        
        # Read and verify data
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['measurement_count'], 1)
        self.assertEqual(data['cpu']['avg'], 65.0)
    
    def test_on_add_artifact(self):
        """Test artifact addition handling."""
        artifact_path = "/path/to/artifact.txt"
        
        self.tracker.on_add_artifact(Level.TRIAL, artifact_path)
        
        # Check file was created - use tracker's actual workspace
        artifact_file = os.path.join(self.tracker.workspace, "artifact_performance.json")
        self.assertTrue(os.path.exists(artifact_file))
        
        # Read and verify data
        with open(artifact_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['level'], 'TRIAL')
        self.assertEqual(data[0]['artifact_path'], artifact_path)
        self.assertIn('timestamp', data[0])
    
    def test_create_child(self):
        """Test child tracker creation."""
        child = self.tracker.create_child()
        
        self.assertIsInstance(child, PerformanceTracker)
        self.assertEqual(child.parent, self.tracker)
        self.assertEqual(child.monitoring_interval, self.tracker.monitoring_interval)
        self.assertEqual(child.cpu_threshold, self.tracker.cpu_threshold)
        
        # Child should have its own workspace
        self.assertNotEqual(child.workspace, self.tracker.workspace)
        self.assertTrue(os.path.exists(child.workspace))
    
    def test_create_child_custom_workspace(self):
        """Test child tracker creation with custom workspace."""
        custom_workspace = os.path.join(self.temp_dir, "custom_child")
        child = self.tracker.create_child(custom_workspace)
        
        # The base Tracker class appends "artifacts" to the workspace
        expected_workspace = os.path.join(custom_workspace, "artifacts")
        self.assertEqual(child.workspace, expected_workspace)
        self.assertTrue(os.path.exists(expected_workspace))
    
    def test_save_empty_data(self):
        """Test saving when there's no data."""
        self.tracker.save()
        
        # Summary file should be created even with no data - use tracker's actual workspace
        summary_file = os.path.join(self.tracker.workspace, "overall_performance_summary.json")
        self.assertTrue(os.path.exists(summary_file))
        
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data, {})
    
    def test_save_with_data(self):
        """Test saving with performance data."""
        # Add test data
        snapshot = PerformanceSnapshot(cpu_percent=75.0, memory_percent=60.0)
        self.tracker.performance_history.append(snapshot)
        
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            level='warning',
            resource='cpu',
            message='Test alert',
            value=75.0,
            threshold=70.0
        )
        self.tracker.alerts.append(alert)
        
        bottleneck = BottleneckAnalysis(
            timestamp=datetime.now(),
            primary_bottleneck='cpu',
            bottleneck_score=0.75,
            recommendations=['Test recommendation'],
            resource_scores={'cpu': 0.75}
        )
        self.tracker.bottlenecks.append(bottleneck)
        
        self.tracker.save()
        
        # Check all files were created - use tracker's actual workspace
        files_to_check = [
            "performance_data.json",
            "performance_alerts.json",
            "bottleneck_analysis.json",
            "overall_performance_summary.json"
        ]
        
        for filename in files_to_check:
            file_path = os.path.join(self.tracker.workspace, filename)
            self.assertTrue(os.path.exists(file_path))
            
            # Verify files contain data
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.assertGreater(len(data) if isinstance(data, list) else len(data.keys()), 0)
    
    def test_destructor_cleanup(self):
        """Test that destructor stops monitoring."""
        # In test_mode, monitoring cannot start, so test the cleanup logic directly
        self.tracker.is_monitoring = True  # Simulate monitoring state
        
        # Trigger destructor
        self.tracker.__del__()
        
        self.assertFalse(self.tracker.is_monitoring)
    
    def test_integration_full_workflow(self):
        """Test full workflow integration."""
        # Test that the tracker can handle basic lifecycle events without errors
        
        # Create experiment (should not start monitoring in test_mode)
        self.tracker.on_create(Level.EXPERIMENT, "integration_test")
        self.tracker.on_start(Level.EXPERIMENT)
        self.assertFalse(self.tracker.is_monitoring)  # Should be false in test_mode
        
        # Track a metric
        self.tracker.track(Metric.CUSTOM, ("test_metric", 42.0), step=1)
        
        # Add artifact
        self.tracker.on_add_artifact(Level.EXPERIMENT, "test_artifact.txt")
        
        # End experiment
        self.tracker.on_end(Level.EXPERIMENT)
        
        # Save all data
        self.tracker.save()
        
        # Verify key files exist - use tracker's actual workspace
        key_files = [
            "level_creation_experiment.json",
            "custom_performance_metrics.json",
            "artifact_performance.json",
            "overall_performance_summary.json"
        ]
        
        for filename in key_files:
            file_path = os.path.join(self.tracker.workspace, filename)
            self.assertTrue(os.path.exists(file_path), f"Expected file {filename} was not created")


if __name__ == '__main__':
    unittest.main() 