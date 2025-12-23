#!/usr/bin/env python3
"""
Test for the multiple CUSTOM_METRICS fix.
This test verifies that multiple custom metrics are properly tracked and stored.
"""

import os
import sys
import tempfile
import json
import sqlite3
from omegaconf import DictConfig

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_manager.pipelines.callbacks.metric_tracker import MetricsTracker
from experiment_manager.trackers.plugins.db_tracker import DBTracker
from experiment_manager.trackers.tracker_manager import TrackerManager
from experiment_manager.common.common import Metric, Level
from experiment_manager.environment import Environment


class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        self.logger = MockLogger()


class MockLogger:
    """Mock logger for testing."""
    
    def info(self, message: str):
        print(f"INFO: {message}")
    
    def debug(self, message: str):
        print(f"DEBUG: {message}")
    
    def error(self, message: str):
        print(f"ERROR: {message}")


def test_metrics_tracker_multiple_custom_metrics():
    """Test that MetricsTracker properly handles multiple custom metrics."""
    
    print("=== Testing MetricsTracker Multiple Custom Metrics ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create mock environment
        env = MockEnvironment(artifact_dir=temp_dir)
        
        # Create a metrics tracker
        tracker = MetricsTracker(env)
        
        # Start tracking
        tracker.on_start()
        
        # Test multiple custom metrics (the bug fix)
        print("\n--- Test: Multiple Custom Metrics ---")
        metrics_multiple = {
            Metric.CUSTOM: [
                ("training_speed", 150.5),
                ("memory_usage", 85.2),
                ("gpu_utilization", 67.8),
                ("learning_rate", 0.001),
                ("batch_size", 32)
            ],
            Metric.TRAIN_LOSS: 0.6,
            Metric.VAL_ACC: 0.8
        }
        
        tracker.on_epoch_end(epoch_idx=0, metrics=metrics_multiple)
        print(f"Tracker metrics after multiple: {tracker.metrics}")
        
        # Check results
        print("\n--- Results Analysis ---")
        
        # Check if all custom metrics were stored
        expected_custom_metrics = [
            "training_speed", 
            "memory_usage", 
            "gpu_utilization",
            "learning_rate",
            "batch_size"
        ]
        
        found_custom_metrics = []
        for key, values in tracker.metrics.items():
            if isinstance(key, str) and key in expected_custom_metrics:
                found_custom_metrics.append(key)
                print(f"✅ Found custom metric '{key}' with {len(values)} values: {values}")
        
        missing_metrics = set(expected_custom_metrics) - set(found_custom_metrics)
        if missing_metrics:
            print(f"❌ Missing custom metrics: {missing_metrics}")
            return False
        else:
            print(f"✅ All {len(expected_custom_metrics)} custom metrics were found!")
        
        # Test the CSV output
        print("\n--- Testing CSV Output ---")
        tracker.on_end({})
        
        metrics_file = os.path.join(temp_dir, "metrics.log")
        if os.path.exists(metrics_file):
            print(f"✅ Metrics CSV file created: {metrics_file}")
            with open(metrics_file, 'r') as f:
                content = f.read()
                print("CSV content:")
                print(content)
        else:
            print(f"❌ Metrics CSV file not found: {metrics_file}")
            return False
        
        return True


def test_db_tracker_multiple_custom_metrics():
    """Test that DBTracker properly handles multiple custom metrics."""
    
    print("\n=== Testing DBTracker Multiple Custom Metrics ===")
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create mock environment
        env = MockEnvironment(artifact_dir=temp_dir)
        
        # Create a DB tracker
        tracker = DBTracker(workspace=temp_dir)
        
        # Mock the tracker ID (normally set by create_child)
        tracker.id = 1
        
        # Initialize the tracker (normally done by create_child)
        tracker.create_child()
        
        # Test multiple custom metrics
        print("\n--- Test: Multiple Custom Metrics ---")
        custom_metrics = [
            ("training_speed", 150.5),
            ("memory_usage", 85.2),
            ("gpu_utilization", 67.8),
            ("learning_rate", 0.001),
            ("batch_size", 32)
        ]
        
        # Track multiple custom metrics
        tracker.track(Metric.CUSTOM, custom_metrics, step=0)
        
        # Check if database was created (DBTracker creates it in artifacts subdirectory)
        db_file = os.path.join(temp_dir, "artifacts", "tracker.db")
        if os.path.exists(db_file):
            print(f"✅ Database file created: {db_file}")
            
            # Query the database to check custom metrics
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Check METRIC table
                cursor.execute("SELECT type, COUNT(*) FROM METRIC GROUP BY type")
                metric_counts = cursor.fetchall()
                
                print(f"   Metrics in database:")
                for metric_type, count in metric_counts:
                    print(f"     - {metric_type}: {count} records")
                
                # Check for custom metrics specifically
                cursor.execute("SELECT type FROM METRIC WHERE type IN (?, ?, ?, ?, ?)", 
                             ["training_speed", "memory_usage", "gpu_utilization", "learning_rate", "batch_size"])
                custom_metrics_in_db = cursor.fetchall()
                print(f"   Custom metrics in database: {len(custom_metrics_in_db)}")
                
                conn.close()
                
                if len(custom_metrics_in_db) >= len(custom_metrics):
                    print("✅ SUCCESS: All custom metrics were stored in database!")
                    return True
                else:
                    print(f"❌ FAILURE: Only {len(custom_metrics_in_db)}/{len(custom_metrics)} custom metrics stored")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Error querying database: {e}")
                return False
        else:
            print(f"❌ Database file not found: {db_file}")
            return False
    finally:
        # Clean up the temporary directory
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def test_tracker_manager_multiple_custom_metrics():
    """Test that TrackerManager properly handles multiple custom metrics."""
    
    print("\n=== Testing TrackerManager Multiple Custom Metrics ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create tracker manager
        manager = TrackerManager(workspace=temp_dir)
        
        # Create mock trackers
        class MockTracker:
            def __init__(self, name):
                self.name = name
                self.tracked_metrics = []
            
            def track(self, metric, value, step=None, *args, **kwargs):
                self.tracked_metrics.append((metric, value, step))
        
        tracker1 = MockTracker("tracker1")
        tracker2 = MockTracker("tracker2")
        
        manager.add_tracker(tracker1)
        manager.add_tracker(tracker2)
        
        # Test multiple custom metrics
        print("\n--- Test: Multiple Custom Metrics ---")
        custom_metrics = [
            ("training_speed", 150.5),
            ("memory_usage", 85.2),
            ("gpu_utilization", 67.8)
        ]
        
        # Track multiple custom metrics
        manager.track(Metric.CUSTOM, custom_metrics, step=0)
        
        # Check results
        print(f"Tracker 1 tracked {len(tracker1.tracked_metrics)} metrics")
        print(f"Tracker 2 tracked {len(tracker2.tracked_metrics)} metrics")
        
        # Both trackers should have received the same metrics
        if len(tracker1.tracked_metrics) == len(tracker2.tracked_metrics) == 1:
            print("✅ SUCCESS: Both trackers received the multiple custom metrics!")
            return True
        else:
            print("❌ FAILURE: Trackers did not receive the expected metrics")
            return False


def main():
    """Run all tests."""
    
    print("🧪 Testing Multiple CUSTOM_METRICS Fix")
    print("=" * 50)
    
    # Run tests
    test1_success = test_metrics_tracker_multiple_custom_metrics()
    test2_success = test_db_tracker_multiple_custom_metrics()
    test3_success = test_tracker_manager_multiple_custom_metrics()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    print(f"MetricsTracker Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"DBTracker Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    print(f"TrackerManager Test: {'✅ PASSED' if test3_success else '❌ FAILED'}")
    
    all_passed = test1_success and test2_success and test3_success
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Multiple CUSTOM_METRICS fix is working correctly!")
    else:
        print("\n❌ SOME TESTS FAILED! Multiple CUSTOM_METRICS fix needs attention.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
