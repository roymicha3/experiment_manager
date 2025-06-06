"""
Test module for analytics environment integration.

This module contains comprehensive tests for the analytics workspace functionality
added to the Environment class, including directory management, path creation,
workspace inheritance, and integration with the existing environment system.
"""

import os
import json
import pytest
import tempfile
import shutil
from omegaconf import OmegaConf
from experiment_manager.environment import Environment, ProductPaths


@pytest.fixture
def analytics_env_config(tmp_path):
    """Create a test configuration for analytics environment testing."""
    return OmegaConf.create({
        "workspace": os.path.join(str(tmp_path), "analytics_test_workspace"),
        "verbose": False,
        "debug": True,
        "trackers": []
    })


@pytest.fixture
def analytics_env(analytics_env_config):
    """Create an Environment instance for analytics testing."""
    return Environment.from_config(analytics_env_config)


class TestAnalyticsDirectoryStructure:
    """Test analytics directory structure and creation."""
    
    def test_analytics_enum_integration(self):
        """Test that ProductPaths enum properly includes analytics directory."""
        assert hasattr(ProductPaths, 'ANALYTICS_DIR')
        assert ProductPaths.ANALYTICS_DIR.value == "analytics"
        
        # Ensure all expected paths are present
        expected_paths = ['CONFIG_FILE', 'LOG_DIR', 'ARTIFACT_DIR', 'CONFIG_DIR', 'ANALYTICS_DIR']
        for path_name in expected_paths:
            assert hasattr(ProductPaths, path_name)
    
    def test_analytics_directory_creation_on_init(self, analytics_env):
        """Test that analytics directories are created during environment initialization."""
        # Main analytics directory
        assert os.path.exists(analytics_env.analytics_dir)
        assert analytics_env.analytics_dir.endswith("analytics")
        
        # Analytics subdirectories  
        assert os.path.exists(analytics_env.analytics_reports_dir)
        assert os.path.exists(analytics_env.analytics_cache_dir)
        assert os.path.exists(analytics_env.analytics_artifacts_dir)
        
        # Verify directory hierarchy
        assert analytics_env.analytics_reports_dir.startswith(analytics_env.analytics_dir)
        assert analytics_env.analytics_cache_dir.startswith(analytics_env.analytics_dir)
        assert analytics_env.analytics_artifacts_dir.startswith(analytics_env.analytics_dir)
    
    def test_analytics_directory_lazy_creation_pattern(self, tmp_path):
        """Test that analytics directories follow lazy creation pattern."""
        workspace = str(tmp_path / "lazy_analytics_test")
        config = OmegaConf.create({
            "workspace": workspace,
            "verbose": False,
            "trackers": []
        })
        
        env = Environment(workspace=workspace, config=config)
        
        # Directories should be created when accessed, not before
        analytics_path = os.path.join(workspace, "analytics")
        
        # Accessing the property should create the directory
        _ = env.analytics_dir
        assert os.path.exists(analytics_path)
        
        # Subdirectories should be created when accessed
        reports_path = os.path.join(analytics_path, "reports")
        _ = env.analytics_reports_dir
        assert os.path.exists(reports_path)


class TestAnalyticsUtilityMethods:
    """Test analytics utility methods and path creation."""
    
    def test_create_analytics_artifact_path(self, analytics_env):
        """Test analytics artifact path creation."""
        # Simple filename
        artifact_path = analytics_env.create_analytics_artifact_path("model_performance.png")
        expected = os.path.join(analytics_env.analytics_artifacts_dir, "model_performance.png")
        assert artifact_path == expected
        
        # Nested path
        nested_path = analytics_env.create_analytics_artifact_path("charts/accuracy_over_time.png")
        expected_nested = os.path.join(analytics_env.analytics_artifacts_dir, "charts/accuracy_over_time.png")
        assert nested_path == expected_nested
        
        # Different file types
        csv_path = analytics_env.create_analytics_artifact_path("results.csv")
        json_path = analytics_env.create_analytics_artifact_path("metadata.json")
        
        assert csv_path.endswith(".csv")
        assert json_path.endswith(".json")
    
    def test_create_analytics_report_path(self, analytics_env):
        """Test analytics report path creation."""
        # Simple report
        report_path = analytics_env.create_analytics_report_path("analysis_summary.json")
        expected = os.path.join(analytics_env.analytics_reports_dir, "analysis_summary.json")
        assert report_path == expected
        
        # Structured report path
        structured_path = analytics_env.create_analytics_report_path("monthly/2024_01_report.html")
        expected_structured = os.path.join(analytics_env.analytics_reports_dir, "monthly/2024_01_report.html")
        assert structured_path == expected_structured
    
    def test_get_analytics_workspace_info(self, analytics_env):
        """Test analytics workspace information retrieval."""
        workspace_info = analytics_env.get_analytics_workspace_info()
        
        # Test structure
        assert isinstance(workspace_info, dict)
        
        # Test required keys
        required_keys = ["analytics_dir", "reports_dir", "cache_dir", "artifacts_dir", "directories_exist"]
        for key in required_keys:
            assert key in workspace_info
        
        # Test path values
        assert workspace_info["analytics_dir"] == analytics_env.analytics_dir
        assert workspace_info["reports_dir"] == analytics_env.analytics_reports_dir
        assert workspace_info["cache_dir"] == analytics_env.analytics_cache_dir
        assert workspace_info["artifacts_dir"] == analytics_env.analytics_artifacts_dir
        
        # Test existence flags
        existence_info = workspace_info["directories_exist"]
        assert existence_info["analytics"] == True
        assert existence_info["reports"] == True
        assert existence_info["cache"] == True
        assert existence_info["artifacts"] == True


class TestAnalyticsWorkspaceOperations:
    """Test workspace operations with analytics integration."""
    
    def test_set_workspace_creates_analytics_dirs(self, analytics_env, tmp_path):
        """Test that set_workspace creates analytics directories in new location."""
        # Change to new workspace
        new_workspace = str(tmp_path / "new_analytics_workspace")
        analytics_env.set_workspace(new_workspace)
        
        # Verify analytics directories are created
        assert os.path.exists(analytics_env.analytics_dir)
        assert os.path.exists(analytics_env.analytics_reports_dir)
        assert os.path.exists(analytics_env.analytics_cache_dir)
        assert os.path.exists(analytics_env.analytics_artifacts_dir)
        
        # Verify correct paths
        assert analytics_env.analytics_dir == os.path.join(new_workspace, "analytics")
        assert new_workspace in analytics_env.analytics_dir
    
    def test_inner_workspace_analytics_support(self, analytics_env, tmp_path):
        """Test inner workspace creation with analytics support."""
        # Set initial workspace
        base_workspace = str(tmp_path / "base_workspace")
        analytics_env.set_workspace(base_workspace)
        
        # Set inner workspace
        analytics_env.set_workspace("experiment_1", inner=True)
        
        # Verify analytics directories in inner workspace
        expected_workspace = os.path.join(base_workspace, "experiment_1")
        assert analytics_env.workspace == expected_workspace
        
        assert os.path.exists(analytics_env.analytics_dir)
        assert analytics_env.analytics_dir == os.path.join(expected_workspace, "analytics")
    
    def test_child_environment_analytics_inheritance(self, analytics_env):
        """Test that child environments properly inherit analytics functionality."""
        # Create child environment
        child_env = analytics_env.create_child("trial_1")
        
        # Verify child has analytics directories
        assert os.path.exists(child_env.analytics_dir)
        assert os.path.exists(child_env.analytics_reports_dir)
        assert os.path.exists(child_env.analytics_cache_dir)
        assert os.path.exists(child_env.analytics_artifacts_dir)
        
        # Verify child paths are independent
        assert "trial_1" in child_env.analytics_dir
        assert child_env.analytics_dir != analytics_env.analytics_dir
        
        # Test multi-level inheritance
        grandchild_env = child_env.create_child("run_1")
        assert os.path.exists(grandchild_env.analytics_dir)
        assert "trial_1" in grandchild_env.analytics_dir
        assert "run_1" in grandchild_env.analytics_dir
    
    def test_environment_copy_with_analytics(self, analytics_env):
        """Test that environment copy includes analytics functionality."""
        env_copy = analytics_env.copy()
        
        # Verify copy has same workspace
        assert env_copy.workspace == analytics_env.workspace
        
        # Verify analytics directories exist in copy
        assert os.path.exists(env_copy.analytics_dir)
        assert env_copy.analytics_dir == analytics_env.analytics_dir
        
        # Verify utility methods work in copy
        artifact_path = env_copy.create_analytics_artifact_path("test.png")
        assert artifact_path.startswith(env_copy.analytics_artifacts_dir)


class TestAnalyticsFileOperations:
    """Test file operations in analytics directories."""
    
    def test_analytics_artifact_file_creation(self, analytics_env):
        """Test creating and managing files in analytics artifacts directory."""
        # Create test artifact
        artifact_path = analytics_env.create_analytics_artifact_path("experiment_results.csv")
        
        test_data = "experiment_id,accuracy,loss\n1,0.95,0.05\n2,0.92,0.08\n"
        
        with open(artifact_path, 'w') as f:
            f.write(test_data)
        
        # Verify file exists and is readable
        assert os.path.exists(artifact_path)
        
        with open(artifact_path, 'r') as f:
            content = f.read()
            assert content == test_data
            assert "experiment_id,accuracy,loss" in content
    
    def test_analytics_report_file_creation(self, analytics_env):
        """Test creating and managing report files."""
        # Create test report
        report_path = analytics_env.create_analytics_report_path("analysis_report.json")
        
        report_data = {
            "experiment_name": "test_experiment",
            "total_trials": 5,
            "best_accuracy": 0.95,
            "metrics": {
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95
            },
            "status": "completed"
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Verify file exists and is readable
        assert os.path.exists(report_path)
        
        with open(report_path, 'r') as f:
            loaded_data = json.load(f)
            assert loaded_data["experiment_name"] == "test_experiment"
            assert loaded_data["best_accuracy"] == 0.95
            assert "metrics" in loaded_data
    
    def test_analytics_cache_operations(self, analytics_env):
        """Test cache directory operations."""
        # Create cache file
        cache_file = os.path.join(analytics_env.analytics_cache_dir, "processed_data.cache")
        
        cache_data = "cached_results_12345"
        
        with open(cache_file, 'w') as f:
            f.write(cache_data)
        
        # Verify cache file operations
        assert os.path.exists(cache_file)
        
        with open(cache_file, 'r') as f:
            content = f.read()
            assert content == cache_data
    
    def test_nested_directory_creation(self, analytics_env):
        """Test creating files in nested directories within analytics workspace."""
        # Create nested artifact path
        nested_artifact = analytics_env.create_analytics_artifact_path("models/checkpoints/best_model.pth")
        
        # Create parent directories
        os.makedirs(os.path.dirname(nested_artifact), exist_ok=True)
        
        # Create test file
        with open(nested_artifact, 'w') as f:
            f.write("mock_model_data")
        
        assert os.path.exists(nested_artifact)
        assert "models/checkpoints" in nested_artifact


class TestAnalyticsBackwardsCompatibility:
    """Test that analytics integration maintains backwards compatibility."""
    
    def test_original_environment_functionality_preserved(self, analytics_env):
        """Test that all original environment functionality still works."""
        # Test original directory properties
        assert os.path.exists(analytics_env.log_dir)
        assert os.path.exists(analytics_env.artifact_dir)
        assert os.path.exists(analytics_env.config_dir)
        
        # Test original methods exist
        original_methods = ['save', 'copy', 'create_child', 'set_workspace']
        for method in original_methods:
            assert hasattr(analytics_env, method)
            assert callable(getattr(analytics_env, method))
    
    def test_original_directory_paths_unchanged(self, analytics_env):
        """Test that original directory paths follow the same patterns."""
        # Test that original paths use ProductPaths enum
        assert analytics_env.log_dir == os.path.join(analytics_env.workspace, ProductPaths.LOG_DIR.value)
        assert analytics_env.artifact_dir == os.path.join(analytics_env.workspace, ProductPaths.ARTIFACT_DIR.value)
        assert analytics_env.config_dir == os.path.join(analytics_env.workspace, ProductPaths.CONFIG_DIR.value)
        
        # Test that analytics follows the same pattern
        assert analytics_env.analytics_dir == os.path.join(analytics_env.workspace, ProductPaths.ANALYTICS_DIR.value)
    
    def test_environment_save_load_compatibility(self, analytics_env):
        """Test that environment save/load works with analytics integration."""
        # Save environment
        analytics_env.save()
        
        # Verify config file exists
        config_path = os.path.join(analytics_env.config_dir, ProductPaths.CONFIG_FILE.value)
        assert os.path.exists(config_path)
        
        # Load and verify config
        loaded_config = OmegaConf.load(config_path)
        assert loaded_config == analytics_env.config
    
    def test_existing_tests_still_pass(self, tmp_path):
        """Test that existing environment patterns still work with analytics integration."""
        # Create environment using original pattern
        config = OmegaConf.create({
            "workspace": str(tmp_path / "compatibility_test"),
            "verbose": True,
            "debug": False,
            "trackers": []
        })
        
        env = Environment.from_config(config)
        
        # Test that original functionality works
        assert os.path.exists(env.workspace)
        assert os.path.exists(env.log_dir)
        assert os.path.exists(env.artifact_dir)
        assert os.path.exists(env.config_dir)
        
        # Test that analytics is added without breaking anything
        assert os.path.exists(env.analytics_dir)
        
        # Test child creation (original pattern)
        child = env.create_child("test_child")
        assert os.path.exists(child.workspace)
        assert os.path.exists(child.analytics_dir)


class TestAnalyticsEdgeCases:
    """Test edge cases and error conditions for analytics integration."""
    
    def test_empty_filename_handling(self, analytics_env):
        """Test handling of edge cases in file path creation."""
        # Test empty filename
        empty_path = analytics_env.create_analytics_artifact_path("")
        # Normalize paths for comparison (handle trailing separators)
        normalized_empty = os.path.normpath(empty_path)
        normalized_artifacts = os.path.normpath(analytics_env.analytics_artifacts_dir)
        assert normalized_empty == normalized_artifacts
        
        # Test path with spaces
        spaced_path = analytics_env.create_analytics_artifact_path("file with spaces.txt")
        assert "file with spaces.txt" in spaced_path
    
    def test_special_characters_in_paths(self, analytics_env):
        """Test handling of special characters in analytics paths."""
        special_chars = ["file-with-dashes.txt", "file_with_underscores.txt", "file.with.dots.txt"]
        
        for filename in special_chars:
            path = analytics_env.create_analytics_artifact_path(filename)
            assert filename in path
            assert path.startswith(analytics_env.analytics_artifacts_dir)
    
    def test_very_long_paths(self, analytics_env):
        """Test handling of very long file paths."""
        long_filename = "a" * 100 + ".txt"
        long_path = analytics_env.create_analytics_artifact_path(long_filename)
        
        assert long_filename in long_path
        assert len(long_path) > 100
    
    def test_workspace_info_with_missing_directories(self, tmp_path):
        """Test workspace info when some directories don't exist yet."""
        workspace = str(tmp_path / "incomplete_workspace")
        config = OmegaConf.create({
            "workspace": workspace,
            "verbose": False,
            "trackers": []
        })
        
        env = Environment(workspace=workspace, config=config)
        
        # Get workspace info (should create directories)
        workspace_info = env.get_analytics_workspace_info()
        
        # All should be created now
        assert all(workspace_info["directories_exist"].values()) 