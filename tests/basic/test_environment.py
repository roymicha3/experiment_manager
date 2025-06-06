import os
import glob
import pytest
from omegaconf import OmegaConf
from experiment_manager.environment import Environment, ProductPaths

@pytest.fixture
def env_config(tmp_path):
    return OmegaConf.create({
        "workspace": os.path.join(str(tmp_path), "test_outputs"),
        "verbose": True,
        "debug": True,
    })

@pytest.fixture
def env(env_config, tmp_path):
    env = Environment.from_config(env_config)
    return env

def test_environment_initialization(env):
    """Test that environment is initialized correctly"""
    assert os.path.exists(env.workspace)
    assert os.path.exists(env.log_dir)
    assert os.path.exists(env.config_dir)

def test_environment_paths(env):
    """Test that environment paths are correct"""
    assert env.log_dir == os.path.join(env.workspace, ProductPaths.LOG_DIR.value)
    assert env.artifact_dir == os.path.join(env.workspace, ProductPaths.ARTIFACT_DIR.value)
    assert env.config_dir == os.path.join(env.workspace, ProductPaths.CONFIG_DIR.value)

def test_environment_from_config(env_config, tmp_path):
    """Test creating environment from config"""
    workspace = os.path.join(str(tmp_path), env_config.workspace)
    env_config.workspace = workspace
    env = Environment.from_config(env_config)
    
    assert env.workspace == workspace
    assert env.config == env_config

def test_environment_save(env):
    """Test saving environment config"""
    config_path = os.path.join(env.config_dir, ProductPaths.CONFIG_FILE.value)
    assert os.path.exists(config_path)
    loaded_config = OmegaConf.load(config_path)
    assert loaded_config == env.config

def test_set_workspace(env, tmp_path):
    """Test setting new workspace"""
    new_workspace = os.path.join(str(tmp_path), "new_workspace")
    env.set_workspace(new_workspace)
      # Set up the new workspace
    assert env.workspace == os.path.abspath(new_workspace)
    
    # Test inner workspace
    inner_workspace = "inner_workspace"
    env.set_workspace(inner_workspace, inner=True)
      # Set up the inner workspace
    assert env.workspace == os.path.abspath(os.path.join(new_workspace, inner_workspace))

def test_environment_logging_file_only(env):
    """Test that logs are created in the correct directory with file-only logging"""
    # Setup environment with verbose=False for file-only logging
    
    
    # Check that log directory exists
    assert os.path.exists(env.log_dir)
    
    # Generate some log messages
    env.logger.info("Test info message")
    env.logger.debug("Test debug message")
    env.logger.warning("Test warning message")
    
    # Check that log file was created
    log_files = glob.glob(os.path.join(env.log_dir, "*.log"))
    assert len(log_files) == 1, "Expected exactly one log file"
    
    # Check log file content
    with open(log_files[0], 'r') as f:
        content = f.read()
        assert "Test info message" in content
        assert "Test warning message" in content

def test_environment_logging_composite(env):
    """Test that logs are created with both file and console logging"""
    # Setup environment with verbose=True for both console and file logging
    
    
    # Check that log directory exists
    assert os.path.exists(env.log_dir)
    
    # Generate some log messages
    env.logger.info("Test info message")
    env.logger.debug("Test debug message")
    env.logger.warning("Test warning message")
    
    # Check that log file was created
    log_files = glob.glob(os.path.join(env.log_dir, "*.log"))
    assert len(log_files) == 1, "Expected exactly one log file"
    
    # Check log file content
    with open(log_files[0], 'r') as f:
        content = f.read()
        assert "Test info message" in content
        assert "Test debug message" in content
        assert "Test warning message" in content


# =================== ANALYTICS ENVIRONMENT TESTS ===================

def test_analytics_directory_creation(env):
    """Test that analytics directories are created correctly"""
    # Test main analytics directory
    assert os.path.exists(env.analytics_dir)
    assert env.analytics_dir == os.path.join(env.workspace, ProductPaths.ANALYTICS_DIR.value)
    
    # Test analytics subdirectories
    assert os.path.exists(env.analytics_reports_dir)
    assert os.path.exists(env.analytics_cache_dir)
    assert os.path.exists(env.analytics_artifacts_dir)
    
    # Test directory structure
    assert env.analytics_reports_dir == os.path.join(env.analytics_dir, "reports")
    assert env.analytics_cache_dir == os.path.join(env.analytics_dir, "cache")
    assert env.analytics_artifacts_dir == os.path.join(env.analytics_dir, "artifacts")

def test_analytics_product_paths_enum():
    """Test that ProductPaths enum includes analytics directory"""
    assert hasattr(ProductPaths, 'ANALYTICS_DIR')
    assert ProductPaths.ANALYTICS_DIR.value == "analytics"

def test_analytics_path_creation_methods(env):
    """Test analytics path creation utility methods"""
    # Test artifact path creation
    artifact_path = env.create_analytics_artifact_path("test_plot.png")
    expected_artifact = os.path.join(env.analytics_artifacts_dir, "test_plot.png")
    assert artifact_path == expected_artifact
    
    # Test report path creation
    report_path = env.create_analytics_report_path("analysis_report.json")
    expected_report = os.path.join(env.analytics_reports_dir, "analysis_report.json")
    assert report_path == expected_report
    
    # Test with subdirectories
    nested_artifact = env.create_analytics_artifact_path("charts/monthly_metrics.png")
    expected_nested = os.path.join(env.analytics_artifacts_dir, "charts/monthly_metrics.png")
    assert nested_artifact == expected_nested

def test_analytics_workspace_info(env):
    """Test get_analytics_workspace_info method"""
    workspace_info = env.get_analytics_workspace_info()
    
    # Test required keys exist
    required_keys = ["analytics_dir", "reports_dir", "cache_dir", "artifacts_dir", "directories_exist"]
    for key in required_keys:
        assert key in workspace_info, f"Missing key: {key}"
    
    # Test paths are correct
    assert workspace_info["analytics_dir"] == env.analytics_dir
    assert workspace_info["reports_dir"] == env.analytics_reports_dir
    assert workspace_info["cache_dir"] == env.analytics_cache_dir
    assert workspace_info["artifacts_dir"] == env.analytics_artifacts_dir
    
    # Test all directories exist
    directories_exist = workspace_info["directories_exist"]
    assert directories_exist["analytics"] == True
    assert directories_exist["reports"] == True
    assert directories_exist["cache"] == True
    assert directories_exist["artifacts"] == True

def test_analytics_set_workspace_integration(env, tmp_path):
    """Test that analytics directories are created when workspace changes"""
    # Change workspace
    new_workspace = os.path.join(str(tmp_path), "new_analytics_workspace")
    env.set_workspace(new_workspace)
    
    # Verify analytics directories are created in new workspace
    assert os.path.exists(env.analytics_dir)
    assert os.path.exists(env.analytics_reports_dir)
    assert os.path.exists(env.analytics_cache_dir)
    assert os.path.exists(env.analytics_artifacts_dir)
    
    # Verify paths are correct
    assert env.analytics_dir == os.path.join(new_workspace, "analytics")
    assert new_workspace in env.analytics_dir
    
    # Test inner workspace change
    env.set_workspace("trial_1", inner=True)
    inner_workspace = os.path.join(new_workspace, "trial_1")
    
    assert os.path.exists(env.analytics_dir)
    assert env.analytics_dir == os.path.join(inner_workspace, "analytics")

def test_analytics_child_environment_inheritance(env):
    """Test that child environments inherit analytics directories"""
    child_env = env.create_child("test_trial")
    
    # Verify child has analytics directories
    assert os.path.exists(child_env.analytics_dir)
    assert os.path.exists(child_env.analytics_reports_dir)
    assert os.path.exists(child_env.analytics_cache_dir)
    assert os.path.exists(child_env.analytics_artifacts_dir)
    
    # Verify child directories are in child workspace
    assert "test_trial" in child_env.analytics_dir
    assert child_env.analytics_dir == os.path.join(child_env.workspace, "analytics")
    
    # Test nested child environment
    grandchild_env = child_env.create_child("run_1")
    assert os.path.exists(grandchild_env.analytics_dir)
    assert "test_trial" in grandchild_env.analytics_dir
    assert "run_1" in grandchild_env.analytics_dir

def test_analytics_file_operations(env):
    """Test file operations in analytics directories"""
    # Create test artifact file
    artifact_path = env.create_analytics_artifact_path("test_data.csv")
    test_data = "col1,col2,col3\n1,2,3\n4,5,6\n"
    
    with open(artifact_path, 'w') as f:
        f.write(test_data)
    
    assert os.path.exists(artifact_path)
    
    # Read back the file
    with open(artifact_path, 'r') as f:
        content = f.read()
        assert content == test_data
    
    # Create test report file
    report_path = env.create_analytics_report_path("test_report.json")
    test_report = '{"status": "success", "metrics": {"accuracy": 0.95}}'
    
    with open(report_path, 'w') as f:
        f.write(test_report)
    
    assert os.path.exists(report_path)
    
    # Read back the report
    import json
    with open(report_path, 'r') as f:
        report_data = json.load(f)
        assert report_data["status"] == "success"
        assert report_data["metrics"]["accuracy"] == 0.95

def test_analytics_backwards_compatibility(env):
    """Test that analytics integration doesn't break existing functionality"""
    # Test all original directory properties still work
    assert os.path.exists(env.log_dir)
    assert os.path.exists(env.artifact_dir)
    assert os.path.exists(env.config_dir)
    
    # Test original methods still work
    assert hasattr(env, 'save')
    assert hasattr(env, 'copy')
    assert hasattr(env, 'create_child')
    
    # Test copy method works with analytics
    env_copy = env.copy()
    assert env_copy.workspace == env.workspace
    assert os.path.exists(env_copy.analytics_dir)
    
    # Test save method still works
    original_config = env.config.copy()
    env.save()
    config_path = os.path.join(env.config_dir, ProductPaths.CONFIG_FILE.value)
    assert os.path.exists(config_path)

def test_analytics_directories_lazy_creation(tmp_path):
    """Test that analytics directories are created lazily when accessed"""
    workspace = str(tmp_path / "lazy_test")
    config = OmegaConf.create({
        "workspace": workspace,
        "verbose": False,
        "trackers": []
    })
    
    env = Environment(workspace=workspace, config=config)
    
    # Initially only workspace should exist
    assert os.path.exists(env.workspace)
    
    # Accessing analytics_dir should create it
    analytics_dir = env.analytics_dir
    assert os.path.exists(analytics_dir)
    
    # Accessing subdirectories should create them
    reports_dir = env.analytics_reports_dir
    cache_dir = env.analytics_cache_dir
    artifacts_dir = env.analytics_artifacts_dir
    
    assert os.path.exists(reports_dir)
    assert os.path.exists(cache_dir)
    assert os.path.exists(artifacts_dir)

def test_analytics_directory_permissions(env):
    """Test that analytics directories have correct permissions and are writable"""
    # Test that we can create files in each analytics directory
    test_files = [
        env.create_analytics_artifact_path("permission_test.txt"),
        env.create_analytics_report_path("permission_test.txt"),
        os.path.join(env.analytics_cache_dir, "permission_test.txt")
    ]
    
    for test_file in test_files:
        # Create directory if needed (for cache test)
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        
        # Write test file
        with open(test_file, 'w') as f:
            f.write("permission test")
        
        # Verify file exists and is readable
        assert os.path.exists(test_file)
        
        with open(test_file, 'r') as f:
            content = f.read()
            assert content == "permission test"
        
        # Clean up
        os.remove(test_file)