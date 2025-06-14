"""Tests for the DataSource factory system integration with DBDataSource."""
import pytest
import tempfile
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from experiment_manager.results.sources.datasource_factory import DataSourceFactory
from experiment_manager.results.sources.db_datasource import DBDataSource
from experiment_manager.results.sources.datasource import ExperimentDataSource
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.db.manager import DatabaseManager


class TestDataSourceFactoryIntegration:
    """Test cases for DataSource factory integration with DBDataSource."""
    
    def test_factory_can_create_db_datasource(self, tmp_path):
        """Test that DataSourceFactory can properly instantiate DBDataSource."""
        # Create a temporary database
        db_path = tmp_path / "test_factory.db"
        
        # Create config for DBDataSource (needs write access to create DB)
        config = DictConfig({
            'db_path': str(db_path),
            'use_sqlite': True,
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'readonly': False
        })
        
        # Test factory creation
        datasource = DataSourceFactory.create("DBDataSource", config)
        
        # Verify the instance
        assert isinstance(datasource, DBDataSource)
        assert isinstance(datasource, ExperimentDataSource)
        assert datasource.db_path == str(db_path)
        assert datasource.db_manager.use_sqlite is True
        
        # Clean up
        datasource.close()
    
    def test_db_datasource_registration(self):
        """Test that DBDataSource is properly registered with YAMLSerializable."""
        # Check that DBDataSource is registered
        assert "DBDataSource" in YAMLSerializable._registry
        
        # Check that we can get the class by name
        db_class = YAMLSerializable.get_by_name("DBDataSource")
        assert db_class is DBDataSource
        
        # Verify it's the correct class
        assert issubclass(db_class, ExperimentDataSource)
        assert issubclass(db_class, YAMLSerializable)
    
    def test_factory_with_various_configurations(self, tmp_path):
        """Test factory creation with different configuration scenarios."""
        db_path = tmp_path / "test_configs.db"
        
        # Test 1: Minimal configuration (SQLite defaults, needs write access)
        minimal_config = DictConfig({
            'db_path': str(db_path),
            'use_sqlite': True,
            'readonly': False
        })
        
        datasource1 = DataSourceFactory.create("DBDataSource", minimal_config)
        assert isinstance(datasource1, DBDataSource)
        assert datasource1.db_manager.use_sqlite is True
        datasource1.close()
        
        # Test 2: Full configuration with all parameters (needs write access)
        full_config = DictConfig({
            'db_path': str(db_path),
            'use_sqlite': True,
            'host': 'test_host',
            'user': 'test_user',
            'password': 'test_pass',
            'readonly': False
        })
        
        datasource2 = DataSourceFactory.create("DBDataSource", full_config)
        assert isinstance(datasource2, DBDataSource)
        assert datasource2.db_path == str(db_path)
        datasource2.close()
        
        # Test 3: Configuration validation (MySQL setup without actual connection)
        mysql_config = DictConfig({
            'db_path': 'test_database',
            'use_sqlite': False,
            'host': 'mysql_host',
            'user': 'mysql_user',
            'password': 'mysql_pass'
        })
        
        # Verify configuration is parsed correctly (but don't try to connect)
        # This will fail during DatabaseManager initialization, which is expected
        try:
            datasource3 = DataSourceFactory.create("DBDataSource", mysql_config)
            datasource3.close()
            # If it doesn't fail, verify it was created correctly
            assert isinstance(datasource3, DBDataSource)
            assert datasource3.db_manager.use_sqlite is False
        except Exception as e:
            # Expected when MySQL server doesn't exist - this is OK for factory testing
            # Should be experiment_manager.db.manager.ConnectionError but we'll catch all
            assert "Unknown MySQL server host" in str(e) or "ConnectionError" in str(type(e))
            pass
    
    def test_yaml_configuration_driven_creation(self, tmp_path):
        """Test factory creation using YAML configuration files."""
        db_path = tmp_path / "test_yaml.db"
        yaml_file = tmp_path / "datasource_config.yaml"
        
        # Create YAML configuration
        yaml_content = f"""
        db_path: {str(db_path)}
        use_sqlite: true
        host: localhost
        user: root
        password: ""
        readonly: False  # Needs write access for test DB creation
        """
        
        # Write YAML file
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        # Load configuration from YAML
        config = OmegaConf.load(yaml_file)
        
        # Create datasource using factory
        datasource = DataSourceFactory.create("DBDataSource", config)
        
        # Verify creation
        assert isinstance(datasource, DBDataSource)
        assert datasource.db_path == str(db_path)
        assert datasource.db_manager.use_sqlite is True
        
        # Clean up
        datasource.close()
    
    def test_factory_error_handling_invalid_name(self):
        """Test factory error handling for invalid datasource names."""
        config = DictConfig({
            'db_path': '/tmp/test.db',
            'use_sqlite': True
        })
        
        # Test with non-existent datasource name
        with pytest.raises(ValueError, match="'InvalidDataSource' is not registered"):
            DataSourceFactory.create("InvalidDataSource", config)
    
    def test_factory_error_handling_invalid_config(self, tmp_path):
        """Test factory error handling for invalid configurations."""
        # Test with missing required parameters
        invalid_config = DictConfig({
            'use_sqlite': True
            # Missing db_path
        })
        
        # This should raise an AttributeError due to new DBDataSource validation
        with pytest.raises(AttributeError, match="db_path.*required configuration field"):
            DataSourceFactory.create("DBDataSource", invalid_config)
    
    def test_factory_lookup_mechanisms(self):
        """Test factory registration and lookup mechanisms."""
        # Test that the registry contains our expected datasources
        registry = YAMLSerializable._registry
        
        # DBDataSource should be registered
        assert "DBDataSource" in registry
        assert registry["DBDataSource"] is DBDataSource
        
        # FileSystemDataSource should also be registered (imported in factory)
        assert "FileSystemDataSource" in registry
        
        # Test multiple lookups work correctly
        for _ in range(3):
            db_class = YAMLSerializable.get_by_name("DBDataSource")
            assert db_class is DBDataSource
    
    def test_factory_with_real_database(self, experiment_db_only):
        """Test factory creation with real MNIST experiment database."""
        # experiment_db_only provides path to real MNIST database
        real_db_path = experiment_db_only
        
        # Create configuration for real database (read-only is sufficient)
        config = DictConfig({
            'db_path': real_db_path,
            'use_sqlite': True,
            'host': 'localhost',
            'user': 'root', 
            'password': '',
            'readonly': True
        })
        
        # Create datasource using factory
        datasource = DataSourceFactory.create("DBDataSource", config)
        
        # Verify it works with real data
        assert isinstance(datasource, DBDataSource)
        
        # Test that it can access real experiment data
        experiment = datasource.get_experiment()
        assert experiment is not None
        assert experiment.id is not None
        assert experiment.name is not None
        assert len(experiment.trials) > 0
        
        # Verify trials have data
        for trial in experiment.trials:
            assert trial.id is not None
            assert len(trial.runs) > 0
        
        print(f"✅ Factory successfully created DBDataSource with real MNIST data")
        print(f"   - Experiment: '{experiment.name}' (ID: {experiment.id})")
        print(f"   - Trials: {len(experiment.trials)}")
        print(f"   - Total runs: {sum(len(trial.runs) for trial in experiment.trials)}")
        
        # Clean up
        datasource.close()
    
    def test_factory_interface_compliance(self, tmp_path):
        """Test that factory-created DBDataSource properly implements ExperimentDataSource interface."""
        db_path = tmp_path / "test_interface.db"
        
        # Create a test database with sample data
        manager = DatabaseManager(database_path=str(db_path), use_sqlite=True, recreate=True)
        
        # Create minimal test data
        experiment = manager.create_experiment("Test Experiment", "Test description")
        trial = manager.create_trial(experiment.id, "Test Trial")
        trial_run = manager.create_trial_run(trial.id)
        
        # Create the DataSource using the factory
        config = DictConfig({
            'db_path': str(db_path),
            'use_sqlite': True
        })
        
        datasource = DataSourceFactory.create("DBDataSource", config)
        
        # Test all required interface methods exist and are callable
        assert hasattr(datasource, 'get_experiment')
        assert hasattr(datasource, 'get_trials') 
        assert hasattr(datasource, 'get_trial_runs')
        assert hasattr(datasource, 'get_metrics')
        assert hasattr(datasource, 'get_artifacts')
        assert hasattr(datasource, 'metrics_dataframe')
        
        # Test methods work (basic functionality)
        experiment = datasource.get_experiment()
        assert experiment is not None
        
        trials = datasource.get_trials(experiment)
        assert isinstance(trials, list)
        
        if trials:
            trial_runs = datasource.get_trial_runs(trials[0])
            assert isinstance(trial_runs, list)
        
        df = datasource.metrics_dataframe(experiment)
        assert hasattr(df, 'columns')  # Should be a DataFrame
        
        # Test context manager interface
        assert hasattr(datasource, '__enter__')
        assert hasattr(datasource, '__exit__')
        
        with datasource as ds:
            assert ds is datasource
        
        # Clean up
        datasource.close()
        
        print(f"✅ Factory-created DBDataSource properly implements ExperimentDataSource interface") 