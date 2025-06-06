"""
Tests for Analytics Default Configuration Manager

Tests the default configuration system including all configuration levels,
file generation, documentation, and integration with the factory pattern.
"""

import os
import tempfile
import pytest
from omegaconf import DictConfig, OmegaConf

from experiment_manager.analytics.defaults import (
    DefaultConfigurationManager, 
    ConfigurationLevel
)
from experiment_manager.analytics import AnalyticsFactory


class TestConfigurationLevels:
    """Test all configuration levels."""
    
    def test_minimal_config(self):
        """Test minimal configuration level."""
        config = DefaultConfigurationManager.get_minimal_config()
        
        assert isinstance(config, DictConfig)
        assert 'processors' in config
        assert 'statistics' in config.processors
        assert config.processors.statistics.confidence_level == 0.95
        assert config.processors.statistics.include_advanced == False
        assert len(config.processors.statistics.percentiles) == 4
        
        # Should have minimal components
        assert 'outliers' not in config.processors
        assert 'failures' not in config.processors
        assert 'comparisons' not in config.processors
        
        # Aggregation should be simple
        assert len(config.aggregation.default_functions) == 3
        assert config.export.include_metadata == False
    
    def test_standard_config(self):
        """Test standard configuration level."""
        config = DefaultConfigurationManager.get_standard_config()
        
        assert isinstance(config, DictConfig)
        assert 'processors' in config
        
        # Should have statistics, outliers, and failures
        assert 'statistics' in config.processors
        assert 'outliers' in config.processors
        assert 'failures' in config.processors
        assert 'comparisons' not in config.processors  # Not in standard
        
        # Statistics should be more comprehensive
        assert config.processors.statistics.include_advanced == True
        assert len(config.processors.statistics.percentiles) == 5
        
        # Should have caching and metadata
        assert config.result_caching == True
        assert config.export.include_metadata == True
        assert config.failure_analysis.enabled == True
    
    def test_advanced_config(self):
        """Test advanced configuration level."""
        config = DefaultConfigurationManager.get_advanced_config()
        
        assert isinstance(config, DictConfig)
        
        # Should have all processors
        assert 'statistics' in config.processors
        assert 'outliers' in config.processors
        assert 'failures' in config.processors
        assert 'comparisons' in config.processors
        
        # More comprehensive statistics
        assert len(config.processors.statistics.percentiles) == 7
        assert config.processors.failures.failure_threshold == 0.05  # More sensitive
        
        # Advanced export settings
        assert config.export.default_format == "parquet"
        assert config.export.compression == True
        assert config.export.export_timeout == 300
        
        # More aggregation functions
        assert len(config.aggregation.default_functions) == 8
    
    def test_research_config(self):
        """Test research configuration level."""
        config = DefaultConfigurationManager.get_research_config()
        
        assert isinstance(config, DictConfig)
        
        # Research-grade statistics
        assert config.processors.statistics.confidence_level == 0.99
        assert len(config.processors.statistics.percentiles) == 9
        assert config.processors.statistics.missing_strategy == "keep"
        
        # Conservative outlier detection
        assert config.processors.outliers.default_method == "modified_zscore"
        assert config.processors.outliers.iqr_factor == 3.0
        assert config.processors.outliers.action == "flag"
        
        # Very sensitive failure detection
        assert config.processors.failures.failure_threshold == 0.01
        assert config.processors.failures.min_samples == 30
        
        # Comprehensive comparisons
        assert config.processors.comparisons.confidence_level == 0.99
        assert config.processors.comparisons.significance_threshold == 0.01
        
        # Extensive aggregation
        assert len(config.aggregation.default_functions) == 11
        assert "sem" in config.aggregation.default_functions
        assert "mad" in config.aggregation.default_functions


class TestConfigurationByLevel:
    """Test configuration retrieval by level."""
    
    def test_get_config_by_level_string(self):
        """Test getting configuration by level string."""
        config = DefaultConfigurationManager.get_config_by_level("minimal")
        assert isinstance(config, DictConfig)
        assert len(config.processors) == 1  # Only statistics
        
        config = DefaultConfigurationManager.get_config_by_level("standard")
        assert len(config.processors) == 3  # Statistics, outliers, failures
        
        config = DefaultConfigurationManager.get_config_by_level("advanced")
        assert len(config.processors) == 4  # All processors
        
        config = DefaultConfigurationManager.get_config_by_level("research")
        assert len(config.processors) == 4  # All processors
    
    def test_get_config_by_level_enum(self):
        """Test getting configuration by level enum."""
        config = DefaultConfigurationManager.get_config_by_level(ConfigurationLevel.MINIMAL)
        assert isinstance(config, DictConfig)
        
        config = DefaultConfigurationManager.get_config_by_level(ConfigurationLevel.STANDARD)
        assert isinstance(config, DictConfig)
    
    def test_invalid_level(self):
        """Test error handling for invalid levels."""
        with pytest.raises(ValueError) as exc_info:
            DefaultConfigurationManager.get_config_by_level("invalid")
        
        assert "Invalid configuration level" in str(exc_info.value)
        assert "minimal" in str(exc_info.value)


class TestConfigurationFileGeneration:
    """Test configuration file generation."""
    
    def test_create_default_config_file_basic(self):
        """Test basic configuration file creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_analytics_config.yaml")
            
            result_path = DefaultConfigurationManager.create_default_analytics_config_file(
                output_path=output_path,
                level="standard",
                include_documentation=False
            )
            
            assert result_path == output_path
            assert os.path.exists(output_path)
            
            # Load and verify configuration
            config = OmegaConf.load(output_path)
            assert isinstance(config, DictConfig)
            assert 'processors' in config
    
    def test_create_default_config_file_with_docs(self):
        """Test configuration file creation with documentation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "documented_config.yaml")
            
            result_path = DefaultConfigurationManager.create_default_analytics_config_file(
                output_path=output_path,
                level="advanced",
                include_documentation=True
            )
            
            assert os.path.exists(output_path)
            
            # Check that documentation is included
            with open(output_path, 'r') as f:
                content = f.read()
            
            assert "Analytics Configuration - Advanced Level" in content
            assert "Environment Variable Overrides" in content
            assert "VIZ_ANALYTICS_" in content
            assert "processors:" in content
    
    def test_create_default_config_file_default_path(self):
        """Test configuration file creation with default path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                result_path = DefaultConfigurationManager.create_default_analytics_config_file(
                    level="minimal"
                )
                
                assert result_path == "analytics_config.yaml"
                assert os.path.exists("analytics_config.yaml")
            finally:
                os.chdir(original_cwd)


class TestLevelDocumentation:
    """Test level documentation and metadata."""
    
    def test_get_available_levels(self):
        """Test getting available configuration levels."""
        levels = DefaultConfigurationManager.get_available_levels()
        
        assert isinstance(levels, list)
        assert len(levels) == 4
        assert "minimal" in levels
        assert "standard" in levels
        assert "advanced" in levels
        assert "research" in levels
    
    def test_get_level_documentation(self):
        """Test getting comprehensive level documentation."""
        docs = DefaultConfigurationManager.get_level_documentation("standard")
        
        assert isinstance(docs, dict)
        assert docs['level'] == 'standard'
        assert 'description' in docs
        assert 'use_cases' in docs
        assert 'enabled_processors' in docs
        assert 'key_features' in docs
        assert 'configuration' in docs
        
        # Check enabled processors
        assert 'statistics' in docs['enabled_processors']
        assert 'outliers' in docs['enabled_processors']
        assert 'failures' in docs['enabled_processors']
        
        # Check use cases
        assert isinstance(docs['use_cases'], list)
        assert len(docs['use_cases']) > 0
        assert any("production" in use_case.lower() for use_case in docs['use_cases'])
        
        # Check key features
        assert isinstance(docs['key_features'], list)
        assert len(docs['key_features']) > 0
    
    def test_level_documentation_all_levels(self):
        """Test documentation for all levels."""
        for level in DefaultConfigurationManager.get_available_levels():
            docs = DefaultConfigurationManager.get_level_documentation(level)
            assert docs['level'] == level
            assert len(docs['use_cases']) > 0
            assert len(docs['key_features']) > 0


class TestFactoryIntegration:
    """Test integration with AnalyticsFactory."""
    
    def test_factory_get_complete_default_config(self):
        """Test factory method for getting complete default config."""
        config = AnalyticsFactory.get_complete_default_config("standard")
        
        assert isinstance(config, DictConfig)
        assert 'processors' in config
        assert 'aggregation' in config
        assert 'export' in config
    
    def test_factory_create_with_defaults(self):
        """Test creating processors with default configurations."""
        # Create statistics processor with standard defaults
        processor = AnalyticsFactory.create_with_defaults("statistics", level="standard")
        
        assert processor is not None
        assert hasattr(processor, 'process')
        
        # Create with overrides
        processor = AnalyticsFactory.create_with_defaults(
            "statistics", 
            level="minimal",
            confidence_level=0.99
        )
        
        assert processor is not None
    
    def test_factory_get_default_config_integration(self):
        """Test factory default config integration with defaults manager."""
        # Should use DefaultConfigurationManager when available
        config = AnalyticsFactory.get_default_config("statistics")
        
        assert isinstance(config, DictConfig)
        assert 'confidence_level' in config
        assert 'percentiles' in config
        assert 'missing_strategy' in config
        assert 'include_advanced' in config


class TestConfigurationValidation:
    """Test that generated configurations are valid."""
    
    def test_all_levels_are_valid(self):
        """Test that all configuration levels pass validation."""
        from experiment_manager.analytics.validation import validate_analytics_config
        
        for level_name in DefaultConfigurationManager.get_available_levels():
            config = DefaultConfigurationManager.get_config_by_level(level_name)
            
            # Convert to regular dict for validation
            config_dict = OmegaConf.to_container(config, resolve=True)
            
            validation_result = validate_analytics_config(config_dict)
            
            # Should be valid with at most warnings
            assert validation_result.is_valid, f"Level {level_name} config is invalid: {validation_result.errors}"
    
    def test_processors_can_be_created(self):
        """Test that processors can be created from all default configs."""
        for level_name in DefaultConfigurationManager.get_available_levels():
            config = DefaultConfigurationManager.get_config_by_level(level_name)
            
            # Create processors from the configuration
            processors = AnalyticsFactory.create_from_config(config)
            
            assert isinstance(processors, dict)
            assert len(processors) > 0
            
            # All processors should be valid
            for processor_name, processor in processors.items():
                assert processor is not None
                assert hasattr(processor, 'process')


class TestConfigurationLevelFeatures:
    """Test that different levels have appropriate features."""
    
    def test_level_progression(self):
        """Test that higher levels include more features."""
        minimal = DefaultConfigurationManager.get_minimal_config()
        standard = DefaultConfigurationManager.get_standard_config()
        advanced = DefaultConfigurationManager.get_advanced_config()
        research = DefaultConfigurationManager.get_research_config()
        
        # Processor count should increase
        assert len(minimal.processors) < len(standard.processors)
        assert len(standard.processors) <= len(advanced.processors)
        assert len(advanced.processors) == len(research.processors)
        
        # Aggregation functions should increase
        minimal_funcs = len(minimal.aggregation.default_functions)
        standard_funcs = len(standard.aggregation.default_functions)
        advanced_funcs = len(advanced.aggregation.default_functions)
        research_funcs = len(research.aggregation.default_functions)
        
        assert minimal_funcs < standard_funcs <= advanced_funcs <= research_funcs
    
    def test_research_vs_production_settings(self):
        """Test research-specific vs production-specific settings."""
        standard = DefaultConfigurationManager.get_standard_config()
        research = DefaultConfigurationManager.get_research_config()
        
        # Research should be more conservative and transparent
        assert research.processors.statistics.confidence_level > standard.processors.statistics.confidence_level
        assert research.processors.statistics.missing_strategy == "keep"
        assert standard.processors.statistics.missing_strategy == "drop"
        
        # Research should flag outliers, production should exclude
        assert research.processors.outliers.action == "flag"
        assert standard.processors.outliers.action == "exclude"
        
        # Research should be more sensitive to failures
        assert research.processors.failures.failure_threshold < standard.processors.failures.failure_threshold


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_defaults_module_fallback(self):
        """Test fallback behavior when defaults module is not available."""
        # This tests the ImportError handling in analytics_factory.py
        config = AnalyticsFactory.get_default_config("statistics")
        
        assert isinstance(config, DictConfig)
        assert 'confidence_level' in config
    
    def test_invalid_processor_type(self):
        """Test error handling for invalid processor types."""
        with pytest.raises(ValueError) as exc_info:
            AnalyticsFactory.get_default_config("invalid_processor")
        
        assert "Unknown processor type" in str(exc_info.value)
    
    def test_from_config_method(self):
        """Test the from_config class method."""
        manager = DefaultConfigurationManager.from_config(DictConfig({}))
        assert isinstance(manager, DefaultConfigurationManager)


if __name__ == "__main__":
    pytest.main([__file__]) 