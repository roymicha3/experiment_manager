"""
Tests for Analytics Configuration Validation System

Tests comprehensive validation including inheritance patterns, override validation,
cross-section validation, and integration with ConfigManager.
"""

import pytest
from typing import Dict, Any

from experiment_manager.analytics.validation import (
    AnalyticsConfigValidator,
    ValidationSeverity,
    ValidationResult,
    validate_analytics_config,
    validate_processor_config,
    validate_config_inheritance
)


class TestValidationResult:
    """Test ValidationResult data class functionality."""
    
    def test_validation_result_categorization(self):
        """Test that ValidationResult correctly categorizes issues."""
        from experiment_manager.analytics.validation import ValidationIssue
        
        issues = [
            ValidationIssue(ValidationSeverity.ERROR, "Error message", "path1", "CODE1"),
            ValidationIssue(ValidationSeverity.WARNING, "Warning message", "path2", "CODE2"),
            ValidationIssue(ValidationSeverity.INFO, "Info message", "path3", "CODE3"),
            ValidationIssue(ValidationSeverity.ERROR, "Another error", "path4", "CODE4")
        ]
        
        result = ValidationResult(True, issues)
        
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.errors[0].severity == ValidationSeverity.ERROR
        assert result.warnings[0].severity == ValidationSeverity.WARNING
        assert not result.is_valid  # Should be False due to errors
    
    def test_validation_result_valid_config(self):
        """Test ValidationResult with valid configuration."""
        from experiment_manager.analytics.validation import ValidationIssue
        
        issues = [
            ValidationIssue(ValidationSeverity.INFO, "Info message", "path1", "CODE1"),
            ValidationIssue(ValidationSeverity.WARNING, "Warning message", "path2", "CODE2")
        ]
        
        result = ValidationResult(True, issues)
        
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.is_valid  # Should be True - no errors


class TestAnalyticsConfigValidator:
    """Test the main AnalyticsConfigValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AnalyticsConfigValidator()
    
    def test_empty_config_validation(self):
        """Test validation of empty configuration."""
        result = self.validator.validate_analytics_config({})
        
        assert result.is_valid
        assert len(result.warnings) == 1
        assert "Empty analytics configuration" in result.warnings[0].message
    
    def test_valid_config_validation(self):
        """Test validation of a complete valid configuration."""
        valid_config = {
            "processors": {
                "statistics": {
                    "confidence_level": 0.95,
                    "percentiles": [25, 50, 75, 90, 95],
                    "missing_strategy": "drop",
                    "include_advanced": True
                },
                "outliers": {
                    "default_method": "iqr",
                    "iqr_factor": 1.5,
                    "action": "exclude"
                }
            },
            "aggregation": {
                "default_functions": ["mean", "median", "std"],
                "group_by_defaults": ["experiment_name"],
                "metric_columns": ["accuracy", "loss"]
            },
            "export": {
                "default_format": "csv",
                "output_directory": "analytics_outputs",
                "export_timeout": 120
            }
        }
        
        result = self.validator.validate_analytics_config(valid_config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_invalid_processor_type(self):
        """Test validation with invalid processor type."""
        invalid_config = {
            "processors": {
                "invalid_processor": {
                    "some_setting": "value"
                }
            }
        }
        
        result = self.validator.validate_analytics_config(invalid_config)
        
        assert not result.is_valid
        assert len(result.errors) >= 1
        assert "Invalid processor type" in result.errors[0].message
        assert "invalid_processor" in result.errors[0].message
    
    def test_missing_required_section(self):
        """Test validation when required section is missing."""
        invalid_config = {
            "export": {
                "default_format": "csv"
            }
            # Missing required 'processors' section
        }
        
        result = self.validator.validate_analytics_config(invalid_config)
        
        assert not result.is_valid
        assert any("Missing required configuration section: 'processors'" in error.message 
                  for error in result.errors)
    
    def test_unknown_section_warning(self):
        """Test validation with unknown configuration section."""
        config_with_unknown = {
            "processors": {
                "statistics": {"confidence_level": 0.95}
            },
            "unknown_section": {
                "some_setting": "value"
            }
        }
        
        result = self.validator.validate_analytics_config(config_with_unknown)
        
        assert result.is_valid  # Should still be valid, just warnings
        assert len(result.warnings) >= 1
        assert any("Unknown configuration section: 'unknown_section'" in warning.message 
                  for warning in result.warnings)


class TestProcessorValidation:
    """Test validation of individual processor configurations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AnalyticsConfigValidator()
    
    def test_statistics_processor_valid(self):
        """Test valid statistics processor configuration."""
        config = {
            "confidence_level": 0.95,
            "percentiles": [25, 50, 75, 90, 95],
            "missing_strategy": "drop",
            "include_advanced": True
        }
        
        result = self.validator.validate_processor_config("statistics", config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_statistics_processor_invalid_confidence_level(self):
        """Test statistics processor with invalid confidence level."""
        config = {
            "confidence_level": 1.5  # Invalid - must be between 0 and 1
        }
        
        result = self.validator.validate_processor_config("statistics", config)
        
        assert not result.is_valid
        assert any("confidence_level must be between 0 and 1" in error.message 
                  for error in result.errors)
    
    def test_statistics_processor_invalid_strategy(self):
        """Test statistics processor with invalid missing strategy."""
        config = {
            "missing_strategy": "invalid_strategy"
        }
        
        result = self.validator.validate_processor_config("statistics", config)
        
        assert not result.is_valid
        assert any("Invalid missing_strategy" in error.message 
                  for error in result.errors)
    
    def test_statistics_processor_invalid_percentiles(self):
        """Test statistics processor with invalid percentiles."""
        config = {
            "percentiles": [25, 50, 150]  # 150 is invalid
        }
        
        result = self.validator.validate_processor_config("statistics", config)
        
        assert not result.is_valid
        assert any("percentiles must be a list of numbers between 0 and 100" in error.message 
                  for error in result.errors)
    
    def test_outliers_processor_valid(self):
        """Test valid outliers processor configuration."""
        config = {
            "default_method": "iqr",
            "iqr_factor": 1.5,
            "zscore_threshold": 3.0,
            "action": "exclude"
        }
        
        result = self.validator.validate_processor_config("outliers", config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_outliers_processor_invalid_method(self):
        """Test outliers processor with invalid method."""
        config = {
            "default_method": "invalid_method"
        }
        
        result = self.validator.validate_processor_config("outliers", config)
        
        assert not result.is_valid
        assert any("Invalid outlier detection method" in error.message 
                  for error in result.errors)
    
    def test_outliers_processor_invalid_iqr_factor(self):
        """Test outliers processor with invalid IQR factor."""
        config = {
            "iqr_factor": -1.0  # Must be positive
        }
        
        result = self.validator.validate_processor_config("outliers", config)
        
        assert not result.is_valid
        assert any("iqr_factor must be positive" in error.message 
                  for error in result.errors)
    
    def test_failures_processor_valid(self):
        """Test valid failures processor configuration."""
        config = {
            "failure_threshold": 0.1,
            "min_samples": 10,
            "time_window": "day",
            "analysis_types": ["rates", "correlations"]
        }
        
        result = self.validator.validate_processor_config("failures", config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_failures_processor_invalid_threshold(self):
        """Test failures processor with invalid threshold."""
        config = {
            "failure_threshold": 1.5  # Must be between 0 and 1
        }
        
        result = self.validator.validate_processor_config("failures", config)
        
        assert not result.is_valid
        assert any("failure_threshold must be between 0 and 1" in error.message 
                  for error in result.errors)
    
    def test_failures_processor_invalid_time_window(self):
        """Test failures processor with invalid time window."""
        config = {
            "time_window": "invalid_window"
        }
        
        result = self.validator.validate_processor_config("failures", config)
        
        assert not result.is_valid
        assert any("Invalid time_window" in error.message 
                  for error in result.errors)
    
    def test_comparisons_processor_valid(self):
        """Test valid comparisons processor configuration."""
        config = {
            "confidence_level": 0.95,
            "significance_threshold": 0.05,
            "min_samples": 5,
            "comparison_types": ["pairwise", "ranking"],
            "baseline_selection": "auto"
        }
        
        result = self.validator.validate_processor_config("comparisons", config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_comparisons_processor_invalid_min_samples(self):
        """Test comparisons processor with invalid min samples."""
        config = {
            "min_samples": 1  # Must be at least 2 for comparisons
        }
        
        result = self.validator.validate_processor_config("comparisons", config)
        
        assert not result.is_valid
        assert any("min_samples must be at least 2 for comparisons" in error.message 
                  for error in result.errors)


class TestCrossSectionValidation:
    """Test validation across different configuration sections."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AnalyticsConfigValidator()
    
    def test_export_format_compatibility_warning(self):
        """Test warning for potentially incompatible export format."""
        config = {
            "processors": {
                "failures": {
                    "analysis_types": ["root_cause"]  # Complex output
                }
            },
            "export": {
                "default_format": "excel"  # May not handle complex outputs well
            }
        }
        
        result = self.validator.validate_analytics_config(config)
        
        assert result.is_valid  # Still valid, just a warning
        assert len(result.warnings) >= 1
        assert any("Root cause analysis may produce complex outputs" in warning.message 
                  for warning in result.warnings)
    
    def test_timeout_inconsistency_warning(self):
        """Test warning for inconsistent timeout settings."""
        config = {
            "processors": {
                "statistics": {"confidence_level": 0.95}
            },
            "query_timeout": 60,  # Short query timeout
            "export": {
                "export_timeout": 120  # Longer export timeout
            }
        }
        
        result = self.validator.validate_analytics_config(config)
        
        assert result.is_valid  # Still valid, just a warning
        assert len(result.warnings) >= 1
        assert any("Export timeout" in warning.message and "query timeout" in warning.message 
                  for warning in result.warnings)


class TestInheritanceValidation:
    """Test configuration inheritance validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AnalyticsConfigValidator()
    
    def test_valid_inheritance(self):
        """Test valid configuration inheritance."""
        parent_config = {
            "processors": {
                "statistics": {
                    "confidence_level": 0.95,
                    "missing_strategy": "drop"
                }
            },
            "export": {
                "default_format": "csv"
            }
        }
        
        child_config = {
            "processors": {
                "outliers": {
                    "default_method": "iqr",
                    "action": "exclude"
                }
            }
        }
        
        result = self.validator.validate_inheritance(parent_config, child_config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_inheritance_conflict_warning(self):
        """Test warning for configuration inheritance conflicts."""
        parent_config = {
            "processors": {
                "statistics": {
                    "confidence_level": 0.95
                }
            }
        }
        
        child_config = {
            "processors": {
                "statistics": {
                    "confidence_level": 0.99  # Override parent value
                }
            }
        }
        
        result = self.validator.validate_inheritance(parent_config, child_config)
        
        assert result.is_valid  # Still valid, just conflicts
        assert len(result.warnings) >= 1
        assert any("Configuration override may cause conflict" in warning.message 
                  for warning in result.warnings)
    
    def test_inconsistent_confidence_levels_info(self):
        """Test info message for inconsistent confidence levels."""
        config = {
            "processors": {
                "statistics": {
                    "confidence_level": 0.95
                },
                "comparisons": {
                    "confidence_level": 0.99  # Different from statistics
                }
            }
        }
        
        result = self.validator.validate_analytics_config(config)
        
        assert result.is_valid
        # Should have an info message about inconsistent confidence levels
        all_issues = result.issues
        info_issues = [issue for issue in all_issues if issue.severity == ValidationSeverity.INFO]
        assert len(info_issues) >= 1
        assert any("Different confidence levels across processors" in issue.message 
                  for issue in info_issues)


class TestConvenienceFunctions:
    """Test the convenience functions for validation."""
    
    def test_validate_analytics_config_function(self):
        """Test the standalone validate_analytics_config function."""
        config = {
            "processors": {
                "statistics": {"confidence_level": 0.95}
            }
        }
        
        result = validate_analytics_config(config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
    
    def test_validate_processor_config_function(self):
        """Test the standalone validate_processor_config function."""
        config = {
            "confidence_level": 0.95,
            "missing_strategy": "drop"
        }
        
        result = validate_processor_config("statistics", config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
    
    def test_validate_config_inheritance_function(self):
        """Test the standalone validate_config_inheritance function."""
        parent_config = {"processors": {"statistics": {"confidence_level": 0.95}}}
        child_config = {"export": {"default_format": "json"}}
        
        result = validate_config_inheritance(parent_config, child_config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AnalyticsConfigValidator()
    
    def test_invalid_processor_type_in_validate_processor_config(self):
        """Test validation with invalid processor type."""
        result = self.validator.validate_processor_config("invalid_type", {})
        
        assert not result.is_valid
        assert len(result.errors) >= 1
        assert "Unknown processor type" in result.errors[0].message
    
    def test_none_config_values(self):
        """Test validation with None values in configuration."""
        config = {
            "processors": {
                "statistics": {
                    "confidence_level": None  # None value
                }
            }
        }
        
        result = self.validator.validate_analytics_config(config)
        
        # Should handle None values gracefully
        assert isinstance(result, ValidationResult)
    
    def test_empty_aggregation_functions(self):
        """Test validation with empty aggregation functions."""
        config = {
            "processors": {"statistics": {"confidence_level": 0.95}},
            "aggregation": {
                "default_functions": []  # Empty list
            }
        }
        
        result = self.validator.validate_analytics_config(config)
        
        assert result.is_valid  # Should be valid but with warnings
        assert len(result.warnings) >= 1
        assert any("No default aggregation functions specified" in warning.message 
                  for warning in result.warnings)
    
    def test_invalid_data_types(self):
        """Test validation with invalid data types."""
        config = {
            "processors": {
                "statistics": {
                    "confidence_level": "not_a_number"  # Should be float
                }
            }
        }
        
        result = self.validator.validate_analytics_config(config)
        
        assert not result.is_valid
        assert len(result.errors) >= 1
    
    def test_nested_inheritance_conflicts(self):
        """Test complex nested inheritance conflicts."""
        parent_config = {
            "processors": {
                "statistics": {
                    "confidence_level": 0.95,
                    "missing_strategy": "drop"
                },
                "outliers": {
                    "default_method": "iqr",
                    "iqr_factor": 1.5
                }
            }
        }
        
        child_config = {
            "processors": {
                "statistics": {
                    "confidence_level": 0.99,  # Override
                    "percentiles": [25, 50, 75]  # Add new setting
                },
                "outliers": {
                    "iqr_factor": 3.0  # Override
                }
            }
        }
        
        result = self.validator.validate_inheritance(parent_config, child_config)
        
        assert result.is_valid
        # Should detect multiple conflicts
        assert len(result.warnings) >= 2  # At least 2 overrides 