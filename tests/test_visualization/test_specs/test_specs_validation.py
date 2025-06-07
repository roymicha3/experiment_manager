"""
Tests for Plot and Data Specification Validation System
"""

import pytest
from experiment_manager.visualization.specs import (
    PlotSpec, PlotType, DataSpec, DataType, DataColumn, DataMapping,
    spec_validator, ValidationSeverity
)
from experiment_manager.visualization.specs.data_spec import DataSource
from experiment_manager.visualization.specs.validation import (
    ValidationResult, ValidationIssue, SpecValidator
)


class TestValidationResult:
    """Test ValidationResult functionality."""
    
    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        issues = [
            ValidationIssue(
                message="Test issue",
                severity=ValidationSeverity.WARNING,
                rule_name="test_rule"
            )
        ]
        
        result = ValidationResult(
            is_valid=True,
            issues=issues
        )
        
        assert result.is_valid is True
        assert len(result.issues) == 1
        assert result.issues[0].message == "Test issue"
        assert result.issues[0].severity == ValidationSeverity.WARNING


class TestValidationIssue:
    """Test ValidationIssue functionality."""
    
    def test_validation_issue_creation(self):
        """Test creating a ValidationIssue."""
        issue = ValidationIssue(
            message="Test validation issue",
            severity=ValidationSeverity.ERROR,
            rule_name="test_rule",
            field_path="test.field",
            suggested_fix="Fix this issue",
            context={"key": "value"}
        )
        
        assert issue.message == "Test validation issue"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.rule_name == "test_rule"
        assert issue.field_path == "test.field"
        assert issue.suggested_fix == "Fix this issue"
        assert issue.context["key"] == "value"


class TestSpecValidator:
    """Test SpecValidator functionality."""
    
    def test_spec_validator_exists(self):
        """Test that spec_validator is available."""
        assert spec_validator is not None
        assert isinstance(spec_validator, SpecValidator)
    
    def test_plot_spec_validation_runs(self):
        """Test that plot spec validation runs without errors."""
        plot_spec = PlotSpec(
            plot_type=PlotType.SCATTER,
            title="Test Plot",
            description="A test plot"
        )
        
        # Should not raise an exception
        result = spec_validator.validate_plot_spec(plot_spec)
        assert isinstance(result, ValidationResult)
    
    def test_data_spec_validation_runs(self):
        """Test that data spec validation runs without errors."""
        columns = [DataColumn(name="x", data_type=DataType.NUMERIC)]
        source = DataSource(source_type="file", path="test.csv")
        
        data_spec = DataSpec(
            spec_id="test-spec",
            name="Test Data",
            columns=columns,
            source=source
        )
        
        # Should not raise an exception
        result = spec_validator.validate_data_spec(data_spec)
        assert isinstance(result, ValidationResult)


class TestValidationSeverityLevels:
    """Test validation severity level handling."""
    
    def test_severity_levels_exist(self):
        """Test that all severity levels are available."""
        assert ValidationSeverity.INFO
        assert ValidationSeverity.WARNING
        assert ValidationSeverity.ERROR
        assert ValidationSeverity.CRITICAL
    
    def test_severity_level_values(self):
        """Test severity level string values."""
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.CRITICAL.value == "critical"


class TestValidationIntegration:
    """Test validation integration with specs."""
    
    def test_plot_spec_validation_with_different_types(self):
        """Test validation with different plot types."""
        plot_types_to_test = [
            PlotType.LINE,
            PlotType.SCATTER,
            PlotType.BAR,
            PlotType.HISTOGRAM,
            PlotType.HEATMAP
        ]
        
        for plot_type in plot_types_to_test:
            plot_spec = PlotSpec(
                plot_type=plot_type,
                title=f"Test {plot_type.value} Plot",
                description=f"Testing {plot_type.value} validation"
            )
            
            # Should not raise an exception for any plot type
            result = spec_validator.validate_plot_spec(plot_spec)
            assert isinstance(result, ValidationResult)
    
    def test_data_spec_validation_with_different_types(self):
        """Test data spec validation with different data types."""
        data_types_to_test = [
            DataType.NUMERIC,
            DataType.CATEGORICAL,
            DataType.TEXT,
            DataType.DATETIME,
            DataType.BOOLEAN
        ]
        
        for data_type in data_types_to_test:
            columns = [DataColumn(name="test_col", data_type=data_type)]
            source = DataSource(source_type="memory")
            
            data_spec = DataSpec(
                spec_id=f"test-{data_type.value}",
                name=f"Test {data_type.value} Data",
                columns=columns,
                source=source
            )
            
            # Should not raise an exception for any data type
            result = spec_validator.validate_data_spec(data_spec)
            assert isinstance(result, ValidationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 