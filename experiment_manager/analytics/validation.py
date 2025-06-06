"""
Analytics Configuration Validation System

Provides comprehensive validation for analytics configurations including
inheritance patterns, override validation, cross-section validation,
and integration with the existing ConfigManager validation framework.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum
import re

from pydantic import ValidationError
from omegaconf import DictConfig

from experiment_manager.analytics.analytics_factory import AnalyticsFactory

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue with context and suggestions."""
    severity: ValidationSeverity
    message: str
    path: str
    code: str
    suggestion: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings: List[ValidationIssue] = None
    errors: List[ValidationIssue] = None
    
    def __post_init__(self):
        """Categorize issues by severity."""
        self.warnings = [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]
        self.errors = [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
        self.is_valid = len(self.errors) == 0


class AnalyticsConfigValidator:
    """
    Comprehensive validator for analytics configurations.
    
    This validator extends the basic Pydantic validation with:
    - Configuration inheritance validation
    - Override compatibility checks  
    - Cross-section validation
    - Clear error messages with suggestions
    - Custom analytics-specific validation rules
    """
    
    # Valid configuration keys and their expected types/values
    VALID_PROCESSOR_TYPES = {'statistics', 'outliers', 'failures', 'comparisons'}
    VALID_EXPORT_FORMATS = {'csv', 'json', 'parquet', 'excel'}
    VALID_MISSING_STRATEGIES = {'drop', 'fill_mean', 'fill_median', 'keep'}
    VALID_OUTLIER_METHODS = {'iqr', 'zscore', 'modified_zscore', 'custom'}
    VALID_OUTLIER_ACTIONS = {'exclude', 'flag', 'keep'}
    VALID_TIME_WINDOWS = {'hour', 'day', 'week', 'month'}
    VALID_COMPARISON_TYPES = {'pairwise', 'ranking', 'ab_test', 'trend'}
    VALID_BASELINE_SELECTIONS = {'auto', 'first', 'largest', 'custom'}
    VALID_ANALYSIS_TYPES = {'rates', 'correlations', 'temporal', 'root_cause'}
    VALID_REPORT_FORMATS = {'summary', 'detailed', 'full'}
    
    def __init__(self):
        """Initialize the validator."""
        self.issues: List[ValidationIssue] = []
    
    def validate_analytics_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate a complete analytics configuration.
        
        Args:
            config: Analytics configuration dictionary
            
        Returns:
            ValidationResult with validation status and issues
        """
        self.issues = []
        
        if not config:
            self._add_issue(
                ValidationSeverity.WARNING,
                "Empty analytics configuration",
                "root",
                "EMPTY_CONFIG",
                "Consider using default analytics configuration or providing processor settings"
            )
            return ValidationResult(True, self.issues)
        
        # Validate overall structure
        self._validate_config_structure(config)
        
        # Validate individual sections
        if 'processors' in config:
            self._validate_processors_section(config['processors'])
        
        if 'aggregation' in config:
            self._validate_aggregation_section(config['aggregation'])
        
        if 'export' in config:
            self._validate_export_section(config['export'])
        
        if 'failure_analysis' in config:
            self._validate_failure_analysis_section(config['failure_analysis'])
        
        if 'workspace' in config:
            self._validate_workspace_section(config['workspace'])
        
        # Cross-section validation
        self._validate_cross_section_compatibility(config)
        
        # Inheritance and override validation
        self._validate_inheritance_patterns(config)
        
        return ValidationResult(len([issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]) == 0, 
                              self.issues)
    
    def validate_processor_config(self, processor_type: str, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration for a specific processor type.
        
        Args:
            processor_type: Type of processor ('statistics', 'outliers', etc.)
            config: Processor-specific configuration
            
        Returns:
            ValidationResult with validation status and issues
        """
        self.issues = []
        
        if processor_type not in self.VALID_PROCESSOR_TYPES:
            self._add_issue(
                ValidationSeverity.ERROR,
                f"Unknown processor type: '{processor_type}'",
                f"processors.{processor_type}",
                "INVALID_PROCESSOR_TYPE",
                f"Valid processor types: {', '.join(self.VALID_PROCESSOR_TYPES)}"
            )
            return ValidationResult(False, self.issues)
        
        # Validate using AnalyticsFactory
        try:
            is_valid = AnalyticsFactory.validate_processor_config(processor_type, DictConfig(config))
            if not is_valid:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid configuration for {processor_type} processor",
                    f"processors.{processor_type}",
                    "INVALID_PROCESSOR_CONFIG",
                    "Check processor configuration parameters against documentation"
                )
        except Exception as e:
            self._add_issue(
                ValidationSeverity.ERROR,
                f"Failed to validate {processor_type} processor: {str(e)}",
                f"processors.{processor_type}",
                "PROCESSOR_VALIDATION_ERROR",
                "Verify configuration format and parameter values"
            )
        
        # Specific processor validations
        if processor_type == 'statistics':
            self._validate_statistics_processor(config)
        elif processor_type == 'outliers':
            self._validate_outliers_processor(config)
        elif processor_type == 'failures':
            self._validate_failures_processor(config)
        elif processor_type == 'comparisons':
            self._validate_comparisons_processor(config)
        
        return ValidationResult(len([issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]) == 0,
                              self.issues)
    
    def validate_inheritance(self, parent_config: Dict[str, Any], 
                           child_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration inheritance patterns.
        
        Args:
            parent_config: Parent/base configuration
            child_config: Child/derived configuration
            
        Returns:
            ValidationResult with validation status and issues
        """
        self.issues = []
        
        # Check for conflicting overrides
        conflicts = self._find_inheritance_conflicts(parent_config, child_config)
        for conflict_path, conflict_info in conflicts.items():
            self._add_issue(
                ValidationSeverity.WARNING,
                f"Configuration override may cause conflict: {conflict_info['description']}",
                conflict_path,
                "INHERITANCE_CONFLICT",
                conflict_info['suggestion']
            )
        
        # Validate merged configuration using a new validator instance
        merged_config = self._merge_configs(parent_config, child_config)
        merged_validator = AnalyticsConfigValidator()
        merged_result = merged_validator.validate_analytics_config(merged_config)
        
        # Add inheritance context to issues from merged validation
        for issue in merged_result.issues:
            if issue.context is None:
                issue.context = {}
            issue.context['inheritance'] = True
            issue.context['parent_config'] = parent_config
            issue.context['child_config'] = child_config
            self.issues.append(issue)
        
        return ValidationResult(len([issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]) == 0,
                              self.issues)
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> None:
        """Validate the overall structure of analytics configuration."""
        required_sections = {'processors'}
        optional_sections = {'aggregation', 'export', 'failure_analysis', 'workspace', 
                           'database_connection', 'query_timeout', 'result_caching', 'batch_size'}
        
        # Check for unknown sections
        for key in config.keys():
            if key not in required_sections and key not in optional_sections:
                self._add_issue(
                    ValidationSeverity.WARNING,
                    f"Unknown configuration section: '{key}'",
                    key,
                    "UNKNOWN_SECTION",
                    f"Valid sections: {', '.join(required_sections | optional_sections)}"
                )
        
        # Check for required sections
        for required in required_sections:
            if required not in config:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Missing required configuration section: '{required}'",
                    required,
                    "MISSING_REQUIRED_SECTION",
                    f"Add '{required}' section to analytics configuration"
                )
    
    def _validate_processors_section(self, processors: Dict[str, Any]) -> None:
        """Validate the processors configuration section."""
        if not processors:
            self._add_issue(
                ValidationSeverity.WARNING,
                "Empty processors configuration",
                "processors",
                "EMPTY_PROCESSORS",
                "Consider adding at least one processor (statistics, outliers, failures, or comparisons)"
            )
            return
        
        for processor_type, processor_config in processors.items():
            if processor_type not in self.VALID_PROCESSOR_TYPES:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid processor type: '{processor_type}'",
                    f"processors.{processor_type}",
                    "INVALID_PROCESSOR_TYPE",
                    f"Valid types: {', '.join(self.VALID_PROCESSOR_TYPES)}"
                )
                continue
            
            if not isinstance(processor_config, dict):
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Processor configuration must be a dictionary: {processor_type}",
                    f"processors.{processor_type}",
                    "INVALID_PROCESSOR_CONFIG_TYPE",
                    "Provide configuration as key-value pairs"
                )
                continue
            
            # Validate specific processor configuration directly without resetting issues
            if processor_type == 'statistics':
                self._validate_statistics_processor(processor_config)
            elif processor_type == 'outliers':
                self._validate_outliers_processor(processor_config)
            elif processor_type == 'failures':
                self._validate_failures_processor(processor_config)
            elif processor_type == 'comparisons':
                self._validate_comparisons_processor(processor_config)
    
    def _validate_statistics_processor(self, config: Dict[str, Any]) -> None:
        """Validate statistics processor configuration."""
        if 'confidence_level' in config:
            conf_level = config['confidence_level']
            if not isinstance(conf_level, (int, float)) or not 0 < conf_level < 1:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"confidence_level must be between 0 and 1, got: {conf_level}",
                    "processors.statistics.confidence_level",
                    "INVALID_CONFIDENCE_LEVEL",
                    "Use a value like 0.95 for 95% confidence"
                )
        
        if 'missing_strategy' in config:
            strategy = config['missing_strategy']
            if strategy not in self.VALID_MISSING_STRATEGIES:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid missing_strategy: '{strategy}'",
                    "processors.statistics.missing_strategy",
                    "INVALID_MISSING_STRATEGY",
                    f"Valid strategies: {', '.join(self.VALID_MISSING_STRATEGIES)}"
                )
        
        if 'percentiles' in config:
            percentiles = config['percentiles']
            if not isinstance(percentiles, list) or not all(isinstance(p, (int, float)) and 0 <= p <= 100 for p in percentiles):
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"percentiles must be a list of numbers between 0 and 100",
                    "processors.statistics.percentiles",
                    "INVALID_PERCENTILES",
                    "Example: [25, 50, 75, 90, 95]"
                )
    
    def _validate_outliers_processor(self, config: Dict[str, Any]) -> None:
        """Validate outliers processor configuration."""
        if 'default_method' in config:
            method = config['default_method']
            if method not in self.VALID_OUTLIER_METHODS:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid outlier detection method: '{method}'",
                    "processors.outliers.default_method",
                    "INVALID_OUTLIER_METHOD",
                    f"Valid methods: {', '.join(self.VALID_OUTLIER_METHODS)}"
                )
        
        if 'action' in config:
            action = config['action']
            if action not in self.VALID_OUTLIER_ACTIONS:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid outlier action: '{action}'",
                    "processors.outliers.action",
                    "INVALID_OUTLIER_ACTION",
                    f"Valid actions: {', '.join(self.VALID_OUTLIER_ACTIONS)}"
                )
        
        if 'iqr_factor' in config:
            factor = config['iqr_factor']
            if not isinstance(factor, (int, float)) or factor <= 0:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"iqr_factor must be positive, got: {factor}",
                    "processors.outliers.iqr_factor",
                    "INVALID_IQR_FACTOR",
                    "Common values are 1.5 (moderate) or 3.0 (conservative)"
                )
        
        if 'zscore_threshold' in config:
            threshold = config['zscore_threshold']
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"zscore_threshold must be positive, got: {threshold}",
                    "processors.outliers.zscore_threshold",
                    "INVALID_ZSCORE_THRESHOLD",
                    "Common value is 3.0 for 99.7% of data"
                )
    
    def _validate_failures_processor(self, config: Dict[str, Any]) -> None:
        """Validate failures processor configuration."""
        if 'failure_threshold' in config:
            threshold = config['failure_threshold']
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"failure_threshold must be between 0 and 1, got: {threshold}",
                    "processors.failures.failure_threshold",
                    "INVALID_FAILURE_THRESHOLD",
                    "Use 0.1 for 10% failure rate threshold"
                )
        
        if 'min_samples' in config:
            min_samples = config['min_samples']
            if not isinstance(min_samples, int) or min_samples < 1:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"min_samples must be at least 1, got: {min_samples}",
                    "processors.failures.min_samples",
                    "INVALID_MIN_SAMPLES",
                    "Use a value like 10 for reliable statistics"
                )
        
        if 'time_window' in config:
            window = config['time_window']
            if window not in self.VALID_TIME_WINDOWS:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid time_window: '{window}'",
                    "processors.failures.time_window",
                    "INVALID_TIME_WINDOW",
                    f"Valid windows: {', '.join(self.VALID_TIME_WINDOWS)}"
                )
        
        if 'analysis_types' in config:
            analysis_types = config['analysis_types']
            if not isinstance(analysis_types, list):
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "analysis_types must be a list",
                    "processors.failures.analysis_types",
                    "INVALID_ANALYSIS_TYPES",
                    f"Valid types: {', '.join(self.VALID_ANALYSIS_TYPES)}"
                )
            else:
                for analysis_type in analysis_types:
                    if analysis_type not in self.VALID_ANALYSIS_TYPES:
                        self._add_issue(
                            ValidationSeverity.ERROR,
                            f"Invalid analysis type: '{analysis_type}'",
                            f"processors.failures.analysis_types.{analysis_type}",
                            "INVALID_ANALYSIS_TYPE",
                            f"Valid types: {', '.join(self.VALID_ANALYSIS_TYPES)}"
                        )
    
    def _validate_comparisons_processor(self, config: Dict[str, Any]) -> None:
        """Validate comparisons processor configuration."""
        if 'confidence_level' in config:
            conf_level = config['confidence_level']
            if not isinstance(conf_level, (int, float)) or not 0 < conf_level < 1:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"confidence_level must be between 0 and 1, got: {conf_level}",
                    "processors.comparisons.confidence_level",
                    "INVALID_CONFIDENCE_LEVEL",
                    "Use a value like 0.95 for 95% confidence"
                )
        
        if 'significance_threshold' in config:
            threshold = config['significance_threshold']
            if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"significance_threshold must be between 0 and 1, got: {threshold}",
                    "processors.comparisons.significance_threshold",
                    "INVALID_SIGNIFICANCE_THRESHOLD",
                    "Common value is 0.05 for 5% significance level"
                )
        
        if 'min_samples' in config:
            min_samples = config['min_samples']
            if not isinstance(min_samples, int) or min_samples < 2:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"min_samples must be at least 2 for comparisons, got: {min_samples}",
                    "processors.comparisons.min_samples",
                    "INVALID_MIN_SAMPLES",
                    "Use at least 5 samples per group for reliable comparisons"
                )
        
        if 'comparison_types' in config:
            comp_types = config['comparison_types']
            if not isinstance(comp_types, list):
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "comparison_types must be a list",
                    "processors.comparisons.comparison_types",
                    "INVALID_COMPARISON_TYPES",
                    f"Valid types: {', '.join(self.VALID_COMPARISON_TYPES)}"
                )
            else:
                for comp_type in comp_types:
                    if comp_type not in self.VALID_COMPARISON_TYPES:
                        self._add_issue(
                            ValidationSeverity.ERROR,
                            f"Invalid comparison type: '{comp_type}'",
                            f"processors.comparisons.comparison_types.{comp_type}",
                            "INVALID_COMPARISON_TYPE",
                            f"Valid types: {', '.join(self.VALID_COMPARISON_TYPES)}"
                        )
        
        if 'baseline_selection' in config:
            baseline = config['baseline_selection']
            if baseline not in self.VALID_BASELINE_SELECTIONS:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid baseline_selection: '{baseline}'",
                    "processors.comparisons.baseline_selection",
                    "INVALID_BASELINE_SELECTION",
                    f"Valid selections: {', '.join(self.VALID_BASELINE_SELECTIONS)}"
                )
    
    def _validate_aggregation_section(self, aggregation: Dict[str, Any]) -> None:
        """Validate aggregation configuration section."""
        if 'default_functions' in aggregation:
            functions = aggregation['default_functions']
            if not isinstance(functions, list):
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "default_functions must be a list",
                    "aggregation.default_functions",
                    "INVALID_FUNCTIONS_TYPE",
                    "Example: ['mean', 'median', 'std', 'min', 'max']"
                )
            elif not functions:
                self._add_issue(
                    ValidationSeverity.WARNING,
                    "No default aggregation functions specified",
                    "aggregation.default_functions",
                    "EMPTY_FUNCTIONS",
                    "Consider adding common functions like 'mean', 'std'"
                )
        
        if 'group_by_defaults' in aggregation:
            group_by = aggregation['group_by_defaults']
            if not isinstance(group_by, list):
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "group_by_defaults must be a list",
                    "aggregation.group_by_defaults",
                    "INVALID_GROUP_BY_TYPE",
                    "Example: ['experiment_name', 'trial_name']"
                )
        
        if 'metric_columns' in aggregation:
            metrics = aggregation['metric_columns']
            if not isinstance(metrics, list):
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "metric_columns must be a list",
                    "aggregation.metric_columns",
                    "INVALID_METRICS_TYPE",
                    "Example: ['accuracy', 'loss', 'metric_total_val']"
                )
    
    def _validate_export_section(self, export: Dict[str, Any]) -> None:
        """Validate export configuration section."""
        if 'default_format' in export:
            format_type = export['default_format']
            if format_type not in self.VALID_EXPORT_FORMATS:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid export format: '{format_type}'",
                    "export.default_format",
                    "INVALID_EXPORT_FORMAT",
                    f"Valid formats: {', '.join(self.VALID_EXPORT_FORMATS)}"
                )
        
        if 'export_timeout' in export:
            timeout = export['export_timeout']
            if not isinstance(timeout, (int, float)) or timeout < 1:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"export_timeout must be at least 1 second, got: {timeout}",
                    "export.export_timeout",
                    "INVALID_EXPORT_TIMEOUT",
                    "Use a reasonable timeout like 120 seconds"
                )
        
        if 'output_directory' in export:
            output_dir = export['output_directory']
            if not isinstance(output_dir, str) or not output_dir.strip():
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "output_directory must be a non-empty string",
                    "export.output_directory",
                    "INVALID_OUTPUT_DIRECTORY",
                    "Example: 'analytics_outputs' or 'exports'"
                )
    
    def _validate_failure_analysis_section(self, failure_analysis: Dict[str, Any]) -> None:
        """Validate failure analysis configuration section."""
        if 'correlation_threshold' in failure_analysis:
            threshold = failure_analysis['correlation_threshold']
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"correlation_threshold must be between 0 and 1, got: {threshold}",
                    "failure_analysis.correlation_threshold",
                    "INVALID_CORRELATION_THRESHOLD",
                    "Use 0.7 for strong correlation requirement"
                )
        
        if 'temporal_window' in failure_analysis:
            window = failure_analysis['temporal_window']
            # Accept both old and new format for temporal window
            valid_windows = {'1h', '1d', '1w', '1m', 'hour', 'day', 'week', 'month'}
            if window not in valid_windows:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid temporal_window: '{window}'",
                    "failure_analysis.temporal_window",
                    "INVALID_TEMPORAL_WINDOW",
                    f"Valid windows: {', '.join(valid_windows)}"
                )
        
        if 'root_cause_depth' in failure_analysis:
            depth = failure_analysis['root_cause_depth']
            if not isinstance(depth, int) or depth < 1:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"root_cause_depth must be at least 1, got: {depth}",
                    "failure_analysis.root_cause_depth",
                    "INVALID_ROOT_CAUSE_DEPTH",
                    "Use 3 for moderate analysis depth"
                )
        
        if 'report_format' in failure_analysis:
            report_format = failure_analysis['report_format']
            if report_format not in self.VALID_REPORT_FORMATS:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid report_format: '{report_format}'",
                    "failure_analysis.report_format",
                    "INVALID_REPORT_FORMAT",
                    f"Valid formats: {', '.join(self.VALID_REPORT_FORMATS)}"
                )
    
    def _validate_workspace_section(self, workspace: Dict[str, Any]) -> None:
        """Validate workspace configuration section."""
        required_dirs = ['analytics_dir', 'reports_dir', 'cache_dir', 'artifacts_dir']
        
        for dir_key in required_dirs:
            if dir_key in workspace:
                dir_value = workspace[dir_key]
                if not isinstance(dir_value, str) or not dir_value.strip():
                    self._add_issue(
                        ValidationSeverity.ERROR,
                        f"{dir_key} must be a non-empty string",
                        f"workspace.{dir_key}",
                        "INVALID_DIRECTORY_NAME",
                        f"Example: '{dir_key.replace('_dir', '')}'"
                    )
    
    def _validate_cross_section_compatibility(self, config: Dict[str, Any]) -> None:
        """Validate compatibility between different configuration sections."""
        # Check if export format is compatible with processor outputs
        if 'export' in config and 'processors' in config:
            export_format = config['export'].get('default_format', 'csv')
            
            # Some processors may have specific output requirements
            if 'failures' in config['processors'] and export_format == 'excel':
                # Check if failure analysis produces complex outputs that need special handling
                analysis_types = config['processors']['failures'].get('analysis_types', [])
                if 'root_cause' in analysis_types:
                    self._add_issue(
                        ValidationSeverity.WARNING,
                        "Root cause analysis may produce complex outputs that don't export well to Excel",
                        "export.default_format",
                        "EXPORT_COMPATIBILITY_WARNING",
                        "Consider using JSON format for complex analysis results"
                    )
        
        # Check timeout consistency
        if 'export' in config and 'query_timeout' in config:
            export_timeout = config['export'].get('export_timeout', 120)
            query_timeout = config.get('query_timeout', 300)
            
            if export_timeout > query_timeout:
                self._add_issue(
                    ValidationSeverity.WARNING,
                    f"Export timeout ({export_timeout}s) is longer than query timeout ({query_timeout}s)",
                    "export.export_timeout",
                    "TIMEOUT_INCONSISTENCY",
                    "Consider adjusting timeouts to be consistent"
                )
    
    def _validate_inheritance_patterns(self, config: Dict[str, Any]) -> None:
        """Validate configuration inheritance patterns."""
        # Check for potentially problematic inheritance patterns
        if 'processors' in config:
            processors = config['processors']
            
            # Check if all processors have compatible settings
            confidence_levels = []
            for proc_name, proc_config in processors.items():
                if 'confidence_level' in proc_config:
                    confidence_levels.append((proc_name, proc_config['confidence_level']))
            
            if len(confidence_levels) > 1:
                levels = [level for _, level in confidence_levels]
                if len(set(levels)) > 1:
                    self._add_issue(
                        ValidationSeverity.INFO,
                        "Different confidence levels across processors may affect result consistency",
                        "processors",
                        "INCONSISTENT_CONFIDENCE_LEVELS",
                        "Consider using consistent confidence levels across all processors"
                    )
    
    def _find_inheritance_conflicts(self, parent_config: Dict[str, Any], 
                                   child_config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Find potential conflicts in configuration inheritance."""
        conflicts = {}
        
        def check_conflicts(parent_dict, child_dict, path=""):
            local_conflicts = {}
            for key, child_value in child_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if key in parent_dict:
                    parent_value = parent_dict[key]
                    
                    if isinstance(parent_value, dict) and isinstance(child_value, dict):
                        # Recurse into nested dictionaries
                        nested_conflicts = check_conflicts(parent_value, child_value, current_path)
                        local_conflicts.update(nested_conflicts)
                    elif parent_value != child_value:
                        # Direct value override
                        local_conflicts[current_path] = {
                            'description': f"Child overrides parent value: {parent_value} -> {child_value}",
                            'suggestion': f"Verify that overriding {key} is intentional"
                        }
            return local_conflicts
        
        conflicts = check_conflicts(parent_config, child_config)
        return conflicts
    
    def _merge_configs(self, parent_config: Dict[str, Any], 
                      child_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge parent and child configurations."""
        def deep_merge(parent, child):
            result = parent.copy()
            for key, value in child.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(parent_config, child_config)
    
    def _add_issue(self, severity: ValidationSeverity, message: str, path: str, 
                  code: str, suggestion: Optional[str] = None) -> None:
        """Add a validation issue to the current validation session."""
        self.issues.append(ValidationIssue(
            severity=severity,
            message=message,
            path=path,
            code=code,
            suggestion=suggestion
        ))


# Convenience functions for common validation scenarios

def validate_analytics_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate a complete analytics configuration.
    
    Args:
        config: Analytics configuration dictionary
        
    Returns:
        ValidationResult with validation status and issues
    """
    validator = AnalyticsConfigValidator()
    return validator.validate_analytics_config(config)


def validate_processor_config(processor_type: str, config: Dict[str, Any]) -> ValidationResult:
    """
    Validate configuration for a specific processor type.
    
    Args:
        processor_type: Type of processor
        config: Processor configuration
        
    Returns:
        ValidationResult with validation status and issues
    """
    validator = AnalyticsConfigValidator()
    return validator.validate_processor_config(processor_type, config)


def validate_config_inheritance(parent_config: Dict[str, Any], 
                               child_config: Dict[str, Any]) -> ValidationResult:
    """
    Validate configuration inheritance.
    
    Args:
        parent_config: Parent configuration
        child_config: Child configuration
        
    Returns:
        ValidationResult with validation status and issues
    """
    validator = AnalyticsConfigValidator()
    return validator.validate_inheritance(parent_config, child_config) 