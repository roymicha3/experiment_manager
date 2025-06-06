"""
Analytics Default Configuration Manager

Provides sensible defaults for analytics operations, making configuration optional
for basic usage scenarios. Includes comprehensive documentation and examples.
"""

import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from omegaconf import DictConfig, OmegaConf

from experiment_manager.common.serializable import YAMLSerializable


class ConfigurationLevel(Enum):
    """Configuration levels for different use cases."""
    MINIMAL = "minimal"      # Bare minimum for basic usage
    STANDARD = "standard"    # Recommended settings for most users
    ADVANCED = "advanced"    # Full configuration with all options
    RESEARCH = "research"    # Optimized for research workflows


@YAMLSerializable.register("DefaultConfigurationManager")
class DefaultConfigurationManager(YAMLSerializable):
    """
    Manages default configurations for analytics operations.
    
    Provides different configuration levels tailored to various use cases,
    from minimal setup for quick starts to comprehensive research-grade configurations.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def get_minimal_config(cls) -> DictConfig:
        """
        Get minimal configuration for basic analytics usage.
        
        Best for: Quick start, learning, small experiments
        
        Returns:
            DictConfig: Minimal analytics configuration
        """
        return DictConfig({
            "processors": {
                "statistics": {
                    "confidence_level": 0.95,
                    "percentiles": [25, 50, 75, 95],
                    "missing_strategy": "drop",
                    "include_advanced": False
                }
            },
            
            "aggregation": {
                "default_functions": ["mean", "std", "count"],
                "group_by_defaults": ["experiment_name"],
                "metric_columns": ["loss", "accuracy"]
            },
            
            "export": {
                "default_format": "csv",
                "output_directory": "analytics_outputs",
                "include_metadata": False
            },
            
            "workspace": {
                "analytics_dir": "analytics",
                "auto_create_dirs": True
            }
        })
    
    @classmethod
    def get_standard_config(cls) -> DictConfig:
        """
        Get standard configuration for most production use cases.
        
        Best for: Production experiments, team collaboration
        
        Returns:
            DictConfig: Standard analytics configuration
        """
        return DictConfig({
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
                    "zscore_threshold": 3.0,
                    "action": "exclude"
                },
                
                "failures": {
                    "failure_threshold": 0.1,
                    "min_samples": 10,
                    "time_window": "day",
                    "analysis_types": ["rates", "correlations"]
                }
            },
            
            "aggregation": {
                "default_functions": ["mean", "median", "std", "min", "max", "count"],
                "group_by_defaults": ["experiment_name", "trial_name"],
                "metric_columns": ["metric_total_val", "loss", "accuracy"]
            },
            
            "export": {
                "default_format": "csv",
                "output_directory": "analytics_outputs",
                "include_metadata": True,
                "include_timestamps": True,
                "compression": False
            },
            
            "database_connection": None,
            "query_timeout": 600,
            "result_caching": True,
            "batch_size": 2000,
            
            "failure_analysis": {
                "enabled": True,
                "auto_detect": True,
                "correlation_threshold": 0.5,
                "temporal_window": "day",
                "root_cause_depth": 5,
                "report_format": "summary"
            },
            
            "workspace": {
                "analytics_dir": "analytics",
                "reports_dir": "reports",
                "cache_dir": "cache",
                "artifacts_dir": "artifacts",
                "auto_create_dirs": True
            }
        })
    
    @classmethod
    def get_advanced_config(cls) -> DictConfig:
        """
        Get advanced configuration for complex analysis scenarios.
        
        Best for: Complex experiments, performance optimization
        
        Returns:
            DictConfig: Advanced analytics configuration
        """
        return DictConfig({
            "processors": {
                "statistics": {
                    "confidence_level": 0.95,
                    "percentiles": [5, 25, 50, 75, 90, 95, 99],
                    "missing_strategy": "drop",
                    "include_advanced": True
                },
                
                "outliers": {
                    "default_method": "iqr",
                    "iqr_factor": 1.5,
                    "zscore_threshold": 3.0,
                    "modified_zscore_threshold": 3.5,
                    "action": "exclude"
                },
                
                "failures": {
                    "failure_threshold": 0.05,
                    "min_samples": 20,
                    "time_window": "hour",
                    "analysis_types": ["rates", "correlations", "temporal", "root_cause"],
                    "config_columns": ["optimizer", "model", "dataset", "learning_rate", "batch_size"]
                },
                
                "comparisons": {
                    "confidence_level": 0.95,
                    "significance_threshold": 0.05,
                    "min_samples": 5,
                    "comparison_types": ["pairwise", "ranking", "ab_test", "trend"],
                    "baseline_selection": "auto"
                }
            },
            
            "aggregation": {
                "default_functions": [
                    "mean", "median", "std", "min", "max", "count",
                    "skew", "kurt"
                ],
                "group_by_defaults": [
                    "experiment_name", "trial_name", "model", "optimizer"
                ],
                "metric_columns": [
                    "metric_total_val", "loss", "accuracy", "f1_score",
                    "precision", "recall"
                ]
            },
            
            "export": {
                "default_format": "parquet",
                "output_directory": "analytics_outputs",
                "include_metadata": True,
                "compression": True,
                "export_timeout": 300
            },
            
            "database_connection": None,
            "query_timeout": 600,
            "result_caching": True,
            "batch_size": 5000,
            
            "failure_analysis": {
                "enabled": True,
                "auto_detect": True,
                "correlation_threshold": 0.3,
                "temporal_window": "hour",
                "root_cause_depth": 8,
                "report_format": "detailed"
            },
            
            "workspace": {
                "analytics_dir": "analytics",
                "reports_dir": "reports",
                "cache_dir": "cache",
                "artifacts_dir": "artifacts",
                "auto_create_dirs": True
            }
        })
    
    @classmethod
    def get_research_config(cls) -> DictConfig:
        """
        Get research-grade configuration for comprehensive analysis.
        
        Best for: Academic research, publication-quality analysis
        
        Returns:
            DictConfig: Research analytics configuration
        """
        return DictConfig({
            "processors": {
                "statistics": {
                    "confidence_level": 0.99,
                    "percentiles": [1, 5, 10, 25, 50, 75, 90, 95, 99],
                    "missing_strategy": "keep",  # Keep for research transparency
                    "include_advanced": True
                },
                
                "outliers": {
                    "default_method": "modified_zscore",
                    "iqr_factor": 3.0,  # More conservative for research
                    "zscore_threshold": 3.0,
                    "modified_zscore_threshold": 3.5,
                    "action": "flag"   # Flag rather than exclude for transparency
                },
                
                "failures": {
                    "failure_threshold": 0.01,  # Very sensitive for research
                    "min_samples": 30,
                    "time_window": "hour",
                    "analysis_types": ["rates", "correlations", "temporal", "root_cause"],
                    "config_columns": [
                        "optimizer", "model", "dataset", "learning_rate", "batch_size",
                        "architecture", "regularization", "scheduler", "loss_function"
                    ]
                },
                
                "comparisons": {
                    "confidence_level": 0.99,
                    "significance_threshold": 0.01,
                    "min_samples": 10,
                    "comparison_types": ["pairwise", "ranking", "ab_test", "trend"],
                    "baseline_selection": "largest"
                }
            },
            
            "aggregation": {
                "default_functions": [
                    "mean", "median", "std", "min", "max", "count",
                    "skew", "kurt", "var", "sem", "mad"
                ],
                "group_by_defaults": [
                    "experiment_name", "trial_name", "model", "optimizer",
                    "dataset", "architecture"
                ],
                "metric_columns": [
                    "metric_total_val", "loss", "accuracy", "f1_score",
                    "precision", "recall", "auc", "mse", "mae"
                ]
            },
            
            "export": {
                "default_format": "parquet",
                "output_directory": "analytics_outputs",
                "include_metadata": True,
                "compression": True,
                "export_timeout": 600
            },
            
            "database_connection": None,
            "query_timeout": 1200,
            "result_caching": True,
            "batch_size": 10000,
            
            "failure_analysis": {
                "enabled": True,
                "auto_detect": True,
                "correlation_threshold": 0.5,
                "temporal_window": "hour",
                "root_cause_depth": 10,
                "report_format": "full"
            },
            
            "workspace": {
                "analytics_dir": "analytics",
                "reports_dir": "reports",
                "cache_dir": "cache",
                "artifacts_dir": "artifacts",
                "auto_create_dirs": True
            }
        })
    
    @classmethod
    def get_config_by_level(cls, level: Union[str, ConfigurationLevel]) -> DictConfig:
        """
        Get configuration by level name or enum.
        
        Args:
            level: Configuration level (string or ConfigurationLevel enum)
            
        Returns:
            DictConfig: Configuration for the specified level
            
        Raises:
            ValueError: If level is not supported
        """
        if isinstance(level, str):
            try:
                level = ConfigurationLevel(level.lower())
            except ValueError:
                valid_levels = [l.value for l in ConfigurationLevel]
                raise ValueError(f"Invalid configuration level: '{level}'. Valid levels: {valid_levels}")
        
        level_map = {
            ConfigurationLevel.MINIMAL: cls.get_minimal_config,
            ConfigurationLevel.STANDARD: cls.get_standard_config,
            ConfigurationLevel.ADVANCED: cls.get_advanced_config,
            ConfigurationLevel.RESEARCH: cls.get_research_config
        }
        
        return level_map[level]()
    
    @classmethod
    def create_default_analytics_config_file(cls, 
                                           output_path: Optional[str] = None,
                                           level: Union[str, ConfigurationLevel] = ConfigurationLevel.STANDARD,
                                           include_documentation: bool = True) -> str:
        """
        Create a default analytics configuration file with documentation.
        
        Args:
            output_path: Path where to save the config file (default: analytics_config.yaml)
            level: Configuration level to use
            include_documentation: Whether to include inline documentation
            
        Returns:
            str: Path to the created configuration file
        """
        if output_path is None:
            output_path = "analytics_config.yaml"
        
        config = cls.get_config_by_level(level)
        
        if include_documentation:
            # Add comprehensive documentation to the configuration
            documented_content = cls._add_configuration_documentation(config, level)
            
            # Write the documented configuration
            with open(output_path, 'w') as f:
                f.write(documented_content)
        else:
            # Save standard YAML without extra documentation
            OmegaConf.save(config, output_path)
        
        return output_path
    
    @classmethod
    def _add_configuration_documentation(cls, config: DictConfig, level: Union[str, ConfigurationLevel]) -> str:
        """
        Add comprehensive documentation to configuration content.
        
        Args:
            config: Configuration to document
            level: Configuration level for level-specific documentation
            
        Returns:
            str: YAML content with documentation
        """
        # Convert string to enum if needed
        if isinstance(level, str):
            level = ConfigurationLevel(level.lower())
        
        header = f"""#####################################################################################
#   Analytics Configuration - {level.value.title()} Level
#####################################################################################
#
# This configuration provides {cls._get_level_description(level)}
#
# Configuration Level: {level.value.upper()}
# Generated by: Experiment Manager Analytics System
#
# For more information, see the Analytics Configuration Guide:
# https://github.com/your-org/experiment-manager/blob/main/docs/analytics_config.md
#
#####################################################################################

"""
        
        # Convert config to YAML string
        yaml_content = OmegaConf.to_yaml(config)
        
        # Add section comments
        documented_yaml = cls._add_section_comments(yaml_content, level)
        
        footer = f"""
#####################################################################################
# Environment Variable Overrides (prefix with VIZ_ANALYTICS_):                    #
# Examples:                                                                        #
# - VIZ_ANALYTICS_PROCESSORS_STATISTICS_CONFIDENCE_LEVEL=0.99                    #
# - VIZ_ANALYTICS_EXPORT_DEFAULT_FORMAT=json                                      #
# - VIZ_ANALYTICS_FAILURE_ANALYSIS_ENABLED=false                                  #
#                                                                                  #
# For complete documentation and examples, visit:                                 #
# https://github.com/your-org/experiment-manager/blob/main/docs/analytics.md     #
#####################################################################################
"""
        
        return header + documented_yaml + footer
    
    @classmethod
    def _get_level_description(cls, level: ConfigurationLevel) -> str:
        """Get description for configuration level."""
        descriptions = {
            ConfigurationLevel.MINIMAL: "basic analytics with minimal configuration for quick start",
            ConfigurationLevel.STANDARD: "comprehensive analytics with recommended settings for most users",
            ConfigurationLevel.ADVANCED: "full analytics functionality with advanced features enabled",
            ConfigurationLevel.RESEARCH: "research-optimized analytics with extensive analysis capabilities"
        }
        return descriptions[level]
    
    @classmethod
    def _add_section_comments(cls, yaml_content: str, level: ConfigurationLevel) -> str:
        """Add detailed comments to YAML sections."""
        lines = yaml_content.split('\n')
        documented_lines = []
        
        for line in lines:
            # Add comments for major sections
            if line.startswith('processors:'):
                documented_lines.append('# Analytics processors configuration')
                documented_lines.append('# Enable different types of analysis (statistics, outliers, failures, comparisons)')
            elif line.startswith('aggregation:'):
                documented_lines.append('')
                documented_lines.append('# Data aggregation settings')
                documented_lines.append('# Define how metrics are grouped and aggregated')
            elif line.startswith('export:'):
                documented_lines.append('')
                documented_lines.append('# Export configuration for analysis results')
                documented_lines.append('# Controls output formats and destinations')
            elif line.startswith('failure_analysis:'):
                documented_lines.append('')
                documented_lines.append('# Failure analysis parameters')
                documented_lines.append('# Settings for detecting and analyzing experiment failures')
            elif line.startswith('workspace:'):
                documented_lines.append('')
                documented_lines.append('# Workspace and artifact management')
                documented_lines.append('# Directory structure for analytics outputs')
            
            documented_lines.append(line)
        
        return '\n'.join(documented_lines)
    
    @classmethod
    def get_available_levels(cls) -> List[str]:
        """
        Get list of available configuration levels.
        
        Returns:
            List[str]: Available configuration level names
        """
        return [level.value for level in ConfigurationLevel]
    
    @classmethod 
    def get_level_documentation(cls, level: Union[str, ConfigurationLevel]) -> Dict[str, Any]:
        """
        Get comprehensive documentation for a configuration level.
        
        Args:
            level: Configuration level to document
            
        Returns:
            Dict containing level documentation
        """
        if isinstance(level, str):
            level = ConfigurationLevel(level.lower())
        
        config = cls.get_config_by_level(level)
        
        docs = {
            'level': level.value,
            'description': cls._get_level_description(level),
            'use_cases': cls._get_level_use_cases(level),
            'enabled_processors': list(config.get('processors', {}).keys()),
            'key_features': cls._get_level_features(level),
            'configuration': OmegaConf.to_container(config, resolve=True)
        }
        
        return docs
    
    @classmethod
    def _get_level_use_cases(cls, level: ConfigurationLevel) -> List[str]:
        """Get use cases for configuration level."""
        use_cases = {
            ConfigurationLevel.MINIMAL: [
                "Quick start with basic analytics",
                "Small-scale experiments",
                "Learning and exploration",
                "Minimal computational overhead"
            ],
            ConfigurationLevel.STANDARD: [
                "Production experiments",
                "Team collaboration",
                "Regular model development",
                "Automated experiment monitoring"
            ],
            ConfigurationLevel.ADVANCED: [
                "Complex experiment analysis",
                "Performance optimization",
                "Detailed failure investigation",
                "Multi-dimensional comparisons"
            ],
            ConfigurationLevel.RESEARCH: [
                "Academic research publications",
                "Comprehensive experiment analysis",
                "Statistical significance testing",
                "Reproducible research workflows"
            ]
        }
        return use_cases[level]
    
    @classmethod
    def _get_level_features(cls, level: ConfigurationLevel) -> List[str]:
        """Get key features for configuration level."""
        features = {
            ConfigurationLevel.MINIMAL: [
                "Basic statistical analysis",
                "Simple CSV exports",
                "Essential metrics tracking",
                "Automatic directory creation"
            ],
            ConfigurationLevel.STANDARD: [
                "Statistical analysis with confidence intervals",
                "Outlier detection",
                "Basic failure analysis",
                "Multiple export formats",
                "Result caching"
            ],
            ConfigurationLevel.ADVANCED: [
                "All standard features",
                "Advanced failure analysis",
                "Cross-experiment comparisons",
                "Custom aggregation functions",
                "Compressed exports",
                "Detailed reporting"
            ],
            ConfigurationLevel.RESEARCH: [
                "All advanced features",
                "Research-grade statistics",
                "Extensive outlier detection",
                "Comprehensive failure analysis",
                "Publication-ready reports",
                "Full transparency and reproducibility"
            ]
        }
        return features[level]
    
    @classmethod
    def from_config(cls, config: DictConfig):
        """Create instance from configuration."""
        return cls() 