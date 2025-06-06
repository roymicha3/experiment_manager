# Analytics Configuration Guide

## Overview

The Experiment Manager analytics system provides powerful data analysis capabilities with flexible configuration options. This guide covers everything you need to know about configuring analytics for your experiments.

## Quick Start

### Minimal Configuration (Zero Setup)

The analytics system works out of the box with no configuration required:

```python
from experiment_manager.analytics import AnalyticsFactory
from experiment_manager.analytics.defaults import DefaultConfigurationManager

# Use minimal defaults - perfect for getting started
config = DefaultConfigurationManager.get_minimal_config()
processors = AnalyticsFactory.create_from_config(config)

# Basic statistical analysis is ready to use
stats = processors['statistics'].process(your_data)
```

### Generate Configuration File

Create a configuration file for your project:

```python
from experiment_manager.analytics.defaults import DefaultConfigurationManager

# Create standard configuration file
config_path = DefaultConfigurationManager.create_default_analytics_config_file(
    output_path="analytics_config.yaml",
    level="standard",  # minimal, standard, advanced, research
    include_documentation=True
)
```

## Configuration Levels

The analytics system provides four configuration levels to match different use cases:

### 1. Minimal Level
**Best for:** Learning, small experiments, minimal setup

```yaml
processors:
  statistics:
    confidence_level: 0.95
    percentiles: [25, 50, 75, 95]
    missing_strategy: "drop"
    include_advanced: false

aggregation:
  default_functions: ["mean", "std", "count"]
  group_by_defaults: ["experiment_name"]
  metric_columns: ["loss", "accuracy"]

export:
  default_format: "csv"
  output_directory: "analytics_outputs"
  include_metadata: false
```

**Features:**
- Basic statistical analysis
- Simple CSV exports
- Essential metrics tracking
- Automatic directory creation

### 2. Standard Level (Recommended)
**Best for:** Production experiments, team collaboration

```yaml
processors:
  statistics:
    confidence_level: 0.95
    percentiles: [25, 50, 75, 90, 95]
    missing_strategy: "drop"
    include_advanced: true
  
  outliers:
    default_method: "iqr"
    iqr_factor: 1.5
    zscore_threshold: 3.0
    action: "exclude"
  
  failures:
    failure_threshold: 0.1
    min_samples: 10
    time_window: "day"
    analysis_types: ["rates", "correlations"]

aggregation:
  default_functions: ["mean", "median", "std", "min", "max", "count"]
  group_by_defaults: ["experiment_name", "trial_name"]
  metric_columns: ["metric_total_val", "loss", "accuracy"]

export:
  default_format: "csv"
  output_directory: "analytics_outputs"
  include_metadata: true
  include_timestamps: true
  compression: false
```

**Features:**
- Statistical analysis with confidence intervals
- Outlier detection
- Basic failure analysis
- Multiple export formats
- Result caching

### 3. Advanced Level
**Best for:** Complex analysis, performance optimization

```yaml
processors:
  statistics:
    confidence_level: 0.95
    percentiles: [5, 25, 50, 75, 90, 95, 99]
    missing_strategy: "drop"
    include_advanced: true
  
  outliers:
    default_method: "iqr"
    iqr_factor: 1.5
    zscore_threshold: 3.0
    modified_zscore_threshold: 3.5
    action: "exclude"
  
  failures:
    failure_threshold: 0.05
    min_samples: 20
    time_window: "hour"
    analysis_types: ["rates", "correlations", "temporal", "root_cause"]
    config_columns: ["optimizer", "model", "dataset", "learning_rate", "batch_size"]
  
  comparisons:
    confidence_level: 0.95
    significance_threshold: 0.05
    min_samples: 5
    comparison_types: ["pairwise", "ranking", "ab_test", "trend"]
    baseline_selection: "auto"

export:
  default_format: "parquet"
  output_directory: "analytics_outputs"
  include_metadata: true
  compression: true
  export_timeout: 300
```

**Features:**
- All standard features
- Advanced failure analysis
- Cross-experiment comparisons
- Custom aggregation functions
- Compressed exports
- Detailed reporting

### 4. Research Level
**Best for:** Academic research, publications, comprehensive analysis

```yaml
processors:
  statistics:
    confidence_level: 0.99
    percentiles: [1, 5, 10, 25, 50, 75, 90, 95, 99]
    missing_strategy: "keep"  # Keep for research transparency
    include_advanced: true
  
  outliers:
    default_method: "modified_zscore"
    iqr_factor: 3.0  # More conservative for research
    action: "flag"   # Flag rather than exclude for transparency
  
  failures:
    failure_threshold: 0.01  # Very sensitive for research
    min_samples: 30
    analysis_types: ["rates", "correlations", "temporal", "root_cause"]
    config_columns: [
      "optimizer", "model", "dataset", "learning_rate", "batch_size",
      "architecture", "regularization", "scheduler", "loss_function"
    ]
  
  comparisons:
    confidence_level: 0.99
    significance_threshold: 0.01
    min_samples: 10
    comparison_types: ["pairwise", "ranking", "ab_test", "trend"]
    baseline_selection: "largest"

aggregation:
  default_functions: [
    "mean", "median", "std", "min", "max", "count",
    "skew", "kurt", "var", "sem", "mad"
  ]
  group_by_defaults: [
    "experiment_name", "trial_name", "model", "optimizer",
    "dataset", "architecture"
  ]
```

**Features:**
- All advanced features
- Research-grade statistics
- Extensive outlier detection
- Comprehensive failure analysis
- Publication-ready reports
- Full transparency and reproducibility

## Configuration Sections

### Processors Configuration

#### Statistics Processor
Provides comprehensive statistical analysis of experiment results.

```yaml
processors:
  statistics:
    confidence_level: 0.95           # Confidence level for intervals (0.0-1.0)
    percentiles: [25, 50, 75, 95]    # Percentiles to calculate
    missing_strategy: "drop"         # How to handle missing data: drop, fill_mean, fill_median, keep
    include_advanced: true           # Include skewness, kurtosis, etc.
```

**Missing Strategy Options:**
- `drop`: Remove rows with missing values (default for most cases)
- `fill_mean`: Replace missing values with column mean
- `fill_median`: Replace missing values with column median  
- `keep`: Keep missing values (useful for research transparency)

#### Outlier Detection Processor
Detects and handles outliers in your data using various methods.

```yaml
processors:
  outliers:
    default_method: "iqr"                    # Detection method: iqr, zscore, modified_zscore, custom
    iqr_factor: 1.5                         # IQR multiplier (1.5 standard, 3.0 conservative)
    zscore_threshold: 3.0                   # Z-score threshold for outlier detection
    modified_zscore_threshold: 3.5          # Modified Z-score threshold
    custom_thresholds: {}                   # Custom thresholds per metric
    action: "exclude"                       # What to do: exclude, flag, keep
```

**Detection Methods:**
- `iqr`: Interquartile Range method (recommended for most cases)
- `zscore`: Standard Z-score method
- `modified_zscore`: Modified Z-score using median absolute deviation
- `custom`: Use custom thresholds per metric

**Actions:**
- `exclude`: Remove outliers from analysis
- `flag`: Mark outliers but keep in analysis
- `keep`: No action, just detect

#### Failure Analysis Processor
Analyzes experiment failures and identifies patterns.

```yaml
processors:
  failures:
    failure_threshold: 0.1                  # Minimum failure rate to consider significant
    min_samples: 10                         # Minimum samples for reliable analysis
    time_window: "day"                      # Time window: hour, day, week, month
    analysis_types:                         # Types of analysis to perform
      - "rates"                             # Calculate failure rates
      - "correlations"                      # Find configuration correlations
      - "temporal"                          # Temporal pattern detection
      - "root_cause"                        # Root cause suggestions
    config_columns:                         # Configuration columns to analyze
      - "optimizer"
      - "model"
      - "dataset"
```

#### Comparison Processor
Enables cross-experiment comparisons and A/B testing.

```yaml
processors:
  comparisons:
    confidence_level: 0.95                  # Statistical confidence level
    significance_threshold: 0.05            # P-value threshold for significance
    min_samples: 5                          # Minimum samples per group
    comparison_types:                       # Types of comparisons
      - "pairwise"                          # Pairwise comparisons
      - "ranking"                           # Performance ranking
      - "ab_test"                           # A/B test analysis
      - "trend"                             # Trend analysis over time
    baseline_selection: "auto"              # Baseline selection: auto, first, largest, custom
```

### Aggregation Configuration

Controls how data is grouped and aggregated across experiments.

```yaml
aggregation:
  default_functions:                        # Aggregation functions to apply
    - "mean"
    - "median"
    - "std"
    - "min"
    - "max"
    - "count"
  custom_functions: {}                      # Custom aggregation functions
  group_by_defaults:                        # Default grouping columns
    - "experiment_name"
    - "trial_name"
  metric_columns:                           # Metrics to analyze
    - "metric_total_val"
    - "loss"
    - "accuracy"
```

### Export Configuration

Controls how analysis results are exported and stored.

```yaml
export:
  default_format: "csv"                     # Export format: csv, json, parquet, excel
  output_directory: "analytics_outputs"     # Output directory
  include_metadata: true                    # Include experiment metadata
  include_timestamps: true                  # Include timestamp information
  compression: false                        # Enable compression
  export_timeout: 120                       # Timeout in seconds
```

**Export Formats:**
- `csv`: Comma-separated values (widely compatible)
- `json`: JavaScript Object Notation (structured data)
- `parquet`: Apache Parquet (efficient for large datasets)
- `excel`: Microsoft Excel format (good for sharing)

### Workspace Configuration

Defines directory structure for analytics artifacts.

```yaml
workspace:
  analytics_dir: "analytics"               # Main analytics directory
  reports_dir: "reports"                   # Analysis reports
  cache_dir: "cache"                       # Cached results
  artifacts_dir: "artifacts"               # Analysis artifacts
  auto_create_dirs: true                   # Automatically create directories
```

## Environment Variable Overrides

Override any configuration value using environment variables with the `VIZ_ANALYTICS_` prefix:

```bash
# Override confidence level
export VIZ_ANALYTICS_PROCESSORS_STATISTICS_CONFIDENCE_LEVEL=0.99

# Change export format
export VIZ_ANALYTICS_EXPORT_DEFAULT_FORMAT=json

# Disable failure analysis
export VIZ_ANALYTICS_FAILURE_ANALYSIS_ENABLED=false

# Set custom output directory
export VIZ_ANALYTICS_EXPORT_OUTPUT_DIRECTORY=/path/to/custom/output
```

## Usage Examples

### Basic Usage with Defaults

```python
from experiment_manager.analytics import AnalyticsFactory
from experiment_manager.analytics.defaults import DefaultConfigurationManager

# Get default configuration
config = DefaultConfigurationManager.get_standard_config()

# Create processors
processors = AnalyticsFactory.create_from_config(config)

# Use statistical processor
stats_processor = processors['statistics']
results = stats_processor.process(your_experiment_data)

print(f"Mean accuracy: {results['mean_accuracy']:.3f}")
print(f"95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
```

### Custom Configuration

```python
from omegaconf import DictConfig
from experiment_manager.analytics import AnalyticsFactory

# Custom configuration
custom_config = DictConfig({
    "processors": {
        "statistics": {
            "confidence_level": 0.99,  # Higher confidence
            "percentiles": [10, 25, 50, 75, 90],
            "missing_strategy": "fill_median"
        },
        "outliers": {
            "default_method": "modified_zscore",
            "action": "flag"  # Flag outliers instead of excluding
        }
    }
})

# Create processors with custom config
processors = AnalyticsFactory.create_from_config(custom_config)
```

### Research Workflow Example

```python
from experiment_manager.analytics.defaults import DefaultConfigurationManager

# Generate research-grade configuration
config_path = DefaultConfigurationManager.create_default_analytics_config_file(
    output_path="research_analytics_config.yaml",
    level="research",
    include_documentation=True
)

# Load and use the configuration
from omegaconf import OmegaConf
config = OmegaConf.load(config_path)

# The research config includes:
# - Higher confidence levels (0.99)
# - More comprehensive statistics
# - Conservative outlier detection
# - Extensive failure analysis
# - Publication-ready exports
```

### Integration with Experiment Manager

```python
from experiment_manager import Environment
from experiment_manager.analytics.defaults import DefaultConfigurationManager

# Set up environment with analytics
env = Environment("my_experiment")

# Get analytics workspace info
workspace_info = env.get_analytics_workspace_info()
print(f"Analytics directory: {workspace_info['analytics_dir']}")

# Use analytics with environment
config = DefaultConfigurationManager.get_standard_config()
config.workspace.analytics_dir = env.analytics_dir

# Create analytics report
report_path = env.create_analytics_report_path("experiment_analysis.csv")
# ... perform analysis and save to report_path
```

## Best Practices

### 1. Choose the Right Configuration Level

- **Minimal**: For learning, prototyping, and simple experiments
- **Standard**: For most production use cases and team collaboration
- **Advanced**: For complex analyses and performance-critical workflows
- **Research**: For academic research and publication-quality analyses

### 2. Processor Selection Guidelines

- **Always include Statistics**: Essential for understanding experiment performance
- **Add Outliers**: Recommended for data quality and reliability
- **Include Failures**: Critical for debugging and understanding failure patterns
- **Add Comparisons**: Essential for A/B testing and cross-experiment analysis

### 3. Missing Data Strategy

- **Drop**: Safe default for most analyses
- **Fill methods**: Use when missing data is systematic and imputation makes sense
- **Keep**: Use for research transparency or when missing data is informative

### 4. Export Format Selection

- **CSV**: Best for Excel integration and human readability
- **Parquet**: Best for large datasets and performance
- **JSON**: Best for structured data and API integration
- **Excel**: Best for sharing with non-technical stakeholders

### 5. Environment Variable Usage

Use environment variables for:
- Deployment-specific settings
- CI/CD pipeline configurations
- User-specific preferences
- Sensitive configuration values

### 6. Directory Structure

```
project/
├── analytics/
│   ├── reports/           # Analysis reports
│   ├── cache/            # Cached results
│   └── artifacts/        # Plots, exports, etc.
├── analytics_config.yaml # Configuration file
└── experiments/          # Experiment data
```

## Troubleshooting

### Common Configuration Issues

1. **Import Errors**
   ```python
   # Ensure analytics module is properly installed
   from experiment_manager.analytics.defaults import DefaultConfigurationManager
   ```

2. **Validation Errors**
   ```python
   # Use validation to check configuration
   from experiment_manager.analytics.validation import validate_analytics_config
   
   result = validate_analytics_config(your_config)
   if not result.is_valid:
       for error in result.errors:
           print(f"Error: {error.message}")
   ```

3. **Missing Directory Errors**
   ```yaml
   workspace:
     auto_create_dirs: true  # Automatically create directories
   ```

4. **Performance Issues**
   ```yaml
   # Adjust batch size and caching
   batch_size: 2000
   result_caching: true
   
   # Use appropriate export format
   export:
     default_format: "parquet"  # More efficient for large data
     compression: true
   ```

### Getting Help

1. **Check validation results**: Use the validation system to identify configuration issues
2. **Review logs**: Enable debug logging to see detailed processing information
3. **Use minimal config**: Start with minimal configuration and gradually add features
4. **Check documentation**: Refer to processor-specific documentation for advanced options

## Migration from Previous Versions

### Version 0.15.x to 0.16.x

```python
# Old way (direct processor creation)
from experiment_manager.analytics.processors import StatisticsProcessor
processor = StatisticsProcessor(confidence_level=0.95)

# New way (factory pattern with configuration)
from experiment_manager.analytics import AnalyticsFactory
from omegaconf import DictConfig

config = DictConfig({"statistics": {"confidence_level": 0.95}})
processor = AnalyticsFactory.create("statistics", config)
```

### Configuration File Migration

```bash
# Convert old config format to new format
python -m experiment_manager.analytics.migrate_config \
    --input old_config.yaml \
    --output new_config.yaml \
    --level standard
```

## Advanced Topics

### Custom Processors

```python
from experiment_manager.analytics.processors.base import DataProcessor
from experiment_manager.common.serializable import YAMLSerializable

@YAMLSerializable.register("CustomProcessor")
class CustomProcessor(DataProcessor):
    def __init__(self, custom_param: float = 1.0):
        super().__init__()
        self.custom_param = custom_param
    
    def process(self, data):
        # Your custom processing logic
        return {"custom_result": data.mean() * self.custom_param}
```

### Custom Aggregation Functions

```yaml
aggregation:
  custom_functions:
    geometric_mean: "lambda x: np.exp(np.log(x).mean())"
    harmonic_mean: "lambda x: len(x) / np.sum(1.0/x)"
```

### Integration with External Tools

```python
# Export to external analysis tools
import pandas as pd

# Process with analytics
results = processor.process(data)

# Export for R analysis
results_df = pd.DataFrame(results)
results_df.to_csv("for_r_analysis.csv")

# Export for Python scientific stack
import numpy as np
np.save("for_numpy_analysis.npy", results['statistics'])
```

---

## Reference

### Complete Configuration Schema

See [Configuration Schema Reference](analytics_schema.md) for the complete YAML schema definition.

### API Documentation

See [Analytics API Reference](analytics_api.md) for detailed API documentation.

### Examples Repository

Find complete examples at [Analytics Examples](https://github.com/your-org/experiment-manager-examples/tree/main/analytics).