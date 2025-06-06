"""
Analytics Processors Module

This module contains data processing classes for analytics operations:
- DataProcessor: Base class for all processors
- StatisticsProcessor: Statistical analysis and metrics
- OutlierProcessor: Outlier detection and handling
- FailureAnalyzer: Failure analysis and correlation
- ComparisonProcessor: Data comparison and A/B testing
"""

from experiment_manager.analytics.processors.base import DataProcessor
from experiment_manager.analytics.processors.statistics import StatisticsProcessor
from experiment_manager.analytics.processors.outliers import OutlierProcessor
from experiment_manager.analytics.processors.failures import FailureAnalyzer
from experiment_manager.analytics.processors.comparisons import ComparisonProcessor

__all__ = [
    'DataProcessor',
    'StatisticsProcessor', 
    'OutlierProcessor',
    'FailureAnalyzer',
    'ComparisonProcessor'
]

__version__ = '1.0.0' 