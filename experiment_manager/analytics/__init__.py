"""
Analytics Module for Experiment Manager

This module provides comprehensive analytics capabilities including:
- Statistical analysis processors
- Outlier detection and handling
- Failure analysis and correlation
- Data comparison and A/B testing
- Analytics factory for processor creation
"""

from experiment_manager.analytics.api import ExperimentAnalytics
from experiment_manager.analytics.analytics_factory import AnalyticsFactory
from experiment_manager.analytics.defaults import DefaultConfigurationManager, ConfigurationLevel
from experiment_manager.analytics.query import AnalyticsQuery, ValidationError
from experiment_manager.analytics.results import AnalyticsResult, QueryMetadata
from experiment_manager.analytics.export import (
    ResultExporter, ResultVisualizer, AnalyticsReportGenerator,
    ExportFormat, VisualizationType, ExportOptions, VisualizationOptions
)

__all__ = [
    'ExperimentAnalytics',
    'AnalyticsFactory',
    'DefaultConfigurationManager',
    'ConfigurationLevel',
    'AnalyticsQuery',
    'ValidationError',
    'AnalyticsResult',
    'QueryMetadata',
    'ResultExporter',
    'ResultVisualizer', 
    'AnalyticsReportGenerator',
    'ExportFormat',
    'VisualizationType',
    'ExportOptions',
    'VisualizationOptions'
]

__version__ = '1.0.0' 