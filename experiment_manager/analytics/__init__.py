"""
Analytics Module for Experiment Manager

This module provides comprehensive analytics capabilities including:
- Statistical analysis processors
- Outlier detection and handling
- Failure analysis and correlation
- Data comparison and A/B testing
- Analytics factory for processor creation
"""

from experiment_manager.analytics.analytics_factory import AnalyticsFactory
from experiment_manager.analytics.defaults import DefaultConfigurationManager, ConfigurationLevel
from experiment_manager.analytics.query import AnalyticsQuery, ValidationError
from experiment_manager.analytics.results import AnalyticsResult, QueryMetadata

__all__ = [
    'AnalyticsFactory',
    'DefaultConfigurationManager',
    'ConfigurationLevel',
    'AnalyticsQuery',
    'ValidationError',
    'AnalyticsResult',
    'QueryMetadata'
]

__version__ = '1.0.0' 