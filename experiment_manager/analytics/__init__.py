"""
Experiment Analytics Module

This module provides sophisticated data analysis capabilities for experiment results,
including statistical analysis, outlier detection, failure analysis, and cross-experiment comparisons.

Main Components:
- AnalyticsEngine: Central orchestrator for analytics operations
- AnalyticsQuery: Fluent query builder API
- AnalyticsResult: Comprehensive result container with export capabilities
- ExperimentAnalytics: High-level user-facing API
"""

from .engine import AnalyticsEngine
from .query_builder import AnalyticsQuery
from .results import AnalyticsResult
from .api import ExperimentAnalytics

__all__ = [
    'AnalyticsEngine',
    'AnalyticsQuery', 
    'AnalyticsResult',
    'ExperimentAnalytics'
] 