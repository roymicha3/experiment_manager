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

__all__ = [
    'AnalyticsFactory'
]

__version__ = '1.0.0' 