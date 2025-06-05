"""
Experiment Manager package initialization.
"""

__version__ = "0.1.0"

# Import main components for easy access
from .experiment import Experiment
from .trial import Trial
from .environment import Environment

# Import analytics module components
from .analytics import ExperimentAnalytics, AnalyticsEngine, AnalyticsQuery, AnalyticsResult

__all__ = [
    'Experiment',
    'Trial', 
    'Environment',
    'ExperimentAnalytics',
    'AnalyticsEngine',
    'AnalyticsQuery',
    'AnalyticsResult'
]
