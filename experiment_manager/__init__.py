"""
Experiment Manager - A Python package for orchestrating ML experiments.

Basic usage:
    from experiment_manager import Experiment, Pipeline, Callback, Metric
"""

__version__ = "0.1.0"

# Core classes
from .experiment import Experiment
from .trial import Trial
from .environment import Environment

# Common enums and utilities
from .common import (
    Metric,
    MetricCategory,
    RunStatus,
    Level,
    FactoryRegistry,
    FactoryType,
)

# Pipeline classes
from .pipelines import Pipeline, Callback

__all__ = [
    # Version
    '__version__',
    # Core
    'Experiment',
    'Trial',
    'Environment',
    # Enums
    'Metric',
    'MetricCategory',
    'RunStatus',
    'Level',
    # Factory
    'FactoryRegistry',
    'FactoryType',
    # Pipeline
    'Pipeline',
    'Callback',
]
