"""
Built-in Data Processors for Visualization Pipeline

This package provides specialized data processing implementations for common
visualization data preparation tasks, particularly for machine learning and
experiment analysis workflows.
"""

from .time_series_smoother import TimeSeriesSmoother
from .missing_data_imputer import MissingDataImputer
from .outlier_detector import OutlierDetector
from .metric_normalizer import MetricNormalizer

def get_specialized_processors():
    """Get dictionary of specialized processor classes."""
    return {
        'time_series_smoother': TimeSeriesSmoother,
        'missing_data_imputer': MissingDataImputer,
        'outlier_detector': OutlierDetector,
        'metric_normalizer': MetricNormalizer,
    }

def register_specialized_processors(pipeline):
    """Register all specialized processors with a pipeline."""
    processors = get_specialized_processors()
    for name, processor_class in processors.items():
        pipeline.register_processor(name, processor_class)

__all__ = [
    "TimeSeriesSmoother",
    "MissingDataImputer", 
    "OutlierDetector",
    "MetricNormalizer",
    "get_specialized_processors",
    "register_specialized_processors",
] 