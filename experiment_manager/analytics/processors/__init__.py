"""
Analytics Data Processors

This module contains specialized data processing components for analytics operations.
All processors implement the DataProcessor interface for consistency and extensibility.

Base Components:
- DataProcessor: Abstract base class for all data processors
- ProcessorManager: Manager for coordinating multiple processors

Processor Types:
- StatisticsProcessor: Basic and advanced statistical calculations
- OutlierProcessor: Multiple outlier detection methods
- FailureAnalyzer: Failure pattern analysis and correlation detection
- ComparisonProcessor: Cross-experiment comparative analysis
"""

from .base import DataProcessor, ProcessorManager

__all__ = [
    'DataProcessor',
    'ProcessorManager'
] 