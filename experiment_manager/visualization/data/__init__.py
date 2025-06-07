"""
Data Processing Pipeline for Visualization Components

This module provides a robust, chainable data processing pipeline system
for preparing visualization data. The pipeline supports caching, error 
handling, rollback capabilities, and performance monitoring.
"""

from .pipeline import (
    DataPipeline,
    DataProcessor,
    ProcessingContext,
    ProcessingResult,
    CacheStrategy,
    PerformanceLevel,
    ResourceMetrics,
    PerformanceProfile,
    PerformanceMonitor,
    PipelineExecutionError,
    ProcessorValidationError,
    ProcessorRegistrationError,
    PerformanceCallback,
    ResourceCallback,
    ThresholdCallback
)

# Import basic processors from basic_processors.py module
from .basic_processors import (
    FilterProcessor,
    AggregationProcessor, 
    NormalizationProcessor,
    get_builtin_processors,
    register_builtin_processors
)

# Import specialized processors from processors package
from .processors.time_series_smoother import TimeSeriesSmoother
from .processors.missing_data_imputer import MissingDataImputer
from .processors.outlier_detector import OutlierDetector
from .processors.metric_normalizer import MetricNormalizer
from .processors import get_specialized_processors, register_specialized_processors

# Import cache system
from .cache import (
    DataCache,
    CacheConfig,
    CacheEntry,
    CacheMetrics,
    EvictionPolicy,
    InvalidationStrategy,
    MemoryCacheBackend,
    CacheManager,
    cache_manager
)

# Import analytics adapter
from .analytics_adapter import (
    AnalyticsDataAdapter,
    QueryOptimization,
    StreamingConfig,
    DataSourceConfig,
    DataSourceInterface,
    AnalyticsEngineSource,
    DatabaseDirectSource,
    create_analytics_adapter
)

__all__ = [
    # Core pipeline
    'DataPipeline',
    'DataProcessor', 
    'ProcessingContext',
    'ProcessingResult',
    'CacheStrategy',
    
    # Performance monitoring
    'PerformanceLevel',
    'ResourceMetrics',
    'PerformanceProfile',
    'PerformanceMonitor',
    'PerformanceCallback',
    'ResourceCallback',
    'ThresholdCallback',
    
    # Exceptions
    'PipelineExecutionError',
    'ProcessorValidationError',
    'ProcessorRegistrationError',
    
    # Built-in processors
    'FilterProcessor',
    'AggregationProcessor',
    'NormalizationProcessor',
    'get_builtin_processors',
    'register_builtin_processors',
    
    # Specialized processors
    'TimeSeriesSmoother',
    'MissingDataImputer',
    'OutlierDetector',
    'MetricNormalizer',
    'get_specialized_processors',
    'register_specialized_processors',
    
    # Cache system
    'DataCache',
    'CacheConfig',
    'CacheEntry',
    'CacheMetrics',
    'EvictionPolicy',
    'InvalidationStrategy',
    'MemoryCacheBackend',
    'CacheManager',
    'cache_manager',
] 