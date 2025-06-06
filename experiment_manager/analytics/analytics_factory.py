"""
Analytics Factory for Data Processors

Provides factory pattern implementation for creating analytics processors from configuration.
Integrates with the YAMLSerializable framework for configuration-driven instantiation.
"""

from typing import Dict, List, Any, Optional, Type
from omegaconf import DictConfig

from experiment_manager.common.factory import Factory
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.analytics.processors.base import DataProcessor

# Import all processor classes to ensure they're registered
from experiment_manager.analytics.processors.statistics import StatisticsProcessor
from experiment_manager.analytics.processors.outliers import OutlierProcessor
from experiment_manager.analytics.processors.failures import FailureAnalyzer
from experiment_manager.analytics.processors.comparisons import ComparisonProcessor


class AnalyticsFactory(Factory):
    """
    Factory class for creating analytics data processors.
    
    This factory provides a unified interface for creating various analytics
    processors from configuration. It integrates with the YAMLSerializable
    framework to enable configuration-driven instantiation.
    
    Supported Processors:
    - StatisticsProcessor: Statistical analysis and calculations
    - OutlierProcessor: Outlier detection using various methods
    - FailureAnalyzer: Failure pattern analysis and root cause detection
    - ComparisonProcessor: Cross-experiment comparison and A/B testing
    """
    
    # Maps processor names to their YAMLSerializable registry keys
    PROCESSOR_REGISTRY = {
        'statistics': 'StatisticsProcessor',
        'outliers': 'OutlierProcessor', 
        'failures': 'FailureAnalyzer',
        'comparisons': 'ComparisonProcessor'
    }
    
    @staticmethod
    def create(processor_type: str, config: DictConfig = None, **kwargs) -> DataProcessor:
        """
        Create an analytics processor instance.
        
        Args:
            processor_type: Type of processor to create ('statistics', 'outliers', 'failures', 'comparisons')
            config: Configuration for the processor
            **kwargs: Additional arguments passed to processor constructor
            
        Returns:
            DataProcessor: Configured processor instance
            
        Raises:
            ValueError: If processor type is unknown or creation fails
        """
        # Normalize processor type
        processor_type = processor_type.lower().strip()
        
        # Check if processor type is supported
        if processor_type not in AnalyticsFactory.PROCESSOR_REGISTRY:
            available_types = list(AnalyticsFactory.PROCESSOR_REGISTRY.keys())
            raise ValueError(
                f"Unknown processor type: '{processor_type}'. "
                f"Available types: {available_types}"
            )
        
        # Get the registry key for YAMLSerializable
        registry_key = AnalyticsFactory.PROCESSOR_REGISTRY[processor_type]
        
        try:
            # Use parent Factory.create method with YAMLSerializable integration
            processor = Factory.create(registry_key, config or DictConfig({}))
            
            # Set additional attributes if provided
            if 'name' in kwargs:
                processor.name = kwargs['name']
            
            return processor
            
        except ValueError as e:
            # Provide more context in error message
            raise ValueError(
                f"Failed to create {processor_type} processor: {e}. "
                f"Ensure the processor class is properly registered with YAMLSerializable."
            ) from e
    
    @staticmethod 
    def create_from_config(analytics_config: DictConfig, **kwargs) -> Dict[str, DataProcessor]:
        """
        Create multiple processors from a complete analytics configuration.
        
        Args:
            analytics_config: Complete analytics configuration containing processor configs
            **kwargs: Additional arguments passed to all processors
            
        Returns:
            Dict[str, DataProcessor]: Dictionary mapping processor names to instances
            
        Example:
            ```python
            config = DictConfig({
                'processors': {
                    'statistics': {'confidence_level': 0.99},
                    'outliers': {'method': 'iqr', 'iqr_factor': 2.0}
                }
            })
            processors = AnalyticsFactory.create_from_config(config)
            # Returns: {'statistics': StatisticsProcessor(...), 'outliers': OutlierProcessor(...)}
            ```
        """
        processors = {}
        
        # Get processors configuration
        processors_config = analytics_config.get('processors', DictConfig({}))
        
        for processor_type, processor_config in processors_config.items():
            try:
                # Create processor with its specific configuration
                processor = AnalyticsFactory.create(
                    processor_type, 
                    DictConfig({processor_type: processor_config}),
                    **kwargs
                )
                processors[processor_type] = processor
                
            except ValueError as e:
                # Log warning but continue with other processors
                print(f"Warning: Failed to create {processor_type} processor: {e}")
                continue
        
        return processors
    
    @staticmethod
    def get_available_processors() -> Dict[str, str]:
        """
        Get a mapping of available processor types to their descriptions.
        
        Returns:
            Dict[str, str]: Mapping of processor types to descriptions
        """
        return {
            'statistics': 'Statistical analysis and calculations (mean, std, percentiles, confidence intervals)',
            'outliers': 'Outlier detection using IQR, Z-score, modified Z-score, and custom thresholds',
            'failures': 'Failure pattern analysis, root cause detection, and configuration correlation',
            'comparisons': 'Cross-experiment comparison, A/B testing, ranking, and trend analysis'
        }
    
    @staticmethod
    def validate_processor_config(processor_type: str, config: DictConfig) -> bool:
        """
        Validate that a processor configuration is valid.
        
        Args:
            processor_type: Type of processor to validate config for
            config: Configuration to validate
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Attempt to create processor with the configuration
            processor = AnalyticsFactory.create(processor_type, config)
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_processor_class(processor_type: str) -> Type[DataProcessor]:
        """
        Get the processor class for a given processor type.
        
        Args:
            processor_type: Type of processor
            
        Returns:
            Type[DataProcessor]: Processor class
            
        Raises:
            ValueError: If processor type is unknown
        """
        processor_type = processor_type.lower().strip()
        
        if processor_type not in AnalyticsFactory.PROCESSOR_REGISTRY:
            available_types = list(AnalyticsFactory.PROCESSOR_REGISTRY.keys())
            raise ValueError(
                f"Unknown processor type: '{processor_type}'. "
                f"Available types: {available_types}"
            )
        
        registry_key = AnalyticsFactory.PROCESSOR_REGISTRY[processor_type]
        return YAMLSerializable.get_by_name(registry_key)
    
    @staticmethod
    def create_pipeline(processor_configs: List[Dict[str, Any]], **kwargs) -> List[DataProcessor]:
        """
        Create a pipeline of processors in sequence.
        
        Args:
            processor_configs: List of processor configurations, each containing 'type' and 'config'
            **kwargs: Additional arguments passed to all processors
            
        Returns:
            List[DataProcessor]: List of configured processors in order
            
        Example:
            ```python
            pipeline_config = [
                {'type': 'outliers', 'config': {'method': 'iqr', 'action': 'exclude'}},
                {'type': 'statistics', 'config': {'confidence_level': 0.95}}
            ]
            pipeline = AnalyticsFactory.create_pipeline(pipeline_config)
            ```
        """
        pipeline = []
        
        for i, proc_config in enumerate(processor_configs):
            if 'type' not in proc_config:
                raise ValueError(f"Pipeline step {i} missing 'type' field")
            
            processor_type = proc_config['type']
            processor_config = proc_config.get('config', DictConfig({}))
            
            # Add step-specific name if not provided
            step_kwargs = kwargs.copy()
            if 'name' not in step_kwargs:
                step_kwargs['name'] = f"{processor_type}_step_{i+1}"
            
            processor = AnalyticsFactory.create(processor_type, processor_config, **step_kwargs)
            pipeline.append(processor)
        
        return pipeline
    
    @staticmethod
    def get_default_config(processor_type: str) -> DictConfig:
        """
        Get default configuration for a processor type.
        
        Args:
            processor_type: Type of processor
            
        Returns:
            DictConfig: Default configuration for the processor
            
        Raises:
            ValueError: If processor type is unknown
        """
        processor_type = processor_type.lower().strip()
        
        # Define default configurations for each processor type
        default_configs = {
            'statistics': DictConfig({
                'confidence_level': 0.95,
                'percentiles': [25, 50, 75, 90, 95],
                'missing_strategy': 'drop',
                'include_advanced': True
            }),
            'outliers': DictConfig({
                'method': 'iqr',
                'iqr_factor': 1.5,
                'zscore_threshold': 3.0,
                'modified_zscore_threshold': 3.5,
                'action': 'exclude'
            }),
            'failures': DictConfig({
                'analysis_type': 'all',
                'failure_threshold': 0.1,
                'min_samples': 10,
                'time_window': 'day'
            }),
            'comparisons': DictConfig({
                'comparison_type': 'pairwise',
                'confidence_level': 0.95,
                'significance_threshold': 0.05,
                'min_samples': 5
            })
        }
        
        if processor_type not in default_configs:
            available_types = list(default_configs.keys())
            raise ValueError(
                f"Unknown processor type: '{processor_type}'. "
                f"Available types: {available_types}"
            )
        
        return default_configs[processor_type] 