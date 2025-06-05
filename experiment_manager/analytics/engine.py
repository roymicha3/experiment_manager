"""
Analytics engine for orchestrating data extraction, processing, and analysis.

This module provides the AnalyticsEngine class which coordinates between
the database layer, data processors, and query building components.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Type
from datetime import datetime, timedelta
import pandas as pd

from .query_builder import AnalyticsQuery
from .results import AnalyticsResult
from .processors.base import ProcessorManager, DataProcessor
from experiment_manager.common.serializable import YAMLSerializable

logger = logging.getLogger(__name__)


class AnalyticsEngine(YAMLSerializable):
    """
    Central orchestrator for analytics operations.
    
    The AnalyticsEngine coordinates data extraction from the database,
    processing through various data processors, and result generation.
    It provides caching, query optimization, and integration with the
    existing experiment tracking system.
    """
    
    def __init__(self, 
                 database_manager,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analytics engine.
        
        Args:
            database_manager: DatabaseManager instance for data access
            config: Optional configuration dictionary
        """
        self.database_manager = database_manager
        self.config = config or {}
        
        # Initialize processor manager
        self.processor_manager = ProcessorManager()
        
        # Cache configuration
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes default
        self._query_cache = {}
        
        # Performance tracking
        self._query_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_execution_time': 0.0
        }
        
        logger.info("AnalyticsEngine initialized with database integration")
    
    def create_query(self) -> AnalyticsQuery:
        """
        Create a new analytics query builder.
        
        Returns:
            AnalyticsQuery: A new query builder instance
        """
        return AnalyticsQuery(self.database_manager)
    
    def execute_analytics_query(self, 
                               experiment_ids: Optional[List[int]] = None,
                               filters: Optional[Dict[str, Any]] = None,
                               use_cache: bool = True) -> AnalyticsResult:
        """
        Execute a comprehensive analytics query using the enhanced database methods.
        
        Args:
            experiment_ids: List of experiment IDs to include
            filters: Additional filters for the query
            use_cache: Whether to use query caching
            
        Returns:
            AnalyticsResult: Results of the analytics query
        """
        start_time = datetime.now()
        
        try:
            # Check cache first if enabled
            cache_key = self._generate_cache_key('analytics_data', experiment_ids, filters)
            if use_cache and self.cache_enabled and cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                if self._is_cache_valid(cached_result['timestamp']):
                    self._query_stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for analytics query: {cache_key}")
                    return cached_result['result']
            
            # Execute the enhanced database query
            logger.info(f"Executing analytics query for experiments: {experiment_ids}")
            data = self.database_manager.get_analytics_data(experiment_ids, filters)
            
            # Create result with metadata
            metadata = {
                'query_type': 'analytics_data',
                'experiment_ids': experiment_ids,
                'filters': filters,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'row_count': len(data),
                'cache_used': False
            }
            
            result = AnalyticsResult(data, metadata)
            
            # Cache the result if enabled
            if use_cache and self.cache_enabled:
                self._query_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            # Update statistics
            self._query_stats['total_queries'] += 1
            self._query_stats['cache_misses'] += 1
            self._query_stats['total_execution_time'] += metadata['execution_time']
            
            logger.info(f"Analytics query completed in {metadata['execution_time']:.3f}s, {len(data)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute analytics query: {e}")
            raise
    
    def get_aggregated_metrics(self,
                              experiment_ids: Optional[List[int]] = None,
                              group_by: str = 'trial',
                              functions: Optional[List[str]] = None,
                              use_cache: bool = True) -> AnalyticsResult:
        """
        Get pre-aggregated metrics using the enhanced database methods.
        
        Args:
            experiment_ids: List of experiment IDs
            group_by: Grouping level ('experiment', 'trial', 'trial_run')
            functions: Aggregation functions to apply
            use_cache: Whether to use query caching
            
        Returns:
            AnalyticsResult: Aggregated metrics results
        """
        start_time = datetime.now()
        
        try:
            # Check cache
            cache_key = self._generate_cache_key('aggregated_metrics', experiment_ids, 
                                                {'group_by': group_by, 'functions': functions})
            if use_cache and self.cache_enabled and cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                if self._is_cache_valid(cached_result['timestamp']):
                    self._query_stats['cache_hits'] += 1
                    return cached_result['result']
            
            # Execute aggregated metrics query
            logger.info(f"Executing aggregated metrics query: group_by={group_by}, functions={functions}")
            data = self.database_manager.get_aggregated_metrics(experiment_ids, group_by, functions)
            
            metadata = {
                'query_type': 'aggregated_metrics',
                'experiment_ids': experiment_ids,
                'group_by': group_by,
                'functions': functions,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'row_count': len(data)
            }
            
            result = AnalyticsResult(data, metadata)
            
            # Cache result
            if use_cache and self.cache_enabled:
                self._query_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            self._query_stats['total_queries'] += 1
            self._query_stats['cache_misses'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute aggregated metrics query: {e}")
            raise
    
    def get_failure_analysis_data(self,
                                 experiment_ids: Optional[List[int]] = None,
                                 include_configs: bool = False,
                                 use_cache: bool = True) -> AnalyticsResult:
        """
        Get failure analysis data using the enhanced database methods.
        
        Args:
            experiment_ids: List of experiment IDs
            include_configs: Whether to include configuration data
            use_cache: Whether to use query caching
            
        Returns:
            AnalyticsResult: Failure analysis data
        """
        start_time = datetime.now()
        
        try:
            # Check cache
            cache_key = self._generate_cache_key('failure_data', experiment_ids, 
                                                {'include_configs': include_configs})
            if use_cache and self.cache_enabled and cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                if self._is_cache_valid(cached_result['timestamp']):
                    self._query_stats['cache_hits'] += 1
                    return cached_result['result']
            
            # Execute failure data query
            logger.info(f"Executing failure analysis query: include_configs={include_configs}")
            data = self.database_manager.get_failure_data(experiment_ids, include_configs)
            
            metadata = {
                'query_type': 'failure_data',
                'experiment_ids': experiment_ids,
                'include_configs': include_configs,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'row_count': len(data)
            }
            
            result = AnalyticsResult(data, metadata)
            
            # Cache result
            if use_cache and self.cache_enabled:
                self._query_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            self._query_stats['total_queries'] += 1
            self._query_stats['cache_misses'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute failure analysis query: {e}")
            raise
    
    def get_epoch_series_data(self,
                             trial_run_ids: List[int],
                             metric_types: Optional[List[str]] = None,
                             use_cache: bool = True) -> AnalyticsResult:
        """
        Get epoch series data for training curve analysis.
        
        Args:
            trial_run_ids: List of trial run IDs
            metric_types: List of metric types to include
            use_cache: Whether to use query caching
            
        Returns:
            AnalyticsResult: Epoch series data
        """
        start_time = datetime.now()
        
        try:
            # Check cache
            cache_key = self._generate_cache_key('epoch_series', trial_run_ids, 
                                                {'metric_types': metric_types})
            if use_cache and self.cache_enabled and cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                if self._is_cache_valid(cached_result['timestamp']):
                    self._query_stats['cache_hits'] += 1
                    return cached_result['result']
            
            # Execute epoch series query
            logger.info(f"Executing epoch series query for {len(trial_run_ids)} trial runs")
            data = self.database_manager.get_epoch_series(trial_run_ids, metric_types)
            
            metadata = {
                'query_type': 'epoch_series',
                'trial_run_ids': trial_run_ids,
                'metric_types': metric_types,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'row_count': len(data)
            }
            
            result = AnalyticsResult(data, metadata)
            
            # Cache result
            if use_cache and self.cache_enabled:
                self._query_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            self._query_stats['total_queries'] += 1
            self._query_stats['cache_misses'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute epoch series query: {e}")
            raise
    
    def execute_custom_query(self, 
                           query: str, 
                           params: Optional[tuple] = None,
                           use_cache: bool = True) -> AnalyticsResult:
        """
        Execute a custom SQL query using the database manager.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            use_cache: Whether to use query caching
            
        Returns:
            AnalyticsResult: Query results
        """
        start_time = datetime.now()
        
        try:
            # Check cache
            cache_key = self._generate_cache_key('custom_query', query, params)
            if use_cache and self.cache_enabled and cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                if self._is_cache_valid(cached_result['timestamp']):
                    self._query_stats['cache_hits'] += 1
                    return cached_result['result']
            
            # Execute custom query
            logger.info("Executing custom analytics query")
            data = self.database_manager.execute_query(query, params)
            
            metadata = {
                'query_type': 'custom_query',
                'query': query,
                'params': params,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'row_count': len(data)
            }
            
            result = AnalyticsResult(data, metadata)
            
            # Cache result
            if use_cache and self.cache_enabled:
                self._query_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            self._query_stats['total_queries'] += 1
            self._query_stats['cache_misses'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute custom query: {e}")
            raise
    
    def process_data(self, 
                    result: AnalyticsResult, 
                    processors: List[Union[str, DataProcessor]],
                    processor_config: Optional[Dict[str, Any]] = None) -> AnalyticsResult:
        """
        Process analytics results through a chain of data processors.
        
        Args:
            result: AnalyticsResult to process
            processors: List of processor names or instances
            processor_config: Configuration for processors
            
        Returns:
            AnalyticsResult: Processed results
        """
        try:
            # Convert result to ProcessedData for processor chain
            processed_data = result.to_processed_data()
            
            # Execute processing chain
            final_data = self.processor_manager.execute_chain(
                processors, processed_data, processor_config or {}
            )
            
            # Create new result with processed data
            new_metadata = result.metadata.copy()
            new_metadata.update({
                'processors_applied': [str(p) for p in processors],
                'processing_config': processor_config,
                'processing_timestamp': datetime.now().isoformat()
            })
            
            return AnalyticsResult(final_data.data, new_metadata, result.query_info)
            
        except Exception as e:
            logger.error(f"Failed to process analytics data: {e}")
            raise
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._query_cache.clear()
        logger.info("Analytics engine cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache and performance statistics."""
        total_queries = self._query_stats['total_queries']
        cache_hit_rate = (self._query_stats['cache_hits'] / total_queries * 100) if total_queries > 0 else 0
        avg_execution_time = (self._query_stats['total_execution_time'] / total_queries) if total_queries > 0 else 0
        
        return {
            'total_queries': total_queries,
            'cache_hits': self._query_stats['cache_hits'],
            'cache_misses': self._query_stats['cache_misses'],
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'cache_size': len(self._query_cache),
            'total_execution_time': round(self._query_stats['total_execution_time'], 3),
            'average_execution_time': round(avg_execution_time, 3)
        }
    
    def initialize_database_indexes(self) -> None:
        """Initialize analytics-optimized database indexes."""
        try:
            logger.info("Creating analytics-optimized database indexes")
            self.database_manager.create_analytics_indexes()
            logger.info("Analytics indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create analytics indexes: {e}")
            raise
    
    def _generate_cache_key(self, query_type: str, *args) -> str:
        """Generate a cache key for the given query parameters."""
        import hashlib
        
        # Convert arguments to string representation
        key_parts = [query_type]
        for arg in args:
            if arg is None:
                key_parts.append('None')
            elif isinstance(arg, (list, tuple)):
                key_parts.append(','.join(str(x) for x in arg))
            elif isinstance(arg, dict):
                key_parts.append(','.join(f"{k}:{v}" for k, v in sorted(arg.items())))
            else:
                key_parts.append(str(arg))
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if a cached result is still valid based on TTL."""
        if not self.cache_enabled:
            return False
        
        age = (datetime.now() - timestamp).total_seconds()
        return age < self.cache_ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert engine configuration to dictionary for YAML serialization."""
        return {
            'cache_enabled': self.cache_enabled,
            'cache_ttl': self.cache_ttl,
            'processor_manager_config': self.processor_manager.to_dict() if hasattr(self.processor_manager, 'to_dict') else {},
            'config': self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], database_manager) -> 'AnalyticsEngine':
        """Create engine instance from dictionary configuration."""
        config = data.get('config', {})
        config.update({
            'cache_enabled': data.get('cache_enabled', True),
            'cache_ttl': data.get('cache_ttl', 300)
        })
        
        engine = cls(database_manager, config)
        
        # Restore processor manager if available
        if 'processor_manager_config' in data and data['processor_manager_config']:
            # Note: This would need ProcessorManager.from_dict implementation
            pass
        
        return engine
    
    def __str__(self) -> str:
        """String representation of the analytics engine."""
        cache_stats = self.get_cache_stats()
        return (f"AnalyticsEngine(cache_enabled={self.cache_enabled}, "
                f"total_queries={cache_stats['total_queries']}, "
                f"cache_hit_rate={cache_stats['cache_hit_rate_percent']}%)")
    
    def __repr__(self) -> str:
        """Detailed representation of the analytics engine."""
        return (f"AnalyticsEngine(database_manager={self.database_manager}, "
                f"cache_enabled={self.cache_enabled}, cache_ttl={self.cache_ttl})") 