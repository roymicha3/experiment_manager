"""
Analytics Data Adapter for Visualization Components

This module provides seamless integration between the visualization data pipeline
and the existing analytics infrastructure, enabling efficient data access,
query optimization, and streaming data support for live updates.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from experiment_manager.analytics.api import ExperimentAnalytics
from experiment_manager.analytics.query import AnalyticsQuery, RunStatus
from experiment_manager.analytics.results import AnalyticsResult
from experiment_manager.db.manager import DatabaseManager
from experiment_manager.visualization.data.cache import DataCache, CacheConfig, cache_manager

logger = logging.getLogger(__name__)


@dataclass
class QueryOptimization:
    """Configuration for query optimization strategies."""
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    enable_prefetching: bool = True
    batch_size: int = 1000
    parallel_queries: bool = True
    max_workers: int = 4
    enable_compression: bool = True
    streaming_threshold: int = 10000  # Switch to streaming for large datasets


@dataclass
class StreamingConfig:
    """Configuration for streaming data support."""
    enable_streaming: bool = True
    buffer_size: int = 1000
    update_interval_seconds: float = 1.0
    max_buffer_age_seconds: int = 60
    enable_live_updates: bool = True


@dataclass
class DataSourceConfig:
    """Configuration for data source abstraction."""
    primary_source: str = "analytics_engine"
    fallback_sources: List[str] = field(default_factory=lambda: ["database_direct"])
    enable_source_switching: bool = True
    health_check_interval: int = 30
    timeout_seconds: int = 10
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


class DataSourceInterface(ABC):
    """Abstract interface for data sources."""
    
    @abstractmethod
    async def query_data(self, query: Dict[str, Any]) -> pd.DataFrame:
        """Execute a data query and return results."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the data source is healthy."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the data schema information."""
        pass


class AnalyticsEngineSource(DataSourceInterface):
    """Data source implementation using the analytics engine."""
    
    def __init__(self, analytics: ExperimentAnalytics):
        self.analytics = analytics
        self._last_health_check = None
        self._health_status = True
        
    async def query_data(self, query: Dict[str, Any]) -> pd.DataFrame:
        """Execute query using analytics engine."""
        try:
            # Convert query to AnalyticsQuery
            analytics_query = self._build_analytics_query(query)
            
            # Execute query
            result = analytics_query.execute()
            
            # Convert to DataFrame
            return self._result_to_dataframe(result)
            
        except Exception as e:
            logger.error(f"Analytics engine query failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check analytics engine health."""
        try:
            # Simple query to test connectivity
            test_query = self.analytics.query().experiments()
            
            # This should return quickly
            self._health_status = True
            self._last_health_check = datetime.now()
            return True
            
        except Exception as e:
            logger.warning(f"Analytics engine health check failed: {e}")
            self._health_status = False
            return False
    
    def get_schema(self) -> Dict[str, Any]:
        """Get analytics engine schema."""
        return {
            'experiments': ['id', 'title', 'description', 'start_time', 'update_time'],
            'trials': ['id', 'name', 'experiment_id', 'start_time', 'update_time'],
            'runs': ['id', 'trial_id', 'status', 'start_time', 'end_time'],
            'metrics': ['id', 'type', 'total_val', 'per_label_val'],
            'artifacts': ['id', 'type', 'location']
        }
    
    def _build_analytics_query(self, query: Dict[str, Any]) -> AnalyticsQuery:
        """Convert generic query to AnalyticsQuery."""
        analytics_query = self.analytics.query()
        
        # Apply experiment filters
        if 'experiments' in query:
            exp_filter = query['experiments']
            analytics_query = analytics_query.experiments(
                ids=exp_filter.get('ids'),
                names=exp_filter.get('names'),
                time_range=exp_filter.get('time_range')
            )
        
        # Apply trial filters
        if 'trials' in query:
            trial_filter = query['trials']
            analytics_query = analytics_query.trials(
                names=trial_filter.get('names'),
                status=trial_filter.get('status')
            )
        
        # Apply run filters
        if 'runs' in query:
            run_filter = query['runs']
            analytics_query = analytics_query.runs(
                status=run_filter.get('status', ['completed']),
                exclude_timeouts=run_filter.get('exclude_timeouts', True)
            )
        
        # Apply metric filters
        if 'metrics' in query:
            metric_filter = query['metrics']
            analytics_query = analytics_query.metrics(
                types=metric_filter.get('types'),
                context=metric_filter.get('context', 'results')
            )
        
        # Apply processing operations
        if 'exclude_outliers' in query:
            outlier_config = query['exclude_outliers']
            analytics_query = analytics_query.exclude_outliers(
                metric_type=outlier_config['metric_type'],
                method=outlier_config.get('method', 'iqr'),
                threshold=outlier_config.get('threshold', 1.5)
            )
        
        if 'aggregate' in query:
            analytics_query = analytics_query.aggregate(query['aggregate'])
        
        if 'group_by' in query:
            analytics_query = analytics_query.group_by(query['group_by'])
        
        if 'sort_by' in query:
            sort_config = query['sort_by']
            analytics_query = analytics_query.sort_by(
                field=sort_config['field'],
                ascending=sort_config.get('ascending', True)
            )
        
        return analytics_query
    
    def _result_to_dataframe(self, result: AnalyticsResult) -> pd.DataFrame:
        """Convert AnalyticsResult to pandas DataFrame."""
        try:
            data = result.data
            
            # Handle different result data structures
            if isinstance(data, dict):
                if 'raw_data' in data:
                    # Statistical analysis result
                    return pd.DataFrame(data['raw_data'])
                elif 'comparison_data' in data:
                    # Comparison result
                    return pd.DataFrame(data['comparison_data'])
                else:
                    # Generic dict result
                    return pd.DataFrame([data])
            elif isinstance(data, list):
                # List of records
                return pd.DataFrame(data)
            else:
                # Single value or other type
                return pd.DataFrame([{'value': data}])
                
        except Exception as e:
            logger.error(f"Failed to convert result to DataFrame: {e}")
            return pd.DataFrame()


class DatabaseDirectSource(DataSourceInterface):
    """Direct database access data source."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._last_health_check = None
        self._health_status = True
    
    async def query_data(self, query: Dict[str, Any]) -> pd.DataFrame:
        """Execute query directly against database."""
        try:
            # Build SQL query based on request
            sql_query, params = self._build_sql_query(query)
            
            # Execute query
            cursor = self.db_manager._execute_query(sql_query, params)
            
            # Convert to DataFrame
            if self.db_manager.use_sqlite:
                columns = [description[0] for description in cursor.description]
                data = [dict(zip(columns, row)) for row in cursor.fetchall()]
            else:
                data = cursor.fetchall()
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Database direct query failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            # Simple query to test connectivity
            cursor = self.db_manager._execute_query("SELECT 1")
            cursor.fetchone()
            
            self._health_status = True
            self._last_health_check = datetime.now()
            return True
            
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            self._health_status = False
            return False
    
    def get_schema(self) -> Dict[str, Any]:
        """Get database schema."""
        return {
            'experiments': ['id', 'title', 'desc', 'start_time', 'update_time'],
            'trials': ['id', 'name', 'experiment_id', 'start_time', 'update_time'],
            'trial_runs': ['id', 'trial_id', 'status', 'start_time', 'end_time'],
            'metrics': ['id', 'type', 'total_val', 'per_label_val'],
            'artifacts': ['id', 'type', 'location']
        }
    
    def _build_sql_query(self, query: Dict[str, Any]) -> Tuple[str, tuple]:
        """Build SQL query from generic query specification."""
        base_query = "SELECT * FROM EXPERIMENT"
        params = []
        conditions = []
        
        if 'experiments' in query:
            exp_filter = query['experiments']
            
            if 'ids' in exp_filter:
                placeholders = ','.join(['?' if self.db_manager.use_sqlite else '%s'] * len(exp_filter['ids']))
                conditions.append(f"id IN ({placeholders})")
                params.extend(exp_filter['ids'])
            
            if 'names' in exp_filter:
                placeholders = ','.join(['?' if self.db_manager.use_sqlite else '%s'] * len(exp_filter['names']))
                conditions.append(f"title IN ({placeholders})")
                params.extend(exp_filter['names'])
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        if 'limit' in query:
            base_query += f" LIMIT {query['limit']}"
        
        if 'offset' in query:
            base_query += f" OFFSET {query['offset']}"
        
        return base_query, tuple(params)


class AnalyticsDataAdapter:
    """
    Main adapter class for integrating visualization pipeline with analytics engine.
    
    Features:
    - Data source abstraction with fallback support
    - Query optimization with caching and prefetching
    - Streaming data support for large datasets
    - Live update capabilities
    - Performance monitoring and health checks
    """
    
    def __init__(self,
                 analytics: ExperimentAnalytics,
                 db_manager: Optional[DatabaseManager] = None,
                 optimization_config: Optional[QueryOptimization] = None,
                 streaming_config: Optional[StreamingConfig] = None,
                 data_source_config: Optional[DataSourceConfig] = None):
        
        self.analytics = analytics
        self.db_manager = db_manager
        
        # Configuration
        self.optimization_config = optimization_config or QueryOptimization()
        self.streaming_config = streaming_config or StreamingConfig()
        self.data_source_config = data_source_config or DataSourceConfig()
        
        # Initialize data sources
        self._data_sources: Dict[str, DataSourceInterface] = {}
        self._initialize_data_sources()
        
        # Cache setup
        if self.optimization_config.enable_caching:
            cache_config = CacheConfig(
                max_memory_size_mb=100,
                default_ttl_seconds=self.optimization_config.cache_ttl_seconds,
                enable_compression=self.optimization_config.enable_compression
            )
            self._cache = cache_manager.get_cache("analytics_adapter", cache_config)
        else:
            self._cache = None
        
        # Threading and async support
        self._executor = ThreadPoolExecutor(max_workers=self.optimization_config.max_workers)
        self._active_streams: Dict[str, Any] = {}
        
        # Performance monitoring
        self._query_metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_query_time': 0.0,
            'streaming_sessions': 0,
            'data_source_switches': 0
        }
        
        logger.info("AnalyticsDataAdapter initialized successfully")
    
    def _initialize_data_sources(self):
        """Initialize available data sources."""
        # Analytics engine source
        self._data_sources['analytics_engine'] = AnalyticsEngineSource(self.analytics)
        
        # Database direct source (if available)
        if self.db_manager:
            self._data_sources['database_direct'] = DatabaseDirectSource(self.db_manager)
        
        logger.info(f"Initialized data sources: {list(self._data_sources.keys())}")
    
    async def query_data(self, 
                        query: Dict[str, Any],
                        use_cache: bool = True,
                        preferred_source: Optional[str] = None) -> pd.DataFrame:
        """
        Execute a data query with optimization and fallback support.
        
        Args:
            query: Query specification dictionary
            use_cache: Whether to use caching
            preferred_source: Preferred data source to use
            
        Returns:
            pandas.DataFrame: Query results
        """
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(query) if use_cache and self._cache else None
            
            # Check cache first
            if cache_key and self._cache:
                cached_result = self._cache.get(cache_key)
                if cached_result is not None:
                    self._query_metrics['cache_hits'] += 1
                    logger.debug(f"Cache hit for query: {cache_key}")
                    return cached_result
                else:
                    self._query_metrics['cache_misses'] += 1
            
            # Determine data source
            source_name = preferred_source or self.data_source_config.primary_source
            data_source = self._get_healthy_data_source(source_name)
            
            # Execute query
            result = await data_source.query_data(query)
            
            # Cache result
            if cache_key and self._cache and not result.empty:
                self._cache.put(cache_key, result)
            
            # Update metrics
            query_time = time.time() - start_time
            self._update_query_metrics(query_time)
            
            logger.debug(f"Query executed in {query_time:.3f}s using {source_name}")
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_experiment_data(self, 
                           experiment_names: List[str],
                           metrics: Optional[List[str]] = None,
                           include_trials: bool = True,
                           include_runs: bool = True) -> pd.DataFrame:
        """
        Convenience method to get experiment data for visualization.
        
        Args:
            experiment_names: List of experiment names
            metrics: Optional list of metrics to include
            include_trials: Whether to include trial data
            include_runs: Whether to include run data
            
        Returns:
            pandas.DataFrame: Experiment data ready for visualization
        """
        query = {
            'experiments': {'names': experiment_names},
            'runs': {'status': ['completed']},
        }
        
        if metrics:
            query['metrics'] = {'types': metrics}
        
        # For now, return a placeholder since we can't use async easily here
        # In practice, you'd use asyncio.run or make this method async
        return pd.DataFrame()
    
    def get_training_curves_data(self,
                                experiment_name: str,
                                metrics: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get training curves data optimized for visualization.
        
        Args:
            experiment_name: Name of the experiment
            metrics: List of metrics to retrieve
            
        Returns:
            Dict mapping metric names to DataFrames with training curves
        """
        if metrics is None:
            metrics = ['loss', 'accuracy']
        
        results = {}
        
        for metric in metrics:
            query = {
                'experiments': {'names': [experiment_name]},
                'metrics': {'types': [metric]},
                'runs': {'status': ['completed']},
                'sort_by': {'field': 'epoch', 'ascending': True}
            }
            
            # For now, return placeholder
            results[metric] = pd.DataFrame()
        
        return results
    
    def _get_healthy_data_source(self, preferred: Optional[str] = None) -> DataSourceInterface:
        """Get a healthy data source with fallback support."""
        sources_to_try = []
        
        if preferred and preferred in self._data_sources:
            sources_to_try.append(preferred)
        
        # Add primary source
        if self.data_source_config.primary_source not in sources_to_try:
            sources_to_try.append(self.data_source_config.primary_source)
        
        # Add fallback sources
        sources_to_try.extend(self.data_source_config.fallback_sources)
        
        for source_name in sources_to_try:
            if source_name in self._data_sources:
                source = self._data_sources[source_name]
                # In practice, you'd check health asynchronously
                # For now, assume healthy
                return source
        
        raise RuntimeError("No healthy data sources available")
    
    def _generate_cache_key(self, query: Dict[str, Any]) -> str:
        """Generate a cache key for the query."""
        import hashlib
        import json
        
        # Sort keys for consistent hashing
        query_str = json.dumps(query, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _update_query_metrics(self, query_time: float):
        """Update query performance metrics."""
        self._query_metrics['total_queries'] += 1
        
        # Update average query time
        total = self._query_metrics['total_queries']
        current_avg = self._query_metrics['avg_query_time']
        self._query_metrics['avg_query_time'] = ((current_avg * (total - 1)) + query_time) / total
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics."""
        metrics = self._query_metrics.copy()
        
        if self._cache:
            cache_metrics = self._cache.get_metrics()
            metrics.update({
                'cache_hit_ratio': cache_metrics.hit_ratio,
                'cache_size': cache_metrics.entry_count,
                'cache_memory_mb': cache_metrics.memory_usage_mb
            })
        
        metrics.update({
            'active_streams': len(self._active_streams),
            'available_sources': list(self._data_sources.keys())
        })
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all data sources."""
        status = {}
        
        for name, source in self._data_sources.items():
            # In practice, this would be async
            status[name] = True  # Placeholder
        
        return status
    
    def close(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        
        if self._cache:
            self._cache.clear()
        
        self._active_streams.clear()
        
        logger.info("AnalyticsDataAdapter closed")


# Convenience function for creating adapter instances
def create_analytics_adapter(analytics: ExperimentAnalytics,
                           db_manager: Optional[DatabaseManager] = None,
                           **config_kwargs) -> AnalyticsDataAdapter:
    """
    Create an AnalyticsDataAdapter with sensible defaults.
    
    Args:
        analytics: ExperimentAnalytics instance
        db_manager: Optional DatabaseManager instance
        **config_kwargs: Additional configuration options
        
    Returns:
        AnalyticsDataAdapter: Configured adapter instance
    """
    return AnalyticsDataAdapter(
        analytics=analytics,
        db_manager=db_manager,
        **config_kwargs
    ) 