"""
Data Processing Pipeline Implementation

This module implements a robust, chainable data processing pipeline system
for visualization data preparation. It provides caching, error handling,
rollback capabilities, and performance monitoring.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, Callable, Set
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import pickle
from pathlib import Path
import psutil
import sys
import traceback
import tracemalloc
from contextlib import contextmanager

from ..core.plugin_registry import BasePlugin, PluginType, PluginRegistryError
from ..core.event_bus import EventBus, Event, EventType, EventPriority

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for pipeline execution."""
    NONE = "none"           # No caching
    MEMORY = "memory"       # In-memory caching  
    DISK = "disk"           # Disk-based caching
    HYBRID = "hybrid"       # Memory + disk caching


class ProcessingStatus(Enum):
    """Status of processing operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class PerformanceLevel(Enum):
    """Performance monitoring levels."""
    NONE = "none"           # No performance monitoring
    BASIC = "basic"         # Basic timing and metrics
    DETAILED = "detailed"   # CPU, memory, and resource tracking
    PROFILING = "profiling" # Full profiling with call stacks


@dataclass
class ResourceMetrics:
    """System resource utilization metrics."""
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    thread_count: int = 0
    open_files: int = 0
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cpu_percent': self.cpu_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_percent': self.memory_percent,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb,
            'network_sent_mb': self.network_sent_mb,
            'network_recv_mb': self.network_recv_mb,
            'thread_count': self.thread_count,
            'open_files': self.open_files,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class PerformanceProfile:
    """Detailed performance profiling information."""
    processor_name: str
    execution_time_ms: float
    cpu_time_ms: float
    peak_memory_mb: float
    memory_delta_mb: float
    cache_operations: int
    resource_metrics_start: Optional[ResourceMetrics] = None
    resource_metrics_end: Optional[ResourceMetrics] = None
    call_stack_depth: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'processor_name': self.processor_name,
            'execution_time_ms': self.execution_time_ms,
            'cpu_time_ms': self.cpu_time_ms,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_delta_mb': self.memory_delta_mb,
            'cache_operations': self.cache_operations,
            'resource_metrics_start': self.resource_metrics_start.to_dict() if self.resource_metrics_start else None,
            'resource_metrics_end': self.resource_metrics_end.to_dict() if self.resource_metrics_end else None,
            'call_stack_depth': self.call_stack_depth,
            'custom_metrics': self.custom_metrics
        }


@dataclass
class ProcessingContext:
    """
    Context information for data processing operations.
    
    Contains metadata, configuration, and state information
    that processors can use during execution.
    """
    pipeline_id: str
    execution_id: str
    processor_name: str
    input_hash: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_enabled: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    
    @property
    def execution_time(self) -> Optional[timedelta]:
        """Calculate execution time if both start and end times are available."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            'pipeline_id': self.pipeline_id,
            'execution_id': self.execution_id,
            'processor_name': self.processor_name,
            'input_hash': self.input_hash,
            'config': self.config,
            'metadata': self.metadata,
            'cache_enabled': self.cache_enabled,
            'cache_strategy': self.cache_strategy.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'execution_time_ms': self.execution_time.total_seconds() * 1000 if self.execution_time else None
        }


@dataclass
class ProcessingResult:
    """
    Result of a data processing operation.
    
    Contains the processed data, processing context, metrics,
    and any errors that occurred during processing.
    """
    data: Any
    context: ProcessingContext
    cache_hit: bool = False
    cache_key: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    rollback_data: Optional[Any] = None
    
    @property
    def is_success(self) -> bool:
        """Check if processing was successful."""
        return self.context.status == ProcessingStatus.COMPLETED and not self.errors
    
    @property
    def has_warnings(self) -> bool:
        """Check if processing had warnings."""
        return len(self.warnings) > 0
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(f"Processing warning: {message}")
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        logger.error(f"Processing error: {message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'context': self.context.to_dict(),
            'cache_hit': self.cache_hit,
            'cache_key': self.cache_key,
            'metrics': self.metrics,
            'warnings': self.warnings,
            'errors': self.errors,
            'is_success': self.is_success,
            'has_warnings': self.has_warnings,
            'data_type': type(self.data).__name__ if self.data is not None else None,
            'data_size': len(str(self.data)) if self.data is not None else 0
                    }


# Performance monitoring callback types (defined after classes to avoid circular references)
PerformanceCallback = Callable[[str, 'PerformanceProfile'], None]
ResourceCallback = Callable[[str, ResourceMetrics], None]
ThresholdCallback = Callable[[str, str, float, float], None]  # pipeline_id, metric_name, current_value, threshold


class DataProcessor(BasePlugin):
    """
    Abstract base class for data processors.
    
    Data processors are plugins that transform data in the pipeline.
    They support caching, validation, error handling, and rollback.
    """
    
    @property
    def plugin_type(self) -> PluginType:
        """Plugin type is always DATA_PROCESSOR."""
        return PluginType.DATA_PROCESSOR
    
    @property
    def supports_parallel(self) -> bool:
        """Whether this processor supports parallel execution."""
        return False
    
    @property
    def supports_caching(self) -> bool:
        """Whether this processor supports caching."""
        return True
    
    @property
    def supports_rollback(self) -> bool:
        """Whether this processor supports rollback operations."""
        return False
    
    @abstractmethod
    def process(self, data: Any, context: ProcessingContext) -> ProcessingResult:
        """
        Process the input data.
        
        Args:
            data: Input data to process
            context: Processing context with metadata and configuration
            
        Returns:
            ProcessingResult containing processed data and metadata
            
        Raises:
            ProcessorValidationError: If input data is invalid
            PipelineExecutionError: If processing fails
        """
        pass
    
    def validate_input(self, data: Any, context: ProcessingContext) -> bool:
        """
        Validate input data before processing.
        
        Args:
            data: Input data to validate
            context: Processing context
            
        Returns:
            True if data is valid, False otherwise
        """
        return True
    
    def get_cache_key(self, data: Any, context: ProcessingContext) -> str:
        """
        Generate cache key for the given input data and context.
        
        Args:
            data: Input data
            context: Processing context
            
        Returns:
            Cache key string
        """
        # Create a hash from processor name, data, and relevant context
        key_data = {
            'processor': self.plugin_name,
            'input_hash': context.input_hash,
            'config': context.config
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def rollback(self, original_data: Any, processed_data: Any, context: ProcessingContext) -> Any:
        """
        Rollback processing operation.
        
        Args:
            original_data: Original input data
            processed_data: Processed data to rollback
            context: Processing context
            
        Returns:
            Rollback data or original data
        """
        return original_data
    
    def get_performance_metrics(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        Get performance metrics for the processing operation.
        
        Args:
            result: Processing result
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        if result.context.execution_time:
            metrics['execution_time_ms'] = result.context.execution_time.total_seconds() * 1000
        
        metrics['cache_hit'] = result.cache_hit
        metrics['processor'] = self.plugin_name
        metrics['success'] = result.is_success
        metrics['warnings_count'] = len(result.warnings)
        metrics['errors_count'] = len(result.errors)
        
        return metrics


class PipelineExecutionError(Exception):
    """Raised when pipeline execution fails."""
    pass


class ProcessorValidationError(Exception):
    """Raised when processor validation fails."""
    pass


class ProcessorRegistrationError(Exception):
    """Raised when processor registration fails."""
    pass


class PerformanceMonitor:
    """
    System resource and performance monitoring utility.
    
    Provides methods to capture system resource metrics, CPU usage,
    memory consumption, and other performance indicators.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self._process = psutil.Process()
        self._baseline_metrics: Optional[ResourceMetrics] = None
        
    def capture_resource_metrics(self) -> ResourceMetrics:
        """Capture current system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = self._process.cpu_percent()
            
            # Memory metrics  
            memory_info = self._process.memory_info()
            memory_used_mb = memory_info.rss / (1024 * 1024)  # RSS in MB
            memory_percent = self._process.memory_percent()
            
            # I/O metrics
            io_counters = self._process.io_counters()
            disk_io_read_mb = io_counters.read_bytes / (1024 * 1024)
            disk_io_write_mb = io_counters.write_bytes / (1024 * 1024)
            
            # Network metrics (system-wide)
            net_io = psutil.net_io_counters()
            network_sent_mb = net_io.bytes_sent / (1024 * 1024)
            network_recv_mb = net_io.bytes_recv / (1024 * 1024)
            
            # Process metrics
            thread_count = self._process.num_threads()
            open_files = len(self._process.open_files())
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_used_mb=memory_used_mb,
                memory_percent=memory_percent,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                thread_count=thread_count,
                open_files=open_files,
                timestamp=datetime.now()
            )
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Failed to capture resource metrics: {e}")
            return ResourceMetrics(timestamp=datetime.now())
    
    def set_baseline(self) -> None:
        """Set baseline metrics for comparison."""
        self._baseline_metrics = self.capture_resource_metrics()
    
    def get_baseline(self) -> Optional[ResourceMetrics]:
        """Get baseline metrics."""
        return self._baseline_metrics
    
    def calculate_delta(self, current: ResourceMetrics) -> Dict[str, float]:
        """Calculate delta from baseline metrics."""
        if not self._baseline_metrics:
            return {}
        
        return {
            'cpu_percent_delta': current.cpu_percent - self._baseline_metrics.cpu_percent,
            'memory_mb_delta': current.memory_used_mb - self._baseline_metrics.memory_used_mb,
            'disk_read_mb_delta': current.disk_io_read_mb - self._baseline_metrics.disk_io_read_mb,
            'disk_write_mb_delta': current.disk_io_write_mb - self._baseline_metrics.disk_io_write_mb,
            'network_sent_mb_delta': current.network_sent_mb - self._baseline_metrics.network_sent_mb,
            'network_recv_mb_delta': current.network_recv_mb - self._baseline_metrics.network_recv_mb,
            'thread_count_delta': current.thread_count - self._baseline_metrics.thread_count,
            'open_files_delta': current.open_files - self._baseline_metrics.open_files
        }
    
    @contextmanager
    def profile_execution(self, processor_name: str):
        """
        Context manager for profiling processor execution.
        
        Args:
            processor_name: Name of the processor being profiled
            
        Yields:
            Dict containing profiling data that can be updated during execution
        """
        # Start profiling
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        start_metrics = self.capture_resource_metrics()
        
        # Start memory tracing if in profiling mode
        tracemalloc.start()
        
        profile_data = {
            'processor_name': processor_name,
            'start_time': start_time,
            'start_cpu_time': start_cpu_time,
            'start_metrics': start_metrics,
            'cache_operations': 0,
            'custom_metrics': {}
        }
        
        try:
            yield profile_data
            
        finally:
            # End profiling
            end_time = time.perf_counter()
            end_cpu_time = time.process_time()
            end_metrics = self.capture_resource_metrics()
            
            # Memory profiling
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            execution_time_ms = (end_time - start_time) * 1000
            cpu_time_ms = (end_cpu_time - start_cpu_time) * 1000
            peak_memory_mb = peak_memory / (1024 * 1024)
            memory_delta_mb = end_metrics.memory_used_mb - start_metrics.memory_used_mb
            
            # Update profile data
            profile_data.update({
                'execution_time_ms': execution_time_ms,
                'cpu_time_ms': cpu_time_ms,
                'peak_memory_mb': peak_memory_mb,
                'memory_delta_mb': memory_delta_mb,
                'end_metrics': end_metrics,
                'call_stack_depth': len(traceback.extract_stack())
            })


class DataPipeline:
    """
    Chainable data processing pipeline.
    
    The DataPipeline allows chaining multiple data processors together
    to create complex data transformation workflows. It supports caching,
    error handling, rollback, and performance monitoring.
    
    Example:
        ```python
        pipeline = DataPipeline("analytics_pipeline")
        
        # Add processors to the pipeline
        pipeline.add_processor("filter_outliers", FilterOutliersProcessor())
        pipeline.add_processor("normalize", NormalizationProcessor())
        pipeline.add_processor("aggregate", AggregationProcessor())
        
        # Execute the pipeline
        result = pipeline.execute(raw_data, config={'cache_strategy': CacheStrategy.MEMORY})
        
        if result.is_success:
            processed_data = result.data
        else:
            logger.error(f"Pipeline failed: {result.errors}")
        ```
    """
    
    def __init__(self, 
                 pipeline_id: str,
                 event_bus: Optional[EventBus] = None,
                 cache_strategy: CacheStrategy = CacheStrategy.MEMORY,
                 max_parallel_processors: int = 4,
                 enable_rollback: bool = True,
                 performance_level: PerformanceLevel = PerformanceLevel.BASIC):
        """
        Initialize the data pipeline.
        
        Args:
            pipeline_id: Unique identifier for this pipeline
            event_bus: Event bus for publishing events (optional)
            cache_strategy: Default caching strategy
            max_parallel_processors: Maximum number of parallel processors
            enable_rollback: Whether to enable rollback functionality
            performance_level: Level of performance monitoring to enable
        """
        self.pipeline_id = pipeline_id
        self.event_bus = event_bus
        self.cache_strategy = cache_strategy
        self.max_parallel_processors = max_parallel_processors
        self.enable_rollback = enable_rollback
        self.performance_level = performance_level
        
        # Pipeline configuration
        self._processors: List[DataProcessor] = []
        self._processor_configs: Dict[str, Dict[str, Any]] = {}
        self._processor_names: List[str] = []
        
        # Cache management
        self._memory_cache: Dict[str, Any] = {}
        self._cache_stats: Dict[str, int] = {'hits': 0, 'misses': 0}
        self._cache_lock = threading.Lock()
        
        # Initialize disk cache directory path (shared between instances with same pipeline_id)
        self._disk_cache_dir = Path.cwd() / '.pipeline_cache' / self.pipeline_id
        
        # Execution tracking
        self._execution_history: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, List[float]] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=max_parallel_processors)
        
        # Performance monitoring
        self._performance_monitor = PerformanceMonitor() if performance_level != PerformanceLevel.NONE else None
        self._performance_callbacks: List[PerformanceCallback] = []
        self._resource_callbacks: List[ResourceCallback] = []
        self._threshold_callbacks: List[ThresholdCallback] = []
        self._performance_profiles: List[PerformanceProfile] = []
        self._performance_thresholds: Dict[str, float] = {
            'max_execution_time_ms': 30000,  # 30 seconds
            'max_cpu_time_ms': 20000,        # 20 seconds
            'max_memory_mb': 1000,           # 1GB
            'max_cpu_percent': 80.0          # 80% CPU
        }
        
        # State management
        self._is_initialized = False
        self._rollback_stack: List[Dict[str, Any]] = []
        
        # Set performance baseline
        if self._performance_monitor:
            self._performance_monitor.set_baseline()
        
        logger.info(f"Created data pipeline: {pipeline_id} (performance level: {performance_level.value})")
    
    def add_processor(self, 
                     name: str, 
                     processor: DataProcessor,
                     config: Optional[Dict[str, Any]] = None) -> 'DataPipeline':
        """
        Add a processor to the pipeline.
        
        Args:
            name: Unique name for the processor in this pipeline
            processor: DataProcessor instance
            config: Processor-specific configuration
            
        Returns:
            Self for method chaining
            
        Raises:
            ProcessorRegistrationError: If processor registration fails
        """
        if name in self._processor_names:
            raise ProcessorRegistrationError(f"Processor '{name}' already exists in pipeline")
        
        if not isinstance(processor, DataProcessor):
            raise ProcessorRegistrationError(f"Processor must be instance of DataProcessor")
        
        try:
            # Initialize processor
            processor.initialize(config or {})
            
            self._processors.append(processor)
            self._processor_names.append(name)
            self._processor_configs[name] = config or {}
            
            # Publish event
            if self.event_bus:
                event = Event(
                    event_type=EventType.PLUGIN_REGISTERED,
                    data={'processor_name': name, 'processor_type': processor.plugin_name},
                    source=f"pipeline.{self.pipeline_id}",
                    priority=EventPriority.NORMAL
                )
                self.event_bus.publish(event)
            
            logger.info(f"Added processor '{name}' to pipeline '{self.pipeline_id}'")
            return self
            
        except Exception as e:
            raise ProcessorRegistrationError(f"Failed to register processor '{name}': {str(e)}")
    
    def remove_processor(self, name: str) -> 'DataPipeline':
        """
        Remove a processor from the pipeline.
        
        Args:
            name: Name of the processor to remove
            
        Returns:
            Self for method chaining
            
        Raises:
            ProcessorRegistrationError: If processor not found
        """
        if name not in self._processor_names:
            raise ProcessorRegistrationError(f"Processor '{name}' not found in pipeline")
        
        try:
            index = self._processor_names.index(name)
            processor = self._processors[index]
            
            # Cleanup processor
            processor.cleanup()
            
            # Remove from lists
            del self._processors[index]
            del self._processor_names[index]
            del self._processor_configs[name]
            
            # Publish event
            if self.event_bus:
                event = Event(
                    event_type=EventType.PLUGIN_UNREGISTERED,
                    data={'processor_name': name},
                    source=f"pipeline.{self.pipeline_id}",
                    priority=EventPriority.NORMAL
                )
                self.event_bus.publish(event)
            
            logger.info(f"Removed processor '{name}' from pipeline '{self.pipeline_id}'")
            return self
            
        except Exception as e:
            raise ProcessorRegistrationError(f"Failed to remove processor '{name}': {str(e)}")
    
    def add_performance_callback(self, callback: PerformanceCallback) -> 'DataPipeline':
        """Add a performance monitoring callback."""
        self._performance_callbacks.append(callback)
        return self
    
    def add_resource_callback(self, callback: ResourceCallback) -> 'DataPipeline':
        """Add a resource monitoring callback."""
        self._resource_callbacks.append(callback)
        return self
    
    def add_threshold_callback(self, callback: ThresholdCallback) -> 'DataPipeline':
        """Add a threshold violation callback."""
        self._threshold_callbacks.append(callback)
        return self
    
    def set_performance_threshold(self, metric_name: str, threshold: float) -> 'DataPipeline':
        """
        Set a performance threshold for monitoring.
        
        Args:
            metric_name: Name of the metric to monitor
            threshold: Threshold value for the metric
        """
        self._performance_thresholds[metric_name] = threshold
        return self
    
    def get_performance_profiles(self) -> List[PerformanceProfile]:
        """Get all performance profiles from pipeline execution."""
        return self._performance_profiles.copy()
    
    def export_performance_data(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Export comprehensive performance data for analysis.
        
        Args:
            filepath: Optional path to save the data as JSON
            
        Returns:
            Dictionary containing all performance data
        """
        performance_data = {
            'pipeline_id': self.pipeline_id,
            'performance_level': self.performance_level.value,
            'execution_count': len(self._execution_history),
            'baseline_metrics': self._performance_monitor.get_baseline().to_dict() if self._performance_monitor and self._performance_monitor.get_baseline() else None,
            'performance_profiles': [profile.to_dict() for profile in self._performance_profiles],
            'performance_thresholds': self._performance_thresholds,
            'cache_statistics': self._cache_stats,
            'execution_history': self._execution_history[-50:],  # Last 50 executions
            'summary_statistics': self._calculate_performance_summary()
        }
        
        if filepath:
            import json
            try:
                with open(filepath, 'w') as f:
                    json.dump(performance_data, f, indent=2, default=str)
                logger.info(f"Performance data exported to: {filepath}")
            except Exception as e:
                logger.error(f"Failed to export performance data: {e}")
        
        return performance_data
    
    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from performance profiles."""
        if not self._performance_profiles:
            return {}
        
        execution_times = [p.execution_time_ms for p in self._performance_profiles]
        cpu_times = [p.cpu_time_ms for p in self._performance_profiles]
        memory_peaks = [p.peak_memory_mb for p in self._performance_profiles]
        
        return {
            'total_executions': len(self._performance_profiles),
            'avg_execution_time_ms': sum(execution_times) / len(execution_times),
            'max_execution_time_ms': max(execution_times),
            'min_execution_time_ms': min(execution_times),
            'avg_cpu_time_ms': sum(cpu_times) / len(cpu_times),
            'max_cpu_time_ms': max(cpu_times),
            'avg_peak_memory_mb': sum(memory_peaks) / len(memory_peaks),
            'max_peak_memory_mb': max(memory_peaks),
            'cache_hit_ratio': self._cache_stats['hits'] / (self._cache_stats['hits'] + self._cache_stats['misses']) if (self._cache_stats['hits'] + self._cache_stats['misses']) > 0 else 0
        }
    
    def _check_performance_thresholds(self, profile: PerformanceProfile) -> None:
        """Check if performance profile violates any thresholds."""
        violations = []
        
        if profile.execution_time_ms > self._performance_thresholds.get('max_execution_time_ms', float('inf')):
            violations.append(('execution_time_ms', profile.execution_time_ms, self._performance_thresholds['max_execution_time_ms']))
        
        if profile.cpu_time_ms > self._performance_thresholds.get('max_cpu_time_ms', float('inf')):
            violations.append(('cpu_time_ms', profile.cpu_time_ms, self._performance_thresholds['max_cpu_time_ms']))
        
        if profile.peak_memory_mb > self._performance_thresholds.get('max_memory_mb', float('inf')):
            violations.append(('peak_memory_mb', profile.peak_memory_mb, self._performance_thresholds['max_memory_mb']))
        
        # Check resource metrics if available
        if profile.resource_metrics_end:
            if profile.resource_metrics_end.cpu_percent > self._performance_thresholds.get('max_cpu_percent', float('inf')):
                violations.append(('cpu_percent', profile.resource_metrics_end.cpu_percent, self._performance_thresholds['max_cpu_percent']))
        
        # Call threshold callbacks for violations
        for metric_name, current_value, threshold in violations:
            for callback in self._threshold_callbacks:
                try:
                    callback(self.pipeline_id, metric_name, current_value, threshold)
                except Exception as e:
                    logger.error(f"Threshold callback failed: {e}")
        
        if violations:
            logger.warning(f"Performance threshold violations in processor '{profile.processor_name}': {violations}")
    
    def execute(self, 
                data: Any, 
                config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Execute the pipeline on the given data.
        
        Args:
            data: Input data to process
            config: Pipeline execution configuration
            
        Returns:
            ProcessingResult with processed data and metadata
            
        Raises:
            PipelineExecutionError: If pipeline execution fails
        """
        config = config or {}
        execution_id = str(uuid.uuid4())
        
        # Create initial context
        input_hash = self._compute_data_hash(data)
        context = ProcessingContext(
            pipeline_id=self.pipeline_id,
            execution_id=execution_id,
            processor_name="pipeline",
            input_hash=input_hash,
            config=config,
            cache_strategy=config.get('cache_strategy', self.cache_strategy),
            start_time=datetime.now(),
            status=ProcessingStatus.RUNNING
        )
        
        try:
            # Publish start event
            if self.event_bus:
                event = Event(
                    event_type=EventType.DATA_PROCESSING,
                    data={'pipeline_id': self.pipeline_id, 'execution_id': execution_id},
                    source=f"pipeline.{self.pipeline_id}",
                    priority=EventPriority.NORMAL
                )
                self.event_bus.publish(event)
            
            # Execute processors in sequence
            current_data = data
            results = []
            rollback_data = []
            
            for i, (processor, name) in enumerate(zip(self._processors, self._processor_names)):
                processor_context = ProcessingContext(
                    pipeline_id=self.pipeline_id,
                    execution_id=execution_id,
                    processor_name=name,
                    input_hash=self._compute_data_hash(current_data),
                    config=self._processor_configs[name],
                    cache_strategy=context.cache_strategy,
                    start_time=datetime.now(),
                    status=ProcessingStatus.RUNNING
                )
                
                try:
                    # Check cache first
                    result = self._execute_processor_with_cache(processor, current_data, processor_context)
                    
                    if not result.is_success:
                        # Handle processor failure
                        if self.enable_rollback:
                            self._perform_rollback(rollback_data, results)
                        raise PipelineExecutionError(f"Processor '{name}' failed: {result.errors}")
                    
                    results.append(result)
                    if self.enable_rollback:
                        rollback_data.append({
                            'processor': processor,
                            'original_data': current_data,
                            'processed_data': result.data,
                            'context': processor_context
                        })
                    
                    current_data = result.data
                    
                except Exception as e:
                    # Handle processor exception
                    if self.enable_rollback:
                        self._perform_rollback(rollback_data, results)
                    raise PipelineExecutionError(f"Processor '{name}' raised exception: {str(e)}")
            
            # Create final result
            context.end_time = datetime.now()
            context.status = ProcessingStatus.COMPLETED
            
            # Aggregate cache information from all processor results
            total_cache_hits = sum(1 for r in results if r.cache_hit)
            has_cache_hit = total_cache_hits > 0
            
            final_result = ProcessingResult(
                data=current_data,
                context=context,
                cache_hit=has_cache_hit,
                cache_key=results[-1].cache_key if results and results[-1].cache_key else None,
                metrics=self._aggregate_metrics(results)
            )
            
            # Store execution history
            self._execution_history.append({
                'execution_id': execution_id,
                'timestamp': context.start_time.isoformat(),
                'duration_ms': context.execution_time.total_seconds() * 1000,
                'processors_count': len(self._processors),
                'success': True,
                'cache_hits': sum(1 for r in results if r.cache_hit),
                'total_warnings': sum(len(r.warnings) for r in results),
                'total_errors': sum(len(r.errors) for r in results)
            })
            
            # Publish completion event
            if self.event_bus:
                event = Event(
                    event_type=EventType.DATA_PROCESSED,
                    data={'pipeline_id': self.pipeline_id, 'execution_id': execution_id, 'success': True},
                    source=f"pipeline.{self.pipeline_id}",
                    priority=EventPriority.NORMAL
                )
                self.event_bus.publish(event)
            
            logger.info(f"Pipeline '{self.pipeline_id}' executed successfully in {context.execution_time}")
            return final_result
            
        except Exception as e:
            context.end_time = datetime.now()
            context.status = ProcessingStatus.FAILED
            
            # Store failed execution
            self._execution_history.append({
                'execution_id': execution_id,
                'timestamp': context.start_time.isoformat(),
                'duration_ms': context.execution_time.total_seconds() * 1000 if context.execution_time else 0,
                'processors_count': len(self._processors),
                'success': False,
                'error': str(e)
            })
            
            # Publish error event
            if self.event_bus:
                event = Event(
                    event_type=EventType.DATA_ERROR,
                    data={'pipeline_id': self.pipeline_id, 'execution_id': execution_id, 'error': str(e)},
                    source=f"pipeline.{self.pipeline_id}",
                    priority=EventPriority.HIGH
                )
                self.event_bus.publish(event)
            
            logger.error(f"Pipeline '{self.pipeline_id}' execution failed: {str(e)}")
            raise
    
    def _execute_processor_with_cache(self, 
                                    processor: DataProcessor, 
                                    data: Any, 
                                    context: ProcessingContext) -> ProcessingResult:
        """Execute processor with caching support and performance monitoring."""
        cache_key = None
        cache_operations = 0
        
        # Check cache if enabled
        if context.cache_enabled and processor.supports_caching:
            cache_key = processor.get_cache_key(data, context)
            cached_result = self._get_from_cache(cache_key, context.cache_strategy)
            cache_operations += 1
            
            if cached_result is not None:
                context.end_time = datetime.now()
                context.status = ProcessingStatus.COMPLETED
                
                with self._cache_lock:
                    self._cache_stats['hits'] += 1
                
                return ProcessingResult(
                    data=cached_result,
                    context=context,
                    cache_hit=True,
                    cache_key=cache_key
                )
        
        # Execute processor with performance monitoring
        result = None
        profile = None
        
        if self._performance_monitor and self.performance_level != PerformanceLevel.NONE:
            # Use performance profiler
            profile_data = None
            with self._performance_monitor.profile_execution(processor.plugin_name) as pd:
                profile_data = pd
                profile_data['cache_operations'] = cache_operations
                
                # Execute processor
                with self._cache_lock:
                    self._cache_stats['misses'] += 1
                    
                result = processor.process(data, context)
                result.cache_key = cache_key
                
                # Store in cache if enabled
                if context.cache_enabled and processor.supports_caching and result.is_success:
                    self._store_in_cache(cache_key, result.data, context.cache_strategy)
                    profile_data['cache_operations'] += 1
            
            # Create performance profile after context manager completes
            if profile_data:
                profile = PerformanceProfile(
                    processor_name=processor.plugin_name,
                    execution_time_ms=profile_data.get('execution_time_ms', 0.0),
                    cpu_time_ms=profile_data.get('cpu_time_ms', 0.0),
                    peak_memory_mb=profile_data.get('peak_memory_mb', 0.0),
                    memory_delta_mb=profile_data.get('memory_delta_mb', 0.0),
                    cache_operations=profile_data.get('cache_operations', 0),
                    resource_metrics_start=profile_data.get('start_metrics'),
                    resource_metrics_end=profile_data.get('end_metrics'),
                    call_stack_depth=profile_data.get('call_stack_depth', 0),
                    custom_metrics=profile_data.get('custom_metrics', {})
                )
                
                # Store profile and check thresholds
                self._performance_profiles.append(profile)
                self._check_performance_thresholds(profile)
                
                # Call performance callbacks
                for callback in self._performance_callbacks:
                    try:
                        callback(self.pipeline_id, profile)
                    except Exception as e:
                        logger.error(f"Performance callback failed: {e}")
                
                # Call resource callbacks
                if profile.resource_metrics_end:
                    for callback in self._resource_callbacks:
                        try:
                            callback(self.pipeline_id, profile.resource_metrics_end)
                        except Exception as e:
                            logger.error(f"Resource callback failed: {e}")
        
        else:
            # Basic execution without detailed profiling
            with self._cache_lock:
                self._cache_stats['misses'] += 1
                
            result = processor.process(data, context)
            result.cache_key = cache_key
            
            # Store in cache if enabled
            if context.cache_enabled and processor.supports_caching and result.is_success:
                self._store_in_cache(cache_key, result.data, context.cache_strategy)
        
        return result
    
    def _get_from_cache(self, cache_key: str, strategy: CacheStrategy) -> Optional[Any]:
        """Get data from cache based on strategy."""
        if strategy == CacheStrategy.NONE:
            return None
            
        # Try memory cache first for MEMORY and HYBRID strategies
        if strategy in (CacheStrategy.MEMORY, CacheStrategy.HYBRID):
            with self._cache_lock:
                memory_result = self._memory_cache.get(cache_key)
                if memory_result is not None:
                    return memory_result
        
        # Try disk cache for DISK and HYBRID strategies
        if strategy in (CacheStrategy.DISK, CacheStrategy.HYBRID):
            disk_result = self._get_from_disk_cache(cache_key)
            if disk_result is not None:
                # For HYBRID strategy, promote to memory cache
                if strategy == CacheStrategy.HYBRID:
                    with self._cache_lock:
                        self._memory_cache[cache_key] = disk_result
                return disk_result
        
        return None
    
    def _store_in_cache(self, cache_key: str, data: Any, strategy: CacheStrategy) -> None:
        """Store data in cache based on strategy."""
        if strategy == CacheStrategy.NONE:
            return
            
        # Store in memory cache for MEMORY and HYBRID strategies
        if strategy in (CacheStrategy.MEMORY, CacheStrategy.HYBRID):
            with self._cache_lock:
                self._memory_cache[cache_key] = data
        
        # Store in disk cache for DISK and HYBRID strategies
        if strategy in (CacheStrategy.DISK, CacheStrategy.HYBRID):
            self._store_in_disk_cache(cache_key, data)
    
    def _get_from_disk_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from disk cache."""
        # Ensure cache directory exists
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = self._disk_cache_dir / f"{cache_key}.pkl"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {str(e)}")
            # Clean up corrupted cache file
            try:
                cache_file.unlink()
            except:
                pass
        
        return None
    
    def _store_in_disk_cache(self, cache_key: str, data: Any) -> None:
        """Store data in disk cache."""
        # Ensure cache directory exists
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = self._disk_cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to store in disk cache: {str(e)}")
    
    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of data for cache keys."""
        try:
            # Try to serialize as JSON first (for simple data types)
            data_str = json.dumps(data, sort_keys=True, default=str)
        except:
            # Fall back to string representation
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _perform_rollback(self, rollback_data: List[Dict[str, Any]], results: List[ProcessingResult]) -> None:
        """Perform rollback of processed data."""
        if not self.enable_rollback:
            return
            
        logger.warning(f"Performing rollback for pipeline '{self.pipeline_id}'")
        
        for rollback_info in reversed(rollback_data):
            processor = rollback_info['processor']
            if processor.supports_rollback:
                try:
                    processor.rollback(
                        rollback_info['original_data'],
                        rollback_info['processed_data'],
                        rollback_info['context']
                    )
                except Exception as e:
                    logger.error(f"Rollback failed for processor '{processor.plugin_name}': {str(e)}")
    
    def _aggregate_metrics(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Aggregate metrics from all processor results."""
        total_time = sum(
            result.context.execution_time.total_seconds() * 1000 
            for result in results 
            if result.context.execution_time
        )
        
        return {
            'total_execution_time_ms': total_time,
            'processors_executed': len(results),
            'cache_hits': sum(1 for r in results if r.cache_hit),
            'total_warnings': sum(len(r.warnings) for r in results),
            'total_errors': sum(len(r.errors) for r in results),
            'memory_cache_size': len(self._memory_cache),
            'cache_hit_ratio': self._cache_stats['hits'] / (self._cache_stats['hits'] + self._cache_stats['misses']) if (self._cache_stats['hits'] + self._cache_stats['misses']) > 0 else 0
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._cache_lock:
            self._memory_cache.clear()
            self._cache_stats = {'hits': 0, 'misses': 0}
        
        # Clear disk cache if it exists
        if self._disk_cache_dir.exists():
            try:
                import shutil
                shutil.rmtree(self._disk_cache_dir)
                logger.info(f"Cleared disk cache for pipeline '{self.pipeline_id}'")
            except Exception as e:
                logger.warning(f"Failed to clear disk cache: {str(e)}")
        
        logger.info(f"Cleared cache for pipeline '{self.pipeline_id}'")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        with self._cache_lock:
            cache_stats = self._cache_stats.copy()
        
        return {
            'pipeline_id': self.pipeline_id,
            'processors_count': len(self._processors),
            'executions_count': len(self._execution_history),
            'cache_stats': cache_stats,
            'memory_cache_size': len(self._memory_cache),
            'recent_executions': self._execution_history[-10:] if self._execution_history else []
        }
    
    def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        try:
            # Cleanup all processors
            for processor in self._processors:
                try:
                    processor.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up processor: {str(e)}")
            
            # Clear caches
            self.clear_cache()
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            logger.info(f"Cleaned up pipeline '{self.pipeline_id}'")
            
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup() 