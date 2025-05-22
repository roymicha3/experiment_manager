"""
Performance Profiler for Database Operations

This module provides performance monitoring and profiling capabilities
for database operations and migrations.
"""
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a database operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    cpu_usage_before: float
    cpu_usage_after: float
    memory_before: int  # bytes
    memory_after: int   # bytes
    query_count: int = 0
    slow_queries: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def cpu_delta(self) -> float:
        """CPU usage change during operation."""
        return self.cpu_usage_after - self.cpu_usage_before

    @property
    def memory_delta(self) -> int:
        """Memory usage change during operation."""
        return self.memory_after - self.memory_before

    @property
    def memory_delta_mb(self) -> float:
        """Memory usage change in MB."""
        return self.memory_delta / (1024 * 1024)


class PerformanceProfiler:
    """
    Performance profiler for database operations.
    
    Monitors timing, CPU usage, memory consumption, and query performance.
    """

    def __init__(self, slow_query_threshold: float = 1.0):
        """
        Initialize the performance profiler.
        
        Args:
            slow_query_threshold: Threshold in seconds for identifying slow queries
        """
        self.slow_query_threshold = slow_query_threshold
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []

    @contextmanager
    def profile_operation(self, operation_name: str):
        """
        Context manager for profiling database operations.
        
        Args:
            operation_name: Name of the operation being profiled
            
        Yields:
            PerformanceMetrics: Metrics object that gets populated during execution
        """
        # Initialize metrics
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=0,
            end_time=0,
            duration=0,
            cpu_usage_before=0,
            cpu_usage_after=0,
            memory_before=0,
            memory_after=0
        )

        # Capture initial state
        metrics.start_time = time.time()
        
        if HAS_PSUTIL:
            process = psutil.Process()
            metrics.cpu_usage_before = process.cpu_percent()
            metrics.memory_before = process.memory_info().rss
        else:
            metrics.cpu_usage_before = 0.0
            metrics.memory_before = 0

        try:
            yield metrics
        finally:
            # Capture final state
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            
            if HAS_PSUTIL:
                process = psutil.Process()
                metrics.cpu_usage_after = process.cpu_percent()
                metrics.memory_after = process.memory_info().rss
            else:
                metrics.cpu_usage_after = 0.0
                metrics.memory_after = 0

            # Store in history
            self.metrics_history.append(metrics)
            
            # Log performance summary
            if HAS_PSUTIL:
                self.logger.info(
                    f"Operation '{operation_name}' completed in {metrics.duration:.3f}s "
                    f"(CPU: {metrics.cpu_delta:+.1f}%, Memory: {metrics.memory_delta_mb:+.1f}MB)"
                )
            else:
                self.logger.info(
                    f"Operation '{operation_name}' completed in {metrics.duration:.3f}s "
                    f"(psutil not available - no CPU/memory metrics)"
                )

    def profile_query(self, query: str, execution_time: float) -> bool:
        """
        Profile a single query execution.
        
        Args:
            query: SQL query that was executed
            execution_time: Time taken to execute the query
            
        Returns:
            bool: True if query was slow (above threshold)
        """
        is_slow = execution_time > self.slow_query_threshold
        
        if is_slow:
            self.logger.warning(f"Slow query detected ({execution_time:.3f}s): {query[:100]}...")
        
        return is_slow

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics from all profiled operations.
        
        Returns:
            Dict containing summary performance statistics
        """
        if not self.metrics_history:
            return {"message": "No operations profiled yet"}

        total_operations = len(self.metrics_history)
        total_duration = sum(m.duration for m in self.metrics_history)
        avg_duration = total_duration / total_operations
        max_duration = max(m.duration for m in self.metrics_history)
        min_duration = min(m.duration for m in self.metrics_history)

        total_cpu_delta = sum(m.cpu_delta for m in self.metrics_history)
        avg_cpu_delta = total_cpu_delta / total_operations

        total_memory_delta = sum(m.memory_delta for m in self.metrics_history)
        avg_memory_delta_mb = (total_memory_delta / total_operations) / (1024 * 1024)

        total_slow_queries = sum(len(m.slow_queries) for m in self.metrics_history)

        return {
            "total_operations": total_operations,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "max_duration": max_duration,
            "min_duration": min_duration,
            "average_cpu_delta": avg_cpu_delta,
            "average_memory_delta_mb": avg_memory_delta_mb,
            "total_slow_queries": total_slow_queries,
            "operations": [
                {
                    "name": m.operation_name,
                    "duration": m.duration,
                    "cpu_delta": m.cpu_delta,
                    "memory_delta_mb": m.memory_delta_mb,
                    "slow_queries": len(m.slow_queries)
                }
                for m in self.metrics_history
            ]
        }

    def clear_history(self):
        """Clear the performance metrics history."""
        self.metrics_history.clear()
        self.logger.info("Performance metrics history cleared")

    def export_metrics(self) -> List[Dict[str, Any]]:
        """
        Export performance metrics for external analysis.
        
        Returns:
            List of performance metrics as dictionaries
        """
        return [
            {
                "operation_name": m.operation_name,
                "start_time": m.start_time,
                "end_time": m.end_time,
                "duration": m.duration,
                "cpu_usage_before": m.cpu_usage_before,
                "cpu_usage_after": m.cpu_usage_after,
                "cpu_delta": m.cpu_delta,
                "memory_before": m.memory_before,
                "memory_after": m.memory_after,
                "memory_delta": m.memory_delta,
                "memory_delta_mb": m.memory_delta_mb,
                "query_count": m.query_count,
                "slow_queries": m.slow_queries,
                "details": m.details
            }
            for m in self.metrics_history
        ] 