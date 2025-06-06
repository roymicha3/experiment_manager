"""
Analytics Fluent Query API

Provides a fluent, chainable interface for building complex analytics queries
that can filter, process, and execute analytics operations on experiment data.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import copy
import hashlib
import json

from experiment_manager.analytics.results import AnalyticsResult, QueryMetadata
from experiment_manager.common.serializable import YAMLSerializable


class RunStatus(Enum):
    """Enumeration for run status values."""
    SUCCESS = "success"
    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ValidationError(Exception):
    """Exception raised when query validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        super().__init__(message)
        self.field = field
        self.suggestion = suggestion
        self.message = message


@dataclass
class FilterState:
    """State tracking for applied filters."""
    experiments: Dict[str, Any] = field(default_factory=dict)
    trials: Dict[str, Any] = field(default_factory=dict)
    runs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingState:
    """State tracking for processing operations."""
    outlier_exclusions: List[Dict[str, Any]] = field(default_factory=list)
    aggregations: List[Dict[str, Any]] = field(default_factory=list)
    groupings: List[str] = field(default_factory=list)
    sorting: Optional[Dict[str, Any]] = None


@dataclass
class QueryState:
    """Complete state of an analytics query."""
    filters: FilterState = field(default_factory=FilterState)
    processing: ProcessingState = field(default_factory=ProcessingState)
    execution_options: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[ValidationError] = field(default_factory=list)
    query_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@YAMLSerializable.register("AnalyticsQuery")
class AnalyticsQuery(YAMLSerializable):
    """
    Fluent query builder for analytics operations.
    
    Provides a chainable interface for constructing complex analytics queries
    with filters, processing operations, and execution options.
    
    Example:
        result = (analytics.query()
                  .experiments(names=['transformer_exp'])
                  .runs(status=['completed'])
                  .metrics(['accuracy', 'loss'])
                  .exclude_outliers('accuracy', method='iqr')
                  .aggregate(['mean', 'std', 'count'])
                  .execute())
    """
    
    def __init__(self, analytics_engine=None):
        """
        Initialize AnalyticsQuery.
        
        Args:
            analytics_engine: Reference to the analytics engine for execution
        """
        self._engine = analytics_engine
        self._state = QueryState()
        self._state.query_id = self._generate_query_id()
        
        # Cache for validation and optimization
        self._validation_cache = {}
        self._optimization_cache = {}
        
        # Execution tracking
        self._execution_count = 0
        self._last_execution_time = None
    
    def _generate_query_id(self) -> str:
        """Generate a unique ID for this query."""
        timestamp = datetime.now().isoformat()
        content = f"query_{timestamp}_{id(self)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _copy_state(self) -> 'AnalyticsQuery':
        """Create a new query instance with copied state."""
        new_query = AnalyticsQuery(self._engine)
        new_query._state = copy.deepcopy(self._state)
        new_query._validation_cache = self._validation_cache.copy()
        new_query._optimization_cache = self._optimization_cache.copy()
        return new_query
    
    # === CHAINABLE FILTER METHODS ===
    
    def experiments(self, 
                    ids: Optional[List[Union[int, str]]] = None,
                    names: Optional[List[str]] = None,
                    time_range: Optional[tuple] = None) -> 'AnalyticsQuery':
        """
        Filter by experiment criteria.
        
        Args:
            ids: List of experiment IDs to include
            names: List of experiment names to include
            time_range: Tuple of (start_time, end_time) for filtering
            
        Returns:
            New AnalyticsQuery instance with experiment filters applied
        """
        new_query = self._copy_state()
        
        filter_params = {}
        if ids is not None:
            filter_params['ids'] = list(ids)
        if names is not None:
            filter_params['names'] = list(names)
        if time_range is not None:
            if len(time_range) != 2:
                raise ValidationError(
                    "time_range must be a tuple of (start_time, end_time)",
                    field="time_range",
                    suggestion="Provide a tuple like (datetime(2023,1,1), datetime(2023,12,31))"
                )
            filter_params['time_range'] = time_range
        
        new_query._state.filters.experiments.update(filter_params)
        return new_query
    
    def trials(self, 
               names: Optional[List[str]] = None,
               status: Optional[List[str]] = None) -> 'AnalyticsQuery':
        """
        Filter by trial criteria.
        
        Args:
            names: List of trial names to include
            status: List of trial statuses to include
            
        Returns:
            New AnalyticsQuery instance with trial filters applied
        """
        new_query = self._copy_state()
        
        filter_params = {}
        if names is not None:
            filter_params['names'] = list(names)
        if status is not None:
            filter_params['status'] = list(status)
        
        new_query._state.filters.trials.update(filter_params)
        return new_query
    
    def runs(self, 
             status: List[str] = None,
             exclude_timeouts: bool = True) -> 'AnalyticsQuery':
        """
        Filter by run criteria.
        
        Args:
            status: List of run statuses to include (default: ['completed'])
            exclude_timeouts: Whether to exclude timeout runs
            
        Returns:
            New AnalyticsQuery instance with run filters applied
        """
        new_query = self._copy_state()
        
        if status is None:
            status = ['completed']
        
        filter_params = {
            'status': list(status),
            'exclude_timeouts': exclude_timeouts
        }
        
        new_query._state.filters.runs.update(filter_params)
        return new_query
    
    def metrics(self, 
                types: Optional[List[str]] = None,
                context: str = 'results') -> 'AnalyticsQuery':
        """
        Filter by metric criteria.
        
        Args:
            types: List of metric types to include
            context: Context for metrics (default: 'results')
            
        Returns:
            New AnalyticsQuery instance with metric filters applied
        """
        new_query = self._copy_state()
        
        filter_params = {
            'context': context
        }
        
        if types is not None:
            filter_params['types'] = list(types)
        
        new_query._state.filters.metrics.update(filter_params)
        return new_query
    
    # === CHAINABLE PROCESSING METHODS ===
    
    def exclude_outliers(self, 
                        metric_type: str,
                        method: str = 'iqr',
                        threshold: float = 1.5) -> 'AnalyticsQuery':
        """
        Add outlier exclusion processing step.
        
        Args:
            metric_type: Type of metric to analyze for outliers
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold value for outlier detection
            
        Returns:
            New AnalyticsQuery instance with outlier exclusion processing
        """
        new_query = self._copy_state()
        
        valid_methods = ['iqr', 'zscore', 'modified_zscore']
        if method not in valid_methods:
            raise ValidationError(
                f"Invalid outlier detection method: {method}",
                field="method",
                suggestion=f"Use one of: {', '.join(valid_methods)}"
            )
        
        outlier_config = {
            'metric_type': metric_type,
            'method': method,
            'threshold': threshold
        }
        
        new_query._state.processing.outlier_exclusions.append(outlier_config)
        return new_query
    
    def aggregate(self, functions: List[str] = None) -> 'AnalyticsQuery':
        """
        Add aggregation processing step.
        
        Args:
            functions: List of aggregation functions ('mean', 'std', 'min', 'max', 'count', etc.)
            
        Returns:
            New AnalyticsQuery instance with aggregation processing
        """
        new_query = self._copy_state()
        
        if functions is None:
            functions = ['mean', 'std']
        
        valid_functions = ['mean', 'median', 'std', 'var', 'min', 'max', 'count', 'sum', 'skew', 'kurt']
        invalid_functions = [f for f in functions if f not in valid_functions]
        
        if invalid_functions:
            raise ValidationError(
                f"Invalid aggregation functions: {', '.join(invalid_functions)}",
                field="functions",
                suggestion=f"Use functions from: {', '.join(valid_functions)}"
            )
        
        aggregation_config = {
            'functions': list(functions)
        }
        
        new_query._state.processing.aggregations.append(aggregation_config)
        return new_query
    
    def group_by(self, field: str = 'trial') -> 'AnalyticsQuery':
        """
        Add grouping processing step.
        
        Args:
            field: Field to group by ('trial', 'experiment', 'run', etc.)
            
        Returns:
            New AnalyticsQuery instance with grouping processing
        """
        new_query = self._copy_state()
        
        valid_fields = ['trial', 'experiment', 'run', 'metric_type', 'model', 'optimizer']
        if field not in valid_fields:
            raise ValidationError(
                f"Invalid grouping field: {field}",
                field="field",
                suggestion=f"Use one of: {', '.join(valid_fields)}"
            )
        
        if field not in new_query._state.processing.groupings:
            new_query._state.processing.groupings.append(field)
        
        return new_query
    
    def sort_by(self, field: str, ascending: bool = True) -> 'AnalyticsQuery':
        """
        Add sorting processing step.
        
        Args:
            field: Field to sort by
            ascending: Sort direction (True for ascending, False for descending)
            
        Returns:
            New AnalyticsQuery instance with sorting processing
        """
        new_query = self._copy_state()
        
        sorting_config = {
            'field': field,
            'ascending': ascending
        }
        
        new_query._state.processing.sorting = sorting_config
        return new_query
    
    # === EXECUTION METHODS ===
    
    def execute(self) -> AnalyticsResult:
        """
        Execute the query and return results.
        
        Returns:
            AnalyticsResult containing the query results
            
        Raises:
            ValidationError: If query validation fails
            RuntimeError: If no analytics engine is configured
        """
        # Validate query before execution
        validation_errors = self.validate()
        if validation_errors:
            error_messages = [str(error) for error in validation_errors]
            raise ValidationError(
                f"Query validation failed: {'; '.join(error_messages)}",
                suggestion="Check the validation errors and fix the query configuration"
            )
        
        if self._engine is None:
            # Return mock result for testing/demo purposes
            # Still track execution for consistency
            start_time = datetime.now()
            self._execution_count += 1
            
            import pandas as pd
            mock_data = pd.DataFrame({'message': ['Query executed successfully']})
            
            # Calculate execution time for mock
            execution_time = (datetime.now() - start_time).total_seconds()
            self._last_execution_time = execution_time
            
            # Create mock metadata
            mock_metadata = QueryMetadata(
                query_type='fluent_query',
                execution_time=execution_time,
                row_count=len(mock_data),
                processing_steps=self._get_processing_steps(),
                filters_applied=self._get_filters_summary(),
                cache_hit=False,
                optimization_applied=False
            )
            
            return AnalyticsResult(mock_data, mock_metadata)
        
        # Track execution
        start_time = datetime.now()
        self._execution_count += 1
        
        try:
            # Execute query through the analytics engine
            result = self._engine._execute_query(self._state)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            self._last_execution_time = execution_time
            
            # Update result metadata
            if isinstance(result, AnalyticsResult):
                result.metadata.execution_time = execution_time
                result.metadata.query_type = 'fluent_query'
                result.metadata.processing_steps = self._get_processing_steps()
                result.metadata.filters_applied = self._get_filters_summary()
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")
    
    def count(self) -> int:
        """
        Get count of results without retrieving full data.
        
        Returns:
            Number of rows that would be returned by the query
        """
        if self._engine is None:
            raise RuntimeError("No analytics engine configured")
        
        # Use optimized count query
        return self._engine._execute_count_query(self._state)
    
    def validate(self) -> List[ValidationError]:
        """
        Validate the current query configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check if at least one filter is applied
        has_filters = (
            bool(self._state.filters.experiments) or
            bool(self._state.filters.trials) or
            bool(self._state.filters.runs) or
            bool(self._state.filters.metrics)
        )
        
        if not has_filters:
            errors.append(ValidationError(
                "Query must have at least one filter applied",
                suggestion="Add filters using .experiments(), .trials(), .runs(), or .metrics()"
            ))
        
        # Validate aggregation dependencies
        if self._state.processing.aggregations and not self._state.processing.groupings:
            errors.append(ValidationError(
                "Aggregation requires grouping to be specified",
                field="aggregation",
                suggestion="Add .group_by() before using .aggregate()"
            ))
        
        # Validate outlier exclusion dependencies
        for outlier_config in self._state.processing.outlier_exclusions:
            metric_type = outlier_config['metric_type']
            metric_types = self._state.filters.metrics.get('types', [])
            
            if metric_types and metric_type not in metric_types:
                errors.append(ValidationError(
                    f"Outlier exclusion metric '{metric_type}' not in filtered metrics: {metric_types}",
                    field="exclude_outliers",
                    suggestion=f"Add '{metric_type}' to .metrics() or remove outlier exclusion"
                ))
        
        # Check for conflicting processing steps
        if len(self._state.processing.aggregations) > 1:
            errors.append(ValidationError(
                "Multiple aggregation steps are not supported",
                field="aggregation",
                suggestion="Combine all aggregation functions into a single .aggregate() call"
            ))
        
        return errors
    
    # === UTILITY METHODS ===
    
    def _get_processing_steps(self) -> List[str]:
        """Get list of processing steps for metadata."""
        steps = []
        
        if self._state.processing.outlier_exclusions:
            steps.extend([f"exclude_outliers_{config['method']}" for config in self._state.processing.outlier_exclusions])
        
        if self._state.processing.groupings:
            steps.append(f"group_by_{','.join(self._state.processing.groupings)}")
        
        if self._state.processing.aggregations:
            for agg_config in self._state.processing.aggregations:
                steps.append(f"aggregate_{','.join(agg_config['functions'])}")
        
        if self._state.processing.sorting:
            direction = "asc" if self._state.processing.sorting['ascending'] else "desc"
            steps.append(f"sort_{self._state.processing.sorting['field']}_{direction}")
        
        return steps
    
    def _get_filters_summary(self) -> Dict[str, Any]:
        """Get summary of applied filters for metadata."""
        return {
            'experiments': dict(self._state.filters.experiments),
            'trials': dict(self._state.filters.trials),
            'runs': dict(self._state.filters.runs),
            'metrics': dict(self._state.filters.metrics)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary representation."""
        return {
            'query_id': self._state.query_id,
            'created_at': self._state.created_at.isoformat(),
            'filters': {
                'experiments': dict(self._state.filters.experiments),
                'trials': dict(self._state.filters.trials),
                'runs': dict(self._state.filters.runs),
                'metrics': dict(self._state.filters.metrics)
            },
            'processing': {
                'outlier_exclusions': self._state.processing.outlier_exclusions,
                'aggregations': self._state.processing.aggregations,
                'groupings': self._state.processing.groupings,
                'sorting': self._state.processing.sorting
            },
            'execution_options': dict(self._state.execution_options),
            'execution_count': self._execution_count,
            'last_execution_time': self._last_execution_time
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert query to JSON representation."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def __repr__(self) -> str:
        """String representation of the query."""
        filters_count = sum([
            len(self._state.filters.experiments),
            len(self._state.filters.trials),
            len(self._state.filters.runs),
            len(self._state.filters.metrics)
        ])
        
        processing_count = (
            len(self._state.processing.outlier_exclusions) +
            len(self._state.processing.aggregations) +
            len(self._state.processing.groupings) +
            (1 if self._state.processing.sorting else 0)
        )
        
        return f"AnalyticsQuery(id={self._state.query_id}, filters={filters_count}, processing={processing_count})"
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AnalyticsQuery':
        """Create AnalyticsQuery from configuration dict."""
        query = cls()
        
        # Restore filters
        if 'filters' in config:
            filters = config['filters']
            if 'experiments' in filters:
                query = query.experiments(**filters['experiments'])
            if 'trials' in filters:
                query = query.trials(**filters['trials'])
            if 'runs' in filters:
                query = query.runs(**filters['runs'])
            if 'metrics' in filters:
                query = query.metrics(**filters['metrics'])
        
        # Restore processing
        if 'processing' in config:
            processing = config['processing']
            
            for outlier_config in processing.get('outlier_exclusions', []):
                query = query.exclude_outliers(**outlier_config)
            
            for agg_config in processing.get('aggregations', []):
                query = query.aggregate(agg_config['functions'])
            
            for grouping in processing.get('groupings', []):
                query = query.group_by(grouping)
            
            if processing.get('sorting'):
                sort_config = processing['sorting']
                query = query.sort_by(sort_config['field'], sort_config['ascending'])
        
        return query
    
    def to_config(self) -> Dict[str, Any]:
        """Convert to configuration dict."""
        return self.to_dict() 