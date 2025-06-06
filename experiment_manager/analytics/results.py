"""
Analytics Results Container

Provides structured access to analytics query results with comprehensive
metadata and data manipulation capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, field
from experiment_manager.common.serializable import YAMLSerializable


@dataclass
class QueryMetadata:
    """Metadata about an analytics query execution."""
    query_type: str
    execution_time: float
    row_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    processing_steps: List[str] = field(default_factory=list)
    cache_hit: bool = False
    optimization_applied: bool = False
    warnings: List[str] = field(default_factory=list)


@YAMLSerializable.register("AnalyticsResult")
class AnalyticsResult(YAMLSerializable):
    """
    Container for analytics query results.
    
    Provides structured access to analytics data with comprehensive metadata,
    data manipulation capabilities, and export functionality.
    """
    
    def __init__(self, 
                 data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]], 
                 metadata: Optional[Union[QueryMetadata, Dict[str, Any]]] = None):
        """
        Initialize AnalyticsResult.
        
        Args:
            data: Query result data (DataFrame, dict, or list of dicts)
            metadata: Query execution metadata
        """
        self.data = self._normalize_data(data)
        self.metadata = self._normalize_metadata(metadata)
    
    def _normalize_data(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Convert data to pandas DataFrame format."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            # Convert dict to DataFrame
            if all(isinstance(v, (list, tuple)) for v in data.values()):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame([data])
        elif isinstance(data, (list, tuple)):
            if data and isinstance(data[0], dict):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame({'value': data})
        else:
            # Single value
            return pd.DataFrame({'value': [data]})
    
    def _normalize_metadata(self, metadata: Optional[Union[QueryMetadata, Dict[str, Any]]]) -> QueryMetadata:
        """Convert metadata to QueryMetadata object."""
        if isinstance(metadata, QueryMetadata):
            return metadata
        elif isinstance(metadata, dict):
            # Create QueryMetadata from dict, filling defaults for missing keys
            return QueryMetadata(
                query_type=metadata.get('query_type', 'unknown'),
                execution_time=metadata.get('execution_time', 0.0),
                row_count=metadata.get('row_count', len(self.data) if hasattr(self, 'data') else 0),
                timestamp=metadata.get('timestamp', datetime.now()),
                filters_applied=metadata.get('filters_applied', {}),
                processing_steps=metadata.get('processing_steps', []),
                cache_hit=metadata.get('cache_hit', False),
                optimization_applied=metadata.get('optimization_applied', False),
                warnings=metadata.get('warnings', [])
            )
        else:
            # Default metadata
            return QueryMetadata(
                query_type='unknown',
                execution_time=0.0,
                row_count=len(self.data) if hasattr(self, 'data') else 0
            )
    
    @property
    def is_empty(self) -> bool:
        """Check if result data is empty."""
        return self.data.empty
    
    @property
    def shape(self) -> tuple:
        """Get shape of the result data."""
        return self.data.shape
    
    @property
    def columns(self) -> List[str]:
        """Get column names from the result data."""
        return list(self.data.columns)
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get first n rows of result data."""
        return self.data.head(n)
    
    def tail(self, n: int = 5) -> pd.DataFrame:
        """Get last n rows of result data."""
        return self.data.tail(n)
    
    def describe(self) -> pd.DataFrame:
        """Get statistical summary of numeric columns."""
        return self.data.describe()
    
    def to_dict(self, orient: str = 'records') -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Convert result data to dictionary format."""
        return self.data.to_dict(orient=orient)
    
    def to_csv(self, path: Optional[str] = None, **kwargs) -> Optional[str]:
        """Export result data to CSV format."""
        if path:
            self.data.to_csv(path, index=False, **kwargs)
            return path
        else:
            return self.data.to_csv(index=False, **kwargs)
    
    def to_json(self, path: Optional[str] = None, **kwargs) -> Optional[str]:
        """Export result data to JSON format."""
        if path:
            self.data.to_json(path, orient='records', **kwargs)
            return path
        else:
            return self.data.to_json(orient='records', **kwargs)
    
    def filter(self, condition) -> 'AnalyticsResult':
        """Filter result data based on condition."""
        filtered_data = self.data[condition]
        
        # Update metadata
        new_metadata = QueryMetadata(
            query_type=f"{self.metadata.query_type}_filtered",
            execution_time=self.metadata.execution_time,
            row_count=len(filtered_data),
            timestamp=datetime.now(),
            filters_applied=self.metadata.filters_applied.copy(),
            processing_steps=self.metadata.processing_steps + ['filter'],
            cache_hit=False,
            optimization_applied=self.metadata.optimization_applied
        )
        
        return AnalyticsResult(filtered_data, new_metadata)
    
    def sort_values(self, by: Union[str, List[str]], ascending: bool = True) -> 'AnalyticsResult':
        """Sort result data by specified columns."""
        sorted_data = self.data.sort_values(by=by, ascending=ascending)
        
        # Update metadata
        new_metadata = QueryMetadata(
            query_type=f"{self.metadata.query_type}_sorted",
            execution_time=self.metadata.execution_time,
            row_count=len(sorted_data),
            timestamp=datetime.now(),
            filters_applied=self.metadata.filters_applied.copy(),
            processing_steps=self.metadata.processing_steps + [f'sort_by_{by}'],
            cache_hit=False,
            optimization_applied=self.metadata.optimization_applied
        )
        
        return AnalyticsResult(sorted_data, new_metadata)
    
    def groupby(self, by: Union[str, List[str]]) -> pd.core.groupby.DataFrameGroupBy:
        """Group result data by specified columns."""
        return self.data.groupby(by)
    
    def aggregate(self, agg_dict: Dict[str, Union[str, List[str]]]) -> 'AnalyticsResult':
        """Aggregate result data using specified functions."""
        if not isinstance(agg_dict, dict):
            raise ValueError("agg_dict must be a dictionary mapping columns to aggregation functions")
        
        aggregated_data = self.data.agg(agg_dict)
        
        # Flatten MultiIndex columns if needed
        if isinstance(aggregated_data.columns, pd.MultiIndex):
            aggregated_data.columns = ['_'.join(col).strip() for col in aggregated_data.columns]
        
        # Convert Series to DataFrame if needed
        if isinstance(aggregated_data, pd.Series):
            aggregated_data = aggregated_data.to_frame().T
        
        # Update metadata
        new_metadata = QueryMetadata(
            query_type=f"{self.metadata.query_type}_aggregated",
            execution_time=self.metadata.execution_time,
            row_count=len(aggregated_data),
            timestamp=datetime.now(),
            filters_applied=self.metadata.filters_applied.copy(),
            processing_steps=self.metadata.processing_steps + ['aggregate'],
            cache_hit=False,
            optimization_applied=self.metadata.optimization_applied
        )
        
        return AnalyticsResult(aggregated_data, new_metadata)
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the result."""
        return {
            'shape': self.shape,
            'columns': self.columns,
            'data_types': dict(self.data.dtypes),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'null_counts': dict(self.data.isnull().sum()),
            'metadata': {
                'query_type': self.metadata.query_type,
                'execution_time': self.metadata.execution_time,
                'timestamp': self.metadata.timestamp.isoformat(),
                'cache_hit': self.metadata.cache_hit,
                'optimization_applied': self.metadata.optimization_applied,
                'processing_steps': self.metadata.processing_steps,
                'warnings': self.metadata.warnings
            }
        }
    
    def __repr__(self) -> str:
        """String representation of AnalyticsResult."""
        return f"AnalyticsResult(shape={self.shape}, query_type='{self.metadata.query_type}')"
    
    def __len__(self) -> int:
        """Get number of rows in result data."""
        return len(self.data)
    
    def __getitem__(self, key):
        """Get data column or slice."""
        return self.data[key]
    
    def __iter__(self):
        """Iterate over result data rows."""
        return self.data.iterrows()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AnalyticsResult':
        """Create AnalyticsResult from configuration."""
        data = config.get('data', pd.DataFrame())
        metadata = config.get('metadata', {})
        return cls(data, metadata)
    
    def to_config(self) -> Dict[str, Any]:
        """Convert AnalyticsResult to configuration dict."""
        return {
            'data': self.data.to_dict('records'),
            'metadata': {
                'query_type': self.metadata.query_type,
                'execution_time': self.metadata.execution_time,
                'row_count': self.metadata.row_count,
                'timestamp': self.metadata.timestamp.isoformat(),
                'filters_applied': self.metadata.filters_applied,
                'processing_steps': self.metadata.processing_steps,
                'cache_hit': self.metadata.cache_hit,
                'optimization_applied': self.metadata.optimization_applied,
                'warnings': self.metadata.warnings
            }
        } 