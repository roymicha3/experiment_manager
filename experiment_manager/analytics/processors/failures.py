"""
Failure Analysis Processor for Analytics

Provides comprehensive failure analysis capabilities including failure rate calculations,
configuration correlation analysis, temporal pattern detection, and root cause suggestions.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from omegaconf import DictConfig

from experiment_manager.analytics.processors.base import DataProcessor
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("FailureAnalyzer")
class FailureAnalyzer(DataProcessor):
    """
    Processor for analyzing failure patterns in experiment data.
    
    Analysis Capabilities:
    - Failure rate calculations by experiment/trial/configuration
    - Configuration correlation analysis
    - Temporal failure pattern detection  
    - Root cause suggestion algorithms
    """
    
    def __init__(self, name: str = "FailureAnalyzer", config: DictConfig = None):
        """Initialize the failure analysis processor."""
        super().__init__(name, config)
        self.required_columns = ['run_status']
        self.optional_columns = ['experiment_name', 'trial_name', 'epoch', 'start_time', 'end_time']
    
    @classmethod
    def from_config(cls, config: DictConfig, **kwargs):
        """Create FailureAnalyzer from configuration."""
        # Get failure-specific configuration
        failure_config = config.get('failures', DictConfig({}))
        name = kwargs.get('name', 'FailureAnalyzer')
        return cls(name=name, config=failure_config)
    
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Analyze failure patterns in the input data.
        
        Args:
            data: Input DataFrame with experiment run data
            **kwargs: Processing parameters:
                - analysis_type: Type of analysis ('rates', 'correlations', 'temporal', 'root_cause', 'all')
                - group_by: Column(s) to group analysis by (optional)
                - time_column: Column containing timestamps for temporal analysis
                - time_window: Time window for temporal analysis ('hour', 'day', 'week')
                - config_columns: Columns to analyze for configuration correlations
                - failure_threshold: Minimum failure rate to consider significant (default: 0.1)
                - min_samples: Minimum samples needed for reliable analysis (default: 10)
                
        Returns:
            pd.DataFrame: Failure analysis results
        """
        # Extract parameters
        analysis_type = kwargs.get('analysis_type', 'all')
        group_by = kwargs.get('group_by', None)
        time_column = kwargs.get('time_column', None)
        time_window = kwargs.get('time_window', 'day')
        config_columns = kwargs.get('config_columns', [])
        failure_threshold = kwargs.get('failure_threshold', 0.1)
        min_samples = kwargs.get('min_samples', 10)
        
        # Standardize failure status
        processed_data = self._standardize_failure_status(data.copy())
        
        results = []
        
        if analysis_type in ['rates', 'all']:
            rates_result = self._analyze_failure_rates(processed_data, group_by, min_samples)
            results.append(rates_result)
        
        if analysis_type in ['correlations', 'all'] and config_columns:
            correlations_result = self._analyze_config_correlations(
                processed_data, config_columns, failure_threshold, min_samples
            )
            results.append(correlations_result)
        
        if analysis_type in ['temporal', 'all'] and time_column:
            temporal_result = self._analyze_temporal_patterns(
                processed_data, time_column, time_window, group_by
            )
            results.append(temporal_result)
        
        if analysis_type in ['root_cause', 'all']:
            root_cause_result = self._analyze_root_causes(
                processed_data, config_columns, failure_threshold, min_samples
            )
            results.append(root_cause_result)
        
        # Combine all results
        if results:
            combined_result = pd.concat(results, ignore_index=True)
            return combined_result
        else:
            return pd.DataFrame()
    
    def _standardize_failure_status(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize failure status values to a consistent format."""
        # Map various status values to standardized ones
        status_mapping = {
            'failed': 'failed',
            'error': 'failed',
            'exception': 'failed',
            'timeout': 'failed',
            'cancelled': 'failed',
            'success': 'success',
            'completed': 'success',
            'finished': 'success',
            'done': 'success',
            'running': 'running',
            'pending': 'pending',
            'queued': 'pending'
        }
        
        # Standardize the run_status column
        if 'run_status' in data.columns:
            data['run_status_std'] = data['run_status'].str.lower().map(status_mapping)
            data['run_status_std'] = data['run_status_std'].fillna('unknown')
        else:
            data['run_status_std'] = 'unknown'
        
        # Create binary failure flag
        data['is_failure'] = data['run_status_std'] == 'failed'
        
        return data
    
    def _analyze_failure_rates(self, data: pd.DataFrame, group_by: Optional[Union[str, List[str]]], 
                              min_samples: int) -> pd.DataFrame:
        """Analyze failure rates overall and by groups."""
        results = []
        
        if group_by:
            # Group-wise failure rate analysis
            if isinstance(group_by, str):
                group_by = [group_by]
            
            # Validate group_by columns exist
            missing_group_cols = set(group_by) - set(data.columns)
            if missing_group_cols:
                raise ValueError(f"Group-by columns not found in data: {missing_group_cols}")
            
            grouped = data.groupby(group_by)
            
            for name, group_data in grouped:
                if len(group_data) < min_samples:
                    continue
                
                failure_stats = self._calculate_failure_stats(group_data)
                
                # Add group identifiers
                if isinstance(name, tuple):
                    for i, col in enumerate(group_by):
                        failure_stats[col] = name[i]
                else:
                    failure_stats[group_by[0]] = name
                
                failure_stats['analysis_type'] = 'rates'
                failure_stats['analysis_scope'] = 'group'
                results.append(failure_stats)
        else:
            # Overall failure rate analysis
            if len(data) >= min_samples:
                overall_stats = self._calculate_failure_stats(data)
                overall_stats['group'] = 'overall'
                overall_stats['analysis_type'] = 'rates'
                overall_stats['analysis_scope'] = 'overall'
                results.append(overall_stats)
        
        return pd.DataFrame(results)
    
    def _calculate_failure_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate failure statistics for a group of data."""
        total_runs = len(data)
        failed_runs = data['is_failure'].sum()
        success_runs = (data['run_status_std'] == 'success').sum()
        
        failure_rate = failed_runs / total_runs if total_runs > 0 else 0
        success_rate = success_runs / total_runs if total_runs > 0 else 0
        
        return {
            'total_runs': int(total_runs),
            'failed_runs': int(failed_runs),
            'success_runs': int(success_runs),
            'failure_rate': float(failure_rate),
            'success_rate': float(success_rate),
            'completion_rate': float((failed_runs + success_runs) / total_runs) if total_runs > 0 else 0
        }
    
    def _analyze_config_correlations(self, data: pd.DataFrame, config_columns: List[str],
                                   failure_threshold: float, min_samples: int) -> pd.DataFrame:
        """Analyze correlations between configuration values and failures."""
        results = []
        
        # Validate config columns exist
        valid_config_cols = [col for col in config_columns if col in data.columns]
        if not valid_config_cols:
            return pd.DataFrame()
        
        for config_col in valid_config_cols:
            # Group by configuration value
            config_groups = data.groupby(config_col)
            
            for config_value, config_data in config_groups:
                if len(config_data) < min_samples:
                    continue
                
                failure_stats = self._calculate_failure_stats(config_data)
                
                if failure_stats['failure_rate'] >= failure_threshold:
                    correlation_result = {
                        'config_column': config_col,
                        'config_value': str(config_value),
                        'analysis_type': 'correlations',
                        'analysis_scope': 'configuration',
                        **failure_stats
                    }
                    results.append(correlation_result)
        
        return pd.DataFrame(results)
    
    def _analyze_temporal_patterns(self, data: pd.DataFrame, time_column: str, 
                                  time_window: str, group_by: Optional[Union[str, List[str]]]) -> pd.DataFrame:
        """Analyze temporal failure patterns."""
        results = []
        
        if time_column not in data.columns:
            return pd.DataFrame()
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            try:
                data[time_column] = pd.to_datetime(data[time_column])
            except:
                return pd.DataFrame()
        
        # Create time windows
        if time_window == 'hour':
            data['time_window'] = data[time_column].dt.floor('H')
            window_name = 'hourly'
        elif time_window == 'day':
            data['time_window'] = data[time_column].dt.floor('D')
            window_name = 'daily'
        elif time_window == 'week':
            data['time_window'] = data[time_column].dt.floor('W')
            window_name = 'weekly'
        else:
            return pd.DataFrame()
        
        if group_by:
            # Group by both time window and specified columns
            if isinstance(group_by, str):
                group_by = [group_by]
            
            group_cols = ['time_window'] + group_by
            temporal_groups = data.groupby(group_cols)
            
            for name, temporal_data in temporal_groups:
                failure_stats = self._calculate_failure_stats(temporal_data)
                
                temporal_result = {
                    'time_window': name[0],
                    'window_type': window_name,
                    'analysis_type': 'temporal',
                    'analysis_scope': 'temporal_group',
                    **failure_stats
                }
                
                # Add group identifiers
                for i, col in enumerate(group_by):
                    temporal_result[col] = name[i + 1]
                
                results.append(temporal_result)
        else:
            # Overall temporal analysis
            temporal_groups = data.groupby('time_window')
            
            for time_window_val, temporal_data in temporal_groups:
                failure_stats = self._calculate_failure_stats(temporal_data)
                
                temporal_result = {
                    'time_window': time_window_val,
                    'window_type': window_name,
                    'analysis_type': 'temporal',
                    'analysis_scope': 'temporal_overall',
                    **failure_stats
                }
                results.append(temporal_result)
        
        return pd.DataFrame(results)
    
    def _analyze_root_causes(self, data: pd.DataFrame, config_columns: List[str],
                           failure_threshold: float, min_samples: int) -> pd.DataFrame:
        """Analyze potential root causes of failures using simple heuristics."""
        results = []
        
        # Only analyze failed runs
        failed_data = data[data['is_failure'] == True]
        if len(failed_data) < min_samples:
            return pd.DataFrame()
        
        # Analyze configuration combinations that lead to high failure rates
        valid_config_cols = [col for col in config_columns if col in data.columns]
        
        if len(valid_config_cols) >= 2:
            # Analyze pairs of configuration values
            for i in range(len(valid_config_cols)):
                for j in range(i + 1, len(valid_config_cols)):
                    col1, col2 = valid_config_cols[i], valid_config_cols[j]
                    
                    # Group by combination of two config values
                    combo_groups = data.groupby([col1, col2])
                    
                    for combo_values, combo_data in combo_groups:
                        if len(combo_data) < min_samples:
                            continue
                        
                        failure_stats = self._calculate_failure_stats(combo_data)
                        
                        if failure_stats['failure_rate'] >= failure_threshold:
                            root_cause_result = {
                                'root_cause_type': 'config_combination',
                                f'{col1}_value': str(combo_values[0]),
                                f'{col2}_value': str(combo_values[1]),
                                'analysis_type': 'root_cause',
                                'analysis_scope': 'configuration_combination',
                                **failure_stats
                            }
                            results.append(root_cause_result)
        
        # Analyze individual high-impact configurations
        for config_col in valid_config_cols:
            config_groups = data.groupby(config_col)
            
            for config_value, config_data in config_groups:
                if len(config_data) < min_samples:
                    continue
                
                failure_stats = self._calculate_failure_stats(config_data)
                
                # Consider this a root cause if failure rate is significantly high
                if failure_stats['failure_rate'] >= failure_threshold * 2:  # Double the threshold for single factors
                    root_cause_result = {
                        'root_cause_type': 'single_config',
                        'config_column': config_col,
                        'config_value': str(config_value),
                        'analysis_type': 'root_cause',
                        'analysis_scope': 'single_configuration',
                        **failure_stats
                    }
                    results.append(root_cause_result)
        
        return pd.DataFrame(results)
    
    def get_failure_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a high-level summary of failure analysis results.
        
        Args:
            data: DataFrame with failure analysis results
            
        Returns:
            Dict with summary statistics
        """
        # Check if this is processed failure analysis data or raw data
        if 'analysis_type' in data.columns:
            # This is processed failure analysis data
            return self._summarize_analysis_results(data)
        else:
            # This is raw data, process it first
            processed_data = self._standardize_failure_status(data.copy())
            return self._summarize_raw_data(processed_data)
    
    def _summarize_analysis_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize processed failure analysis results."""
        summary = {'analysis_types': data['analysis_type'].unique().tolist()}
        
        # Rates analysis summary
        rates_data = data[data['analysis_type'] == 'rates']
        if not rates_data.empty:
            summary['rates'] = {
                'avg_failure_rate': float(rates_data['failure_rate'].mean()),
                'max_failure_rate': float(rates_data['failure_rate'].max()),
                'groups_analyzed': len(rates_data)
            }
        
        # Correlations summary
        corr_data = data[data['analysis_type'] == 'correlations']
        if not corr_data.empty:
            summary['correlations'] = {
                'high_risk_configs': len(corr_data),
                'config_columns_analyzed': corr_data['config_column'].nunique()
            }
        
        # Root causes summary
        root_cause_data = data[data['analysis_type'] == 'root_cause']
        if not root_cause_data.empty:
            summary['root_causes'] = {
                'identified_causes': len(root_cause_data),
                'cause_types': root_cause_data['root_cause_type'].value_counts().to_dict()
            }
        
        return summary
    
    def _summarize_raw_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize raw experiment data for failure patterns."""
        total_runs = len(data)
        failed_runs = data['is_failure'].sum()
        
        return {
            'total_runs': int(total_runs),
            'failed_runs': int(failed_runs),
            'failure_rate': float(failed_runs / total_runs) if total_runs > 0 else 0,
            'status_distribution': data['run_status_std'].value_counts().to_dict()
        } 