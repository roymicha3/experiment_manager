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
from .base import DataProcessor


class FailureAnalyzer(DataProcessor):
    """
    Processor for analyzing failure patterns in experiment data.
    
    Analysis Capabilities:
    - Failure rate calculations by experiment/trial/configuration
    - Configuration correlation analysis
    - Temporal failure pattern detection  
    - Root cause suggestion algorithms
    """
    
    def __init__(self, name: str = "FailureAnalyzer"):
        """Initialize the failure analysis processor."""
        super().__init__(name)
        self.required_columns = ['run_status']
    
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
            corr_result = self._analyze_configuration_correlations(
                processed_data, config_columns, failure_threshold, min_samples
            )
            results.append(corr_result)
        
        if analysis_type in ['temporal', 'all'] and time_column:
            temp_result = self._analyze_temporal_patterns(
                processed_data, time_column, time_window, group_by
            )
            results.append(temp_result)
        
        if analysis_type in ['root_cause', 'all']:
            root_cause_result = self._suggest_root_causes(
                processed_data, config_columns, time_column, failure_threshold
            )
            results.append(root_cause_result)
        
        # Combine all results
        if len(results) == 1:
            return results[0]
        else:
            # Create a comprehensive summary
            return self._combine_analysis_results(results, analysis_type)
    
    def _standardize_failure_status(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize failure status to boolean values."""
        # Common failure indicators
        failure_indicators = ['failed', 'error', 'cancelled', 'timeout', 'stopped']
        success_indicators = ['success', 'completed', 'done', 'finished']
        
        # Create standardized failure column
        data['is_failure'] = False
        
        for indicator in failure_indicators:
            mask = data['run_status'].str.lower().str.contains(indicator, na=False)
            data.loc[mask, 'is_failure'] = True
        
        return data
    
    def _analyze_failure_rates(self, data: pd.DataFrame, group_by: Optional[Union[str, List[str]]], 
                             min_samples: int) -> pd.DataFrame:
        """Analyze failure rates across different groupings."""
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            
            # Group-wise failure rates
            results = []
            for group_key, group_data in data.groupby(group_by):
                if len(group_data) < min_samples:
                    continue
                
                # Calculate failure statistics
                total_runs = len(group_data)
                failure_count = group_data['is_failure'].sum()
                failure_rate = failure_count / total_runs
                
                # Group identifiers
                if isinstance(group_key, tuple):
                    group_dict = dict(zip(group_by, group_key))
                else:
                    group_dict = {group_by[0]: group_key}
                
                result_dict = {
                    **group_dict,
                    'total_runs': total_runs,
                    'failure_count': int(failure_count),
                    'success_count': int(total_runs - failure_count),
                    'failure_rate': float(failure_rate),
                    'success_rate': float(1 - failure_rate),
                    'analysis_type': 'failure_rates'
                }
                
                results.append(result_dict)
            
            return pd.DataFrame(results)
        
        else:
            # Overall failure rates
            total_runs = len(data)
            failure_count = data['is_failure'].sum()
            failure_rate = failure_count / total_runs if total_runs > 0 else 0
            
            return pd.DataFrame([{
                'total_runs': total_runs,
                'failure_count': int(failure_count),
                'success_count': int(total_runs - failure_count),
                'failure_rate': float(failure_rate),
                'success_rate': float(1 - failure_rate),
                'analysis_type': 'failure_rates'
            }])
    
    def _analyze_configuration_correlations(self, data: pd.DataFrame, config_columns: List[str],
                                          failure_threshold: float, min_samples: int) -> pd.DataFrame:
        """Analyze correlations between configuration parameters and failures."""
        results = []
        
        for config_col in config_columns:
            if config_col not in data.columns:
                continue
            
            # Analyze each configuration value
            for config_value, config_data in data.groupby(config_col):
                if len(config_data) < min_samples:
                    continue
                
                failure_rate = config_data['is_failure'].mean()
                overall_failure_rate = data['is_failure'].mean()
                
                # Calculate correlation strength
                correlation_strength = abs(failure_rate - overall_failure_rate)
                
                if correlation_strength >= failure_threshold:
                    result_dict = {
                        'config_parameter': config_col,
                        'config_value': config_value,
                        'total_runs': len(config_data),
                        'failure_count': int(config_data['is_failure'].sum()),
                        'failure_rate': float(failure_rate),
                        'overall_failure_rate': float(overall_failure_rate),
                        'correlation_strength': float(correlation_strength),
                        'risk_level': self._classify_risk_level(failure_rate),
                        'analysis_type': 'configuration_correlation'
                    }
                    
                    results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def _analyze_temporal_patterns(self, data: pd.DataFrame, time_column: str, 
                                 time_window: str, group_by: Optional[Union[str, List[str]]]) -> pd.DataFrame:
        """Analyze temporal failure patterns."""
        if time_column not in data.columns:
            return pd.DataFrame()
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])
        
        # Create time windows
        if time_window == 'hour':
            data['time_window'] = data[time_column].dt.floor('H')
        elif time_window == 'day':
            data['time_window'] = data[time_column].dt.date
        elif time_window == 'week':
            data['time_window'] = data[time_column].dt.to_period('W')
        else:
            raise ValueError(f"Unsupported time window: {time_window}")
        
        # Analyze patterns by time window
        grouping_cols = ['time_window']
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            grouping_cols.extend(group_by)
        
        results = []
        for group_key, group_data in data.groupby(grouping_cols):
            if len(group_data) < 3:  # Need minimum samples for temporal analysis
                continue
            
            failure_rate = group_data['is_failure'].mean()
            
            # Create result dictionary
            if isinstance(group_key, tuple):
                if len(grouping_cols) == 1:
                    result_dict = {'time_window': group_key[0]}
                else:
                    result_dict = dict(zip(grouping_cols, group_key))
            else:
                result_dict = {'time_window': group_key}
            
            result_dict.update({
                'total_runs': len(group_data),
                'failure_count': int(group_data['is_failure'].sum()),
                'failure_rate': float(failure_rate),
                'time_window_type': time_window,
                'analysis_type': 'temporal_pattern'
            })
            
            results.append(result_dict)
        
        temporal_df = pd.DataFrame(results)
        
        # Add trend analysis
        if len(temporal_df) > 1:
            temporal_df = temporal_df.sort_values('time_window')
            temporal_df['failure_rate_trend'] = temporal_df['failure_rate'].diff()
            temporal_df['is_trending_up'] = temporal_df['failure_rate_trend'] > 0
        
        return temporal_df
    
    def _suggest_root_causes(self, data: pd.DataFrame, config_columns: List[str], 
                           time_column: Optional[str], failure_threshold: float) -> pd.DataFrame:
        """Generate root cause suggestions based on failure patterns."""
        suggestions = []
        
        # Overall failure rate
        overall_failure_rate = data['is_failure'].mean()
        
        # 1. High overall failure rate
        if overall_failure_rate > failure_threshold:
            suggestions.append({
                'root_cause_type': 'high_overall_failure_rate',
                'description': f'Overall failure rate ({overall_failure_rate:.2%}) exceeds threshold',
                'severity': 'high' if overall_failure_rate > 0.3 else 'medium',
                'suggested_action': 'Review system configuration and resource allocation',
                'evidence': f'Failure rate: {overall_failure_rate:.2%}',
                'confidence': 'high',
                'analysis_type': 'root_cause'
            })
        
        # 2. Configuration-based issues
        for config_col in config_columns:
            if config_col in data.columns:
                config_failure_rates = data.groupby(config_col)['is_failure'].agg(['mean', 'count'])
                
                # Find problematic configurations
                problematic_configs = config_failure_rates[
                    (config_failure_rates['mean'] > failure_threshold) & 
                    (config_failure_rates['count'] >= 5)
                ]
                
                for config_value, row in problematic_configs.iterrows():
                    suggestions.append({
                        'root_cause_type': 'configuration_issue',
                        'description': f'High failure rate for {config_col}={config_value}',
                        'severity': 'high' if row['mean'] > 0.5 else 'medium',
                        'suggested_action': f'Review {config_col} parameter settings',
                        'evidence': f'{config_col}={config_value}: {row["mean"]:.2%} failure rate ({int(row["count"])} runs)',
                        'confidence': 'high' if row['count'] >= 10 else 'medium',
                        'analysis_type': 'root_cause'
                    })
        
        # 3. Temporal patterns
        if time_column and time_column in data.columns:
            # Check for recent increase in failures
            data_sorted = data.sort_values(time_column)
            if len(data_sorted) >= 20:
                recent_data = data_sorted.tail(int(len(data_sorted) * 0.2))  # Last 20% of data
                early_data = data_sorted.head(int(len(data_sorted) * 0.2))   # First 20% of data
                
                recent_failure_rate = recent_data['is_failure'].mean()
                early_failure_rate = early_data['is_failure'].mean()
                
                if recent_failure_rate > early_failure_rate + 0.1:  # 10% increase
                    suggestions.append({
                        'root_cause_type': 'degrading_performance',
                        'description': 'Increasing failure rate over time detected',
                        'severity': 'high',
                        'suggested_action': 'Investigate system degradation, check for resource leaks or capacity issues',
                        'evidence': f'Recent failure rate: {recent_failure_rate:.2%}, Early failure rate: {early_failure_rate:.2%}',
                        'confidence': 'medium',
                        'analysis_type': 'root_cause'
                    })
        
        # 4. Sample size issues
        if len(data) < 10:
            suggestions.append({
                'root_cause_type': 'insufficient_data',
                'description': 'Insufficient data for reliable failure analysis',
                'severity': 'low',
                'suggested_action': 'Collect more data before drawing conclusions',
                'evidence': f'Only {len(data)} samples available',
                'confidence': 'high',
                'analysis_type': 'root_cause'
            })
        
        return pd.DataFrame(suggestions)
    
    def _classify_risk_level(self, failure_rate: float) -> str:
        """Classify risk level based on failure rate."""
        if failure_rate >= 0.5:
            return 'very_high'
        elif failure_rate >= 0.3:
            return 'high'
        elif failure_rate >= 0.1:
            return 'medium'
        else:
            return 'low'
    
    def _combine_analysis_results(self, results: List[pd.DataFrame], analysis_type: str) -> pd.DataFrame:
        """Combine multiple analysis results into a single DataFrame."""
        combined_results = []
        
        for result_df in results:
            if not result_df.empty:
                combined_results.append(result_df)
        
        if combined_results:
            final_df = pd.concat(combined_results, ignore_index=True, sort=False)
            return final_df
        else:
            return pd.DataFrame({'analysis_type': [analysis_type], 'message': ['No significant patterns found']})
    
    def get_failure_summary(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Get a high-level summary of failure analysis results.
        
        Args:
            data: Input DataFrame  
            **kwargs: Analysis parameters
            
        Returns:
            Dict containing failure analysis summary
        """
        result_df = self._process_implementation(data, **kwargs)
        
        summary = {
            'total_analyses': len(result_df),
            'analysis_types': result_df['analysis_type'].unique().tolist() if 'analysis_type' in result_df.columns else []
        }
        
        # Add type-specific summaries
        if 'analysis_type' in result_df.columns:
            for analysis_type in result_df['analysis_type'].unique():
                type_data = result_df[result_df['analysis_type'] == analysis_type]
                
                if analysis_type == 'failure_rates':
                    summary[f'{analysis_type}_summary'] = {
                        'avg_failure_rate': float(type_data['failure_rate'].mean()) if 'failure_rate' in type_data.columns else 0,
                        'max_failure_rate': float(type_data['failure_rate'].max()) if 'failure_rate' in type_data.columns else 0,
                        'groups_analyzed': len(type_data)
                    }
                elif analysis_type == 'root_cause':
                    summary[f'{analysis_type}_summary'] = {
                        'total_suggestions': len(type_data),
                        'high_severity_count': int((type_data['severity'] == 'high').sum()) if 'severity' in type_data.columns else 0,
                        'high_confidence_count': int((type_data['confidence'] == 'high').sum()) if 'confidence' in type_data.columns else 0
                    }
        
        return summary 