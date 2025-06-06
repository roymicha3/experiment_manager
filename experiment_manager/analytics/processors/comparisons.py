"""
Comparison Processor for Analytics

Provides comprehensive comparison analysis capabilities for cross-experiment analysis,
A/B testing, performance ranking, and improvement tracking.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from .base import DataProcessor


class ComparisonProcessor(DataProcessor):
    """
    Processor for performing comparative analysis between experiments, configurations, or groups.
    
    Comparison Capabilities:
    - Cross-experiment performance comparison
    - A/B testing statistical analysis
    - Performance ranking and benchmarking
    - Improvement tracking over time
    """
    
    def __init__(self, name: str = "ComparisonProcessor"):
        """Initialize the comparison processor."""
        super().__init__(name)
        self.required_columns = ['metric_total_val']
    
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Perform comparative analysis on the input data.
        
        Args:
            data: Input DataFrame with experiment data
            **kwargs: Processing parameters:
                - comparison_type: Type of comparison ('pairwise', 'ranking', 'ab_test', 'trend', 'all')
                - group_by: Column(s) to use for grouping comparisons
                - baseline_group: Reference group for comparisons (optional)
                - metric_columns: List of metric columns to compare (optional)
                - confidence_level: Statistical confidence level (default: 0.95)
                - significance_threshold: P-value threshold for significance (default: 0.05)
                - min_samples: Minimum samples per group (default: 5)
                - time_column: Column for trend analysis (optional)
                
        Returns:
            pd.DataFrame: Comparison analysis results
        """
        # Extract parameters
        comparison_type = kwargs.get('comparison_type', 'pairwise')
        group_by = kwargs.get('group_by', None)
        baseline_group = kwargs.get('baseline_group', None)
        metric_columns = kwargs.get('metric_columns', ['metric_total_val'])
        confidence_level = kwargs.get('confidence_level', 0.95)
        significance_threshold = kwargs.get('significance_threshold', 0.05)
        min_samples = kwargs.get('min_samples', 5)
        time_column = kwargs.get('time_column', None)
        
        if not group_by:
            raise ValueError("group_by parameter is required for comparison analysis")
        
        # Ensure metric_columns is a list
        if isinstance(metric_columns, str):
            metric_columns = [metric_columns]
        
        # Validate that metric columns exist
        missing_cols = set(metric_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Metric columns not found in data: {missing_cols}")
        
        results = []
        
        if comparison_type in ['pairwise', 'all']:
            pairwise_result = self._perform_pairwise_comparisons(
                data, group_by, metric_columns, confidence_level, significance_threshold, min_samples
            )
            results.append(pairwise_result)
        
        if comparison_type in ['ranking', 'all']:
            ranking_result = self._perform_ranking_analysis(
                data, group_by, metric_columns, min_samples
            )
            results.append(ranking_result)
        
        if comparison_type in ['ab_test', 'all']:
            ab_test_result = self._perform_ab_test_analysis(
                data, group_by, baseline_group, metric_columns, confidence_level, significance_threshold, min_samples
            )
            results.append(ab_test_result)
        
        if comparison_type in ['trend', 'all'] and time_column:
            trend_result = self._perform_trend_comparison(
                data, group_by, time_column, metric_columns, min_samples
            )
            results.append(trend_result)
        
        # Combine results
        if len(results) == 1:
            return results[0]
        else:
            return self._combine_comparison_results(results, comparison_type)
    
    def _perform_pairwise_comparisons(self, data: pd.DataFrame, group_by: Union[str, List[str]],
                                    metric_columns: List[str], confidence_level: float,
                                    significance_threshold: float, min_samples: int) -> pd.DataFrame:
        """Perform pairwise statistical comparisons between all groups."""
        if isinstance(group_by, str):
            group_by = [group_by]
        
        results = []
        
        # Get groups with sufficient data
        valid_groups = []
        group_data = {}
        
        for group_key, group_df in data.groupby(group_by):
            if len(group_df) >= min_samples:
                valid_groups.append(group_key)
                group_data[group_key] = group_df
        
        # Perform pairwise comparisons
        for i, group_a in enumerate(valid_groups):
            for j, group_b in enumerate(valid_groups):
                if i >= j:  # Avoid duplicate comparisons and self-comparison
                    continue
                
                data_a = group_data[group_a]
                data_b = group_data[group_b]
                
                for metric_col in metric_columns:
                    values_a = data_a[metric_col].dropna()
                    values_b = data_b[metric_col].dropna()
                    
                    if len(values_a) < 2 or len(values_b) < 2:
                        continue
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(values_a, values_b)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(values_a) - 1) * values_a.var() + 
                                        (len(values_b) - 1) * values_b.var()) / 
                                       (len(values_a) + len(values_b) - 2))
                    effect_size = (values_a.mean() - values_b.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    # Calculate confidence interval for difference
                    mean_diff = values_a.mean() - values_b.mean()
                    se_diff = np.sqrt(values_a.var() / len(values_a) + values_b.var() / len(values_b))
                    df = len(values_a) + len(values_b) - 2
                    t_crit = stats.t.ppf((1 + confidence_level) / 2, df)
                    ci_margin = t_crit * se_diff
                    
                    # Group identifiers
                    if isinstance(group_a, tuple):
                        group_a_dict = dict(zip([f'group_a_{col}' for col in group_by], group_a))
                        group_b_dict = dict(zip([f'group_b_{col}' for col in group_by], group_b))
                    else:
                        group_a_dict = {f'group_a_{group_by[0]}': group_a}
                        group_b_dict = {f'group_b_{group_by[0]}': group_b}
                    
                    result_dict = {
                        **group_a_dict,
                        **group_b_dict,
                        'metric': metric_col,
                        'group_a_mean': float(values_a.mean()),
                        'group_b_mean': float(values_b.mean()),
                        'group_a_std': float(values_a.std()),
                        'group_b_std': float(values_b.std()),
                        'group_a_count': len(values_a),
                        'group_b_count': len(values_b),
                        'mean_difference': float(mean_diff),
                        'effect_size': float(effect_size),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'is_significant': bool(p_value < significance_threshold),
                        'confidence_level': confidence_level,
                        'ci_lower': float(mean_diff - ci_margin),
                        'ci_upper': float(mean_diff + ci_margin),
                        'effect_interpretation': self._interpret_effect_size(effect_size),
                        'comparison_type': 'pairwise'
                    }
                    
                    results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def _perform_ranking_analysis(self, data: pd.DataFrame, group_by: Union[str, List[str]],
                                metric_columns: List[str], min_samples: int) -> pd.DataFrame:
        """Perform ranking analysis across all groups."""
        if isinstance(group_by, str):
            group_by = [group_by]
        
        results = []
        
        for metric_col in metric_columns:
            group_stats = []
            
            for group_key, group_df in data.groupby(group_by):
                if len(group_df) < min_samples:
                    continue
                
                values = group_df[metric_col].dropna()
                if len(values) < 2:
                    continue
                
                # Group identifiers
                if isinstance(group_key, tuple):
                    group_dict = dict(zip(group_by, group_key))
                else:
                    group_dict = {group_by[0]: group_key}
                
                stat_dict = {
                    **group_dict,
                    'metric': metric_col,
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'count': len(values),
                    'min': float(values.min()),
                    'max': float(values.max())
                }
                
                group_stats.append(stat_dict)
            
            # Convert to DataFrame for ranking
            if group_stats:
                stats_df = pd.DataFrame(group_stats)
                
                # Add rankings (lower values get better rank for metrics like loss, higher for accuracy)
                stats_df['rank_by_mean'] = stats_df['mean'].rank(ascending=False, method='dense')
                stats_df['rank_by_median'] = stats_df['median'].rank(ascending=False, method='dense')
                
                # Calculate percentile scores
                stats_df['percentile_score'] = stats_df['mean'].rank(pct=True) * 100
                
                # Add performance categories
                stats_df['performance_category'] = stats_df['percentile_score'].apply(
                    lambda x: 'excellent' if x >= 80 else 
                             'good' if x >= 60 else 
                             'average' if x >= 40 else 
                             'below_average' if x >= 20 else 'poor'
                )
                
                # Add relative performance
                best_mean = stats_df['mean'].max()
                stats_df['relative_performance'] = (stats_df['mean'] / best_mean) * 100
                
                # Add comparison type
                stats_df['comparison_type'] = 'ranking'
                
                results.append(stats_df)
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _perform_ab_test_analysis(self, data: pd.DataFrame, group_by: Union[str, List[str]],
                                baseline_group: Optional[Any], metric_columns: List[str],
                                confidence_level: float, significance_threshold: float,
                                min_samples: int) -> pd.DataFrame:
        """Perform A/B test analysis comparing all groups to a baseline."""
        if isinstance(group_by, str):
            group_by = [group_by]
        
        results = []
        
        # Determine baseline
        if baseline_group is None:
            # Use the first group as baseline
            baseline_group = data.groupby(group_by).size().index[0]
        
        baseline_data = data[data[group_by[0] if len(group_by) == 1 else group_by[0]] == baseline_group]
        
        if len(baseline_data) < min_samples:
            raise ValueError(f"Baseline group '{baseline_group}' has insufficient data ({len(baseline_data)} < {min_samples})")
        
        for metric_col in metric_columns:
            baseline_values = baseline_data[metric_col].dropna()
            
            if len(baseline_values) < 2:
                continue
            
            for group_key, group_df in data.groupby(group_by):
                # Skip baseline group (comparing to itself)
                if group_key == baseline_group:
                    continue
                
                if len(group_df) < min_samples:
                    continue
                
                test_values = group_df[metric_col].dropna()
                
                if len(test_values) < 2:
                    continue
                
                # Perform statistical test
                t_stat, p_value = stats.ttest_ind(test_values, baseline_values)
                
                # Calculate lift (percentage improvement)
                baseline_mean = baseline_values.mean()
                test_mean = test_values.mean()
                lift = ((test_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
                
                # Calculate statistical power (simplified)
                effect_size = abs(test_mean - baseline_mean) / np.sqrt(
                    (baseline_values.var() + test_values.var()) / 2
                )
                
                # Group identifiers
                if isinstance(group_key, tuple):
                    group_dict = dict(zip(group_by, group_key))
                else:
                    group_dict = {group_by[0]: group_key}
                
                result_dict = {
                    **group_dict,
                    'metric': metric_col,
                    'baseline_group': baseline_group,
                    'test_mean': float(test_mean),
                    'baseline_mean': float(baseline_mean),
                    'test_std': float(test_values.std()),
                    'baseline_std': float(baseline_values.std()),
                    'test_count': len(test_values),
                    'baseline_count': len(baseline_values),
                    'lift_percent': float(lift),
                    'absolute_improvement': float(test_mean - baseline_mean),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'is_significant': bool(p_value < significance_threshold),
                    'effect_size': float(effect_size),
                    'confidence_level': confidence_level,
                    'significance_threshold': significance_threshold,
                    'test_result': 'significant_improvement' if p_value < significance_threshold and test_mean > baseline_mean else
                                 'significant_degradation' if p_value < significance_threshold and test_mean < baseline_mean else
                                 'no_significant_difference',
                    'comparison_type': 'ab_test'
                }
                
                results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def _perform_trend_comparison(self, data: pd.DataFrame, group_by: Union[str, List[str]],
                                time_column: str, metric_columns: List[str], min_samples: int) -> pd.DataFrame:
        """Perform trend comparison analysis over time."""
        if isinstance(group_by, str):
            group_by = [group_by]
        
        if time_column not in data.columns:
            return pd.DataFrame()
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])
        
        results = []
        
        for metric_col in metric_columns:
            for group_key, group_df in data.groupby(group_by):
                if len(group_df) < min_samples:
                    continue
                
                # Sort by time
                group_df_sorted = group_df.sort_values(time_column)
                values = group_df_sorted[metric_col].dropna()
                
                if len(values) < 3:  # Need at least 3 points for trend
                    continue
                
                # Calculate trend using linear regression
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                # Calculate percentage change
                first_value = values.iloc[0]
                last_value = values.iloc[-1]
                percent_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                
                # Determine trend direction
                if abs(slope) < std_err:
                    trend_direction = 'stable'
                elif slope > 0:
                    trend_direction = 'improving'
                else:
                    trend_direction = 'declining'
                
                # Group identifiers
                if isinstance(group_key, tuple):
                    group_dict = dict(zip(group_by, group_key))
                else:
                    group_dict = {group_by[0]: group_key}
                
                result_dict = {
                    **group_dict,
                    'metric': metric_col,
                    'start_value': float(first_value),
                    'end_value': float(last_value),
                    'start_time': group_df_sorted[time_column].iloc[0],
                    'end_time': group_df_sorted[time_column].iloc[-1],
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'trend_p_value': float(p_value),
                    'trend_significant': bool(p_value < 0.05),
                    'percent_change': float(percent_change),
                    'trend_direction': trend_direction,
                    'sample_count': len(values),
                    'comparison_type': 'trend'
                }
                
                results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return 'negligible'
        elif abs_effect < 0.5:
            return 'small'
        elif abs_effect < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _combine_comparison_results(self, results: List[pd.DataFrame], comparison_type: str) -> pd.DataFrame:
        """Combine multiple comparison results into a single DataFrame."""
        combined_results = []
        
        for result_df in results:
            if not result_df.empty:
                combined_results.append(result_df)
        
        if combined_results:
            return pd.concat(combined_results, ignore_index=True, sort=False)
        else:
            return pd.DataFrame({'comparison_type': [comparison_type], 'message': ['No comparisons could be performed']})
    
    def get_comparison_summary(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Get a high-level summary of comparison analysis results.
        
        Args:
            data: Input DataFrame
            **kwargs: Analysis parameters
            
        Returns:
            Dict containing comparison analysis summary
        """
        result_df = self._process_implementation(data, **kwargs)
        
        summary = {
            'total_comparisons': len(result_df),
            'comparison_types': result_df['comparison_type'].unique().tolist() if 'comparison_type' in result_df.columns else []
        }
        
        # Add type-specific summaries
        if 'comparison_type' in result_df.columns:
            for comp_type in result_df['comparison_type'].unique():
                type_data = result_df[result_df['comparison_type'] == comp_type]
                
                if comp_type == 'pairwise':
                    summary[f'{comp_type}_summary'] = {
                        'total_pairs': len(type_data),
                        'significant_comparisons': int((type_data['is_significant'] == True).sum()) if 'is_significant' in type_data.columns else 0,
                        'large_effects': int((type_data['effect_interpretation'] == 'large').sum()) if 'effect_interpretation' in type_data.columns else 0
                    }
                elif comp_type == 'ranking':
                    summary[f'{comp_type}_summary'] = {
                        'groups_ranked': len(type_data),
                        'excellent_performers': int((type_data['performance_category'] == 'excellent').sum()) if 'performance_category' in type_data.columns else 0,
                        'poor_performers': int((type_data['performance_category'] == 'poor').sum()) if 'performance_category' in type_data.columns else 0
                    }
                elif comp_type == 'ab_test':
                    summary[f'{comp_type}_summary'] = {
                        'total_tests': len(type_data),
                        'significant_improvements': int((type_data['test_result'] == 'significant_improvement').sum()) if 'test_result' in type_data.columns else 0,
                        'significant_degradations': int((type_data['test_result'] == 'significant_degradation').sum()) if 'test_result' in type_data.columns else 0
                    }
                elif comp_type == 'trend':
                    summary[f'{comp_type}_summary'] = {
                        'total_trends': len(type_data),
                        'improving_trends': int((type_data['trend_direction'] == 'improving').sum()) if 'trend_direction' in type_data.columns else 0,
                        'declining_trends': int((type_data['trend_direction'] == 'declining').sum()) if 'trend_direction' in type_data.columns else 0
                    }
        
        return summary 