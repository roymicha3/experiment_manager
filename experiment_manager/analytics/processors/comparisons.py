"""
Comparison Processor for Analytics

Provides comprehensive comparison analysis capabilities for cross-experiment analysis,
A/B testing, performance ranking, and improvement tracking.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from omegaconf import DictConfig

from experiment_manager.analytics.processors.base import DataProcessor
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("ComparisonProcessor")
class ComparisonProcessor(DataProcessor):
    """
    Processor for performing comparative analysis between experiments, configurations, or groups.
    
    Comparison Capabilities:
    - Cross-experiment performance comparison
    - A/B testing statistical analysis
    - Performance ranking and benchmarking
    - Improvement tracking over time
    """
    
    def __init__(self, name: str = "ComparisonProcessor", config: DictConfig = None):
        """Initialize the comparison processor."""
        super().__init__(name, config)
        self.required_columns = ['metric_total_val']
        self.optional_columns = ['experiment_name', 'trial_name', 'epoch', 'group']
    
    @classmethod
    def from_config(cls, config: DictConfig, **kwargs):
        """Create ComparisonProcessor from configuration."""
        # Get comparison-specific configuration
        comparison_config = config.get('comparisons', DictConfig({}))
        name = kwargs.get('name', 'ComparisonProcessor')
        return cls(name=name, config=comparison_config)
    
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
            pairwise_results = self._perform_pairwise_comparisons(
                data, group_by, metric_columns, confidence_level, significance_threshold, min_samples
            )
            results.append(pairwise_results)
        
        if comparison_type in ['ranking', 'all']:
            ranking_results = self._perform_ranking_analysis(
                data, group_by, metric_columns, min_samples
            )
            results.append(ranking_results)
        
        if comparison_type in ['ab_test', 'all']:
            ab_test_results = self._perform_ab_test_analysis(
                data, group_by, baseline_group, metric_columns, 
                confidence_level, significance_threshold, min_samples
            )
            results.append(ab_test_results)
        
        if comparison_type in ['trend', 'all'] and time_column:
            trend_results = self._perform_trend_analysis(
                data, group_by, time_column, metric_columns, min_samples
            )
            results.append(trend_results)
        
        # Combine all results
        if results:
            combined_result = pd.concat(results, ignore_index=True)
            return combined_result
        else:
            return pd.DataFrame()
    
    def _perform_pairwise_comparisons(self, data: pd.DataFrame, group_by: Union[str, List[str]],
                                    metric_columns: List[str], confidence_level: float,
                                    significance_threshold: float, min_samples: int) -> pd.DataFrame:
        """Perform pairwise statistical comparisons between all groups."""
        if isinstance(group_by, str):
            group_by = [group_by]
        
        results = []
        
        # Get all groups
        groups = list(data.groupby(group_by))
        
        for metric_col in metric_columns:
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    group1_name, group1_data = groups[i]
                    group2_name, group2_data = groups[j]
                    
                    # Check minimum sample sizes
                    if len(group1_data) < min_samples or len(group2_data) < min_samples:
                        continue
                    
                    # Extract metric values
                    values1 = group1_data[metric_col].dropna()
                    values2 = group2_data[metric_col].dropna()
                    
                    if len(values1) < min_samples or len(values2) < min_samples:
                        continue
                    
                    # Perform statistical tests
                    comparison_result = self._compare_two_groups(
                        values1, values2, confidence_level, significance_threshold
                    )
                    
                    # Format group names
                    group1_str = str(group1_name) if not isinstance(group1_name, tuple) else '_'.join(map(str, group1_name))
                    group2_str = str(group2_name) if not isinstance(group2_name, tuple) else '_'.join(map(str, group2_name))
                    
                    # Add metadata
                    comparison_result.update({
                        'metric_column': metric_col,
                        'group1': group1_str,
                        'group2': group2_str,
                        'group1_size': len(values1),
                        'group2_size': len(values2),
                        'comparison_type': 'pairwise',
                        'analysis_type': 'comparison'
                    })
                    
                    results.append(comparison_result)
        
        return pd.DataFrame(results)
    
    def _compare_two_groups(self, values1: pd.Series, values2: pd.Series, 
                          confidence_level: float, significance_threshold: float) -> Dict[str, Any]:
        """Compare two groups using appropriate statistical tests."""
        # Basic statistics
        mean1, mean2 = values1.mean(), values2.mean()
        std1, std2 = values1.std(), values2.std()
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(values1) - 1) * std1**2 + (len(values2) - 1) * std2**2) / 
                           (len(values1) + len(values2) - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Statistical tests
        # 1. Welch's t-test (doesn't assume equal variances)
        t_stat, t_pvalue = stats.ttest_ind(values1, values2, equal_var=False)
        
        # 2. Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        
        # 3. Levene's test for equal variances
        levene_stat, levene_pvalue = stats.levene(values1, values2)
        
        # Determine which test to trust more
        if levene_pvalue > 0.05:  # Equal variances
            # Use standard t-test
            t_stat_equal, t_pvalue_equal = stats.ttest_ind(values1, values2, equal_var=True)
            recommended_test = 'ttest_equal_var'
            recommended_pvalue = t_pvalue_equal
        else:
            # Use Welch's t-test
            recommended_test = 'ttest_welch'
            recommended_pvalue = t_pvalue
        
        # Confidence interval for difference in means
        se_diff = np.sqrt(std1**2/len(values1) + std2**2/len(values2))
        dof = len(values1) + len(values2) - 2
        t_critical = stats.t.ppf((1 + confidence_level) / 2, dof)
        mean_diff = mean1 - mean2
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        return {
            'group1_mean': float(mean1),
            'group2_mean': float(mean2),
            'mean_difference': float(mean_diff),
            'group1_std': float(std1),
            'group2_std': float(std2),
            'cohens_d': float(cohens_d),
            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
            'ttest_welch_statistic': float(t_stat),
            'ttest_welch_pvalue': float(t_pvalue),
            'mannwhitney_statistic': float(u_stat),
            'mannwhitney_pvalue': float(u_pvalue),
            'levene_statistic': float(levene_stat),
            'levene_pvalue': float(levene_pvalue),
            'recommended_test': recommended_test,
            'recommended_pvalue': float(recommended_pvalue),
            'is_significant': recommended_pvalue < significance_threshold,
            'confidence_interval_lower': float(ci_lower),
            'confidence_interval_upper': float(ci_upper),
            'confidence_level': confidence_level
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        if effect_size < 0.2:
            return 'negligible'
        elif effect_size < 0.5:
            return 'small'
        elif effect_size < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _perform_ranking_analysis(self, data: pd.DataFrame, group_by: Union[str, List[str]],
                                metric_columns: List[str], min_samples: int) -> pd.DataFrame:
        """Perform ranking analysis of groups based on performance metrics."""
        if isinstance(group_by, str):
            group_by = [group_by]
        
        results = []
        
        for metric_col in metric_columns:
            group_stats = []
            
            # Calculate statistics for each group
            for group_name, group_data in data.groupby(group_by):
                if len(group_data) < min_samples:
                    continue
                
                values = group_data[metric_col].dropna()
                if len(values) < min_samples:
                    continue
                
                group_str = str(group_name) if not isinstance(group_name, tuple) else '_'.join(map(str, group_name))
                
                stats_dict = {
                    'group': group_str,
                    'metric_column': metric_col,
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'count': len(values),
                    'min': float(values.min()),
                    'max': float(values.max())
                }
                group_stats.append(stats_dict)
            
            # Sort by mean performance (descending - assuming higher is better)
            group_stats.sort(key=lambda x: x['mean'], reverse=True)
            
            # Add ranking information
            for rank, stats_dict in enumerate(group_stats, 1):
                stats_dict.update({
                    'rank_by_mean': rank,
                    'comparison_type': 'ranking',
                    'analysis_type': 'comparison'
                })
                results.append(stats_dict)
            
            # Add percentile scores
            if group_stats:
                means = [g['mean'] for g in group_stats]
                for stats_dict in group_stats:
                    percentile = (1 - (stats_dict['rank_by_mean'] - 1) / len(group_stats)) * 100
                    stats_dict['percentile_score'] = float(percentile)
        
        return pd.DataFrame(results)
    
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
            if len(baseline_values) < min_samples:
                continue
            
            # Compare each group to baseline
            for group_name, group_data in data.groupby(group_by):
                if len(group_data) < min_samples:
                    continue
                
                group_str = str(group_name) if not isinstance(group_name, tuple) else '_'.join(map(str, group_name))
                
                # Skip if this is the baseline group
                if group_name == baseline_group:
                    continue
                
                test_values = group_data[metric_col].dropna()
                if len(test_values) < min_samples:
                    continue
                
                # Perform A/B test comparison
                comparison_result = self._compare_two_groups(
                    baseline_values, test_values, confidence_level, significance_threshold
                )
                
                # Calculate relative improvement
                baseline_mean = baseline_values.mean()
                test_mean = test_values.mean()
                relative_improvement = (test_mean - baseline_mean) / baseline_mean * 100 if baseline_mean != 0 else 0
                
                # Add A/B test specific metadata
                comparison_result.update({
                    'metric_column': metric_col,
                    'baseline_group': str(baseline_group),
                    'test_group': group_str,
                    'baseline_size': len(baseline_values),
                    'test_size': len(test_values),
                    'relative_improvement_percent': float(relative_improvement),
                    'comparison_type': 'ab_test',
                    'analysis_type': 'comparison'
                })
                
                results.append(comparison_result)
        
        return pd.DataFrame(results)
    
    def _perform_trend_analysis(self, data: pd.DataFrame, group_by: Union[str, List[str]],
                              time_column: str, metric_columns: List[str], 
                              min_samples: int) -> pd.DataFrame:
        """Perform trend analysis over time for different groups."""
        if isinstance(group_by, str):
            group_by = [group_by]
        
        if time_column not in data.columns:
            return pd.DataFrame()
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            try:
                data[time_column] = pd.to_datetime(data[time_column])
            except:
                return pd.DataFrame()
        
        results = []
        
        for metric_col in metric_columns:
            # Analyze trend for each group
            for group_name, group_data in data.groupby(group_by):
                if len(group_data) < min_samples:
                    continue
                
                group_str = str(group_name) if not isinstance(group_name, tuple) else '_'.join(map(str, group_name))
                
                # Sort by time and calculate trend
                group_data_sorted = group_data.sort_values(time_column)
                values = group_data_sorted[metric_col].dropna()
                times = group_data_sorted.loc[values.index, time_column]
                
                if len(values) < min_samples:
                    continue
                
                # Convert times to numeric for regression
                time_numeric = (times - times.min()).dt.total_seconds()
                
                # Linear regression for trend
                if len(time_numeric) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
                    
                    # Calculate trend direction and strength
                    trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                    correlation_strength = self._interpret_correlation(abs(r_value))
                    
                    trend_result = {
                        'metric_column': metric_col,
                        'group': group_str,
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'r_squared': float(r_value**2),
                        'correlation': float(r_value),
                        'p_value': float(p_value),
                        'trend_direction': trend_direction,
                        'correlation_strength': correlation_strength,
                        'is_significant_trend': p_value < 0.05,
                        'sample_count': len(values),
                        'time_span_days': (times.max() - times.min()).total_seconds() / 86400,
                        'comparison_type': 'trend',
                        'analysis_type': 'comparison'
                    }
                    
                    results.append(trend_result)
        
        return pd.DataFrame(results)
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength."""
        if correlation < 0.1:
            return 'negligible'
        elif correlation < 0.3:
            return 'weak'
        elif correlation < 0.5:
            return 'moderate'
        elif correlation < 0.7:
            return 'strong'
        else:
            return 'very_strong'
    
    def get_comparison_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of comparison analysis results.
        
        Args:
            data: DataFrame with comparison analysis results
            
        Returns:
            Dict with summary statistics
        """
        if 'comparison_type' not in data.columns:
            return {'error': 'Data does not appear to be comparison analysis results'}
        
        summary = {
            'total_comparisons': len(data),
            'comparison_types': data['comparison_type'].value_counts().to_dict()
        }
        
        # Pairwise comparison summary
        pairwise_data = data[data['comparison_type'] == 'pairwise']
        if not pairwise_data.empty:
            summary['pairwise'] = {
                'total_pairs': len(pairwise_data),
                'significant_differences': int(pairwise_data['is_significant'].sum()),
                'avg_effect_size': float(pairwise_data['cohens_d'].abs().mean()) if 'cohens_d' in pairwise_data.columns else None
            }
        
        # Ranking summary
        ranking_data = data[data['comparison_type'] == 'ranking']
        if not ranking_data.empty:
            summary['ranking'] = {
                'groups_ranked': int(ranking_data['group'].nunique()),
                'top_performer': ranking_data[ranking_data['rank_by_mean'] == 1]['group'].iloc[0] if len(ranking_data) > 0 else None
            }
        
        # A/B test summary
        ab_test_data = data[data['comparison_type'] == 'ab_test']
        if not ab_test_data.empty:
            summary['ab_test'] = {
                'total_tests': len(ab_test_data),
                'significant_improvements': int(ab_test_data[
                    (ab_test_data['is_significant'] == True) & 
                    (ab_test_data['relative_improvement_percent'] > 0)
                ].shape[0]),
                'avg_improvement': float(ab_test_data['relative_improvement_percent'].mean())
            }
        
        # Trend analysis summary
        trend_data = data[data['comparison_type'] == 'trend']
        if not trend_data.empty:
            summary['trend'] = {
                'groups_analyzed': int(trend_data['group'].nunique()),
                'significant_trends': int(trend_data['is_significant_trend'].sum()),
                'improving_groups': int(trend_data[
                    (trend_data['trend_direction'] == 'increasing') & 
                    (trend_data['is_significant_trend'] == True)
                ].shape[0])
            }
        
        return summary 