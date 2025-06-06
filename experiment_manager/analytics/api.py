"""
High-level user-facing API for experiment analytics.

This module provides the ExperimentAnalytics class which offers simple, 
intuitive methods for common analytics operations without requiring 
detailed knowledge of the underlying analytics engine.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
from pathlib import Path

from .engine import AnalyticsEngine
from .query_builder import AnalyticsQuery
from .results import AnalyticsResult
from experiment_manager.common.common import Metric, RunStatus


class ExperimentAnalytics:
    """
    High-level API for experiment analytics operations.
    
    This class provides simple, intuitive methods for common analytics tasks
    such as extracting results, calculating statistics, analyzing failures,
    and comparing experiments.
    """
    
    def __init__(self, database_manager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ExperimentAnalytics API.
        
        Args:
            database_manager: DatabaseManager instance for data access
            config: Optional configuration for the analytics engine
        """
        self.database_manager = database_manager
        self.engine = AnalyticsEngine(database_manager, config)
    
    # Core Analytics Operations
    
    def extract_results(self, experiment_name: str, 
                       include_failed: bool = False,
                       metric_types: Optional[List[str]] = None) -> AnalyticsResult:
        """
        Extract experiment results for analysis.
        
        Args:
            experiment_name: Name of the experiment to analyze
            include_failed: Whether to include failed runs in results
            metric_types: Specific metric types to extract (None for all)
            
        Returns:
            AnalyticsResult: Extracted experiment data
        """
        query = self.engine.create_query()
        
        # Filter by experiment name
        query = query.experiments(names=[experiment_name])
        
        # Filter by run status
        if not include_failed:
            query = query.runs(status=[RunStatus.SUCCESS])
        
        # Filter by metric types if specified
        if metric_types:
            query = query.metrics(types=metric_types)
        
        return query.execute()
    
    def calculate_statistics(self, experiment_id: int,
                           metric_types: Optional[List[str]] = None,
                           group_by: str = 'trial') -> Dict[str, Any]:
        """
        Calculate statistical summaries for experiment metrics.
        
        Args:
            experiment_id: ID of the experiment to analyze
            metric_types: Specific metric types to analyze (None for all)
            group_by: Grouping level ('trial', 'experiment', 'run')
            
        Returns:
            Dict[str, Any]: Statistical summaries
        """
        query = self.engine.create_query()
        
        # Configure query
        query = (query
                .experiments(ids=[experiment_id])
                .runs(status=[RunStatus.SUCCESS]))
        
        if metric_types:
            query = query.metrics(types=metric_types)
        
        # Add aggregation and grouping
        query = (query
                .aggregate(['mean', 'std', 'min', 'max', 'count'])
                .group_by(group_by))
        
        result = query.execute()
        
        # Convert result to summary format
        summary = result.get_summary()
        summary['statistics_by_group'] = result.to_dataframe().to_dict('records')
        
        return summary
    
    def analyze_failures(self, experiment_id: int,
                        correlation_analysis: bool = True) -> Dict[str, Any]:
        """
        Analyze failure patterns in experiment runs.
        
        Args:
            experiment_id: ID of the experiment to analyze
            correlation_analysis: Whether to perform correlation analysis
            
        Returns:
            Dict[str, Any]: Failure analysis results
        """
        # Query for all runs including failures
        query = self.engine.create_query()
        query = query.experiments(ids=[experiment_id])
        
        all_runs = query.execute()
        
        # Query for only successful runs
        success_query = self.engine.create_query()
        success_query = (success_query
                        .experiments(ids=[experiment_id])
                        .runs(status=[RunStatus.SUCCESS]))
        
        successful_runs = success_query.execute()
        
        # Calculate failure statistics
        total_runs = len(all_runs)
        successful_count = len(successful_runs)
        failed_count = total_runs - successful_count
        failure_rate = failed_count / total_runs if total_runs > 0 else 0
        
        analysis = {
            'failure_statistics': {
                'total_runs': total_runs,
                'successful_runs': successful_count,
                'failed_runs': failed_count,
                'failure_rate': failure_rate
            },
            'patterns': {
                'most_common_failure_points': [],  # TODO: Implement in processors
                'configuration_correlations': []   # TODO: Implement in processors
            }
        }
        
        if correlation_analysis:
            # TODO: Implement detailed correlation analysis in failure processor
            analysis['correlation_analysis'] = {
                'enabled': True,
                'note': 'Detailed correlation analysis will be implemented in Task #19'
            }
        
        return analysis
    
    def compare_experiments(self, experiment_ids: List[int],
                          metric_type: str = 'test_acc') -> AnalyticsResult:
        """
        Compare performance across multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric_type: Metric type for comparison
            
        Returns:
            AnalyticsResult: Comparison results
        """
        query = self.engine.create_query()
        
        query = (query
                .experiments(ids=experiment_ids)
                .runs(status=[RunStatus.SUCCESS])
                .metrics(types=[metric_type])
                .aggregate(['mean', 'std', 'min', 'max', 'count'])
                .group_by('experiment_id')
                .sort_by('mean', ascending=False))
        
        result = query.execute()
        
        # Add comparison metadata
        result.metadata['comparison_type'] = 'multi_experiment'
        result.metadata['experiments_compared'] = experiment_ids
        result.metadata['primary_metric'] = metric_type
        
        # Add summary statistics
        if not result.data.empty and 'mean' in result.data.columns:
            best_experiment = result.data.iloc[0]['experiment_id'] if len(result.data) > 0 else None
            worst_experiment = result.data.iloc[-1]['experiment_id'] if len(result.data) > 0 else None
            
            result.add_summary_statistic('best_experiment_id', best_experiment, 
                                        'Experiment with highest mean performance')
            result.add_summary_statistic('worst_experiment_id', worst_experiment,
                                        'Experiment with lowest mean performance')
        
        return result
    
    # Advanced Operations
    
    def detect_outliers(self, experiment_id: int,
                       metric_type: str = 'test_acc',
                       method: str = 'iqr',
                       threshold: float = 1.5) -> List[int]:
        """
        Detect outlier runs in an experiment.
        
        Args:
            experiment_id: ID of the experiment to analyze
            metric_type: Metric type to analyze for outliers
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            List[int]: List of run IDs identified as outliers
        """
        # First get all data
        all_data_query = self.engine.create_query()
        all_data_query = (all_data_query
                         .experiments(ids=[experiment_id])
                         .runs(status=[RunStatus.SUCCESS])
                         .metrics(types=[metric_type]))
        
        all_data = all_data_query.execute()
        
        # Then get data with outliers excluded
        no_outliers_query = self.engine.create_query()
        no_outliers_query = (no_outliers_query
                            .experiments(ids=[experiment_id])
                            .runs(status=[RunStatus.SUCCESS])
                            .metrics(types=[metric_type])
                            .exclude_outliers(metric_type, method, threshold))
        
        filtered_data = no_outliers_query.execute()
        
        # Find the difference to identify outliers
        if 'run_id' in all_data.data.columns and 'run_id' in filtered_data.data.columns:
            all_run_ids = set(all_data.data['run_id'].tolist())
            filtered_run_ids = set(filtered_data.data['run_id'].tolist())
            outlier_run_ids = list(all_run_ids - filtered_run_ids)
            return outlier_run_ids
        
        return []  # Return empty list if run_id column not available
    
    def analyze_training_curves(self, trial_run_ids: List[int],
                               metric_types: Optional[List[str]] = None) -> AnalyticsResult:
        """
        Analyze training curves for convergence patterns.
        
        Args:
            trial_run_ids: List of trial run IDs to analyze
            metric_types: Metric types to include in analysis
            
        Returns:
            AnalyticsResult: Training curve analysis
        """
        # Use the epoch series data method directly for training curves
        result = self.engine.get_epoch_series_data(trial_run_ids, metric_types)
        
        # Add training curve specific metadata
        result.metadata['analysis_type'] = 'training_curves'
        result.metadata['analyzed_runs'] = trial_run_ids
        
        # TODO: Add convergence detection analysis in processors
        result.add_summary_statistic('runs_analyzed', len(trial_run_ids),
                                    'Number of training runs analyzed')
        
        return result
    
    def generate_summary_report(self, experiment_id: int,
                               output_format: str = 'html') -> str:
        """
        Generate a comprehensive summary report for an experiment.
        
        Args:
            experiment_id: ID of the experiment to summarize
            output_format: Output format ('html', 'json', 'text')
            
        Returns:
            str: Report content or file path
        """
        # Gather comprehensive data
        overview = self.calculate_statistics(experiment_id)
        failures = self.analyze_failures(experiment_id)
        
        # Get basic experiment data
        basic_data = self.extract_results(str(experiment_id), include_failed=True)
        
        if output_format == 'html':
            # Generate HTML report
            report_path = f"experiment_{experiment_id}_report.html"
            basic_data.to_html_report(report_path, 
                                     title=f"Experiment {experiment_id} Summary Report")
            return report_path
        
        elif output_format == 'json':
            # Create comprehensive JSON report
            report = {
                'experiment_id': experiment_id,
                'overview': overview,
                'failure_analysis': failures,
                'data_summary': basic_data.get_summary(),
                'generated_at': basic_data.created_at.isoformat()
            }
            return str(report)
        
        else:  # text format
            # Create simple text summary
            text_report = f"""
            Experiment {experiment_id} Summary Report
            ========================================
            
            Total Runs: {overview.get('overview', {}).get('row_count', 'N/A')}
            Failure Rate: {failures.get('failure_statistics', {}).get('failure_rate', 'N/A'):.2%}
            
            Generated: {basic_data.created_at}
            """
            return text_report.strip()
    
    # Utility Methods
    
    def create_custom_query(self) -> AnalyticsQuery:
        """
        Create a custom analytics query for advanced users.
        
        Returns:
            AnalyticsQuery: New query instance for custom configuration
        """
        return self.engine.create_query()
    
    def get_available_metrics(self, experiment_id: Optional[int] = None) -> List[str]:
        """
        Get list of available metric types.
        
        Args:
            experiment_id: Optional experiment ID to filter metrics
            
        Returns:
            List[str]: Available metric type names
        """
        # TODO: Implement actual database query to get available metrics
        # For now, return common metric types
        common_metrics = [
            'test_acc', 'test_loss', 'val_acc', 'val_loss',
            'train_acc', 'train_loss', 'learning_rate'
        ]
        return common_metrics
    
    def get_experiment_info(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get basic information about an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dict[str, Any]: Experiment information
        """
        # TODO: Implement actual database query for experiment info
        # This is a placeholder implementation
        return {
            'experiment_id': experiment_id,
            'name': f'experiment_{experiment_id}',
            'status': 'completed',
            'total_trials': 0,  # Will be implemented with database integration
            'total_runs': 0,    # Will be implemented with database integration
            'created_at': None, # Will be implemented with database integration
            'note': 'Full implementation will be available in Task #18'
        }
    
    def clear_cache(self):
        """Clear the analytics engine cache."""
        self.engine.clear_cache()
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get analytics engine statistics and status.
        
        Returns:
            Dict[str, Any]: Engine statistics
        """
        return self.engine.get_engine_info()
    
    def export_results(self, result: AnalyticsResult, 
                      filepath: Union[str, Path],
                      format: str = 'csv',
                      **kwargs) -> str:
        """
        Export analytics results to file.
        
        Args:
            result: AnalyticsResult to export
            filepath: Output file path
            format: Export format ('csv', 'json', 'excel', 'html')
            **kwargs: Additional export options
            
        Returns:
            str: Path to exported file
        """
        if format.lower() == 'csv':
            return result.to_csv(filepath, **kwargs)
        elif format.lower() == 'json':
            return result.to_json(filepath, **kwargs)
        elif format.lower() == 'excel':
            return result.to_excel(filepath, **kwargs)
        elif format.lower() == 'html':
            return result.to_html_report(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def __str__(self) -> str:
        """String representation of the analytics API."""
        return f"ExperimentAnalytics(database={'connected' if self.database_manager else 'not connected'})"
    
    def __repr__(self) -> str:
        """Detailed representation of the analytics API."""
        return (f"ExperimentAnalytics(database_manager={self.database_manager}, "
                f"engine={self.engine})") 