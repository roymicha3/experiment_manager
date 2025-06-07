"""
ExperimentAnalytics High-Level API

This module provides the main user-facing API class for experiment analytics.
It offers simple, intuitive methods for common analytics operations while
leveraging the underlying analytics infrastructure.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

from omegaconf import DictConfig

from experiment_manager.analytics.analytics_factory import AnalyticsFactory
from experiment_manager.analytics.query import AnalyticsQuery, ValidationError
from experiment_manager.analytics.results import AnalyticsResult, QueryMetadata
from experiment_manager.analytics.defaults import DefaultConfigurationManager, ConfigurationLevel
from experiment_manager.analytics.export import (
    ResultExporter, ResultVisualizer, AnalyticsReportGenerator,
    ExportFormat, VisualizationType, ExportOptions, VisualizationOptions
)
from experiment_manager.db.manager import DatabaseManager
from experiment_manager.common.serializable import YAMLSerializable

logger = logging.getLogger(__name__)


@YAMLSerializable.register("ExperimentAnalytics")
class ExperimentAnalytics(YAMLSerializable):
    """
    High-level API for experiment analytics operations.
    
    This class provides simple, intuitive methods for common analytics tasks
    including statistical analysis, outlier detection, failure analysis, and
    experiment comparisons. It serves as the main entry point for users.
    
    Example:
        # Basic usage
        analytics = ExperimentAnalytics(db_manager)
        
        # Quick experiment comparison
        result = analytics.compare_experiments(['exp1', 'exp2'], metrics=['accuracy', 'loss'])
        
        # Statistical analysis with outlier exclusion
        stats = analytics.analyze_statistics('accuracy', experiments=['my_exp'], exclude_outliers=True)
        
        # Failure analysis
        failures = analytics.analyze_failures(experiment_names=['failed_exp'])
    """
    
    def __init__(self, 
                 db_manager: DatabaseManager = None,
                 config: Optional[DictConfig] = None,
                 configuration_level: ConfigurationLevel = ConfigurationLevel.STANDARD):
        """
        Initialize ExperimentAnalytics.
        
        Args:
            db_manager: Database manager instance for data access
            config: Optional analytics configuration
            configuration_level: Default configuration level to use
        """
        self.db_manager = db_manager
        self.configuration_level = configuration_level
        
        # Initialize configuration
        if config is None:
            self.config = DefaultConfigurationManager.get_config_by_level(configuration_level)
        else:
            self.config = config
        
        # Initialize processors with default configurations
        self._processors = {}
        self._initialize_processors()
        
        # Query cache for optimization
        self._query_cache = {}
        
        # Initialize export and visualization components
        self.exporter = ResultExporter()
        self.visualizer = ResultVisualizer()
        self.report_generator = AnalyticsReportGenerator()
        
        logger.info(f"ExperimentAnalytics initialized with {configuration_level.value} configuration level")
    
    def _initialize_processors(self):
        """Initialize analytics processors with default configurations."""
        try:
            self._processors = AnalyticsFactory.create_from_config(self.config)
            logger.debug(f"Initialized processors: {list(self._processors.keys())}")
        except Exception as e:
            logger.warning(f"Failed to initialize some processors: {e}")
            # Initialize with minimal set
            self._processors = {}
    
    def query(self) -> AnalyticsQuery:
        """
        Create a new fluent analytics query.
        
        Returns:
            AnalyticsQuery: New query builder instance
            
        Example:
            result = (analytics.query()
                      .experiments(names=['my_exp'])
                      .runs(status=['completed'])
                      .metrics(['accuracy'])
                      .execute())
        """
        return AnalyticsQuery(analytics_engine=self)
    
    # === QUICK ANALYSIS METHODS ===
    
    def analyze_statistics(self,
                          metric: str,
                          experiments: Optional[List[str]] = None,
                          exclude_outliers: bool = False,
                          confidence_level: float = 0.95) -> AnalyticsResult:
        """
        Perform statistical analysis on a metric across experiments.
        
        Args:
            metric: Metric type to analyze
            experiments: List of experiment names (None for all)
            exclude_outliers: Whether to exclude outliers from analysis
            confidence_level: Confidence level for confidence intervals
            
        Returns:
            AnalyticsResult: Statistical analysis results
        """
        if not self.db_manager:
            raise ValueError("Database manager not initialized. Cannot perform statistics analysis.")
        
        try:
            # Get metric data directly from database
            experiment_ids = None
            if experiments:
                experiment_ids = self._resolve_experiment_ids(experiments)
            
            metrics_data = self.db_manager.get_metrics_by_type(metric, experiment_ids)
            
            # Get statistics from database
            stats = self.db_manager.get_metric_statistics(metric, experiment_ids)
            
            # Combine raw data and statistics
            result_data = {
                'raw_data': metrics_data,
                'statistics': stats,
                'metric': metric,
                'experiments': experiments or 'all',
                'exclude_outliers': exclude_outliers,
                'confidence_level': confidence_level
            }
            
            return AnalyticsResult(
                data=result_data,
                metadata=QueryMetadata(
                    query_type="statistics_analysis",
                    execution_time=0.0,
                    row_count=len(metrics_data),
                    processing_steps=['database_query', 'statistics_calculation']
                )
            )
            
        except Exception as e:
            logger.error(f"Statistics analysis failed: {e}")
            raise
    
    def compare_experiments(self,
                           experiment_names: List[str],
                           metrics: List[str],
                           comparison_method: str = 'statistical') -> AnalyticsResult:
        """
        Compare multiple experiments across specified metrics.
        
        Args:
            experiment_names: List of experiment names to compare
            metrics: List of metrics to compare
            comparison_method: Comparison method ('statistical', 'ranking', 'trend')
            
        Returns:
            AnalyticsResult: Comparison analysis results
        """
        if not self.db_manager:
            raise ValueError("Database manager not initialized. Cannot compare experiments.")
        
        if len(experiment_names) < 2:
            raise ValueError("At least 2 experiments required for comparison")
        
        try:
            # Resolve experiment names to IDs
            experiment_ids = self._resolve_experiment_ids(experiment_names)
            
            comparison_data = {}
            
            for metric in metrics:
                # Get metrics data for each experiment
                metric_data = self.db_manager.get_metrics_by_type(metric, experiment_ids)
                comparison_data[metric] = metric_data
                
                # Get statistics for each experiment
                for exp_name, exp_id in zip(experiment_names, experiment_ids):
                    stats = self.db_manager.get_metric_statistics(metric, [exp_id])
                    comparison_data[f"{metric}_{exp_name}_stats"] = stats
            
            result_data = {
                'experiments': experiment_names,
                'metrics': metrics,
                'comparison_method': comparison_method,
                'comparison_data': comparison_data
            }
            
            return AnalyticsResult(
                data=result_data,
                metadata=QueryMetadata(
                    query_type="experiment_comparison",
                    execution_time=0.0,
                    row_count=len(experiment_ids),
                    processing_steps=['database_query', 'experiment_comparison']
                )
            )
            
        except Exception as e:
            logger.error(f"Experiment comparison failed: {e}")
            raise
    
    def detect_outliers(self,
                       metric: str,
                       experiments: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> AnalyticsResult:
        """
        Detect outliers in metric values.
        
        Args:
            metric: Metric type to analyze for outliers
            experiments: List of experiment names (None for all)
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            AnalyticsResult: Outlier detection results
        """
        query = self.query().metrics([metric])
        
        if experiments:
            query = query.experiments(names=experiments)
        
        # Configure outlier processor
        if 'outliers' in self._processors:
            processor = self._processors['outliers']
            processor.method = method
            processor.threshold = threshold
        
        return query.execute()
    
    def analyze_failures(self,
                        experiment_names: Optional[List[str]] = None,
                        time_range: Optional[Tuple[datetime, datetime]] = None,
                        include_root_cause: bool = True) -> AnalyticsResult:
        """
        Analyze experiment failures and identify patterns.
        
        Args:
            experiment_names: List of experiment names (None for all)
            time_range: Time range to analyze failures
            include_root_cause: Whether to include root cause analysis
            
        Returns:
            AnalyticsResult: Failure analysis results
        """
        query = self.query().runs(status=['failed', 'timeout', 'cancelled'])
        
        if experiment_names:
            query = query.experiments(names=experiment_names)
        
        if time_range:
            query = query.experiments(time_range=time_range)
        
        # Configure failure analyzer
        if 'failures' in self._processors:
            processor = self._processors['failures']
            processor.include_root_cause = include_root_cause
        
        return query.execute()
    
    def get_experiment_summary(self,
                              experiment_name: str,
                              include_outliers: bool = False) -> AnalyticsResult:
        """
        Get comprehensive summary for a single experiment.
        
        Args:
            experiment_name: Name of the experiment to summarize
            include_outliers: Whether to include outlier analysis
            
        Returns:
            AnalyticsResult: Comprehensive experiment summary
        """
        if not self.db_manager:
            raise ValueError("Database manager not initialized. Cannot get experiment summary.")
        
        try:
            # Resolve experiment name to ID
            experiment_ids = self._resolve_experiment_ids([experiment_name])
            if not experiment_ids:
                raise ValueError(f"No experiment found with name: {experiment_name}")
            
            experiment_id = experiment_ids[0]
            
            # Get comprehensive summary
            summary = self.db_manager.get_experiment_summary(experiment_id)
            performance_data = self.db_manager.get_experiment_performance_data(experiment_id)
            
            result_data = {
                'experiment_name': experiment_name,
                'experiment_id': experiment_id,
                'summary': summary,
                'performance_data': performance_data,
                'include_outliers': include_outliers
            }
            
            return AnalyticsResult(
                data=result_data,
                metadata=QueryMetadata(
                    query_type="experiment_summary",
                    execution_time=0.0,
                    row_count=1,
                    processing_steps=['database_query', 'summary_compilation']
                )
            )
            
        except Exception as e:
            logger.error(f"Experiment summary failed: {e}")
            raise
    
    def find_best_experiments(self,
                             metric: str,
                             top_k: int = 5,
                             higher_is_better: bool = True) -> AnalyticsResult:
        """
        Find the best performing experiments for a given metric.
        
        Args:
            metric: Metric to rank experiments by
            top_k: Number of top experiments to return
            higher_is_better: Whether higher values are better
            
        Returns:
            AnalyticsResult: Top performing experiments
        """
        query = (self.query()
                 .metrics([metric])
                 .runs(status=['completed'])
                 .exclude_outliers(metric, method='iqr')
                 .aggregate(['mean'])
                 .sort_by(metric, ascending=not higher_is_better))
        
        result = query.execute()
        
        # Limit to top_k results
        if hasattr(result, 'data') and isinstance(result.data, list):
            result.data = result.data[:top_k]
        
        return result
    
    def analyze_trends(self,
                      metric: str,
                      experiments: Optional[List[str]] = None,
                      time_granularity: str = 'daily') -> AnalyticsResult:
        """
        Analyze trends in metric values over time.
        
        Args:
            metric: Metric to analyze trends for
            experiments: List of experiment names (None for all)
            time_granularity: Time granularity ('daily', 'weekly', 'monthly')
            
        Returns:
            AnalyticsResult: Trend analysis results
        """
        query = self.query().metrics([metric])
        
        if experiments:
            query = query.experiments(names=experiments)
        
        # Configure comparison processor for trend analysis
        if 'comparisons' in self._processors:
            processor = self._processors['comparisons']
            processor.comparison_method = 'trend'
            processor.time_granularity = time_granularity
        
        return query.group_by('time').execute()
    
    # === CONFIGURATION AND MANAGEMENT ===
    
    def update_configuration(self, 
                           config: DictConfig,
                           reinitialize_processors: bool = True):
        """
        Update analytics configuration.
        
        Args:
            config: New configuration
            reinitialize_processors: Whether to reinitialize processors
        """
        self.config = config
        
        if reinitialize_processors:
            self._initialize_processors()
        
        logger.info("Analytics configuration updated")
    
    def get_available_processors(self) -> Dict[str, str]:
        """
        Get information about available analytics processors.
        
        Returns:
            Dict[str, str]: Mapping of processor types to descriptions
        """
        return AnalyticsFactory.get_available_processors()
    
    def get_processor_status(self) -> Dict[str, bool]:
        """
        Get status of initialized processors.
        
        Returns:
            Dict[str, bool]: Mapping of processor types to initialization status
        """
        available = AnalyticsFactory.get_available_processors()
        return {proc_type: proc_type in self._processors for proc_type in available}
    
    def validate_query(self, query: AnalyticsQuery) -> List[ValidationError]:
        """
        Validate an analytics query before execution.
        
        Args:
            query: Query to validate
            
        Returns:
            List[ValidationError]: List of validation errors (empty if valid)
        """
        return query.validate()
    
    def clear_cache(self):
        """Clear the query cache."""
        self._query_cache.clear()
        logger.debug("Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get query cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        return {
            'cache_size': len(self._query_cache),
            'cache_keys': list(self._query_cache.keys())
        }
    
    # === EXPORT AND VISUALIZATION METHODS ===
    
    def export_result(self,
                     result: AnalyticsResult,
                     output_path: Union[str, Path],
                     format: ExportFormat = ExportFormat.CSV,
                     **export_options) -> str:
        """
        Export analytics result to file.
        
        Args:
            result: Analytics result to export
            output_path: Path where to save the exported file
            format: Export format
            **export_options: Additional export options
            
        Returns:
            str: Path to the exported file
        """
        options = ExportOptions(format=format, **export_options)
        return self.exporter.export(result, output_path, options)
    
    def visualize_result(self,
                        result: AnalyticsResult,
                        visualization_type: VisualizationType = VisualizationType.LINE_PLOT,
                        save_path: Optional[str] = None,
                        **viz_options) -> Optional[str]:
        """
        Create visualization from analytics result.
        
        Args:
            result: Analytics result to visualize
            visualization_type: Type of visualization to create
            save_path: Path to save visualization (optional)
            **viz_options: Additional visualization options
            
        Returns:
            Optional[str]: Path to saved visualization file (if save_path specified)
        """
        options = VisualizationOptions(
            type=visualization_type,
            save_path=save_path,
            **viz_options
        )
        return self.visualizer.visualize(result, options)
    
    def generate_report(self,
                       results: Union[AnalyticsResult, List[AnalyticsResult]],
                       output_dir: Union[str, Path],
                       report_name: str = "analytics_report",
                       include_visualizations: bool = True,
                       export_formats: List[ExportFormat] = None) -> Dict[str, str]:
        """
        Generate comprehensive analytics report.
        
        Args:
            results: Analytics result(s) to include in report
            output_dir: Directory to save report files
            report_name: Base name for report files
            include_visualizations: Whether to include visualizations
            export_formats: List of export formats
            
        Returns:
            Dict[str, str]: Mapping of file types to file paths
        """
        return self.report_generator.generate_report(
            results=results,
            output_dir=output_dir,
            report_name=report_name,
            include_visualizations=include_visualizations,
            export_formats=export_formats
        )
    
    def quick_export(self,
                    metric: str,
                    experiments: Optional[List[str]] = None,
                    output_path: Union[str, Path] = "analytics_results.csv",
                    format: ExportFormat = ExportFormat.CSV) -> str:
        """
        Quick export of statistical analysis for a metric.
        
        Args:
            metric: Metric to analyze and export
            experiments: List of experiment names (None for all)
            output_path: Path where to save the exported file
            format: Export format
            
        Returns:
            str: Path to the exported file
        """
        result = self.analyze_statistics(metric, experiments)
        return self.export_result(result, output_path, format)
    
    def quick_visualize(self,
                       metric: str,
                       experiments: Optional[List[str]] = None,
                       visualization_type: VisualizationType = VisualizationType.BAR_CHART,
                       save_path: Optional[str] = None) -> Optional[str]:
        """
        Quick visualization of statistical analysis for a metric.
        
        Args:
            metric: Metric to analyze and visualize
            experiments: List of experiment names (None for all)
            visualization_type: Type of visualization to create
            save_path: Path to save visualization (optional)
            
        Returns:
            Optional[str]: Path to saved visualization file (if save_path specified)
        """
        result = self.analyze_statistics(metric, experiments)
        return self.visualize_result(result, visualization_type, save_path)
    
    # === UTILITY METHODS ===
    
    def execute_query(self, query: AnalyticsQuery) -> AnalyticsResult:
        """
        Execute an analytics query with caching support.
        
        Args:
            query: Query to execute
            
        Returns:
            AnalyticsResult: Query results
        """
        if not self.db_manager:
            raise ValueError("Database manager not initialized. Cannot execute analytics queries.")
        
        # Generate cache key
        cache_key = query._state.query_id
        
        # Check cache
        if cache_key in self._query_cache:
            logger.debug(f"Returning cached result for query {cache_key}")
            return self._query_cache[cache_key]
        
        # Validate query
        errors = self.validate_query(query)
        if errors:
            error_msg = "; ".join([e.message for e in errors])
            raise ValidationError(f"Query validation failed: {error_msg}")
        
        # Execute query
        try:
            start_time = datetime.now()
            
            # Execute query using database manager
            data = self._execute_database_query(query)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create result with real data
            result = AnalyticsResult(
                data=data,
                metadata=QueryMetadata(
                    query_type="analytics_query",
                    execution_time=execution_time,
                    row_count=len(data) if isinstance(data, list) else 1,
                    processing_steps=query._get_processing_steps()
                )
            )
            
            # Cache result
            self._query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _execute_database_query(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """
        Execute query against the database manager.
        
        Args:
            query: Analytics query to execute
            
        Returns:
            List[Dict[str, Any]]: Raw data from database
        """
        # Extract query parameters
        state = query._state
        
        # Handle different query types based on what's requested
        if state.metrics and len(state.metrics) == 1:
            # Single metric query - use get_metrics_by_type
            metric_type = state.metrics[0]
            experiment_ids = self._resolve_experiment_ids(state.experiment_names) if state.experiment_names else None
            return self.db_manager.get_metrics_by_type(metric_type, experiment_ids)
        
        elif state.experiment_names and len(state.experiment_names) == 1:
            # Single experiment query - use get_experiment_performance_data
            experiment_id = self._resolve_experiment_ids([state.experiment_names[0]])[0]
            perf_data = self.db_manager.get_experiment_performance_data(experiment_id)
            return perf_data.get('timeline', [])
        
        elif state.experiment_names:
            # Multiple experiments - get data for each
            experiment_ids = self._resolve_experiment_ids(state.experiment_names)
            all_data = []
            for exp_id in experiment_ids:
                summary = self.db_manager.get_experiment_summary(exp_id)
                all_data.append(summary)
            return all_data
        
        else:
            # General query - get all experiments
            return self.db_manager.get_all_experiments()
    
    def _resolve_experiment_ids(self, experiment_names: List[str]) -> List[int]:
        """
        Resolve experiment names to IDs.
        
        Args:
            experiment_names: List of experiment names
            
        Returns:
            List[int]: List of experiment IDs
        """
        # Simple implementation - search by title pattern
        experiment_ids = []
        
        for name in experiment_names:
            experiments = self.db_manager.search_experiments(title_pattern=name)
            if experiments:
                experiment_ids.extend([exp['id'] for exp in experiments])
            else:
                logger.warning(f"No experiments found matching pattern: {name}")
        
        return experiment_ids
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ExperimentAnalytics':
        """
        Create ExperimentAnalytics instance from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ExperimentAnalytics: Configured instance
        """
        # Extract configuration parameters
        db_config = config.get('database', {})
        analytics_config = DictConfig(config.get('analytics', {}))
        level_str = config.get('configuration_level', 'standard')
        
        # Create database manager if configuration provided
        db_manager = None
        if db_config:
            db_manager = DatabaseManager(**db_config)
        
        # Parse configuration level
        try:
            configuration_level = ConfigurationLevel(level_str)
        except ValueError:
            configuration_level = ConfigurationLevel.STANDARD
        
        return cls(
            db_manager=db_manager,
            config=analytics_config,
            configuration_level=configuration_level
        )
    
    def to_config(self) -> Dict[str, Any]:
        """
        Export current configuration.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            'analytics': dict(self.config),
            'configuration_level': self.configuration_level.value,
            'processors': list(self._processors.keys())
        }
    
    def __repr__(self) -> str:
        """String representation of ExperimentAnalytics."""
        processor_count = len(self._processors)
        cache_size = len(self._query_cache)
        return (f"ExperimentAnalytics(processors={processor_count}, "
                f"cache_size={cache_size}, level={self.configuration_level.value})") 