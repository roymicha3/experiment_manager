"""
Training Curves Plot Plugin

Specialized plot plugin for visualizing training progression across epochs/batches
with support for multiple metrics, confidence bands, smoothing, and annotations.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

from experiment_manager.visualization.plugins.plot_plugin import PlotPlugin, PlotData, PlotResult
from experiment_manager.visualization.plugins.theme_plugin import ThemeConfig


logger = logging.getLogger(__name__)


class TrainingCurvesPlotPlugin(PlotPlugin):
    """
    Specialized plot plugin for training curve visualization.
    
    Features:
    - Multi-metric support (loss, accuracy, learning rate, etc.)
    - Confidence bands from multiple runs
    - Smoothing options (moving average, gaussian, exponential)
    - Epoch/batch alignment across multiple experiments
    - Performance metric overlays and annotations
    - Support for train/validation splits
    - Milestone annotations and phase markers
    """
    
    def __init__(self):
        super().__init__()
        self._supported_smoothing = {
            'moving_average': self._moving_average_smooth,
            'gaussian': self._gaussian_smooth,
            'exponential': self._exponential_smooth,
            'none': lambda x, **kwargs: x
        }
        
    @property
    def plugin_name(self) -> str:
        """Name of the training curves plot plugin."""
        return "training_curves"
    
    @property
    def supported_data_types(self) -> List[str]:
        """Data types this plugin can handle."""
        return ['timeseries', 'training_metrics', 'experiment_logs']
    
    @property
    def plot_dimensions(self) -> str:
        """Dimensionality of plots this plugin creates."""
        return '2D'
    
    @property
    def required_data_columns(self) -> List[str]:
        """Required columns for training curve data."""
        return ['step', 'metric_name', 'value']
    
    @property
    def optional_data_columns(self) -> List[str]:
        """Optional columns that enhance the visualization."""
        return ['run_id', 'experiment_id', 'phase', 'epoch', 'batch', 
                'learning_rate', 'split', 'timestamp', 'std', 'confidence_lower', 
                'confidence_upper']
    
    def initialize(self) -> bool:
        """Initialize the training curves plot plugin."""
        try:
            # Verify matplotlib is available
            plt.figure()
            plt.close()
            logger.info("TrainingCurvesPlotPlugin initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TrainingCurvesPlotPlugin: {e}")
            return False
    
    def can_handle_data(self, data: PlotData) -> bool:
        """Check if this plugin can handle the given data."""
        try:
            if isinstance(data.data, pd.DataFrame):
                required_cols = set(self.required_data_columns)
                available_cols = set(data.data.columns)
                
                # Check if we have required columns or reasonable alternatives
                has_step = 'step' in available_cols or 'epoch' in available_cols or 'batch' in available_cols
                has_metric = 'metric_name' in available_cols or len([col for col in available_cols if any(metric in col.lower() for metric in ['loss', 'acc', 'error'])]) > 0
                has_value = 'value' in available_cols or len(data.data.select_dtypes(include=[np.number]).columns) > 0
                
                return has_step and has_metric and has_value
                
            elif isinstance(data.data, dict):
                # Check for metric time series data
                for key, value in data.data.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > 1:
                        return True
                return False
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking data compatibility: {e}")
            return False
    
    def preprocess_data(self, data: PlotData) -> PlotData:
        """Preprocess data for training curve visualization."""
        try:
            if isinstance(data.data, pd.DataFrame):
                df = data.data.copy()
                
                # Normalize column names
                if 'epoch' in df.columns and 'step' not in df.columns:
                    df['step'] = df['epoch']
                elif 'batch' in df.columns and 'step' not in df.columns:
                    df['step'] = df['batch']
                
                # Handle metric name extraction from columns
                if 'metric_name' not in df.columns:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    metric_cols = [col for col in numeric_cols if col not in ['step', 'epoch', 'batch']]
                    
                    if metric_cols:
                        # Melt the dataframe to long format
                        id_vars = ['step'] + [col for col in ['run_id', 'experiment_id', 'phase', 'split'] if col in df.columns]
                        df = df.melt(id_vars=id_vars, value_vars=metric_cols, 
                                   var_name='metric_name', value_name='value')
                
                # Ensure required columns exist
                if 'value' not in df.columns and len(df.select_dtypes(include=[np.number]).columns) > 0:
                    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
                    df['value'] = df[numeric_col]
                
                return PlotData(df, data.metadata)
                
            elif isinstance(data.data, dict):
                # Convert dict to DataFrame format
                rows = []
                for metric_name, values in data.data.items():
                    if isinstance(values, (list, np.ndarray)):
                        for step, value in enumerate(values):
                            rows.append({
                                'step': step,
                                'metric_name': metric_name,
                                'value': value
                            })
                
                df = pd.DataFrame(rows)
                return PlotData(df, data.metadata)
            
            return data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return data
    
    def generate_plot(self, 
                     data: PlotData,
                     config: Optional[Dict[str, Any]] = None) -> PlotResult:
        """Generate training curves plot from the provided data."""
        try:
            # Apply default configuration
            config = config or {}
            plot_config = self._apply_default_config(config)
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            if not isinstance(processed_data.data, pd.DataFrame):
                raise ValueError("Preprocessed data must be a DataFrame")
            
            df = processed_data.data
            
            # Create figure and subplots
            metrics = df['metric_name'].unique()
            num_metrics = len(metrics)
            
            # Determine subplot layout
            if num_metrics == 1:
                fig, axes = plt.subplots(1, 1, figsize=plot_config['figsize'])
                axes = [axes]
            elif num_metrics <= 4:
                rows = int(np.ceil(num_metrics / 2))
                fig, axes = plt.subplots(rows, 2, figsize=plot_config['figsize'])
                axes = axes.flatten() if num_metrics > 1 else [axes]
            else:
                rows = int(np.ceil(num_metrics / 3))
                fig, axes = plt.subplots(rows, 3, figsize=plot_config['figsize'])
                axes = axes.flatten()
            
            # Apply theme
            theme_config = plot_config.get('theme_config')
            if theme_config:
                self._apply_theme(fig, theme_config)
            
            # Plot each metric
            for i, metric in enumerate(metrics):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                metric_data = df[df['metric_name'] == metric]
                
                self._plot_metric(ax, metric_data, metric, plot_config)
            
            # Hide unused subplots
            for i in range(num_metrics, len(axes)):
                axes[i].set_visible(False)
            
            # Add title and adjust layout
            if plot_config.get('title'):
                fig.suptitle(plot_config['title'], fontsize=plot_config.get('title_fontsize', 16))
            
            plt.tight_layout()
            
            # Add annotations if specified
            if plot_config.get('annotations'):
                self._add_annotations(fig, axes, plot_config['annotations'])
            
            # Generate metadata
            metadata = {
                'metrics': list(metrics),
                'num_points': len(df),
                'config': plot_config,
                'data_shape': df.shape
            }
            
            return PlotResult(
                plot_object=fig,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error generating training curves plot: {e}")
            return PlotResult(
                plot_object=None,
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _plot_metric(self, ax, metric_data: pd.DataFrame, metric_name: str, config: Dict[str, Any]):
        """Plot a single metric with confidence bands and smoothing."""
        # Handle train/validation splits first - if splits exist, use split-based plotting exclusively
        if 'split' in metric_data.columns:
            self._plot_splits(ax, metric_data, config)
            return
        
        # Group by run/experiment if available
        if 'run_id' in metric_data.columns or 'experiment_id' in metric_data.columns:
            group_col = 'run_id' if 'run_id' in metric_data.columns else 'experiment_id'
            groups = metric_data.groupby(group_col)
            
            # Plot individual runs
            all_steps = []
            all_values = []
            
            for group_id, group_data in groups:
                steps = group_data['step'].values
                values = group_data['value'].values
                
                # Apply smoothing if requested
                smoothing = config.get('smoothing', 'none')
                if smoothing != 'none':
                    values = self._apply_smoothing(values, smoothing, config.get('smoothing_params', {}))
                
                # Plot individual run
                alpha = config.get('individual_alpha', 0.3)
                ax.plot(steps, values, alpha=alpha, linewidth=1, 
                       color=config.get('base_color', 'blue'))
                
                all_steps.extend(steps)
                all_values.extend(values)
            
            # Calculate and plot mean with confidence bands
            if config.get('show_confidence', True) and len(groups) > 1:
                self._plot_confidence_bands(ax, metric_data, config)
            
        else:
            # Single run
            steps = metric_data['step'].values
            values = metric_data['value'].values
            
            # Apply smoothing
            smoothing = config.get('smoothing', 'none')
            if smoothing != 'none':
                values = self._apply_smoothing(values, smoothing, config.get('smoothing_params', {}))
            
            ax.plot(steps, values, linewidth=config.get('linewidth', 2),
                   color=config.get('base_color', 'blue'))
        
        # Customize axes
        ax.set_xlabel(config.get('xlabel', 'Step'))
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name.replace('_', ' ').title()}")
        ax.grid(config.get('grid', True), alpha=0.3)
        
        # Set axis limits if specified
        if config.get('ylim'):
            ax.set_ylim(config['ylim'])
        if config.get('xlim'):
            ax.set_xlim(config['xlim'])
    
    def _plot_confidence_bands(self, ax, metric_data: pd.DataFrame, config: Dict[str, Any]):
        """Plot confidence bands for multiple runs."""
        group_col = 'run_id' if 'run_id' in metric_data.columns else 'experiment_id'
        
        # Calculate statistics at each step
        stats_df = metric_data.groupby('step')['value'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate confidence intervals
        confidence_level = config.get('confidence_level', 0.95)
        alpha = 1 - confidence_level
        
        stats_df['sem'] = stats_df['std'] / np.sqrt(stats_df['count'])
        stats_df['ci_lower'] = stats_df['mean'] - stats.t.ppf(1 - alpha/2, stats_df['count'] - 1) * stats_df['sem']
        stats_df['ci_upper'] = stats_df['mean'] + stats.t.ppf(1 - alpha/2, stats_df['count'] - 1) * stats_df['sem']
        
        # Apply smoothing to mean if requested
        smoothing = config.get('smoothing', 'none')
        mean_values = stats_df['mean'].values
        if smoothing != 'none':
            mean_values = self._apply_smoothing(mean_values, smoothing, config.get('smoothing_params', {}))
        
        # Plot mean line
        ax.plot(stats_df['step'], mean_values, 
               linewidth=config.get('mean_linewidth', 3),
               color=config.get('mean_color', 'darkblue'),
               label='Mean')
        
        # Plot confidence band
        ax.fill_between(stats_df['step'], stats_df['ci_lower'], stats_df['ci_upper'],
                       alpha=config.get('confidence_alpha', 0.2),
                       color=config.get('confidence_color', 'blue'),
                       label=f'{int(confidence_level*100)}% CI')
        
        ax.legend()
    
    def _plot_splits(self, ax, metric_data: pd.DataFrame, config: Dict[str, Any]):
        """Plot train/validation splits with different styles."""
        splits = metric_data['split'].unique()
        colors = config.get('split_colors', {'train': 'blue', 'val': 'red', 'validation': 'red', 'test': 'green'})
        
        # Check if we have multiple runs for confidence bands
        has_multiple_runs = ('run_id' in metric_data.columns and len(metric_data['run_id'].unique()) > 1) or \
                           ('experiment_id' in metric_data.columns and len(metric_data['experiment_id'].unique()) > 1)
        
        for split in splits:
            split_data = metric_data[metric_data['split'] == split]
            color = colors.get(split, f'C{len(colors)}')
            
            if has_multiple_runs and config.get('show_confidence', True):
                # Plot individual runs with low alpha
                group_col = 'run_id' if 'run_id' in split_data.columns else 'experiment_id'
                groups = split_data.groupby(group_col)
                
                # Plot individual runs
                for group_id, group_data in groups:
                    steps = group_data['step'].values
                    values = group_data['value'].values
                    
                    # Apply smoothing if requested
                    smoothing = config.get('smoothing', 'none')
                    if smoothing != 'none':
                        values = self._apply_smoothing(values, smoothing, config.get('smoothing_params', {}))
                    
                    ax.plot(steps, values, alpha=config.get('individual_alpha', 0.3), 
                           linewidth=1, color=color)
                
                # Plot mean and confidence bands for this split
                self._plot_split_confidence_bands(ax, split_data, split, color, config)
                
            else:
                # Single run or no confidence bands
                steps = split_data['step'].values
                values = split_data['value'].values
                
                # Apply smoothing
                smoothing = config.get('smoothing', 'none')
                if smoothing != 'none':
                    values = self._apply_smoothing(values, smoothing, config.get('smoothing_params', {}))
                
                ax.plot(steps, values, label=split.title(), color=color,
                       linewidth=config.get('linewidth', 2))
        
        ax.legend()
    
    def _plot_split_confidence_bands(self, ax, split_data: pd.DataFrame, split_name: str, color: str, config: Dict[str, Any]):
        """Plot confidence bands for a specific split (train/val)."""
        # Calculate statistics at each step
        stats_df = split_data.groupby('step')['value'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate confidence intervals
        confidence_level = config.get('confidence_level', 0.95)
        alpha = 1 - confidence_level
        
        stats_df['sem'] = stats_df['std'] / np.sqrt(stats_df['count'])
        stats_df['ci_lower'] = stats_df['mean'] - stats.t.ppf(1 - alpha/2, stats_df['count'] - 1) * stats_df['sem']
        stats_df['ci_upper'] = stats_df['mean'] + stats.t.ppf(1 - alpha/2, stats_df['count'] - 1) * stats_df['sem']
        
        # Apply smoothing to mean if requested
        smoothing = config.get('smoothing', 'none')
        mean_values = stats_df['mean'].values
        if smoothing != 'none':
            mean_values = self._apply_smoothing(mean_values, smoothing, config.get('smoothing_params', {}))
        
        # Plot mean line
        ax.plot(stats_df['step'], mean_values, 
               linewidth=config.get('mean_linewidth', 3),
               color=color,
               label=f'{split_name.title()} Mean')
        
        # Plot confidence band
        ax.fill_between(stats_df['step'], stats_df['ci_lower'], stats_df['ci_upper'],
                       alpha=config.get('confidence_alpha', 0.2),
                       color=color,
                       label=f'{split_name.title()} {int(confidence_level*100)}% CI')
    
    def _apply_smoothing(self, values: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
        """Apply smoothing to the values."""
        if method not in self._supported_smoothing:
            logger.warning(f"Unsupported smoothing method: {method}")
            return values
        
        return self._supported_smoothing[method](values, **params)
    
    def _moving_average_smooth(self, values: np.ndarray, window: int = 5, **kwargs) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(values) <= window:
            return values
        
        smoothed = np.convolve(values, np.ones(window)/window, mode='same')
        # Handle edges
        for i in range(window//2):
            smoothed[i] = np.mean(values[:i+window//2+1])
            smoothed[-(i+1)] = np.mean(values[-(i+window//2+1):])
        
        return smoothed
    
    def _gaussian_smooth(self, values: np.ndarray, sigma: float = 1.0, **kwargs) -> np.ndarray:
        """Apply Gaussian smoothing."""
        return gaussian_filter1d(values, sigma=sigma)
    
    def _exponential_smooth(self, values: np.ndarray, alpha: float = 0.3, **kwargs) -> np.ndarray:
        """Apply exponential smoothing."""
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed
    
    def _add_annotations(self, fig, axes, annotations: List[Dict[str, Any]]):
        """Add annotations to the plot."""
        for annotation in annotations:
            ax_idx = annotation.get('axis', 0)
            if ax_idx >= len(axes):
                continue
                
            ax = axes[ax_idx]
            
            if annotation['type'] == 'vertical_line':
                x = annotation['x']
                ax.axvline(x, color=annotation.get('color', 'red'),
                          linestyle=annotation.get('linestyle', '--'),
                          alpha=annotation.get('alpha', 0.7),
                          label=annotation.get('label'))
                
            elif annotation['type'] == 'horizontal_line':
                y = annotation['y']
                ax.axhline(y, color=annotation.get('color', 'red'),
                          linestyle=annotation.get('linestyle', '--'),
                          alpha=annotation.get('alpha', 0.7),
                          label=annotation.get('label'))
                
            elif annotation['type'] == 'text':
                ax.annotate(annotation['text'],
                           xy=(annotation['x'], annotation['y']),
                           xytext=(annotation.get('text_x', annotation['x']),
                                 annotation.get('text_y', annotation['y'])),
                           arrowprops=annotation.get('arrow_props'),
                           fontsize=annotation.get('fontsize', 10),
                           color=annotation.get('color', 'black'))
                
            elif annotation['type'] == 'phase_marker':
                x_start = annotation['x_start']
                x_end = annotation['x_end']
                ax.axvspan(x_start, x_end,
                          alpha=annotation.get('alpha', 0.2),
                          color=annotation.get('color', 'yellow'),
                          label=annotation.get('label'))
    
    def _apply_theme(self, fig, theme_config: ThemeConfig):
        """Apply theme configuration to the figure."""
        try:
            # Apply background colors
            fig.patch.set_facecolor(theme_config.background_color)
            
            # Apply to all axes
            for ax in fig.get_axes():
                ax.set_facecolor(theme_config.plot_background_color)
                
                # Apply grid styling
                if hasattr(theme_config, 'grid_color'):
                    ax.grid(True, color=theme_config.grid_color, alpha=0.3)
                
                # Apply text colors
                if hasattr(theme_config, 'text_color'):
                    ax.tick_params(colors=theme_config.text_color)
                    ax.xaxis.label.set_color(theme_config.text_color)
                    ax.yaxis.label.set_color(theme_config.text_color)
                    ax.title.set_color(theme_config.text_color)
                
        except Exception as e:
            logger.warning(f"Error applying theme: {e}")
    
    def _apply_default_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default configuration values."""
        defaults = {
            'figsize': (12, 8),
            'smoothing': 'none',
            'smoothing_params': {'window': 5, 'sigma': 1.0, 'alpha': 0.3},
            'show_confidence': True,
            'confidence_level': 0.95,
            'confidence_alpha': 0.2,
            'individual_alpha': 0.3,
            'linewidth': 2,
            'mean_linewidth': 3,
            'grid': True,
            'base_color': 'blue',
            'mean_color': 'darkblue',
            'confidence_color': 'blue',
            'split_colors': {'train': 'blue', 'val': 'red', 'validation': 'red', 'test': 'green'},
            'xlabel': 'Step',
            'title_fontsize': 16
        }
        
        # Merge with user config
        result = defaults.copy()
        result.update(config)
        return result
    
    def get_default_style(self) -> Dict[str, Any]:
        """Get default styling configuration for training curves."""
        return {
            "width": 1200,
            "height": 800,
            "title": "Training Curves",
            "theme": "default",
            "smoothing": "moving_average",
            "show_confidence": True,
            "grid": True
        }
    
    def cleanup(self):
        """Clean up resources used by the plugin."""
        plt.close('all')
        logger.info("TrainingCurvesPlotPlugin cleaned up") 