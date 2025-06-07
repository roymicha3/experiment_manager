"""
Tests for the matplotlib renderer plugin.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np

from experiment_manager.visualization.renderers.matplotlib_renderer import MatplotlibRendererPlugin
from experiment_manager.visualization.themes.registry import ThemeRegistry
from experiment_manager.visualization.plugins.theme_plugin import ThemeConfig


class TestMatplotlibRendererPlugin:
    """Test suite for MatplotlibRendererPlugin."""
    
    @pytest.fixture
    def mock_theme_registry(self):
        """Create a mock theme registry."""
        registry = Mock(spec=ThemeRegistry)
        return registry
    
    @pytest.fixture
    def sample_theme(self):
        """Create a sample theme for testing."""
        theme = Mock(spec=ThemeConfig)
        theme.name = "test_theme"
        theme.get_color = Mock(side_effect=lambda key, default: {
            'background': '#ffffff',
            'text': '#000000',
            'grid': '#cccccc'
        }.get(key, default))
        theme.get_font = Mock(side_effect=lambda key, default: {
            'family': 'Arial'
        }.get(key, default))
        theme.fonts = {'size': 12}
        return theme
    
    @pytest.fixture
    def renderer(self, mock_theme_registry):
        """Create a matplotlib renderer instance."""
        return MatplotlibRendererPlugin(theme_registry=mock_theme_registry)
    
    def test_initialization(self, renderer):
        """Test renderer initialization."""
        assert renderer.name == "matplotlib"
        assert isinstance(renderer.supported_plot_types, list)
        assert len(renderer.supported_plot_types) > 0
        assert renderer.default_dpi == 300
        assert renderer.default_figsize == (10, 6)
        assert 'png' in renderer.supported_formats
        assert 'svg' in renderer.supported_formats
        assert 'pdf' in renderer.supported_formats
    
    def test_initialization_without_theme_registry(self):
        """Test renderer initialization without theme registry."""
        renderer = MatplotlibRendererPlugin()
        assert renderer.theme_registry is not None
        assert isinstance(renderer.theme_registry, ThemeRegistry)
    
    def test_supported_plot_types(self, renderer):
        """Test that all expected plot types are supported."""
        expected_types = [
            'line', 'scatter', 'bar', 'histogram', 'box', 'heatmap',
            'area', 'pie', 'violin', 'density'
        ]
        for plot_type in expected_types:
            assert plot_type in renderer.supported_plot_types
    
    def test_set_theme(self, renderer, sample_theme):
        """Test setting a theme."""
        with patch('matplotlib.pyplot.rcParams') as mock_rcparams:
            renderer.set_theme(sample_theme)
            
            assert renderer._current_theme == sample_theme
            mock_rcparams.update.assert_called_once()
            
            # Check that theme parameters were applied
            call_args = mock_rcparams.update.call_args[0][0]
            assert call_args['figure.facecolor'] == '#ffffff'
            assert call_args['text.color'] == '#000000'
            assert call_args['font.family'] == 'Arial'
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_line_plot(self, mock_close, mock_subplots, renderer):
        """Test rendering a line plot."""
        # Setup mocks
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'line',
            'data': {
                'x': [1, 2, 3, 4],
                'y': [1, 4, 2, 3]
            },
            'config': {
                'title': 'Test Line Plot',
                'xlabel': 'X Axis',
                'ylabel': 'Y Axis'
            }
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data') as mock_save:
            result = renderer.render(plot_spec)
            
            # Verify plot was created
            mock_subplots.assert_called_once()
            mock_ax.plot.assert_called_once()
            mock_ax.set_title.assert_called_with('Test Line Plot')
            mock_ax.set_xlabel.assert_called_with('X Axis')
            mock_ax.set_ylabel.assert_called_with('Y Axis')
            
            # Verify cleanup
            mock_close.assert_called_with('all')
            
            # Verify result
            assert result == b'test_data'
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_scatter_plot(self, mock_close, mock_subplots, renderer):
        """Test rendering a scatter plot."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'scatter',
            'data': {
                'x': [1, 2, 3, 4],
                'y': [1, 4, 2, 3],
                'color': ['red', 'blue', 'green', 'yellow'],
                'size': [20, 40, 60, 80]
            },
            'config': {
                'scatter_style': {'alpha': 0.7}
            }
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            renderer.render(plot_spec)
            
            mock_ax.scatter.assert_called_once()
            call_args = mock_ax.scatter.call_args
            assert np.array_equal(call_args[0][0], [1, 2, 3, 4])  # x
            assert np.array_equal(call_args[0][1], [1, 4, 2, 3])  # y
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_bar_plot(self, mock_close, mock_subplots, renderer):
        """Test rendering a bar plot."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'bar',
            'data': {
                'x': ['A', 'B', 'C', 'D'],
                'y': [1, 4, 2, 3]
            },
            'config': {
                'orientation': 'vertical'
            }
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            renderer.render(plot_spec)
            
            mock_ax.bar.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_horizontal_bar_plot(self, mock_close, mock_subplots, renderer):
        """Test rendering a horizontal bar plot."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'bar',
            'data': {
                'x': ['A', 'B', 'C', 'D'],
                'y': [1, 4, 2, 3]
            },
            'config': {
                'orientation': 'horizontal'
            }
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            renderer.render(plot_spec)
            
            mock_ax.barh.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_histogram(self, mock_close, mock_subplots, renderer):
        """Test rendering a histogram."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'histogram',
            'data': {
                'values': [1, 2, 2, 3, 3, 3, 4, 4, 5]
            },
            'config': {
                'bins': 5
            }
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            renderer.render(plot_spec)
            
            mock_ax.hist.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_box_plot(self, mock_close, mock_subplots, renderer):
        """Test rendering a box plot."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'box',
            'data': {
                'values': {
                    'Group A': [1, 2, 3, 4, 5],
                    'Group B': [2, 3, 4, 5, 6]
                }
            }
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            renderer.render(plot_spec)
            
            mock_ax.boxplot.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.colorbar')
    def test_render_heatmap(self, mock_colorbar, mock_close, mock_subplots, renderer):
        """Test rendering a heatmap."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_im = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_ax.imshow.return_value = mock_im
        
        plot_spec = {
            'type': 'heatmap',
            'data': {
                'matrix': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            },
            'config': {
                'colorbar': True
            }
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            renderer.render(plot_spec)
            
            mock_ax.imshow.assert_called_once()
            mock_colorbar.assert_called_once_with(mock_im, ax=mock_ax)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_area_plot(self, mock_close, mock_subplots, renderer):
        """Test rendering an area plot."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'area',
            'data': {
                'x': [1, 2, 3, 4],
                'y': [1, 4, 2, 3]
            }
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            renderer.render(plot_spec)
            
            mock_ax.fill_between.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_pie_chart(self, mock_close, mock_subplots, renderer):
        """Test rendering a pie chart."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'pie',
            'data': {
                'values': [30, 25, 25, 20],
                'labels': ['A', 'B', 'C', 'D']
            }
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            renderer.render(plot_spec)
            
            mock_ax.pie.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_violin_plot(self, mock_close, mock_subplots, renderer):
        """Test rendering a violin plot."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'violin',
            'data': {
                'values': {
                    'Group A': [1, 2, 3, 4, 5],
                    'Group B': [2, 3, 4, 5, 6]
                }
            }
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            renderer.render(plot_spec)
            
            mock_ax.violinplot.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_density_plot_with_pandas(self, mock_close, mock_subplots, renderer):
        """Test rendering a density plot with pandas available."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'density',
            'data': {
                'values': [1, 2, 2, 3, 3, 3, 4, 4, 5]
            }
        }
        
        with patch('pandas.Series') as mock_series:
            mock_series_instance = Mock()
            mock_series.return_value = mock_series_instance
            mock_series_instance.plot.density = Mock()
            
            with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
                renderer.render(plot_spec)
                
                mock_series.assert_called_once()
                mock_series_instance.plot.density.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_density_plot_fallback(self, mock_close, mock_subplots, renderer):
        """Test rendering a density plot without pandas (fallback to histogram)."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_spec = {
            'type': 'density',
            'data': {
                'values': [1, 2, 2, 3, 3, 3, 4, 4, 5]
            }
        }
        
        with patch('pandas.Series', side_effect=ImportError):
            with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
                renderer.render(plot_spec)
                
                mock_ax.hist.assert_called_once()
                call_kwargs = mock_ax.hist.call_args[1]
                assert call_kwargs['density'] is True
    
    def test_render_unsupported_plot_type(self, renderer):
        """Test rendering with unsupported plot type raises error."""
        plot_spec = {
            'type': 'unsupported_type',
            'data': {},
            'config': {}
        }
        
        with pytest.raises(ValueError, match="Unsupported plot type"):
            renderer.render(plot_spec)
    
    def test_save_to_file(self, renderer):
        """Test saving plot to file."""
        mock_fig = Mock()
        config = {'format': 'png', 'dpi': 150}
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            result = renderer._save_to_file(mock_fig, output_path, config)
            
            assert result == output_path
            mock_fig.savefig.assert_called_once()
            
            # Check save arguments
            call_args = mock_fig.savefig.call_args
            assert call_args[0][0] == output_path
            assert call_args[1]['format'] == 'png'
            assert call_args[1]['dpi'] == 150
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_save_to_bytes(self, renderer):
        """Test saving plot to bytes."""
        mock_fig = Mock()
        config = {'format': 'png'}
        
        with patch('io.BytesIO') as mock_bytesio:
            mock_buffer = Mock()
            mock_bytesio.return_value = mock_buffer
            mock_buffer.getvalue.return_value = b'test_image_data'
            
            result = renderer._save_to_bytes(mock_fig, config)
            
            assert result == b'test_image_data'
            # Check that savefig was called with buffer and format, allow additional kwargs
            call_args = mock_fig.savefig.call_args
            assert call_args[0] == (mock_buffer,)  # First positional arg is buffer
            assert call_args[1]['format'] == 'png'  # format keyword arg
            mock_buffer.seek.assert_called_once_with(0)
    
    def test_get_format_kwargs_png(self, renderer):
        """Test format-specific kwargs for PNG."""
        config = {'dpi': 200, 'format': 'png', 'transparent': True}
        kwargs = renderer._get_format_kwargs('png', config)
        
        assert kwargs['dpi'] == 200
        assert kwargs['transparent'] is True
        assert kwargs['bbox_inches'] == 'tight'
        assert kwargs['facecolor'] == 'white'
    
    def test_get_format_kwargs_svg(self, renderer):
        """Test format-specific kwargs for SVG."""
        config = {'format': 'svg'}
        kwargs = renderer._get_format_kwargs('svg', config)
        
        assert kwargs['transparent'] is True
    
    def test_get_format_kwargs_pdf(self, renderer):
        """Test format-specific kwargs for PDF."""
        config = {'format': 'pdf'}
        kwargs = renderer._get_format_kwargs('pdf', config)
        
        assert 'metadata' in kwargs
        assert kwargs['metadata']['Creator'] == 'Experiment Manager Visualization System'
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_render_multiple_plots(self, mock_tight_layout, mock_close, mock_subplots, renderer):
        """Test rendering multiple plots in one figure."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        plot_specs = [
            {
                'type': 'line',
                'data': {'x': [1, 2, 3], 'y': [1, 2, 3]},
                'config': {'title': 'Plot 1'}
            },
            {
                'type': 'scatter',
                'data': {'x': [1, 2, 3], 'y': [3, 2, 1]},
                'config': {'title': 'Plot 2'}
            }
        ]
        
        layout = {
            'rows': 1,
            'cols': 2,
            'figsize': (12, 6)
        }
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            result = renderer.render_multiple(plot_specs, layout)
            
            # Verify subplots creation
            mock_subplots.assert_called_once_with(1, 2, figsize=(12, 6))
            
            # Verify each plot was rendered
            assert mock_axes[0].plot.called or mock_axes[0].scatter.called
            assert mock_axes[1].plot.called or mock_axes[1].scatter.called
            
            # Verify layout
            mock_tight_layout.assert_called_once()
            
            assert result == b'test_data'
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_multiple_plots_auto_layout(self, mock_close, mock_subplots, renderer):
        """Test rendering multiple plots with automatic layout."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        plot_specs = [
            {'type': 'line', 'data': {'x': [1, 2], 'y': [1, 2]}, 'config': {}},
            {'type': 'line', 'data': {'x': [1, 2], 'y': [2, 1]}, 'config': {}},
            {'type': 'line', 'data': {'x': [1, 2], 'y': [1, 1]}, 'config': {}},
            {'type': 'line', 'data': {'x': [1, 2], 'y': [2, 2]}, 'config': {}}
        ]
        
        with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
            renderer.render_multiple(plot_specs)
            
            # Should create 2 rows, 3 cols for 4 plots (auto-layout)
            mock_subplots.assert_called_once()
            call_args = mock_subplots.call_args[0]
            rows, cols = call_args[0], call_args[1]
            assert rows * cols >= len(plot_specs)
    
    def test_get_capabilities(self, renderer):
        """Test getting renderer capabilities."""
        capabilities = renderer.get_capabilities()
        
        assert capabilities['name'] == 'matplotlib'
        assert 'plot_types' in capabilities
        assert 'output_formats' in capabilities
        assert 'features' in capabilities
        assert 'version' in capabilities
        
        # Check that all supported plot types are listed
        for plot_type in renderer.supported_plot_types:
            assert plot_type in capabilities['plot_types']
        
        # Check that all supported formats are listed
        for format_type in renderer.supported_formats:
            assert format_type in capabilities['output_formats']
    
    @patch('matplotlib.pyplot.close')
    def test_cleanup(self, mock_close, renderer):
        """Test cleanup functionality."""
        # Add some items to cache
        renderer._figure_cache['test'] = Mock()
        
        renderer.cleanup()
        
        # Verify cleanup
        mock_close.assert_called_once_with('all')
        assert len(renderer._figure_cache) == 0
    
    def test_render_with_output_path(self, renderer):
        """Test rendering with output path specified."""
        plot_spec = {
            'type': 'line',
            'data': {'x': [1, 2, 3], 'y': [1, 2, 3]},
            'config': {}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            with patch.object(renderer, '_save_to_file', return_value=output_path) as mock_save:
                result = renderer.render(plot_spec, output_path=output_path)
                
                mock_save.assert_called_once()
                assert result == output_path
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_render_with_theme_applied(self, renderer, sample_theme):
        """Test rendering with theme applied."""
        renderer.set_theme(sample_theme)
        
        plot_spec = {
            'type': 'line',
            'data': {'x': [1, 2, 3], 'y': [1, 2, 3]},
            'config': {}
        }
        
        with patch('matplotlib.pyplot.rcParams') as mock_rcparams:
            with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
                renderer.render(plot_spec)
                
                # Theme should be applied during rendering
                assert mock_rcparams.update.call_count >= 1
    
    def test_render_line_plot_multiple_series(self, renderer):
        """Test rendering line plot with multiple series."""
        plot_spec = {
            'type': 'line',
            'data': {
                'x': [1, 2, 3, 4],
                'y': {
                    'Series 1': [1, 2, 3, 4],
                    'Series 2': [4, 3, 2, 1]
                }
            },
            'config': {}
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
                renderer.render(plot_spec)
                
                # Should call plot twice (once for each series)
                assert mock_ax.plot.call_count == 2
                # Should add legend for multiple series
                mock_ax.legend.assert_called_once()
    
    def test_render_stacked_area_plot(self, renderer):
        """Test rendering stacked area plot."""
        plot_spec = {
            'type': 'area',
            'data': {
                'x': [1, 2, 3, 4],
                'y': {
                    'Series 1': [1, 2, 3, 4],
                    'Series 2': [2, 1, 2, 1]
                }
            },
            'config': {}
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            with patch.object(renderer, '_save_to_bytes', return_value=b'test_data'):
                renderer.render(plot_spec)
                
                # Should call fill_between twice (once for each series)
                assert mock_ax.fill_between.call_count == 2
                # Should add legend for multiple series
                mock_ax.legend.assert_called_once()
    
    def test_apply_plot_styling_with_limits(self, renderer):
        """Test applying plot styling with axis limits."""
        mock_ax = Mock()
        config = {
            'xlim': [0, 10],
            'ylim': [-5, 5],
            'grid': True,
            'grid_alpha': 0.5
        }
        
        renderer._apply_plot_styling(mock_ax, config)
        
        mock_ax.set_xlim.assert_called_once_with([0, 10])
        mock_ax.set_ylim.assert_called_once_with([-5, 5])
        mock_ax.grid.assert_called_once_with(True, alpha=0.5)
    
    def test_error_handling_in_render(self, renderer):
        """Test error handling during rendering."""
        plot_spec = {
            'type': 'line',
            'data': {'x': [1, 2, 3], 'y': [1, 2, 3]},
            'config': {}
        }
        
        with patch('matplotlib.pyplot.subplots', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                renderer.render(plot_spec)
    
    def test_format_detection_from_path(self, renderer):
        """Test format detection from file path."""
        mock_fig = Mock()
        config = {}
        
        # Test with .svg extension
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_file:
            svg_path = tmp_file.name
        
        try:
            renderer._save_to_file(mock_fig, svg_path, config)
            
            call_kwargs = mock_fig.savefig.call_args[1]
            assert call_kwargs['format'] == 'svg'
        finally:
            Path(svg_path).unlink(missing_ok=True)
    
    def test_unsupported_format_fallback(self, renderer):
        """Test fallback to PNG for unsupported formats."""
        mock_fig = Mock()
        config = {'format': 'unsupported_format'}
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            renderer._save_to_file(mock_fig, output_path, config)
            
            call_kwargs = mock_fig.savefig.call_args[1]
            assert call_kwargs['format'] == 'png'
        finally:
            Path(output_path).unlink(missing_ok=True) 