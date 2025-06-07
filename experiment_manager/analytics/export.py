"""
Analytics Results Export and Visualization Support

This module provides comprehensive export capabilities and basic visualization
support for analytics results. It supports multiple formats and includes
customizable visualization options.
"""

import json
import csv
import pickle
import io
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from experiment_manager.analytics.results import AnalyticsResult, QueryMetadata
from experiment_manager.common.serializable import YAMLSerializable

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "xlsx"
    PICKLE = "pkl"
    PARQUET = "parquet"
    HTML = "html"
    YAML = "yaml"


class VisualizationType(Enum):
    """Supported visualization types."""
    LINE_PLOT = "line"
    BAR_CHART = "bar"
    HISTOGRAM = "histogram"
    SCATTER_PLOT = "scatter"
    BOX_PLOT = "box"
    HEATMAP = "heatmap"
    CORRELATION_MATRIX = "correlation"


@dataclass
class ExportOptions:
    """Configuration options for export operations."""
    format: ExportFormat = ExportFormat.CSV
    include_metadata: bool = True
    include_timestamps: bool = True
    compression: Optional[str] = None  # 'gzip', 'bz2', 'xz'
    encoding: str = 'utf-8'
    
    # CSV specific options
    csv_delimiter: str = ','
    csv_quoting: int = csv.QUOTE_MINIMAL
    
    # JSON specific options
    json_indent: int = 2
    
    # Excel specific options
    excel_sheet_name: str = 'Analytics Results'
    excel_include_charts: bool = False


@dataclass
class VisualizationOptions:
    """Configuration options for visualization."""
    type: VisualizationType = VisualizationType.LINE_PLOT
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    
    # General options
    width: int = 800
    height: int = 600
    dpi: int = 100
    style: str = 'default'
    
    # Color options
    color_palette: Optional[List[str]] = None
    background_color: str = 'white'
    
    # Interactive options (for Plotly)
    interactive: bool = True
    show_hover: bool = True
    
    # File options
    save_format: str = 'png'
    save_path: Optional[str] = None


@YAMLSerializable.register("ResultExporter")
class ResultExporter(YAMLSerializable):
    """
    Comprehensive exporter for analytics results.
    
    Supports multiple export formats and provides flexible configuration
    options for different use cases.
    """
    
    def __init__(self, default_options: Optional[ExportOptions] = None):
        """
        Initialize ResultExporter.
        
        Args:
            default_options: Default export options to use
        """
        self.default_options = default_options or ExportOptions()
        
        # Check for optional dependencies
        self._check_dependencies()
        
        logger.debug("ResultExporter initialized")
    
    def _check_dependencies(self):
        """Check availability of optional dependencies."""
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available. Excel and Parquet exports will be limited.")
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Visualization features will be limited.")
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Interactive visualizations will be unavailable.")
    
    def export(self, 
               result: AnalyticsResult,
               output_path: Union[str, Path],
               options: Optional[ExportOptions] = None) -> str:
        """
        Export analytics result to specified format.
        
        Args:
            result: Analytics result to export
            output_path: Path where to save the exported file
            options: Export options (uses default if None)
            
        Returns:
            str: Path to the exported file
        """
        options = options or self.default_options
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for export
        export_data = self._prepare_export_data(result, options)
        
        # Export based on format
        if options.format == ExportFormat.CSV:
            return self._export_csv(export_data, output_path, options)
        elif options.format == ExportFormat.JSON:
            return self._export_json(export_data, output_path, options)
        elif options.format == ExportFormat.EXCEL:
            return self._export_excel(export_data, output_path, options)
        elif options.format == ExportFormat.PICKLE:
            return self._export_pickle(export_data, output_path, options)
        elif options.format == ExportFormat.PARQUET:
            return self._export_parquet(export_data, output_path, options)
        elif options.format == ExportFormat.HTML:
            return self._export_html(export_data, output_path, options)
        elif options.format == ExportFormat.YAML:
            return self._export_yaml(export_data, output_path, options)
        else:
            raise ValueError(f"Unsupported export format: {options.format}")
    
    def _prepare_export_data(self, result: AnalyticsResult, options: ExportOptions) -> Dict[str, Any]:
        """Prepare data for export."""
        # Convert DataFrame to appropriate format for export
        if hasattr(result.data, 'to_dict'):
            # It's a pandas DataFrame
            data_for_export = result.data.to_dict('records')
        else:
            data_for_export = result.data
        
        export_data = {
            'data': data_for_export
        }
        
        if options.include_metadata and result.metadata:
            export_data['metadata'] = {
                'query_type': result.metadata.query_type,
                'execution_time': result.metadata.execution_time,
                'row_count': result.metadata.row_count,
                'timestamp': result.metadata.timestamp.isoformat() if result.metadata.timestamp else None,
                'processing_steps': result.metadata.processing_steps,
                'cache_hit': result.metadata.cache_hit,
                'optimization_applied': result.metadata.optimization_applied
            }
        
        if options.include_timestamps:
            export_data['exported_at'] = datetime.now().isoformat()
        
        return export_data
    
    def _export_csv(self, data: Dict[str, Any], output_path: Path, options: ExportOptions) -> str:
        """Export to CSV format."""
        if not PANDAS_AVAILABLE:
            # Fallback to basic CSV export
            return self._export_csv_basic(data, output_path, options)
        
        # Convert to DataFrame for better CSV handling
        if isinstance(data['data'], list) and data['data'] and isinstance(data['data'][0], dict):
            # List of dictionaries - most common case
            df = pd.DataFrame(data['data'])
        elif isinstance(data['data'], dict):
            df = pd.DataFrame([data['data']])
        elif isinstance(data['data'], list):
            df = pd.DataFrame({'value': data['data']})
        else:
            df = pd.DataFrame({'value': [data['data']]})
        
        # Add metadata as additional columns if requested (but only if it won't cause conflicts)
        if options.include_metadata and 'metadata' in data and len(df) == 1:
            for key, value in data['metadata'].items():
                df[f'metadata_{key}'] = value
        
        # Export to CSV
        df.to_csv(
            output_path,
            sep=options.csv_delimiter,
            quoting=options.csv_quoting,
            encoding=options.encoding,
            index=False,
            compression=options.compression
        )
        
        logger.info(f"Exported to CSV: {output_path}")
        return str(output_path)
    
    def _export_csv_basic(self, data: Dict[str, Any], output_path: Path, options: ExportOptions) -> str:
        """Basic CSV export without pandas."""
        with open(output_path, 'w', newline='', encoding=options.encoding) as csvfile:
            writer = csv.writer(csvfile, delimiter=options.csv_delimiter, quoting=options.csv_quoting)
            
            # Write data
            if isinstance(data['data'], dict):
                writer.writerow(data['data'].keys())
                writer.writerow(data['data'].values())
            elif isinstance(data['data'], list) and data['data']:
                if isinstance(data['data'][0], dict):
                    # List of dictionaries
                    writer.writerow(data['data'][0].keys())
                    for row in data['data']:
                        writer.writerow(row.values())
                else:
                    # List of values
                    writer.writerow(['value'])
                    for value in data['data']:
                        writer.writerow([value])
        
        return str(output_path)
    
    def _export_json(self, data: Dict[str, Any], output_path: Path, options: ExportOptions) -> str:
        """Export to JSON format."""
        with open(output_path, 'w', encoding=options.encoding) as jsonfile:
            json.dump(data, jsonfile, indent=options.json_indent, default=str)
        
        logger.info(f"Exported to JSON: {output_path}")
        return str(output_path)
    
    def _export_excel(self, data: Dict[str, Any], output_path: Path, options: ExportOptions) -> str:
        """Export to Excel format."""
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas is required for Excel export")
        
        # Convert to DataFrame
        if isinstance(data['data'], dict):
            df = pd.DataFrame([data['data']])
        elif isinstance(data['data'], list):
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame({'value': [data['data']]})
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=options.excel_sheet_name, index=False)
            
            # Add metadata sheet if requested
            if options.include_metadata and 'metadata' in data:
                metadata_df = pd.DataFrame([data['metadata']])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        logger.info(f"Exported to Excel: {output_path}")
        return str(output_path)
    
    def _export_pickle(self, data: Dict[str, Any], output_path: Path, options: ExportOptions) -> str:
        """Export to Pickle format."""
        with open(output_path, 'wb') as picklefile:
            pickle.dump(data, picklefile)
        
        logger.info(f"Exported to Pickle: {output_path}")
        return str(output_path)
    
    def _export_parquet(self, data: Dict[str, Any], output_path: Path, options: ExportOptions) -> str:
        """Export to Parquet format."""
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas is required for Parquet export")
        
        # Convert to DataFrame
        if isinstance(data['data'], dict):
            df = pd.DataFrame([data['data']])
        elif isinstance(data['data'], list):
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame({'value': [data['data']]})
        
        df.to_parquet(output_path, compression=options.compression, index=False)
        
        logger.info(f"Exported to Parquet: {output_path}")
        return str(output_path)
    
    def _export_html(self, data: Dict[str, Any], output_path: Path, options: ExportOptions) -> str:
        """Export to HTML format."""
        html_content = self._generate_html_report(data)
        
        with open(output_path, 'w', encoding=options.encoding) as htmlfile:
            htmlfile.write(html_content)
        
        logger.info(f"Exported to HTML: {output_path}")
        return str(output_path)
    
    def _export_yaml(self, data: Dict[str, Any], output_path: Path, options: ExportOptions) -> str:
        """Export to YAML format."""
        import yaml
        
        with open(output_path, 'w', encoding=options.encoding) as yamlfile:
            yaml.dump(data, yamlfile, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Exported to YAML: {output_path}")
        return str(output_path)
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report from data."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analytics Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .data {{ margin-top: 20px; }}
                .metadata {{ margin-top: 20px; background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Analytics Results Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="data">
                <h2>Data</h2>
                {self._data_to_html_table(data['data'])}
            </div>
            
            {self._metadata_to_html(data.get('metadata', {})) if 'metadata' in data else ''}
        </body>
        </html>
        """
        return html
    
    def _data_to_html_table(self, data: Any) -> str:
        """Convert data to HTML table."""
        if isinstance(data, dict):
            table = "<table><tr><th>Key</th><th>Value</th></tr>"
            for key, value in data.items():
                table += f"<tr><td>{key}</td><td>{value}</td></tr>"
            table += "</table>"
            return table
        elif isinstance(data, list) and data:
            if isinstance(data[0], dict):
                # List of dictionaries
                table = "<table><tr>"
                for key in data[0].keys():
                    table += f"<th>{key}</th>"
                table += "</tr>"
                for row in data:
                    table += "<tr>"
                    for value in row.values():
                        table += f"<td>{value}</td>"
                    table += "</tr>"
                table += "</table>"
                return table
            else:
                # List of values
                table = "<table><tr><th>Value</th></tr>"
                for value in data:
                    table += f"<tr><td>{value}</td></tr>"
                table += "</table>"
                return table
        else:
            return f"<p>{data}</p>"
    
    def _metadata_to_html(self, metadata: Dict[str, Any]) -> str:
        """Convert metadata to HTML."""
        if not metadata:
            return ""
        
        html = '<div class="metadata"><h2>Metadata</h2><table>'
        for key, value in metadata.items():
            html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
        html += "</table></div>"
        return html


@YAMLSerializable.register("ResultVisualizer")
class ResultVisualizer(YAMLSerializable):
    """
    Basic visualization support for analytics results.
    
    Provides simple plotting capabilities using matplotlib and plotly
    for common visualization needs.
    """
    
    def __init__(self, default_options: Optional[VisualizationOptions] = None):
        """
        Initialize ResultVisualizer.
        
        Args:
            default_options: Default visualization options
        """
        self.default_options = default_options or VisualizationOptions()
        
        # Check for visualization dependencies
        self._check_dependencies()
        
        logger.debug("ResultVisualizer initialized")
    
    def _check_dependencies(self):
        """Check availability of visualization dependencies."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Static visualizations will be unavailable.")
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Interactive visualizations will be unavailable.")
    
    def visualize(self,
                  result: AnalyticsResult,
                  options: Optional[VisualizationOptions] = None) -> Optional[str]:
        """
        Create visualization from analytics result.
        
        Args:
            result: Analytics result to visualize
            options: Visualization options
            
        Returns:
            Optional[str]: Path to saved visualization file (if save_path specified)
        """
        options = options or self.default_options
        
        # Extract data for visualization
        data = self._extract_visualization_data(result)
        
        if options.interactive and PLOTLY_AVAILABLE:
            return self._create_plotly_visualization(data, options)
        elif MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_visualization(data, options)
        else:
            logger.error("No visualization libraries available")
            return None
    
    def _extract_visualization_data(self, result: AnalyticsResult) -> Dict[str, Any]:
        """Extract data suitable for visualization."""
        if isinstance(result.data, dict):
            # Convert dict to list format for easier plotting
            if all(isinstance(v, (int, float)) for v in result.data.values()):
                return {
                    'x': list(result.data.keys()),
                    'y': list(result.data.values()),
                    'type': 'categorical'
                }
        elif isinstance(result.data, list):
            if all(isinstance(item, dict) for item in result.data):
                # List of dictionaries - extract common numeric fields
                numeric_fields = []
                for item in result.data:
                    for key, value in item.items():
                        if isinstance(value, (int, float)) and key not in numeric_fields:
                            numeric_fields.append(key)
                
                if numeric_fields:
                    return {
                        'data': result.data,
                        'numeric_fields': numeric_fields,
                        'type': 'dataframe'
                    }
            elif all(isinstance(item, (int, float)) for item in result.data):
                return {
                    'x': list(range(len(result.data))),
                    'y': result.data,
                    'type': 'series'
                }
        
        # Fallback
        return {'data': result.data, 'type': 'raw'}
    
    def _create_matplotlib_visualization(self, data: Dict[str, Any], options: VisualizationOptions) -> Optional[str]:
        """Create visualization using matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        plt.style.use(options.style)
        fig, ax = plt.subplots(figsize=(options.width/100, options.height/100), dpi=options.dpi)
        
        if data['type'] == 'categorical':
            if options.type == VisualizationType.BAR_CHART:
                ax.bar(data['x'], data['y'], color=options.color_palette)
            elif options.type == VisualizationType.LINE_PLOT:
                ax.plot(data['x'], data['y'], marker='o', color=options.color_palette[0] if options.color_palette else None)
        elif data['type'] == 'series':
            if options.type == VisualizationType.LINE_PLOT:
                ax.plot(data['x'], data['y'], marker='o')
            elif options.type == VisualizationType.HISTOGRAM:
                ax.hist(data['y'], bins=20, alpha=0.7)
        
        # Set labels and title
        if options.title:
            ax.set_title(options.title)
        if options.xlabel:
            ax.set_xlabel(options.xlabel)
        if options.ylabel:
            ax.set_ylabel(options.ylabel)
        
        plt.tight_layout()
        
        # Save if path specified
        if options.save_path:
            plt.savefig(options.save_path, format=options.save_format, dpi=options.dpi)
            plt.close()
            return options.save_path
        else:
            plt.show()
            return None
    
    def _create_plotly_visualization(self, data: Dict[str, Any], options: VisualizationOptions) -> Optional[str]:
        """Create interactive visualization using plotly."""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = None
        
        if data['type'] == 'categorical':
            if options.type == VisualizationType.BAR_CHART:
                fig = px.bar(x=data['x'], y=data['y'], title=options.title)
            elif options.type == VisualizationType.LINE_PLOT:
                fig = px.line(x=data['x'], y=data['y'], title=options.title)
        elif data['type'] == 'series':
            if options.type == VisualizationType.LINE_PLOT:
                fig = px.line(x=data['x'], y=data['y'], title=options.title)
            elif options.type == VisualizationType.HISTOGRAM:
                fig = px.histogram(x=data['y'], title=options.title)
        
        if fig:
            fig.update_layout(
                width=options.width,
                height=options.height,
                xaxis_title=options.xlabel,
                yaxis_title=options.ylabel
            )
            
            if options.save_path:
                fig.write_html(options.save_path)
                return options.save_path
            else:
                fig.show()
                return None
        
        return None


@YAMLSerializable.register("AnalyticsReportGenerator")
class AnalyticsReportGenerator(YAMLSerializable):
    """
    Comprehensive report generator for analytics results.
    
    Combines export and visualization capabilities to generate
    comprehensive analytics reports.
    """
    
    def __init__(self):
        """Initialize AnalyticsReportGenerator."""
        self.exporter = ResultExporter()
        self.visualizer = ResultVisualizer()
        
        logger.debug("AnalyticsReportGenerator initialized")
    
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
            export_formats: List of export formats (default: CSV, JSON, HTML)
            
        Returns:
            Dict[str, str]: Mapping of file types to file paths
        """
        if export_formats is None:
            export_formats = [ExportFormat.CSV, ExportFormat.JSON, ExportFormat.HTML]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # Ensure results is a list
        if not isinstance(results, list):
            results = [results]
        
        # Export in each requested format
        for fmt in export_formats:
            for i, result in enumerate(results):
                suffix = f"_{i}" if len(results) > 1 else ""
                file_name = f"{report_name}{suffix}.{fmt.value}"
                file_path = output_dir / file_name
                
                options = ExportOptions(format=fmt)
                exported_path = self.exporter.export(result, file_path, options)
                generated_files[f"{fmt.value}{suffix}"] = exported_path
        
        # Generate visualizations if requested
        if include_visualizations:
            for i, result in enumerate(results):
                suffix = f"_{i}" if len(results) > 1 else ""
                
                # Create multiple visualization types
                viz_types = [VisualizationType.LINE_PLOT, VisualizationType.BAR_CHART]
                
                for viz_type in viz_types:
                    viz_name = f"{report_name}_viz_{viz_type.value}{suffix}.html"
                    viz_path = output_dir / viz_name
                    
                    options = VisualizationOptions(
                        type=viz_type,
                        title=f"Analytics Results - {viz_type.value.title()}",
                        save_path=str(viz_path)
                    )
                    
                    saved_path = self.visualizer.visualize(result, options)
                    if saved_path:
                        generated_files[f"viz_{viz_type.value}{suffix}"] = saved_path
        
        logger.info(f"Generated analytics report with {len(generated_files)} files in {output_dir}")
        return generated_files 