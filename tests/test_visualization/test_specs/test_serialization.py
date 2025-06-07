"""
Tests for Plot Specification Serialization System
"""

import pytest
import json
import tempfile
from pathlib import Path

from experiment_manager.visualization.specs import (
    PlotSpec, PlotType, DataSpec, DataType, DataColumn, DataMapping,
    spec_serializer, SerializationFormat
)
from experiment_manager.visualization.specs.data_spec import DataSource
from experiment_manager.visualization.specs.serialization import (
    SpecSerializer, SerializationError
)


class TestSpecSerializer:
    """Test SpecSerializer functionality."""
    
    def test_spec_serializer_exists(self):
        """Test that spec_serializer is available."""
        assert spec_serializer is not None
        assert isinstance(spec_serializer, SpecSerializer)
    
    def test_serialization_formats_exist(self):
        """Test that serialization formats are available."""
        assert SerializationFormat.JSON
        assert SerializationFormat.YAML
        assert SerializationFormat.PICKLE
        assert SerializationFormat.BINARY


class TestPlotSpecSerialization:
    """Test PlotSpec serialization functionality."""
    
    def test_plot_spec_to_json(self):
        """Test PlotSpec JSON serialization using to_json method."""
        plot_spec = PlotSpec(
            plot_type=PlotType.LINE,
            title="JSON Test",
            description="Testing JSON serialization"
        )
        
        json_str = plot_spec.to_json()
        assert isinstance(json_str, str)
        
        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["plot_type"] == "line"
        assert data["title"] == "JSON Test"
    
    def test_plot_spec_from_json(self):
        """Test PlotSpec JSON deserialization using from_json method."""
        json_data = {
            "plot_type": "scatter",
            "title": "From JSON Test",
            "description": "Test from JSON"
        }
        
        plot_spec = PlotSpec.from_json(json.dumps(json_data))
        assert plot_spec.plot_type == "scatter"
        assert plot_spec.title == "From JSON Test"
        assert plot_spec.description == "Test from JSON"
    
    def test_plot_spec_to_yaml(self):
        """Test PlotSpec YAML serialization using to_yaml method."""
        plot_spec = PlotSpec(
            plot_type=PlotType.BAR,
            title="YAML Test",
            description="Testing YAML serialization"
        )
        
        yaml_str = plot_spec.to_yaml()
        assert isinstance(yaml_str, str)
        assert "plot_type: bar" in yaml_str
        assert "title: YAML Test" in yaml_str
    
    def test_plot_spec_from_yaml(self):
        """Test PlotSpec YAML deserialization using from_yaml method."""
        yaml_data = """
        plot_type: histogram
        title: From YAML Test
        description: Test from YAML
        """
        
        plot_spec = PlotSpec.from_yaml(yaml_data)
        assert plot_spec.plot_type == "histogram"
        assert plot_spec.title == "From YAML Test"
        assert plot_spec.description == "Test from YAML"
    
    def test_plot_spec_serialization_roundtrip(self):
        """Test that PlotSpec survives serialization roundtrip."""
        original = PlotSpec(
            plot_type=PlotType.HEATMAP,
            title="Roundtrip Test",
            description="Testing serialization roundtrip"
        )
        
        # JSON roundtrip
        json_str = original.to_json()
        from_json = PlotSpec.from_json(json_str)
        assert from_json.title == original.title
        assert from_json.plot_type == original.plot_type
        assert from_json.description == original.description
        
        # YAML roundtrip
        yaml_str = original.to_yaml()
        from_yaml = PlotSpec.from_yaml(yaml_str)
        assert from_yaml.title == original.title
        assert from_yaml.plot_type == original.plot_type
        assert from_yaml.description == original.description


class TestDataSpecSerialization:
    """Test DataSpec serialization functionality."""
    
    def test_data_spec_to_json(self):
        """Test DataSpec JSON serialization using to_json method."""
        columns = [DataColumn(name="test", data_type=DataType.NUMERIC)]
        source = DataSource(source_type="file", path="test.csv")
        
        data_spec = DataSpec(
            spec_id="json-test",
            name="JSON Test",
            columns=columns,
            source=source
        )
        
        json_str = data_spec.to_json()
        assert isinstance(json_str, str)
        
        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["spec_id"] == "json-test"
        assert data["name"] == "JSON Test"
        assert len(data["columns"]) == 1
    
    def test_data_spec_from_json(self):
        """Test DataSpec JSON deserialization using from_json method."""
        json_data = {
            "spec_id": "from-json-test",
            "name": "From JSON Test", 
            "columns": [
                {
                    "name": "test_col",
                    "data_type": "numeric",
                    "description": "Test column"
                }
            ],
            "source": {
                "source_type": "memory"
            }
        }
        
        data_spec = DataSpec.from_json(json.dumps(json_data))
        assert data_spec.spec_id == "from-json-test"
        assert data_spec.name == "From JSON Test"
        assert len(data_spec.columns) == 1
        assert data_spec.columns[0].name == "test_col"
    
    def test_data_spec_to_yaml(self):
        """Test DataSpec YAML serialization using to_yaml method."""
        columns = [DataColumn(name="test", data_type=DataType.TEXT)]
        source = DataSource(source_type="api", url="https://example.com")
        
        data_spec = DataSpec(
            spec_id="yaml-test",
            name="YAML Test",
            columns=columns,
            source=source
        )
        
        yaml_str = data_spec.to_yaml()
        assert isinstance(yaml_str, str)
        assert "spec_id: yaml-test" in yaml_str
        assert "name: YAML Test" in yaml_str
    
    def test_data_spec_serialization_roundtrip(self):
        """Test that DataSpec survives serialization roundtrip."""
        columns = [
            DataColumn(name="x", data_type=DataType.NUMERIC),
            DataColumn(name="y", data_type=DataType.CATEGORICAL)
        ]
        source = DataSource(source_type="database", connection_string="test://db")
        
        original = DataSpec(
            spec_id="roundtrip-test",
            name="Roundtrip Test",
            columns=columns,
            source=source
        )
        
        # JSON roundtrip
        json_str = original.to_json()
        from_json = DataSpec.from_json(json_str)
        assert from_json.spec_id == original.spec_id
        assert from_json.name == original.name
        assert len(from_json.columns) == len(original.columns)
        
        # YAML roundtrip
        yaml_str = original.to_yaml()
        from_yaml = DataSpec.from_yaml(yaml_str)
        assert from_yaml.spec_id == original.spec_id
        assert from_yaml.name == original.name
        assert len(from_yaml.columns) == len(original.columns)


class TestFileOperations:
    """Test file save/load operations."""
    
    def test_plot_spec_save_to_file(self):
        """Test saving PlotSpec to file."""
        plot_spec = PlotSpec(
            plot_type=PlotType.PIE,
            title="File Test",
            description="Testing file operations"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            plot_spec.save_to_file(temp_path)
            assert Path(temp_path).exists()
            
            # Verify file content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "File Test" in content
                assert "pie" in content
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_plot_spec_load_from_file(self):
        """Test loading PlotSpec from file using JSON methods."""
        plot_spec = PlotSpec(
            plot_type=PlotType.VIOLIN,
            title="Load Test",
            description="Testing file loading"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save first
            plot_spec.save_to_file(temp_path)
            
            # Load using JSON method
            with open(temp_path, 'r') as f:
                content = f.read()
            loaded_spec = PlotSpec.from_json(content)
            assert loaded_spec.title == plot_spec.title
            assert loaded_spec.plot_type == plot_spec.plot_type
            assert loaded_spec.description == plot_spec.description
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_data_spec_save_to_file(self):
        """Test saving DataSpec to file."""
        columns = [DataColumn(name="file_test", data_type=DataType.BOOLEAN)]
        source = DataSource(source_type="custom", config={"test": "data"})
        
        data_spec = DataSpec(
            spec_id="file-test",
            name="File Test",
            columns=columns,
            source=source
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            data_spec.save_to_file(temp_path)
            assert Path(temp_path).exists()
            
            # Verify file content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "file-test" in content
                assert "File Test" in content
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_data_spec_load_from_file(self):
        """Test loading DataSpec from file using JSON methods."""
        columns = [DataColumn(name="load_test", data_type=DataType.DATETIME)]
        source = DataSource(source_type="analytics", query="SELECT * FROM test")
        
        data_spec = DataSpec(
            spec_id="load-test",
            name="Load Test",
            columns=columns,
            source=source
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save first
            data_spec.save_to_file(temp_path)
            
            # Load using JSON method
            with open(temp_path, 'r') as f:
                content = f.read()
            loaded_spec = DataSpec.from_json(content)
            assert loaded_spec.spec_id == data_spec.spec_id
            assert loaded_spec.name == data_spec.name
            assert len(loaded_spec.columns) == len(data_spec.columns)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestErrorHandling:
    """Test serialization error handling."""
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON."""
        from experiment_manager.visualization.specs.plot_spec import PlotSpecValidationError
        with pytest.raises(PlotSpecValidationError):
            PlotSpec.from_json("invalid json {")
    
    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML."""
        with pytest.raises(Exception):  # YAML parsing error
            PlotSpec.from_yaml("invalid: yaml: {")
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Missing plot_type
        incomplete_json = '{"title": "Test"}'
        
        with pytest.raises(Exception):  # Validation error
            PlotSpec.from_json(incomplete_json)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 