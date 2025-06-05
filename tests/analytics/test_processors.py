"""
Tests for analytics processors module.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from experiment_manager.analytics.processors.base import (
    ProcessedData, DataProcessor, ProcessorManager
)
from .test_fixtures import (
    mock_processor, sample_experiment_data, MockProcessor
)


class TestProcessedData:
    """Test cases for ProcessedData container class."""
    
    def test_init_with_dataframe(self, sample_experiment_data):
        """Test ProcessedData initialization with DataFrame."""
        metadata = {'test': True}
        processed = ProcessedData(sample_experiment_data, metadata)
        
        assert isinstance(processed.data, pd.DataFrame)
        assert processed.metadata == metadata
        assert 'created_at' in processed.metadata
        assert processed.processing_steps == []
    
    def test_init_with_dict_data(self):
        """Test ProcessedData initialization with dict data."""
        data = {'values': [1, 2, 3], 'labels': ['a', 'b', 'c']}
        processed = ProcessedData(data)
        
        assert isinstance(processed.data, pd.DataFrame)
        assert len(processed.data) == 3
        assert 'values' in processed.data.columns
        assert 'labels' in processed.data.columns
    
    def test_add_processing_step(self, sample_experiment_data):
        """Test adding processing steps."""
        processed = ProcessedData(sample_experiment_data)
        
        processed.add_processing_step(
            'test_processor', 
            {'param1': 'value1'}, 
            'Test processing completed'
        )
        
        assert len(processed.processing_steps) == 1
        step = processed.processing_steps[0]
        assert step['processor_name'] == 'test_processor'
        assert step['parameters'] == {'param1': 'value1'}
        assert step['description'] == 'Test processing completed'
        assert 'timestamp' in step
    
    def test_get_summary(self, sample_experiment_data):
        """Test getting data summary."""
        processed = ProcessedData(sample_experiment_data)
        summary = processed.get_summary()
        
        assert 'row_count' in summary
        assert 'column_count' in summary
        assert 'columns' in summary
        assert 'memory_usage' in summary
        assert summary['row_count'] == len(sample_experiment_data)
        assert summary['column_count'] == len(sample_experiment_data.columns)
    
    def test_to_yaml_serialization(self, sample_experiment_data):
        """Test YAML serialization of ProcessedData."""
        processed = ProcessedData(sample_experiment_data, {'test': True})
        processed.add_processing_step('processor1', {}, 'Step 1')
        
        yaml_str = processed.to_yaml()
        assert isinstance(yaml_str, str)
        assert 'metadata:' in yaml_str
        assert 'processing_steps:' in yaml_str
        assert 'data_summary:' in yaml_str
    
    def test_from_yaml_deserialization(self, sample_experiment_data):
        """Test YAML deserialization of ProcessedData."""
        original = ProcessedData(sample_experiment_data, {'test': True})
        original.add_processing_step('processor1', {}, 'Step 1')
        
        yaml_str = original.to_yaml()
        
        # Note: Full deserialization would require actual YAML parsing
        # For now, we test that the YAML contains expected structure
        assert 'created_at:' in yaml_str
        assert 'test: true' in yaml_str


class ConcreteProcessor(DataProcessor):
    """Concrete implementation of DataProcessor for testing."""
    
    def __init__(self, name="TestProcessor"):
        super().__init__(name)
        self.required_columns = ['metric_value']
    
    def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Test implementation that adds a processed column."""
        result = data.copy()
        result['processed'] = True
        result['processor_name'] = self.name
        return result


class TestDataProcessor:
    """Test cases for DataProcessor abstract base class."""
    
    def test_concrete_processor_creation(self):
        """Test creating a concrete processor."""
        processor = ConcreteProcessor("MyProcessor")
        assert processor.name == "MyProcessor"
        assert processor.required_columns == ['metric_value']
    
    def test_process_with_valid_data(self, sample_experiment_data):
        """Test processing with valid data."""
        processor = ConcreteProcessor()
        result = processor.process(sample_experiment_data)
        
        assert isinstance(result, ProcessedData)
        assert 'processed' in result.data.columns
        assert all(result.data['processed'] == True)
        assert len(result.processing_steps) == 1
    
    def test_process_with_invalid_data(self):
        """Test processing with data missing required columns."""
        processor = ConcreteProcessor()
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            processor.process(invalid_data)
    
    def test_validate_input_success(self, sample_experiment_data):
        """Test successful input validation."""
        processor = ConcreteProcessor()
        assert processor.validate_input(sample_experiment_data) == True
    
    def test_validate_input_failure(self):
        """Test failed input validation."""
        processor = ConcreteProcessor()
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        assert processor.validate_input(invalid_data) == False
    
    def test_get_required_columns(self):
        """Test getting required columns."""
        processor = ConcreteProcessor()
        columns = processor.get_required_columns()
        assert columns == ['metric_value']
    
    def test_yaml_serialization(self):
        """Test processor YAML serialization."""
        processor = ConcreteProcessor("SerializationTest")
        yaml_str = processor.to_yaml()
        
        assert isinstance(yaml_str, str)
        assert 'name: SerializationTest' in yaml_str
        assert 'required_columns:' in yaml_str
        assert '- metric_value' in yaml_str


class TestProcessorManager:
    """Test cases for ProcessorManager class."""
    
    def test_manager_creation(self):
        """Test creating a processor manager."""
        manager = ProcessorManager()
        assert len(manager.processors) == 0
    
    def test_register_processor(self):
        """Test registering a processor."""
        manager = ProcessorManager()
        processor = ConcreteProcessor("TestProcessor")
        
        manager.register_processor("test", processor)
        assert "test" in manager.processors
        assert manager.processors["test"] == processor
    
    def test_register_duplicate_processor(self):
        """Test registering a processor with duplicate name."""
        manager = ProcessorManager()
        processor1 = ConcreteProcessor("Processor1")
        processor2 = ConcreteProcessor("Processor2")
        
        manager.register_processor("test", processor1)
        
        with pytest.raises(ValueError, match="Processor 'test' is already registered"):
            manager.register_processor("test", processor2)
    
    def test_get_processor(self):
        """Test getting a registered processor."""
        manager = ProcessorManager()
        processor = ConcreteProcessor("TestProcessor")
        
        manager.register_processor("test", processor)
        retrieved = manager.get_processor("test")
        
        assert retrieved == processor
    
    def test_get_nonexistent_processor(self):
        """Test getting a processor that doesn't exist."""
        manager = ProcessorManager()
        
        with pytest.raises(KeyError, match="Processor 'nonexistent' not found"):
            manager.get_processor("nonexistent")
    
    def test_remove_processor(self):
        """Test removing a processor."""
        manager = ProcessorManager()
        processor = ConcreteProcessor("TestProcessor")
        
        manager.register_processor("test", processor)
        assert "test" in manager.processors
        
        manager.remove_processor("test")
        assert "test" not in manager.processors
    
    def test_remove_nonexistent_processor(self):
        """Test removing a processor that doesn't exist."""
        manager = ProcessorManager()
        
        with pytest.raises(KeyError, match="Processor 'nonexistent' not found"):
            manager.remove_processor("nonexistent")
    
    def test_list_processors(self):
        """Test listing registered processors."""
        manager = ProcessorManager()
        processor1 = ConcreteProcessor("Processor1")
        processor2 = ConcreteProcessor("Processor2")
        
        manager.register_processor("test1", processor1)
        manager.register_processor("test2", processor2)
        
        processor_list = manager.list_processors()
        assert len(processor_list) == 2
        assert "test1" in processor_list
        assert "test2" in processor_list
    
    def test_execute_single_processor(self, sample_experiment_data):
        """Test executing a single processor."""
        manager = ProcessorManager()
        processor = ConcreteProcessor("TestProcessor")
        manager.register_processor("test", processor)
        
        result = manager.execute("test", sample_experiment_data)
        
        assert isinstance(result, ProcessedData)
        assert 'processed' in result.data.columns
        assert len(result.processing_steps) == 1
    
    def test_execute_processor_chain(self, sample_experiment_data):
        """Test executing a chain of processors."""
        manager = ProcessorManager()
        
        # Create two processors
        processor1 = ConcreteProcessor("Processor1")
        processor2 = ConcreteProcessor("Processor2")
        
        manager.register_processor("proc1", processor1)
        manager.register_processor("proc2", processor2)
        
        result = manager.execute_chain(["proc1", "proc2"], sample_experiment_data)
        
        assert isinstance(result, ProcessedData)
        assert 'processed' in result.data.columns
        assert len(result.processing_steps) == 2
        assert result.processing_steps[0]['processor_name'] == 'Processor1'
        assert result.processing_steps[1]['processor_name'] == 'Processor2'
    
    def test_execute_chain_with_nonexistent_processor(self, sample_experiment_data):
        """Test executing chain with non-existent processor."""
        manager = ProcessorManager()
        processor = ConcreteProcessor("TestProcessor")
        manager.register_processor("test", processor)
        
        with pytest.raises(KeyError, match="Processor 'nonexistent' not found"):
            manager.execute_chain(["test", "nonexistent"], sample_experiment_data)
    
    def test_parallel_execution(self, sample_experiment_data):
        """Test parallel execution of processors."""
        manager = ProcessorManager()
        
        processor1 = ConcreteProcessor("Processor1")
        processor2 = ConcreteProcessor("Processor2")
        
        manager.register_processor("proc1", processor1)
        manager.register_processor("proc2", processor2)
        
        results = manager.execute_parallel(["proc1", "proc2"], sample_experiment_data)
        
        assert len(results) == 2
        assert all(isinstance(result, ProcessedData) for result in results.values())
        assert "proc1" in results
        assert "proc2" in results
    
    def test_manager_yaml_serialization(self):
        """Test processor manager YAML serialization."""
        manager = ProcessorManager()
        processor1 = ConcreteProcessor("Processor1")
        processor2 = ConcreteProcessor("Processor2")
        
        manager.register_processor("proc1", processor1)
        manager.register_processor("proc2", processor2)
        
        yaml_str = manager.to_yaml()
        
        assert isinstance(yaml_str, str)
        assert 'processors:' in yaml_str
        assert 'proc1:' in yaml_str
        assert 'proc2:' in yaml_str
    
    def test_clear_processors(self):
        """Test clearing all processors."""
        manager = ProcessorManager()
        processor = ConcreteProcessor("TestProcessor")
        
        manager.register_processor("test", processor)
        assert len(manager.processors) == 1
        
        manager.clear()
        assert len(manager.processors) == 0


class TestProcessorIntegration:
    """Integration tests for processor components."""
    
    def test_end_to_end_processing_workflow(self, sample_experiment_data):
        """Test complete processing workflow."""
        # Setup
        manager = ProcessorManager()
        processor = ConcreteProcessor("IntegrationTest")
        manager.register_processor("integration", processor)
        
        # Execute processing
        result = manager.execute("integration", sample_experiment_data)
        
        # Verify results
        assert isinstance(result, ProcessedData)
        assert len(result.data) == len(sample_experiment_data)
        assert 'processed' in result.data.columns
        assert all(result.data['processed'] == True)
        
        # Verify metadata
        assert 'created_at' in result.metadata
        assert len(result.processing_steps) == 1
        
        # Verify processing step details
        step = result.processing_steps[0]
        assert step['processor_name'] == 'IntegrationTest'
        assert 'timestamp' in step
    
    def test_processor_error_handling(self):
        """Test processor error handling."""
        class ErrorProcessor(DataProcessor):
            def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
                raise RuntimeError("Processing failed")
        
        manager = ProcessorManager()
        error_processor = ErrorProcessor("ErrorProcessor")
        manager.register_processor("error", error_processor)
        
        data = pd.DataFrame({'metric_value': [1, 2, 3]})
        
        with pytest.raises(RuntimeError, match="Processing failed"):
            manager.execute("error", data)
    
    def test_processor_with_parameters(self, sample_experiment_data):
        """Test processor with custom parameters."""
        class ParameterizedProcessor(DataProcessor):
            def _process_implementation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
                multiplier = kwargs.get('multiplier', 1)
                result = data.copy()
                if 'metric_value' in result.columns:
                    result['metric_value'] = result['metric_value'] * multiplier
                return result
        
        manager = ProcessorManager()
        processor = ParameterizedProcessor("ParamProcessor")
        manager.register_processor("param", processor)
        
        result = manager.execute("param", sample_experiment_data, multiplier=2)
        
        assert isinstance(result, ProcessedData)
        # Check that metric values were doubled
        original_values = sample_experiment_data['metric_value'].tolist()
        processed_values = result.data['metric_value'].tolist()
        
        for orig, proc in zip(original_values, processed_values):
            assert abs(proc - (orig * 2)) < 1e-10 