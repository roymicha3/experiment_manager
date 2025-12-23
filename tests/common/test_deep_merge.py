#!/usr/bin/env python3
"""
Comprehensive tests for the deep merge configuration functionality.
"""

import pytest
from omegaconf import OmegaConf, DictConfig
from experiment_manager.common.yaml_utils import deep_merge_configs


class TestDeepMergeConfigs:
    """Test cases for deep_merge_configs function."""
    
    def test_basic_dictionary_merge(self):
        """Test basic dictionary merging."""
        base = {"a": 1, "b": 2}
        trial = {"b": 3, "c": 4}
        result = deep_merge_configs(base, trial)
        
        assert result.a == 1
        assert result.b == 3  # Overridden by trial
        assert result.c == 4  # Added by trial
    
    def test_nested_dictionary_merge(self):
        """Test nested dictionary merging."""
        base = {
            "model": {
                "type": "Network",
                "layers": [{"size": 100, "activation": "relu"}]
            }
        }
        trial = {
            "model": {
                "type": "CustomNetwork"  # Override type
            }
        }
        result = deep_merge_configs(base, trial)
        
        assert result.model.type == "CustomNetwork"
        assert result.model.layers[0].size == 100
        assert result.model.layers[0].activation == "relu"
    
    def test_array_element_merge(self):
        """Test the main use case: merging array elements at field level."""
        base = {
            "model": {
                "layers": [
                    {"size": 100, "activation": "relu"},
                    {"size": 2, "activation": "softmax"}
                ]
            }
        }
        trial = {
            "model": {
                "layers": [
                    {"activation": "tanh"}  # Should only affect first layer
                ]
            }
        }
        result = deep_merge_configs(base, trial)
        
        # First layer should have size from base and activation from trial
        assert result.model.layers[0].size == 100
        assert result.model.layers[0].activation == "tanh"
        
        # Second layer should be unchanged
        assert result.model.layers[1].size == 2
        assert result.model.layers[1].activation == "softmax"
    
    def test_array_with_more_trial_elements(self):
        """Test when trial array has more elements than base."""
        base = {
            "layers": [
                {"size": 100, "activation": "relu"}
            ]
        }
        trial = {
            "layers": [
                {"activation": "tanh"},
                {"size": 50, "activation": "sigmoid"}
            ]
        }
        result = deep_merge_configs(base, trial)
        
        assert len(result.layers) == 2
        assert result.layers[0].size == 100
        assert result.layers[0].activation == "tanh"
        assert result.layers[1].size == 50
        assert result.layers[1].activation == "sigmoid"
    
    def test_array_with_fewer_trial_elements(self):
        """Test when trial array has fewer elements than base."""
        base = {
            "layers": [
                {"size": 100, "activation": "relu"},
                {"size": 2, "activation": "softmax"}
            ]
        }
        trial = {
            "layers": [
                {"activation": "tanh"}
            ]
        }
        result = deep_merge_configs(base, trial)
        
        assert len(result.layers) == 2
        assert result.layers[0].size == 100
        assert result.layers[0].activation == "tanh"
        assert result.layers[1].size == 2
        assert result.layers[1].activation == "softmax"
    
    def test_complex_nested_structure(self):
        """Test complex nested structure with multiple arrays and dictionaries."""
        base = {
            "model": {
                "type": "Network",
                "layers": [
                    {"size": 100, "activation": "relu", "dropout": 0.1},
                    {"size": 50, "activation": "relu", "dropout": 0.2},
                    {"size": 2, "activation": "softmax", "dropout": 0.0}
                ]
            },
            "optimizer": {
                "type": "Adam",
                "lr": 0.001
            }
        }
        trial = {
            "model": {
                "layers": [
                    {"activation": "tanh"},  # Only override activation of first layer
                    {"dropout": 0.5}         # Only override dropout of second layer
                ]
            },
            "optimizer": {
                "lr": 0.01  # Override learning rate
            }
        }
        result = deep_merge_configs(base, trial)
        
        # First layer: size and dropout from base, activation from trial
        assert result.model.layers[0].size == 100
        assert result.model.layers[0].activation == "tanh"
        assert result.model.layers[0].dropout == 0.1
        
        # Second layer: size and activation from base, dropout from trial
        assert result.model.layers[1].size == 50
        assert result.model.layers[1].activation == "relu"
        assert result.model.layers[1].dropout == 0.5
        
        # Third layer: unchanged
        assert result.model.layers[2].size == 2
        assert result.model.layers[2].activation == "softmax"
        assert result.model.layers[2].dropout == 0.0
        
        # Optimizer: type from base, lr from trial
        assert result.optimizer.type == "Adam"
        assert result.optimizer.lr == 0.01
    
    def test_empty_trial_config(self):
        """Test with empty trial configuration."""
        base = {"a": 1, "b": 2}
        trial = {}
        result = deep_merge_configs(base, trial)
        
        assert result.a == 1
        assert result.b == 2
    
    def test_none_trial_config(self):
        """Test with None trial configuration."""
        base = {"a": 1, "b": 2}
        trial = None
        result = deep_merge_configs(base, trial)
        
        assert result.a == 1
        assert result.b == 2
    
    def test_non_dict_trial_config(self):
        """Test with non-dictionary trial configuration."""
        base = {"a": 1, "b": 2}
        trial = "not a dict"
        result = deep_merge_configs(base, trial)
        
        assert result.a == 1
        assert result.b == 2
    
    def test_omega_conf_objects(self):
        """Test with OmegaConf objects."""
        base = OmegaConf.create({"a": 1, "b": 2})
        trial = OmegaConf.create({"b": 3, "c": 4})
        result = deep_merge_configs(base, trial)
        
        assert isinstance(result, DictConfig)
        assert result.a == 1
        assert result.b == 3
        assert result.c == 4
    
    def test_preserves_base_structure(self):
        """Test that base structure is preserved when trial doesn't override everything."""
        base = {
            "model": {
                "type": "Network",
                "layers": [
                    {"size": 100, "activation": "relu", "dropout": 0.1},
                    {"size": 2, "activation": "softmax", "dropout": 0.0}
                ],
                "regularization": {
                    "l1": 0.01,
                    "l2": 0.001
                }
            }
        }
        trial = {
            "model": {
                "layers": [
                    {"activation": "tanh"}
                ]
            }
        }
        result = deep_merge_configs(base, trial)
        
        # Model type should be preserved
        assert result.model.type == "Network"
        
        # Regularization should be preserved
        assert result.model.regularization.l1 == 0.01
        assert result.model.regularization.l2 == 0.001
        
        # Layers should be merged correctly
        assert result.model.layers[0].size == 100
        assert result.model.layers[0].activation == "tanh"
        assert result.model.layers[0].dropout == 0.1
        assert result.model.layers[1].size == 2
        assert result.model.layers[1].activation == "softmax"
        assert result.model.layers[1].dropout == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
