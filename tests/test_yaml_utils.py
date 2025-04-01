from omegaconf import OmegaConf, DictConfig, ListConfig
from experiment_manager.common.yaml_utils import insert_value, multiply

def test_insert_value():
    # Create a test config with placeholder values
    config = OmegaConf.create({
        "param1": "?",
        "nested": {
            "param2": "?"
        }
    })
    
    # Test inserting value
    insert_value(config, 42)
    assert config.param1 == 42
    assert config.nested.param2 == 42

def test_multiply():
    # Create test configs
    config1 = OmegaConf.create([
        {"model": "A", "size": 1},
        {"model": "B", "size": 2}
    ])
    config2 = OmegaConf.create([
        {"dataset": "X"},
        {"dataset": "Y"}
    ])
    
    # Test multiplication
    result = multiply(config1, config2)
    print("\nResult:", result)  # Print result for debugging
    
    # Verify it's a ListConfig
    assert isinstance(result, ListConfig), f"Expected ListConfig but got {type(result)}"
    
    # Convert to list for easier assertions
    result_list = OmegaConf.to_container(result)
    assert isinstance(result_list, list), "Result should be a list"
    assert len(result_list) == 4, f"Expected 4 combinations but got {len(result_list)}"
    
    # Verify all combinations exist
    expected_combinations = [
        {"model": "A", "size": 1, "dataset": "X"},
        {"model": "A", "size": 1, "dataset": "Y"},
        {"model": "B", "size": 2, "dataset": "X"},
        {"model": "B", "size": 2, "dataset": "Y"}
    ]
    
    for expected in expected_combinations:
        assert any(all(item[k] == v for k, v in expected.items()) 
                  for item in result_list), f"Missing combination: {expected}"
