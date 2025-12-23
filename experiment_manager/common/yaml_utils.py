from omegaconf import DictConfig, OmegaConf, ListConfig
from typing import Union, Any
import copy

def insert_value(config: DictConfig, value):
    """
    Recursively replaces any '?' value in a DictConfig object with the given value.
    
    Args:
        config (DictConfig): The configuration dictionary to process.
        value: The value to replace '?' with.
    """
    for key, val in config.items():
        if isinstance(val, DictConfig):
            insert_value(val, value)  # Recursive call for nested dictionaries
        elif val == "?":
            config[key] = value  # Replace '?' with the given value
            
def multiply(first_config: Union[DictConfig, ListConfig], 
            second_config: Union[DictConfig, ListConfig]) -> ListConfig:
    """
    Creates a cartesian product of configurations by merging each entry from first_config
    with each entry from second_config.
    
    Args:
        first_config (Union[DictConfig, ListConfig]): First configuration list
        second_config (Union[DictConfig, ListConfig]): Second configuration list
        
    Returns:
        ListConfig: List of merged configurations
    """
    res = []
    for first_config_entry in first_config:
        for second_config_entry in second_config:
            merged = OmegaConf.merge(first_config_entry, second_config_entry)
            res.append(merged)
    return OmegaConf.create(res)


def deep_merge_configs(base_config: Union[DictConfig, dict], 
                       trial_config: Union[DictConfig, dict]) -> DictConfig:
    """
    Deep merge configuration that handles array elements at the field level.
    
    This function performs a deep merge where:
    - For dictionaries: merges field by field
    - For arrays: merges individual elements rather than replacing the entire array
    - Preserves base configuration structure while applying trial overrides
    
    Args:
        base_config: The base configuration to merge with
        trial_config: The trial configuration containing overrides
        
    Returns:
        DictConfig: The merged configuration
        
    Example:
        base = {"model": {"layers": [{"size": 100, "activation": "relu"}, {"size": 2, "activation": "softmax"}]}}
        trial = {"model": {"layers": [{"activation": "tanh"}]}}
        result = deep_merge_configs(base, trial)
        # Result: {"model": {"layers": [{"size": 100, "activation": "tanh"}, {"size": 2, "activation": "softmax"}]}}
    """
    # Convert to OmegaConf if needed
    if not isinstance(base_config, (DictConfig, ListConfig)):
        base_config = OmegaConf.create(base_config)
    if not isinstance(trial_config, (DictConfig, ListConfig)):
        trial_config = OmegaConf.create(trial_config)
    
    # Create a deep copy of base config to avoid modifying original
    result = copy.deepcopy(base_config)
    
    # If trial config is None or empty, return base config
    if not trial_config:
        return result
    
    # Handle the merging
    _deep_merge_recursive(result, trial_config)
    
    return result


def _deep_merge_recursive(base: Any, trial: Any) -> None:
    """
    Recursively merge trial configuration into base configuration.
    
    Args:
        base: The base configuration (modified in place)
        trial: The trial configuration to merge
    """
    if not isinstance(trial, (DictConfig, ListConfig, dict, list)):
        return
    
    if isinstance(trial, (ListConfig, list)):
        _merge_arrays(base, trial)
    elif isinstance(trial, (DictConfig, dict)):
        _merge_dictionaries(base, trial)


def _merge_arrays(base: Any, trial: Union[ListConfig, list]) -> None:
    """
    Merge trial array into base array at the element level.
    
    Args:
        base: Base array (modified in place)
        trial: Trial array to merge
    """
    if not isinstance(base, (ListConfig, list)):
        return
    
    # Convert to list for easier manipulation
    base_list = list(base) if hasattr(base, '__iter__') else []
    trial_list = list(trial) if hasattr(trial, '__iter__') else []
    
    # For each element in trial array, merge with corresponding base element
    for i, trial_element in enumerate(trial_list):
        if i < len(base_list):
            # Merge with existing base element
            if isinstance(trial_element, (DictConfig, dict)) and isinstance(base_list[i], (DictConfig, dict)):
                _merge_dictionaries(base_list[i], trial_element)
            else:
                # Replace if not both dictionaries
                base_list[i] = trial_element
        else:
            # Add new element if trial array is longer than base
            base_list.append(trial_element)
    
    # Update the base array
    if isinstance(base, ListConfig):
        # Clear and repopulate ListConfig
        base.clear()
        for item in base_list:
            base.append(item)
    else:
        # For regular lists, replace contents
        base.clear()
        base.extend(base_list)


def _merge_dictionaries(base: Any, trial: Union[DictConfig, dict]) -> None:
    """
    Merge trial dictionary into base dictionary at the field level.
    
    Args:
        base: Base dictionary (modified in place)
        trial: Trial dictionary to merge
    """
    if not isinstance(base, (DictConfig, dict)) or not isinstance(trial, (DictConfig, dict)):
        return
    
    for key, trial_value in trial.items():
        if key in base:
            # Key exists in base, need to merge
            if isinstance(trial_value, (DictConfig, dict)) and isinstance(base[key], (DictConfig, dict)):
                # Both are dictionaries, merge recursively
                _merge_dictionaries(base[key], trial_value)
            elif isinstance(trial_value, (ListConfig, list)) and isinstance(base[key], (ListConfig, list)):
                # Both are arrays, merge arrays
                _merge_arrays(base[key], trial_value)
            else:
                # Replace with trial value
                base[key] = trial_value
        else:
            # Key doesn't exist in base, add it
            base[key] = trial_value