from omegaconf import DictConfig, OmegaConf, ListConfig
from typing import Union

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