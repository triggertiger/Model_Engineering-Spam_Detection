"""
utils.py

Utility functions for reading and updating the model configuration file (YAML).
These functions are used to track the latest registered model and its version.
"""

import yaml
from pathlib import Path

def read_model_config(file = "model_config.yml"):
    """
    Read the model configuration from a YAML file.

    Args:
        file (str): Name of the YAML config file. Defaults to "model_config.yml".

    Returns:
        dict: Dictionary containing the model configuration.
    """
    config_path = Path(__file__).resolve().parents[1] / file

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config['model']

def update_model_config(name, version, file = "model_config.yml"):
    """
    Update the model_config.yml file with the latest registered model information.

    Args:
        name (str): Name of the registered model.
        version (int): Version number of the registered model.
        file (str): Name of the YAML config file. Defaults to "model_config.yml".

    Effects:
        Updates the YAML file with new model name and last_registered_version.
        Prints a confirmation message.
    """
    
    config_path = Path(__file__).resolve().parents[1] / file

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["model"]["name"] = name
    config["model"]["last_registered_version"] = version
    #config["model"]["stage"] = stage

    # Write back safely
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"✅ Updated config.yaml with model '{name}' (v{version}")#, stage={stage})")




