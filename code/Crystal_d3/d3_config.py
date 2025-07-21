#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D3 Configuration Management Module
----------------------------------
This module provides JSON-based configuration management for CRYSTAL D3 calculations.
It allows saving, loading, and applying configuration settings for various D3 calculation types.

Features:
    - Save D3 calculation settings to JSON files
    - Load settings from JSON files for batch processing
    - Support for all D3 calculation types (BAND, DOSS, ECH3, POT3, etc.)
    - Validate configuration completeness and compatibility
    - Default configurations for common calculation types

Usage:
    # Save configuration to file
    save_d3_config(config_dict, "my_band_config.json")
    
    # Load configuration from file
    config = load_d3_config("my_band_config.json")
    
    # Apply configuration to D3 generation
    d3_generator = CRYSTALOptToD3(...)
    d3_generator.apply_config(config)
"""

import json
import os
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path


def clean_config_for_saving(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean configuration for saving by preserving 'auto' settings.
    
    Removes calculated values that should be recalculated for each material,
    keeping only the settings that indicate 'auto' behavior.
    
    Args:
        config: Original configuration dictionary
        
    Returns:
        dict: Cleaned configuration suitable for saving
    """
    import copy
    cleaned = copy.deepcopy(config)
    
    calc_type = cleaned.get("calculation_type", "")
    
    if calc_type == "BAND":
        # For BAND calculations, preserve auto indicators
        if cleaned.get("auto_path", False):
            # Remove calculated path values, keep only the auto indicators
            cleaned["path"] = "auto"
            cleaned["labels"] = "auto"
            if "path_labels" in cleaned:
                del cleaned["path_labels"]
            if "segments" in cleaned:
                del cleaned["segments"]
            # If path was auto, shrink should be auto too (unless it's labels mode)
            if cleaned.get("path_method") == "coordinates":
                cleaned["shrink"] = "auto"
        
        # Keep shrink as "auto" if it was auto-detected
        if cleaned.get("shrink") == "auto" or cleaned.get("auto_shrink", False):
            cleaned["shrink"] = "auto"
        
        # Keep bands as "auto" if it was auto-detected
        if cleaned.get("bands") == "auto" or cleaned.get("auto_bands", False):
            cleaned["bands"] = "auto"
            if "band_range" in cleaned and cleaned.get("bands") == "auto":
                del cleaned["band_range"]
            # Remove specific band numbers when using auto
            if "first_band" in cleaned:
                del cleaned["first_band"]
            if "last_band" in cleaned:
                del cleaned["last_band"]
    
    elif calc_type == "DOSS":
        # For DOSS, if projections were auto-generated, don't save them
        if cleaned.get("project_orbital_types", False):
            # Remove the specific calculated projections
            if "projections" in cleaned:
                cleaned["projections"] = []
            # Set projection_type based on the configuration
            if cleaned.get("element_only", False):
                cleaned["projection_type"] = 2
            elif cleaned.get("include_element_totals", True):
                cleaned["projection_type"] = 3
            else:
                cleaned["projection_type"] = 4
        elif cleaned.get("project_all_atoms", False):
            cleaned["projection_type"] = 5
            # Remove specific atom projections if using all atoms
            if "project_atoms" in cleaned:
                del cleaned["project_atoms"]
        elif "project_atoms" in cleaned:
            cleaned["projection_type"] = 5
        elif "npro" in cleaned and cleaned["npro"] == 0:
            cleaned["projection_type"] = 1
        
        # For manual projections (type 6), keep the projections
        if cleaned.get("projection_type") == 6:
            # Keep manual projections as-is
            pass
    
    elif calc_type == "TRANSPORT":
        # For TRANSPORT, if using Fermi reference, save relative range
        if cleaned.get("mu_reference") == "fermi":
            # Keep the relative range, remove absolute values
            if "mu_relative_range" in cleaned:
                mu_min_rel, mu_max_rel = cleaned["mu_relative_range"]
                mu_step = cleaned.get("mu_range", (0, 0, 0.01))[2]
                cleaned["mu_range"] = (mu_min_rel, mu_max_rel, mu_step)
                cleaned["mu_range_type"] = "auto_fermi"
    
    # Remove any internal processing flags
    keys_to_remove = ["auto_shrink", "auto_bands", "auto_labels", "mu_relative_range"]
    for key in keys_to_remove:
        if key in cleaned:
            del cleaned[key]
    
    return cleaned


def save_d3_config(config: Dict[str, Any], filename: str, overwrite: bool = True) -> bool:
    """
    Save D3 configuration to JSON file.
    
    Args:
        config: Dictionary containing D3 configuration settings
        filename: Path to save the JSON file
        overwrite: Whether to overwrite existing file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if file exists and overwrite is False
        if os.path.exists(filename) and not overwrite:
            response = input(f"{filename} exists. Overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                print("Save cancelled.")
                return False
        
        # Clean configuration before saving
        cleaned_config = clean_config_for_saving(config)
        
        # Add metadata
        config_with_metadata = {
            "version": "1.0",
            "type": "d3_configuration",
            "calculation_type": cleaned_config.get("calculation_type", "unknown"),
            "configuration": cleaned_config
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(config_with_metadata, f, indent=2)
        
        print(f"\nD3 configuration saved to: {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving D3 configuration: {e}")
        return False


def load_d3_config(filename: str) -> Optional[Dict[str, Any]]:
    """
    Load D3 configuration from JSON file.
    
    Args:
        filename: Path to the JSON file to load
        
    Returns:
        dict: Configuration dictionary or None if error
    """
    try:
        if not os.path.exists(filename):
            print(f"Error: Configuration file {filename} not found.")
            return None
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Check if it's a D3 configuration file
        if data.get("type") != "d3_configuration":
            print(f"Error: {filename} is not a valid D3 configuration file.")
            return None
        
        config = data.get("configuration", {})
        print(f"\nLoaded D3 configuration from: {filename}")
        print(f"Calculation type: {config.get('calculation_type', 'unknown')}")
        
        return config
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filename}: {e}")
        return None
    except Exception as e:
        print(f"Error loading D3 configuration: {e}")
        return None


def validate_d3_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate D3 configuration for completeness and compatibility.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check for calculation type
    if "calculation_type" not in config:
        errors.append("Missing calculation_type")
    
    calc_type = config.get("calculation_type", "")
    
    # Validate based on calculation type
    if calc_type == "BAND":
        required = ["n_points", "bands"]
        for key in required:
            if key not in config:
                errors.append(f"Missing required field for BAND: {key}")
    
    elif calc_type == "DOSS":
        required = ["projection_type", "energy_range", "n_points"]
        for key in required:
            if key not in config:
                errors.append(f"Missing required field for DOSS: {key}")
    
    elif calc_type in ["CHARGE", "POTENTIAL", "CHARGE+POTENTIAL"]:
        if calc_type == "CHARGE+POTENTIAL":
            if "charge_config" not in config or "potential_config" not in config:
                errors.append("Missing charge_config or potential_config for CHARGE+POTENTIAL")
        else:
            required = ["type", "n_points"]
            for key in required:
                if key not in config:
                    errors.append(f"Missing required field for {calc_type}: {key}")
    
    elif calc_type == "WANNIER":
        required = ["wannier_functions", "plot_bands"]
        for key in required:
            if key not in config:
                errors.append(f"Missing required field for WANNIER: {key}")
    
    elif calc_type == "DENSITY_MATRIX":
        if "shells" not in config:
            errors.append("Missing shells specification for DENSITY_MATRIX")
    
    elif calc_type == "RAMAN":
        required = ["raman_modes", "temperature", "wavelength"]
        for key in required:
            if key not in config:
                errors.append(f"Missing required field for RAMAN: {key}")
    
    elif calc_type == "DFT_EXCHANGE":
        if "exchange_options" not in config:
            errors.append("Missing exchange_options for DFT_EXCHANGE")
    
    elif calc_type == "TRANSPORT":
        required = ["transport_type", "temperature_range"]
        for key in required:
            if key not in config:
                errors.append(f"Missing required field for TRANSPORT: {key}")
    
    else:
        errors.append(f"Unknown calculation type: {calc_type}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_default_d3_config(calc_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific D3 calculation type.
    
    Args:
        calc_type: Type of calculation (BAND, DOSS, etc.)
        
    Returns:
        dict: Default configuration for the calculation type
    """
    defaults = {
        "BAND": {
            "calculation_type": "BAND",
            "n_points": 100,
            "bands": "auto",
            "path": "auto",
            "labels": "auto",
            "shrink": "auto"
        },
        "DOSS": {
            "calculation_type": "DOSS",
            "projection_type": 1,  # Total DOS
            "energy_range": "bands",
            "band_range": [0, 999],
            "n_points": 1000,
            "print_integrated": True,
            "output_format": 0,
            "projections": []
        },
        "CHARGE": {
            "calculation_type": "CHARGE",
            "type": "ECH3",
            "n_points": 100,
            "scale": 3.0,
            "use_range": False
        },
        "POTENTIAL": {
            "calculation_type": "POTENTIAL",
            "type": "POT3",
            "n_points": 100,
            "scale": 3.0,
            "use_range": False
        },
        "CHARGE+POTENTIAL": {
            "calculation_type": "CHARGE+POTENTIAL",
            "charge_config": {
                "type": "ECH3",
                "n_points": 100,
                "scale": 3.0,
                "use_range": False
            },
            "potential_config": {
                "type": "POT3",
                "n_points": 100,
                "scale": 3.0,
                "use_range": False
            }
        },
        "WANNIER": {
            "calculation_type": "WANNIER",
            "wannier_functions": "all",
            "plot_bands": True,
            "localization_method": "boys",
            "max_iterations": 100
        },
        "DENSITY_MATRIX": {
            "calculation_type": "DENSITY_MATRIX",
            "shells": "all",
            "print_overlap": True,
            "print_kinetic": False
        },
        "RAMAN": {
            "calculation_type": "RAMAN",
            "raman_modes": "all",
            "temperature": 298.15,
            "wavelength": 532.0,
            "polarization": "unpolarized"
        },
        "DFT_EXCHANGE": {
            "calculation_type": "DFT_EXCHANGE",
            "exchange_options": {
                "exact_exchange": 0.25,
                "range_separation": False
            }
        },
        "TRANSPORT": {
            "calculation_type": "TRANSPORT",
            "transport_type": "conductivity",
            "temperature_range": [100, 1000],
            "n_temperatures": 10,
            "carrier_concentration": "auto"
        }
    }
    
    return defaults.get(calc_type, {"calculation_type": calc_type})


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base settings
        
    Returns:
        dict: Merged configuration
    """
    import copy
    
    result = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def print_d3_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the D3 configuration.
    
    Args:
        config: Configuration dictionary to summarize
    """
    print("\n" + "="*60)
    print("D3 CONFIGURATION SUMMARY")
    print("="*60)
    
    calc_type = config.get("calculation_type", "Unknown")
    print(f"Calculation Type: {calc_type}")
    print("-"*60)
    
    if calc_type == "BAND":
        print(f"Number of points: {config.get('n_points', 'N/A')}")
        print(f"Band range: {config.get('bands', 'N/A')}")
        print(f"Path type: {config.get('path', 'N/A')}")
        
    elif calc_type == "DOSS":
        proj_type = config.get("projection_type", 0)
        proj_names = {
            1: "Total DOS only",
            2: "Projected on AO shells", 
            3: "Element and orbital projections",
            4: "Orbital projections only (no element totals)",
            5: "Projected on atoms",
            6: "Manual orbital projections"
        }
        print(f"Projection type: {proj_names.get(proj_type, 'Unknown')}")
        print(f"Energy range: {config.get('energy_range', 'N/A')}")
        print(f"Number of points: {config.get('n_points', 'N/A')}")
        
        if proj_type == 6 and "projections" in config:
            print("Manual projections:")
            for proj in config["projections"]:
                print(f"  - {proj}")
    
    elif calc_type in ["CHARGE", "POTENTIAL"]:
        print(f"Type: {config.get('type', 'N/A')}")
        print(f"Number of points: {config.get('n_points', 'N/A')}")
        print(f"Scale factor: {config.get('scale', 'N/A')}")
        
    elif calc_type == "CHARGE+POTENTIAL":
        if "charge_config" in config:
            print("Charge density:")
            print(f"  Type: {config['charge_config'].get('type', 'N/A')}")
            print(f"  Points: {config['charge_config'].get('n_points', 'N/A')}")
        if "potential_config" in config:
            print("Electrostatic potential:")
            print(f"  Type: {config['potential_config'].get('type', 'N/A')}")
            print(f"  Points: {config['potential_config'].get('n_points', 'N/A')}")
    
    elif calc_type == "WANNIER":
        print(f"Functions: {config.get('wannier_functions', 'N/A')}")
        print(f"Plot bands: {config.get('plot_bands', 'N/A')}")
        print(f"Localization: {config.get('localization_method', 'N/A')}")
        
    elif calc_type == "RAMAN":
        print(f"Modes: {config.get('raman_modes', 'N/A')}")
        print(f"Temperature: {config.get('temperature', 'N/A')} K")
        print(f"Wavelength: {config.get('wavelength', 'N/A')} nm")
    
    print("="*60)


def create_d3_config_from_interactive(calc_type: str, input_file: str = None) -> Dict[str, Any]:
    """
    Create D3 configuration by calling the interactive configuration.
    
    Args:
        calc_type: Type of D3 calculation
        input_file: Path to input file (for context)
        
    Returns:
        dict: Configuration dictionary
    """
    from d3_interactive import configure_d3_calculation
    
    # Get configuration through interactive prompts
    config = configure_d3_calculation(calc_type, input_file)
    
    # Ensure calculation_type is set
    if "calculation_type" not in config:
        config["calculation_type"] = calc_type
    
    return config


def save_d3_options_prompt(config: Dict[str, Any], default_filename: str = None, 
                         skip_prompt: bool = False) -> bool:
    """
    Interactive prompt to save D3 configuration options.
    
    Args:
        config: Configuration dictionary to save
        default_filename: Default filename to suggest
        skip_prompt: If True, skip the initial save prompt and proceed directly
        
    Returns:
        bool: True if saved successfully
    """
    from d12_constants import yes_no_prompt
    
    if skip_prompt or yes_no_prompt("\nSave these D3 options to file for future use?", "yes"):
        if default_filename is None:
            calc_type = config.get("calculation_type", "d3").lower()
            default_filename = f"d3_{calc_type}_config.json"
        
        filename = input(f"Enter filename (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename
        
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        return save_d3_config(config, filename)
    
    return False


def list_available_d3_configs(directory: str = ".") -> List[str]:
    """
    List available D3 configuration files in a directory.
    
    Args:
        directory: Directory to search for config files
        
    Returns:
        list: List of available configuration files
    """
    config_files = []
    
    try:
        for file in os.listdir(directory):
            if file.endswith('.json') and 'd3' in file.lower():
                filepath = os.path.join(directory, file)
                try:
                    # Try to load and validate it's a D3 config
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if data.get("type") == "d3_configuration":
                        config_files.append(file)
                except:
                    pass
    except Exception as e:
        print(f"Error listing configuration files: {e}")
    
    return sorted(config_files)


def select_d3_config_file(directory: str = ".") -> Optional[str]:
    """
    Interactive selection of D3 configuration file.
    
    Args:
        directory: Directory to search for config files
        
    Returns:
        str: Selected filename or None
    """
    configs = list_available_d3_configs(directory)
    
    if not configs:
        print("No D3 configuration files found in current directory.")
        return None
    
    print("\nAvailable D3 configuration files:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config}")
    
    try:
        choice = int(input("\nSelect configuration file (number): ").strip())
        if 1 <= choice <= len(configs):
            return configs[choice - 1]
        else:
            print("Invalid selection.")
            return None
    except (ValueError, KeyboardInterrupt):
        return None