#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D12 Configuration Management Module
============================================================
This module provides JSON-based configuration management for CRYSTAL D12 calculations.
It allows saving, loading, and applying configuration settings for various D12 calculation types.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

Features:
    - Save D12 calculation settings to JSON files
    - Load settings from JSON files for batch processing
    - Support for all calculation types (SP, OPT, FREQ)
    - Validate configuration completeness and compatibility
    - Default configurations for common calculation types

Usage:
    # Save configuration to file
    save_d12_config(config_dict, "my_dft_opt_config.json")
    
    # Load configuration from file
    config = load_d12_config("my_dft_opt_config.json")
    
    # Get default configuration for a calculation type
    config = get_default_d12_config("standard_dft_opt")
"""

import json
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


def get_default_d12_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get all default D12 configuration templates.
    
    Returns:
        dict: Dictionary of configuration templates
    """
    return {
        "standard_dft_opt": {
            "name": "standard_dft_opt",
            "description": "Standard DFT geometry optimization with B3LYP-D3",
            "calculation_type": "OPT",
            "optimization_type": "FULLOPTG",
            "method": "DFT",
            "functional": "B3LYP",
            "dispersion": True,
            "basis_set": "POB-TZVP-REV2",
            "basis_set_type": "INTERNAL",
            "dft_grid": "XLGRID",
            "tolerances": {
                "TOLINTEG": "7 7 7 7 14",
                "TOLDEE": 7
            },
            "optimization_settings": {
                "toldeg": 0.0003,
                "toldex": 0.0012,
                "toldee": 7,
                "maxcycle": 800
            },
            "scf_settings": {
                "method": "DIIS",
                "maxcycle": 800,
                "fmixing": 30
            },
            "spin_polarized": False
        },
        "high_accuracy_sp": {
            "name": "high_accuracy_sp",
            "description": "High-accuracy single point with HSE06-D3",
            "calculation_type": "SP",
            "method": "DFT",
            "functional": "HSE06",
            "dispersion": True,
            "basis_set": "POB-TZVP-REV2",
            "basis_set_type": "INTERNAL",
            "dft_grid": "XXLGRID",
            "tolerances": {
                "TOLINTEG": "9 9 9 11 38",
                "TOLDEE": 11
            },
            "scf_settings": {
                "method": "DIIS",
                "maxcycle": 1000,
                "fmixing": 20
            },
            "spin_polarized": False
        },
        "3c_composite": {
            "name": "3c_composite",
            "description": "Fast composite method with PBEH3C",
            "calculation_type": "OPT",
            "optimization_type": "FULLOPTG",
            "method": "DFT",
            "functional": "PBEH3C",
            "is_3c_method": True,
            "dispersion": False,
            "basis_set": "MINIX",
            "basis_set_type": "INTERNAL",
            "dft_grid": None,
            "tolerances": {
                "TOLINTEG": "7 7 7 7 14",
                "TOLDEE": 7
            },
            "optimization_settings": {
                "toldeg": 0.0003,
                "toldex": 0.0012,
                "toldee": 7,
                "maxcycle": 800
            }
        },
        "freq_analysis": {
            "name": "freq_analysis",
            "description": "Vibrational frequency analysis with tight convergence",
            "calculation_type": "FREQ",
            "method": "DFT",
            "functional": "B3LYP",
            "dispersion": True,
            "basis_set": "POB-TZVP-REV2",
            "basis_set_type": "INTERNAL",
            "dft_grid": "XLGRID",
            "tolerances": {
                "TOLINTEG": "9 9 9 11 38",
                "TOLDEE": 11
            },
            "freq_settings": {
                "freq_mode": "FREQCALC",
                "intensities": True,
                "raman": False,
                "temprange": [20, 0, 400],
                "template": "raman_minimal"
            }
        },
        "surface_slab": {
            "name": "surface_slab",
            "description": "2D slab calculation for surfaces",
            "dimensionality": "SLAB",
            "calculation_type": "OPT",
            "optimization_type": "ATOMONLY",
            "method": "DFT",
            "functional": "PBE",
            "dispersion": True,
            "basis_set": "POB-DZVP-REV2",
            "basis_set_type": "INTERNAL",
            "dft_grid": "LGRID",
            "use_smearing": True,
            "smearing_width": 0.01
        },
        "metallic_system": {
            "name": "metallic_system",
            "description": "Settings optimized for metallic systems",
            "calculation_type": "OPT",
            "method": "DFT",
            "functional": "PBE",
            "dispersion": False,
            "basis_set": "POB-TZVP-REV2",
            "basis_set_type": "INTERNAL",
            "dft_grid": "XLGRID",
            "spin_polarized": True,
            "use_smearing": True,
            "smearing_width": 0.02,
            "scf_settings": {
                "method": "DIIS",
                "maxcycle": 1200,
                "fmixing": 15
            }
        },
        "quick_screen": {
            "name": "quick_screen",
            "description": "Fast screening calculations with minimal basis",
            "calculation_type": "SP",
            "method": "HF",
            "functional": None,
            "dispersion": False,
            "basis_set": "STO-3G",
            "basis_set_type": "INTERNAL",
            "dft_grid": None,
            "tolerances": {
                "TOLINTEG": "6 6 6 6 12",
                "TOLDEE": 6
            }
        }
    }


def save_d12_config(config: Dict[str, Any], filename: str, 
                    config_dir: Optional[str] = None) -> str:
    """
    Save D12 configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        filename: Output filename (will add .json if not present)
        config_dir: Directory to save config (default: ./d12_configs)
        
    Returns:
        str: Full path to saved configuration file
    """
    # Ensure filename has .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Set default directory
    if config_dir is None:
        config_dir = os.path.join(os.getcwd(), 'd12_configs')
    
    # Create directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    # Add metadata
    config_with_meta = {
        "version": "1.0",
        "type": "d12_configuration",
        "configuration": config
    }
    
    # Full path
    filepath = os.path.join(config_dir, filename)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(config_with_meta, f, indent=2)
    
    print(f"Configuration saved to: {filepath}")
    return filepath


def load_d12_config(filename: str, config_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load D12 configuration from a JSON file.
    
    Args:
        filename: Configuration filename
        config_dir: Directory containing config (searches multiple locations)
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file not found
        ValueError: If configuration format is invalid
    """
    # Possible search paths
    search_paths = []
    
    # If absolute path, use it directly
    if os.path.isabs(filename):
        search_paths.append(filename)
    else:
        # Add config_dir if specified
        if config_dir:
            search_paths.append(os.path.join(config_dir, filename))
        
        # Add common search locations
        search_paths.extend([
            filename,  # Current directory
            os.path.join(os.getcwd(), 'd12_configs', filename),
            os.path.join(os.path.dirname(__file__), 'example_configs', filename),
            os.path.join(os.path.dirname(__file__), 'd12_configs', filename),
        ])
    
    # Try to find and load the file
    config_data = None
    loaded_from = None
    
    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config_data = json.load(f)
                loaded_from = path
                break
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}: {e}")
    
    if config_data is None:
        raise FileNotFoundError(
            f"Configuration file '{filename}' not found. Searched locations:\n" + 
            "\n".join(f"  - {p}" for p in search_paths)
        )
    
    # Validate format
    if not isinstance(config_data, dict):
        raise ValueError(f"Invalid configuration format in {loaded_from}")
    
    # Check for version compatibility
    version = config_data.get("version", "0.0")
    if version not in ["1.0"]:
        print(f"Warning: Configuration version {version} may not be fully compatible")
    
    # Extract configuration
    if "configuration" in config_data:
        config = config_data["configuration"]
    else:
        # Assume entire file is configuration (backward compatibility)
        config = config_data
    
    print(f"Loaded configuration from: {loaded_from}")
    return config


def validate_d12_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate D12 configuration for completeness and compatibility.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        tuple: (is_valid, list_of_issues)
    """
    issues = []
    
    # Required fields for all calculations
    required_base = ["calculation_type", "method"]
    
    # Check base requirements
    for field in required_base:
        if field not in config:
            issues.append(f"Missing required field: {field}")
    
    if not issues:  # Only continue if base fields exist
        calc_type = config["calculation_type"]
        method = config["method"]
        
        # Method-specific requirements
        if method == "DFT":
            if "functional" not in config:
                issues.append("DFT method requires 'functional' field")
            if "dft_grid" not in config and not config.get("is_3c_method", False):
                issues.append("DFT method requires 'dft_grid' field (unless 3C method)")
        
        # Calculation type specific requirements
        if calc_type == "OPT":
            if "optimization_type" not in config:
                issues.append("OPT calculation requires 'optimization_type' field")
            if "optimization_settings" not in config:
                issues.append("OPT calculation requires 'optimization_settings' field")
                
        elif calc_type == "FREQ":
            if "freq_settings" not in config:
                issues.append("FREQ calculation requires 'freq_settings' field")
        
        # Basis set requirements
        if "basis_set" not in config:
            issues.append("Missing required field: basis_set")
        if "basis_set_type" not in config:
            issues.append("Missing required field: basis_set_type")
        
        # Tolerance requirements
        if "tolerances" not in config:
            issues.append("Missing required field: tolerances")
        elif isinstance(config["tolerances"], dict):
            if "TOLINTEG" not in config["tolerances"]:
                issues.append("tolerances must include TOLINTEG")
            if "TOLDEE" not in config["tolerances"]:
                issues.append("tolerances must include TOLDEE")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def list_available_configs(config_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """
    List all available configuration files.
    
    Args:
        config_dir: Directory to search (default: searches multiple locations)
        
    Returns:
        list: List of dictionaries with 'name', 'path', and 'description'
    """
    configs = []
    
    # Search locations
    search_dirs = []
    if config_dir:
        search_dirs.append(config_dir)
    else:
        search_dirs.extend([
            os.path.join(os.getcwd(), 'd12_configs'),
            os.path.join(os.path.dirname(__file__), 'example_configs'),
            os.path.join(os.path.dirname(__file__), 'd12_configs'),
        ])
    
    # Search for JSON files
    for dir_path in search_dirs:
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(dir_path, filename)
                    try:
                        config = load_d12_config(filepath)
                        configs.append({
                            'name': filename,
                            'path': filepath,
                            'description': config.get('description', 'No description'),
                            'calc_type': config.get('calculation_type', 'Unknown')
                        })
                    except:
                        # Skip invalid files
                        pass
    
    # Add built-in templates
    for name, config in get_default_d12_configs().items():
        configs.append({
            'name': f"{name} (built-in)",
            'path': "built-in",
            'description': config.get('description', 'No description'),
            'calc_type': config.get('calculation_type', 'Unknown')
        })
    
    return configs


def get_default_d12_config(template_name: str) -> Dict[str, Any]:
    """
    Get a default configuration template by name.
    
    Args:
        template_name: Name of the template
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        ValueError: If template not found
    """
    templates = get_default_d12_configs()
    
    if template_name in templates:
        return templates[template_name].copy()
    else:
        available = list(templates.keys())
        raise ValueError(
            f"Template '{template_name}' not found. Available templates:\n" +
            "\n".join(f"  - {t}" for t in available)
        )


def apply_config_to_options(config: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply configuration settings to an options dictionary.
    
    This maps configuration fields to the format expected by d12 creation scripts.
    
    Args:
        config: Configuration dictionary from JSON
        options: Options dictionary to update
        
    Returns:
        dict: Updated options dictionary
    """
    # Direct mappings
    direct_mappings = [
        'calculation_type', 'method', 'functional', 'dispersion',
        'basis_set', 'basis_set_type', 'basis_set_path',
        'dft_grid', 'spin_polarized', 'spinlock',
        'use_smearing', 'smearing_width', 'dimensionality',
        'symmetry_handling', 'write_only_unique', 'validate_symmetry',
        'trigonal_axes', 'origin_setting', 'is_3c_method'
    ]
    
    for field in direct_mappings:
        if field in config:
            options[field] = config[field]
    
    # Handle nested configurations
    if 'tolerances' in config:
        options['tolerances'] = config['tolerances'].copy()
    
    if 'optimization_settings' in config and config.get('calculation_type') == 'OPT':
        options['optimization_type'] = config.get('optimization_type', 'FULLOPTG')
        for key, value in config['optimization_settings'].items():
            options[f'opt_{key}'] = value
    
    if 'scf_settings' in config:
        for key, value in config['scf_settings'].items():
            options[f'scf_{key}'] = value
    
    if 'freq_settings' in config and config.get('calculation_type') == 'FREQ':
        options['freq_settings'] = config['freq_settings'].copy()
    
    # Handle k-points if specified
    if 'k_points' in config:
        options['k_points'] = config['k_points']
    
    return options


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of a configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("D12 Configuration Summary")
    print("="*60)
    
    # Basic info
    print(f"Name: {config.get('name', 'Unnamed')}")
    print(f"Description: {config.get('description', 'No description')}")
    print(f"Calculation Type: {config.get('calculation_type', 'Unknown')}")
    
    # Method and theory
    print(f"\nMethod: {config.get('method', 'Unknown')}")
    if config.get('method') == 'DFT':
        print(f"Functional: {config.get('functional', 'Unknown')}")
        print(f"Dispersion: {config.get('dispersion', False)}")
        print(f"DFT Grid: {config.get('dft_grid', 'Default')}")
    
    # Basis set
    print(f"\nBasis Set: {config.get('basis_set', 'Unknown')}")
    print(f"Basis Set Type: {config.get('basis_set_type', 'Unknown')}")
    
    # Convergence
    if 'tolerances' in config:
        print(f"\nTolerances:")
        print(f"  TOLINTEG: {config['tolerances'].get('TOLINTEG', 'Default')}")
        print(f"  TOLDEE: {config['tolerances'].get('TOLDEE', 'Default')}")
    
    # Calculation specific
    if config.get('calculation_type') == 'OPT' and 'optimization_settings' in config:
        print(f"\nOptimization Settings:")
        opt = config['optimization_settings']
        print(f"  Convergence: TOLDEG={opt.get('toldeg', 'Default')} TOLDEX={opt.get('toldex', 'Default')}")
        print(f"  Max Cycles: {opt.get('maxcycle', 'Default')}")
    
    print("="*60 + "\n")


# Make validate function importable with correct name
from typing import Tuple