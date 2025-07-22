#!/usr/bin/env python3
"""
MACE Configuration Module
========================

Central configuration for paths and settings used throughout MACE.
This module provides consistent path resolution for all MACE components.

Author: MACE Development Team
"""

import os
from pathlib import Path

# Get the MACE root directory (where this config file is located)
MACE_ROOT = Path(__file__).parent.resolve()

# Environment variable override support
if os.environ.get('MACE_HOME'):
    MACE_ROOT = Path(os.environ['MACE_HOME']).resolve()

# Core directory paths
CRYSTAL_D12_DIR = MACE_ROOT / "Crystal_d12"
CRYSTAL_D3_DIR = MACE_ROOT / "Crystal_d3"
MACE_DIR = MACE_ROOT / "mace"

# Basis set directories (located in Crystal_d12/basis_sets folder)
BASIS_SETS_DIR = CRYSTAL_D12_DIR / "basis_sets"
BASIS_DOUBLEZETA_DIR = BASIS_SETS_DIR / "full.basis.doublezeta"
BASIS_TRIPLEZETA_DIR = BASIS_SETS_DIR / "full.basis.triplezeta"
BASIS_STUTTGART_DIR = BASIS_SETS_DIR / "stuttgart"

# Data directories
RCSR_DIR = CRYSTAL_D3_DIR / "RCSR"
RCSR_2P_DIR = RCSR_DIR / "2P"
RCSR_3P_DIR = RCSR_DIR / "3P"

# Template directories
BAND_PLOT_DIR = CRYSTAL_D3_DIR / "Archived" / "band_plot"
D3_INPUT_DIR = CRYSTAL_D3_DIR / "Archived" / "d3_input"
PARAMETERS_DIR = CRYSTAL_D3_DIR / "Archived" / "parameters"

# Configuration files
EXAMPLE_CONFIGS_DIR = CRYSTAL_D3_DIR / "example_configs"

# Default paths for backward compatibility
DEFAULT_DZ_PATH = str(BASIS_DOUBLEZETA_DIR) + "/"
DEFAULT_TZ_PATH = str(BASIS_TRIPLEZETA_DIR) + "/"
DEFAULT_STUTTGART_PATH = str(BASIS_STUTTGART_DIR) + "/"

def get_basis_path(basis_type="doublezeta"):
    """
    Get the path to a basis set directory.
    
    Args:
        basis_type: One of "doublezeta", "triplezeta", or "stuttgart"
        
    Returns:
        Path object to the basis set directory
    """
    basis_map = {
        "doublezeta": BASIS_DOUBLEZETA_DIR,
        "dz": BASIS_DOUBLEZETA_DIR,
        "triplezeta": BASIS_TRIPLEZETA_DIR,
        "tz": BASIS_TRIPLEZETA_DIR,
        "stuttgart": BASIS_STUTTGART_DIR
    }
    return basis_map.get(basis_type.lower(), BASIS_DOUBLEZETA_DIR)

def get_template_path(template_type, filename=None):
    """
    Get the path to a template file.
    
    Args:
        template_type: One of "band", "d3_input", or "parameters"
        filename: Optional specific filename
        
    Returns:
        Path object to the template directory or file
    """
    template_map = {
        "band": BAND_PLOT_DIR,
        "d3_input": D3_INPUT_DIR,
        "parameters": PARAMETERS_DIR
    }
    
    base_path = template_map.get(template_type.lower())
    if base_path and filename:
        return base_path / filename
    return base_path

def get_rcsr_path(dimension="2P", topology=None):
    """
    Get the path to RCSR data files.
    
    Args:
        dimension: "2P" or "3P"
        topology: Optional specific topology file
        
    Returns:
        Path object to the RCSR directory or file
    """
    if dimension == "2P":
        base_path = RCSR_2P_DIR
    elif dimension == "3P":
        base_path = RCSR_3P_DIR
    else:
        base_path = RCSR_DIR
        
    if topology:
        return base_path / topology
    return base_path

def ensure_paths_exist():
    """
    Check that all required directories exist.
    
    Returns:
        Dictionary of path statuses
    """
    paths_to_check = {
        "MACE_ROOT": MACE_ROOT,
        "Crystal_d12": CRYSTAL_D12_DIR,
        "Crystal_d3": CRYSTAL_D3_DIR,
        "Basis Sets Dir": BASIS_SETS_DIR,
        "Basis DZ": BASIS_DOUBLEZETA_DIR,
        "Basis TZ": BASIS_TRIPLEZETA_DIR,
        "Stuttgart": BASIS_STUTTGART_DIR,
        "RCSR": RCSR_DIR,
        "Templates": D3_INPUT_DIR
    }
    
    status = {}
    for name, path in paths_to_check.items():
        status[name] = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False
        }
    
    return status

# Legacy compatibility - these variables match what scripts expect
# This allows gradual migration without breaking existing code
basis_doublezeta_path = DEFAULT_DZ_PATH
basis_triplezeta_path = DEFAULT_TZ_PATH
stuttgart_path = DEFAULT_STUTTGART_PATH

if __name__ == "__main__":
    # When run directly, display configuration status
    print("MACE Configuration Status")
    print("=" * 60)
    print(f"MACE Root: {MACE_ROOT}")
    print(f"Environment MACE_HOME: {os.environ.get('MACE_HOME', 'Not set')}")
    print()
    
    print("Directory Status:")
    print("-" * 60)
    status = ensure_paths_exist()
    for name, info in status.items():
        exists_str = "✓" if info["exists"] else "✗"
        print(f"{exists_str} {name:15} {info['path']}")
    
    print("\nBasis Set Paths:")
    print("-" * 60)
    print(f"Doublezeta: {DEFAULT_DZ_PATH}")
    print(f"Triplezeta: {DEFAULT_TZ_PATH}")
    print(f"Stuttgart:  {DEFAULT_STUTTGART_PATH}")