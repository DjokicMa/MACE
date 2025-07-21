#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Calculation Configuration Module for CRYSTAL23
---------------------------------------------------
This module handles single point (SP) and optimization (OPT) calculations.

This is part of the refactored D12 creation system where calculation-specific
logic is separated into dedicated modules for better maintainability.
"""

from typing import Dict, Any, Optional


# ==============================================================================
# Single Point Calculations
# ==============================================================================

def configure_single_point(current_settings: Optional[Dict[str, Any]] = None, 
                          shared_mode: bool = False) -> Dict[str, Any]:
    """
    Get single point calculation configuration from user.
    
    Args:
        current_settings: Current settings extracted from files (for CRYSTALOptToD12)
        shared_mode: If True, configuration will be used for multiple files
        
    Returns:
        Dict containing SP calculation settings
    """
    from d12_constants import yes_no_prompt
    
    sp_config = {}
    
    # For SP, ask if user wants tight convergence
    use_tight = yes_no_prompt(
        "Use tight convergence for SP calculation? (Recommended for accurate energies)",
        "no"
    )
    
    if use_tight:
        sp_config["tolerances"] = {
            "TOLINTEG": "9 9 9 11 38",
            "TOLDEE": 11
        }
        sp_config["use_tight_sp"] = True
        print("Using tight convergence tolerances for SP calculation")
    else:
        # Use default tolerances
        sp_config["use_tight_sp"] = False
        
    return sp_config


def get_sp_configuration(current_settings: Optional[Dict[str, Any]] = None, 
                        shared_mode: bool = False) -> Dict[str, Any]:
    """Legacy function name for compatibility"""
    return configure_single_point(current_settings, shared_mode)


def write_sp_calculation(f, calc_settings: Dict[str, Any]):
    """
    Write single point calculation section to D12 file.
    
    For SP calculations, there's no specific calculation block to write
    (unlike OPT which has OPTGEOM or FREQ which has FREQCALC).
    The SP calculation is defined by the absence of these blocks.
    
    Args:
        f: File handle
        calc_settings: Dictionary containing all calculation settings
    """
    # SP calculations don't need any special calculation-specific keywords
    # The calculation type is implicit - if there's no OPTGEOM or FREQCALC,
    # CRYSTAL performs a single point calculation
    pass


# ==============================================================================
# Optimization Calculations
# ==============================================================================

# Optimization types
OPT_TYPES = {
    "1": "FULLOPTG",
    "2": "CELLONLY", 
    "3": "ITATOCELL",  # Fixed: was INTONLY
    "4": "ITATOCEL",
    "5": "CVOLOPT"
}

# Default optimization settings
DEFAULT_OPT_SETTINGS = {
    "type": "FULLOPTG",
    "maxcycle": 200,
    "convergence": "TIGHTOPT",
    "toldeg": 0.00003,
    "toldex": 0.00012,
    "toldee": 7,
}


def configure_optimization(current_settings: Optional[Dict[str, Any]] = None,
                          shared_mode: bool = False,
                          is_already_optimized: bool = False) -> Dict[str, Any]:
    """
    Get optimization calculation configuration from user.
    
    Args:
        current_settings: Current settings extracted from files (for CRYSTALOptToD12)
        shared_mode: If True, configuration will be used for multiple files
        is_already_optimized: If True, warn user that geometry is already optimized
        
    Returns:
        Dict containing optimization calculation settings
    """
    from d12_constants import yes_no_prompt, get_user_input
    
    opt_config = {}
    
    # If geometry is already optimized (from CRYSTALOptToD12), warn user
    if is_already_optimized:
        print("\nWarning: The geometry is already optimized. Are you sure you want to run another optimization?")
        if not yes_no_prompt("Continue with geometry optimization?", "no"):
            # User decided not to optimize again - return SP config instead
            opt_config["calculation_type"] = "SP"
            return opt_config
    
    # Get optimization type
    print("\nOptimization types:")
    print("1. FULLOPTG - Full optimization (cell + coordinates)")
    print("2. CELLONLY - Optimize only unit cell")
    print("3. ITATOCELL - Optimize only internal coordinates iteratively")
    print("4. ITATOCEL - Iterative cell optimization")
    print("5. CVOLOPT - Constant volume optimization")
    
    opt_choice = get_user_input("Select optimization type", OPT_TYPES, "1")
    opt_config["type"] = OPT_TYPES[opt_choice]
    
    # Ask about using default optimization settings
    use_default_opt = yes_no_prompt(
        "\nUse default optimization settings? (TOLDEG=0.00003, TOLDEX=0.00012, TOLDEE=7, MAXCYCLE=800)",
        "yes"
    )
    
    if use_default_opt:
        opt_config["convergence"] = "TIGHTOPT"
        opt_config["toldeg"] = 0.00003
        opt_config["toldex"] = 0.00012
        opt_config["toldee"] = 7
        opt_config["maxcycle"] = 800
    else:
        # Ask about convergence criteria
        print("\nOptimization Convergence Criteria:")
        print("\nConvergence tolerances control when the optimization stops:")
        print("  TOLDEG: RMS of the gradient (forces)")
        print("  TOLDEX: RMS of the displacement")
        print("  TOLDEE: Energy change between cycles")
        print("  MAXCYCLE: Maximum optimization steps")
        print("\n1. Default - Standard convergence")
        print("   TOLDEG=0.0003, TOLDEX=0.0012, TOLDEE=7, MAXCYCLE=200")
        print("   Suitable for most calculations, good balance of accuracy and speed")
        print("\n2. TIGHTOPT - Tight convergence")
        print("   TOLDEG=0.00003, TOLDEX=0.00012, TOLDEE=7, MAXCYCLE=200")
        print("   10x tighter gradient/displacement criteria for high-precision structures")
        print("\n3. Custom - Set your own criteria")
        print("   Define all tolerances manually")
        
        conv_choice = input("\nSelect convergence (1-3) [1]: ").strip() or "1"
        
        if conv_choice == "1":
            opt_config["convergence"] = "DEFAULT"
            opt_config["toldeg"] = 0.0003
            opt_config["toldex"] = 0.0012
            opt_config["toldee"] = 7
            opt_config["maxcycle"] = 200
        elif conv_choice == "2":
            opt_config["convergence"] = "TIGHTOPT"
            opt_config["toldeg"] = 0.00003
            opt_config["toldex"] = 0.00012
            opt_config["toldee"] = 7
            opt_config["maxcycle"] = 200
        else:
            opt_config["convergence"] = "CUSTOM"
            
            # Get custom tolerances
            toldeg_input = input("Enter TOLDEG (RMS of gradient, default 0.00003): ").strip()
            opt_config["toldeg"] = float(toldeg_input) if toldeg_input else 0.00003
            
            toldex_input = input("Enter TOLDEX (RMS of displacement, default 0.00012): ").strip()
            opt_config["toldex"] = float(toldex_input) if toldex_input else 0.00012
            
            toldee_input = input("Enter TOLDEE (energy difference exponent, default 7): ").strip()
            opt_config["toldee"] = int(toldee_input) if toldee_input else 7
            
            maxcycle_input = input("Enter MAXCYCLE (max optimization steps, default 200): ").strip()
            opt_config["maxcycle"] = int(maxcycle_input) if maxcycle_input else 200
    
    # PREOPT is not a valid CRYSTAL keyword - removed
    
    # Ask about MAXTRADIUS
    use_maxtradius = yes_no_prompt(
        "\nSet maximum step size (MAXTRADIUS) for geometry optimization?", "no"
    )
    
    if use_maxtradius:
        maxtradius_input = input("Enter MAXTRADIUS (max displacement, default 0.25): ").strip()
        maxtradius = float(maxtradius_input) if maxtradius_input else 0.25
        opt_config["maxtradius"] = maxtradius
    
    return opt_config


def get_optimization_configuration(current_settings: Optional[Dict[str, Any]] = None,
                                 shared_mode: bool = False,
                                 is_already_optimized: bool = False) -> Dict[str, Any]:
    """Legacy function name for compatibility"""
    return configure_optimization(current_settings, shared_mode, is_already_optimized)


def write_optimization_calculation(f, calc_settings: Dict[str, Any]):
    """
    Write optimization calculation section to D12 file.
    
    Args:
        f: File handle
        calc_settings: Dictionary containing all calculation settings
    """
    opt_type = calc_settings.get("optimization_type", "FULLOPTG")
    opt_settings = calc_settings.get("optimization_settings", DEFAULT_OPT_SETTINGS)
    
    write_optimization_section(f, opt_type, opt_settings)


def write_optimization_section(f, optimization_type, optimization_settings):
    """Write the optimization section of the D12 file"""
    from d12_constants import format_crystal_float, DEFAULT_OPT_SETTINGS
    
    print("OPTGEOM", file=f)
    
    # PREOPT is not a valid CRYSTAL keyword - removed
    
    # Ensure optimization_settings is a dictionary
    if not isinstance(optimization_settings, dict):
        print(f"Warning: optimization_settings is not a dictionary (type: {type(optimization_settings)}), using defaults")
        optimization_settings = DEFAULT_OPT_SETTINGS
    
    # For backwards compatibility, check both old and new field names
    opt_type = optimization_type
    if "type" in optimization_settings:
        opt_type = optimization_settings["type"]
    
    print(opt_type, file=f)
    
    # Note: TIGHTOPT is not a valid CRYSTAL keyword
    # We achieve tight optimization by using tight tolerance values
    
    # Always write MAXCYCLE
    maxcycle = optimization_settings.get("maxcycle") or optimization_settings.get("MAXCYCLE", 200)
    print("MAXCYCLE", file=f)
    print(maxcycle, file=f)
    
    # Write tolerances
    toldeg = optimization_settings.get("toldeg") or optimization_settings.get("TOLDEG", 0.00003)
    print("TOLDEG", file=f)
    # TOLDEG should not use scientific notation
    if toldeg < 0.0001:
        print(f"{toldeg:.6f}".rstrip('0').rstrip('.'), file=f)
    else:
        print(format_crystal_float(toldeg), file=f)
    
    toldex = optimization_settings.get("toldex") or optimization_settings.get("TOLDEX", 0.00012)
    print("TOLDEX", file=f)
    # TOLDEX should not use scientific notation
    if toldex < 0.0001:
        print(f"{toldex:.6f}".rstrip('0').rstrip('.'), file=f)
    else:
        print(format_crystal_float(toldex), file=f)
    
    toldee = optimization_settings.get("toldee") or optimization_settings.get("TOLDEE", 7)
    print("TOLDEE", file=f)
    print(toldee, file=f)
    
    # Add MAXTRADIUS if specified
    maxtradius = optimization_settings.get("maxtradius") or optimization_settings.get("MAXTRADIUS")
    if maxtradius:
        print("MAXTRADIUS", file=f)
        print(format_crystal_float(maxtradius), file=f)
    
    print("ENDOPT", file=f)