#!/usr/bin/env python3
"""
Interactive configuration functions for D12 file creation.
Consolidates user interaction and configuration from NewCifToD12.py and CRYSTALOptToD12.py.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

import os
import sys
from typing import Dict, List, Optional, Any, Tuple
import json

# Import necessary constants and utilities
from d12_constants import (
    DEFAULT_SETTINGS, DEFAULT_OPT_SETTINGS, DEFAULT_TOLERANCES,
    FUNCTIONAL_CATEGORIES, COMMON_FUNCTIONALS, DFT_GRID_OPTIONS,
    DISPERSION_OPTIONS, SMEARING_OPTIONS, DEFAULT_FREQ_SETTINGS,
    SPACEGROUP_SYMBOLS, PRINT_OPTIONS, D3_FUNCTIONALS, yes_no_prompt, get_valid_input,
    configure_tolerances, configure_scf_settings, select_basis_set,
    configure_dft_grid, configure_dispersion, configure_spin_polarization,
    configure_smearing, safe_float, safe_int
)

# Import calculation-specific modules
from d12_calc_basic import configure_single_point, configure_optimization, write_optimization_section
from d12_calc_freq import (
    get_frequency_configuration, get_advanced_frequency_settings,
    write_frequency_section, write_anharm_section
)


def display_default_settings():
    """Display default calculation settings"""
    print("\n" + "="*60)
    print("DEFAULT RECOMMENDED SETTINGS FOR CRYSTAL23")
    print("="*60)
    
    print("\n### SYMMETRY SETTINGS ###")
    print(f"Symmetry handling: CIF (Use symmetry as defined in the CIF file)")
    print(f"Trigonal axes: AUTO (Use setting as detected in CIF)")
    print(f"High symmetry space groups: AUTO (Use origin as detected in CIF)")
    print(f"Atom writing: Unique atoms only (use symmetry operations)")
    
    print("\n### CALCULATION SETTINGS ###")
    print(f"Dimensionality: CRYSTAL (3D periodic system)")
    print(f"Calculation type: OPT (Geometry optimization)")
    print(f"Optimization type: FULLOPTG (Full geometry optimization)")
    print(f"Optimization parameters:")
    print(f"  - TOLDEG: {DEFAULT_OPT_SETTINGS['toldeg']} (RMS of gradient)")
    print(f"  - TOLDEX: {DEFAULT_OPT_SETTINGS['toldex']} (RMS of displacement)")
    print(f"  - TOLDEE: {DEFAULT_OPT_SETTINGS['toldee']} (Energy convergence)")
    print(f"  - MAXCYCLE: 800 (Max optimization steps)")
    print(f"  - MAXTRADIUS: Not set (default)")
    
    print("\n### BASIS SET AND DFT SETTINGS ###")
    print(f"Method: {DEFAULT_SETTINGS['method_type']}")
    print(f"Basis set type: {DEFAULT_SETTINGS['basis_set_type']}")
    print(f"Basis set: {DEFAULT_SETTINGS['basis_set']} (Triple-ζ + polarization, revised)")
    print(f"DFT functional: {DEFAULT_SETTINGS['functional']}-D3 (Screened hybrid with D3 dispersion)")
    print(f"DFT grid: {DEFAULT_SETTINGS['dft_grid']} (Extra large grid, most accurate)")
    print(f"Spin polarized: {DEFAULT_SETTINGS['spin_polarized']}")
    print(f"Fermi smearing: No")
    
    print("\n### SCF SETTINGS ###")
    print(f"Tolerances: TOLINTEG={DEFAULT_TOLERANCES['TOLINTEG']}, TOLDEE={DEFAULT_TOLERANCES['TOLDEE']}")
    print(f"SCF method: DIIS")
    print(f"SCF max cycles: 800")
    print(f"FMIXING: {DEFAULT_SETTINGS.get('fmixing', 30)}%")
    
    print("\n### EXAMPLE D12 OUTPUT ###")
    print("-"*70)
    print("EXAMPLE_STRUCTURE")
    print("CRYSTAL")
    print("0 0 0")
    print("225")
    print("5.46500 #a=b=c cubic alpha = beta = gamma = 90")
    print("2")
    print("11 0.0000000000 0.0000000000 0.0000000000 Biso 1.000000 Na")
    print("17 0.5000000000 0.5000000000 0.5000000000 Biso 1.000000 Cl")
    print("OPTGEOM")
    print("FULLOPTG")
    print("TOLDEG")
    print("0.0003")
    print("TOLDEX")
    print("0.0012")
    print("TOLDEE")
    print("7")
    print("MAXCYCLE")
    print("800")
    print("ENDOPT")
    print("BASISSET")
    print("POB-TZVP-REV2")
    print("DFT")
    print("SPIN")
    print("HSE06-D3")
    print("XLGRID")
    print("ENDDFT")
    print("SHRINK")
    print("0 24")
    print("12 12 12")
    print("SCFDIR")
    print("TOLINTEG")
    print("7 7 7 7 14")
    print("TOLDEE")
    print("7")
    print("MAXCYCLE")
    print("800")
    print("FMIXING")
    print("30")
    print("DIIS")
    print("HISTDIIS")
    print("100")
    print("PPAN")
    print("BIPOSIZE")
    print("110000000")
    print("EXCHSIZE")
    print("110000000")
    print("END")
    print("-"*70)


def display_current_settings(settings: Dict[str, Any], extracted: bool = False, expert_mode: bool = False):
    """Display current calculation settings from parsed files
    
    Args:
        settings: Dictionary of settings to display
        extracted: If True, shows as "EXTRACTED CALCULATION SETTINGS"
        expert_mode: If True, shows all available settings in detail
    """
    print("\n" + "="*60)
    if extracted:
        if expert_mode:
            print("EXTRACTED CALCULATION SETTINGS (EXPERT MODE)")
        else:
            print("EXTRACTED CALCULATION SETTINGS")
    else:
        print("CURRENT CALCULATION SETTINGS")
    print("="*60)
    
    # Structure information (only for extracted settings)
    if extracted:
        print(f"\n### STRUCTURE INFORMATION ###")
        print(f"Dimensionality: {settings.get('dimensionality', 'CRYSTAL')}")
        print(f"Space group: {settings.get('spacegroup', 'N/A')}")
        print(f"Origin setting: {settings.get('origin_setting', '0 0 0')}")
        if settings.get("conventional_cell"):
            cell = settings["conventional_cell"]
            # Ensure all cell parameters are floats
            try:
                a, b, c = float(cell[0]), float(cell[1]), float(cell[2])
                alpha, beta, gamma = float(cell[3]), float(cell[4]), float(cell[5])
                print(f"Unit cell: a={a:.4f}, b={b:.4f}, c={c:.4f}")
                print(f"           α={alpha:.2f}°, β={beta:.2f}°, γ={gamma:.2f}°")
            except (TypeError, ValueError, IndexError) as e:
                # If conversion fails, just print the raw values
                print(f"Unit cell: {cell}")
    
    # Method settings
    print(f"\n### METHOD AND BASIS SET ###")
    if settings.get("functional"):
        func = settings["functional"]
        # Determine if this is a Hartree-Fock method or DFT
        if func in ["RHF", "UHF", "HF3C", "HFSOL3C"]:
            print(f"Method: Hartree-Fock ({func})")
            if settings.get("is_3c_method"):
                print(f"  (3c composite method)")
        else:
            # It's a DFT functional
            if settings.get("dispersion") and "-D3" not in func:
                func += "-D3"
            print(f"Method: DFT")
            print(f"Functional: {func}")
            if settings.get("is_3c_method"):
                print(f"  (3c composite method)")
    else:
        print(f"Method: Hartree-Fock (RHF)")
    
    print(f"Basis set: {settings.get('basis_set', 'N/A')} ({settings.get('basis_set_type', 'INTERNAL')})")
    
    if settings.get("dft_grid"):
        print(f"DFT grid: {settings.get('dft_grid')}")
    elif settings.get("functional") and settings["functional"] not in ["HF", "RHF", "UHF"]:
        print(f"DFT grid: DEFAULT")
    
    # Electronic structure settings
    print(f"\n### ELECTRONIC STRUCTURE ###")
    print(f"Spin polarized: {settings.get('spin_polarized', False)}")
    if settings.get('spin_polarized') and expert_mode:
        print(f"SPINLOCK: {settings.get('spinlock', 0)}")
    
    if settings.get("smearing"):
        print(f"Fermi smearing: Yes (width={settings.get('smearing_width', 0.01)})")
    else:
        print(f"Fermi smearing: No")
    
    # Tolerances
    print(f"\n### CONVERGENCE CRITERIA ###")
    if settings.get("tolerances"):
        tolinteg = settings['tolerances'].get('TOLINTEG')
        toldee = settings['tolerances'].get('TOLDEE')
        
        # Handle TOLINTEG as either string or list
        if isinstance(tolinteg, list):
            tolinteg_str = ' '.join(map(str, tolinteg))
        else:
            tolinteg_str = str(tolinteg) if tolinteg else 'N/A'
            
        print(f"TOLINTEG: {tolinteg_str}")
        print(f"TOLDEE: {toldee if toldee else 'N/A'}")
    
    # SCF settings
    if settings.get("scf_settings"):
        scf = settings["scf_settings"]
        print(f"\n### SCF SETTINGS ###")
        print(f"SCF method: {scf.get('method', 'DIIS')}")
        print(f"SCF max cycles: {scf.get('maxcycle', 800)}")
        fmixing_val = scf.get('fmixing', 30)
        # Ensure fmixing is a number
        try:
            fmixing_val = int(fmixing_val)
        except (TypeError, ValueError):
            fmixing_val = 30  # Default
        print(f"FMIXING: {fmixing_val}%")
        
        if expert_mode:
            if scf.get('histdiis'):
                print(f"HISTDIIS: {scf['histdiis']}")
            if scf.get('ppan'):
                print(f"PPAN: Yes (parallel diagonalization)")
            if scf.get('biposize'):
                print(f"BIPOSIZE: {scf['biposize']}")
            if scf.get('exchsize'):
                print(f"EXCHSIZE: {scf['exchsize']}")
            if scf.get('broyden_w0'):
                print(f"BROYDEN W0: {scf['broyden_w0']}")
    
    # K-points
    if settings.get("k_points"):
        print(f"\n### K-POINTS ###")
        k_points = settings['k_points']
        if isinstance(k_points, str):
            print(f"K-points: {k_points}")
        else:
            print(f"K-points: {k_points}")
    
    # Calculation-specific settings (expert mode)
    if expert_mode:
        # Optimization settings
        if settings.get("optimization_settings"):
            opt = settings["optimization_settings"]
            print(f"\n### OPTIMIZATION SETTINGS ###")
            print(f"Optimization type: {opt.get('type', settings.get('optimization_type', 'FULLOPTG'))}")
            if opt.get('toldeg') is not None:
                print(f"TOLDEG: {opt['toldeg']} (RMS gradient)")
            if opt.get('toldex') is not None:
                print(f"TOLDEX: {opt['toldex']} (RMS displacement)")
            if opt.get('toldee') is not None:
                print(f"TOLDEE: {opt['toldee']} (energy convergence)")
            if opt.get('maxcycle') is not None:
                print(f"MAXCYCLE: {opt['maxcycle']}")
            if opt.get('maxtradius') is not None:
                print(f"MAXTRADIUS: {opt['maxtradius']}")
            if opt.get('convergence'):
                print(f"Convergence preset: {opt['convergence']}")
        
        # Frequency settings
        if settings.get("freq_settings"):
            freq = settings["freq_settings"]
            print(f"\n### FREQUENCY SETTINGS ###")
            freq_mode = freq.get('freq_mode', 'FREQCALC')
            print(f"Frequency mode: {freq_mode}")
            
            if freq_mode == 'ANHARM':
                if freq.get('h_atom'):
                    print(f"H atom label: {freq['h_atom']}")
                if freq.get('keep_symmetry') is not None:
                    print(f"Keep symmetry: {freq['keep_symmetry']}")
                if freq.get('points26'):
                    print(f"Points: 26")
                else:
                    print(f"Points: 7")
                if freq.get('isotopes'):
                    print(f"Isotopes: {freq['isotopes']}")
            else:
                # FREQCALC mode
                if freq.get('template'):
                    print(f"Template: {freq['template']}")
                if freq.get('dispersion'):
                    print(f"Dispersion calculation: Yes")
                if freq.get('ir_raman'):
                    print(f"IR/Raman intensities: Yes")
                if freq.get('modes'):
                    print(f"Selected modes: {freq['modes']}")
                if freq.get('temprange'):
                    temps = freq['temprange']
                    if isinstance(temps, tuple) and len(temps) == 3:
                        print(f"Temperature range: {temps[1]}-{temps[2]}K ({temps[0]} points)")
                if freq.get('pressrange'):
                    press = freq['pressrange']
                    if isinstance(press, tuple) and len(press) == 3:
                        print(f"Pressure range: {press[1]}-{press[2]} GPa ({press[0]} points)")
                
                # Phonon bands
                if freq.get('bands'):
                    bands = freq['bands']
                    print(f"\nPhonon band structure:")
                    print(f"  SHRINK: {bands.get('shrink', 16)}")
                    print(f"  Points per segment: {bands.get('npoints', 100)}")
                    if bands.get('path') == 'AUTO':
                        print(f"  Path: AUTO (from space group)")
                    elif bands.get('path'):
                        print(f"  Path: {bands['path']}")
                
                # Phonon DOS
                if freq.get('pdos'):
                    pdos = freq['pdos']
                    print(f"\nPhonon DOS:")
                    print(f"  Max frequency: {pdos.get('max_freq', 2000)} cm⁻¹")
                    print(f"  Number of bins: {pdos.get('nbins', 200)}")
                    print(f"  Projected: {pdos.get('projected', True)}")
        
        # Additional settings
        if settings.get('levshift'):
            print(f"\n### LEVEL SHIFTING ###")
            print(f"LEVSHIFT: {settings['levshift'].get('shift', 5.0)} Hartree")
            print(f"Cycles: {settings['levshift'].get('ncycles', 30)}")
        
        if settings.get('print_options'):
            print(f"\n### PRINT OPTIONS ###")
            print(f"Options: {', '.join(settings['print_options'])}")
        
        if settings.get('ghost_atoms'):
            print(f"\n### GHOST ATOMS ###")
            print(f"Type: {settings['ghost_atoms'].get('type')}")
            if settings['ghost_atoms'].get('distance'):
                print(f"Distance: {settings['ghost_atoms']['distance']} Å")
            if settings['ghost_atoms'].get('positions'):
                print(f"Positions: {settings['ghost_atoms']['positions']}")
            if settings['ghost_atoms'].get('atom_numbers'):
                print(f"Atom numbers: {settings['ghost_atoms']['atom_numbers']}")
        
        if settings.get('external_pressure'):
            print(f"\n### EXTERNAL PRESSURE ###")
            print(f"Pressure: {settings['external_pressure'].get('pressure', 0.0)} GPa")
    
    print("="*60)


def get_user_choice(prompt: str, options: List[Tuple[str, str]], default: str = "1") -> str:
    """Get user choice from a list of options with concise prompting
    
    Args:
        prompt: Base prompt like "Select calculation type"
        options: List of (key, description) tuples
        default: Default choice key
        
    Returns:
        Selected key
    """
    # Don't print the prompt here as it's already printed in the calling code
    valid_keys = [opt[0] for opt in options]
    choice = input(f"{prompt} (default: {default}): ").strip() or default
    
    while choice not in valid_keys:
        print(f"Invalid input. Please choose from: {', '.join(valid_keys)}")
        choice = input(f"{prompt} (default: {default}): ").strip() or default
    
    return choice


def get_calculation_type() -> str:
    """Get the type of calculation from user"""
    calc_options = [
        ("1", "Single Point (SP)"),
        ("2", "Optimization (OPT)"),
        ("3", "Frequency (FREQ)")
    ]
    
    calc_types = {
        "1": "SP",
        "2": "OPT",
        "3": "FREQ"
    }
    
    print("\nSelect calculation type:")
    for key, desc in calc_options:
        print(f"{key}. {desc}")
    
    choice = get_user_choice("Select calculation type", calc_options, "2")
    return calc_types[choice]


def select_basis_set_with_defaults(elements: List[int], method: str = "DFT", 
                                 functional: Optional[str] = None,
                                 shared_mode: bool = False,
                                 current_basis_type: str = "INTERNAL",
                                 current_basis: str = "Not set") -> Dict[str, Any]:
    """
    Wrapper for select_basis_set that preserves current settings as defaults.
    
    This function temporarily patches the get_user_input function to use
    appropriate defaults based on current settings.
    """
    import d12_constants
    
    # Store the original get_user_input function
    original_get_user_input = d12_constants.get_user_input
    
    # Create a patched version that uses our defaults
    def patched_get_user_input(prompt, options, default):
        # For basis type selection
        if "Select basis set type" in prompt:
            if current_basis_type == "EXTERNAL":
                return "1"
            else:
                return "2"
        
        # For external basis selection
        elif "Select external basis set" in prompt and current_basis_type == "EXTERNAL":
            if "DZVP" in current_basis or "doublezeta" in current_basis:
                return "1"
            elif "TZVP" in current_basis or "triplezeta" in current_basis:
                return "2"
            else:
                return default
        
        # For internal basis selection - need to check the prompt more carefully
        elif "Enter your choice" in prompt and current_basis_type == "INTERNAL":
            # Map current basis to selection number
            basis_map = {
                "STO-3G": "1", "STO-6G": "2", "POB-DZVP": "3", "POB-DZVPP": "4",
                "POB-TZVP": "5", "POB-DZVP-REV2": "6", "POB-TZVP-REV2": "7",
                "MINIS": "8", "6-31G*": "9", "def2-SV(P)": "10", "def2-SVP": "11",
                "def-TZVP": "12", "def2-TZVP": "13"
            }
            
            # Try exact match first
            if current_basis in basis_map:
                return basis_map[current_basis]
            
            # Try partial matches for special cases
            for basis_name, choice in basis_map.items():
                if basis_name in current_basis or current_basis in basis_name:
                    return choice
            
            # Default to TZVP-REV2 if not found
            return "7"
        
        # For all other prompts, use the original function
        return original_get_user_input(prompt, options, default)
    
    # Temporarily replace the function
    d12_constants.get_user_input = patched_get_user_input
    
    try:
        # Call the original select_basis_set with our patched input function
        result = select_basis_set(elements, method, functional, shared_mode)
    finally:
        # Always restore the original function
        d12_constants.get_user_input = original_get_user_input
    
    return result


def configure_dft_grid_with_defaults(functional: str, current_grid: str = "XLGRID", 
                                   shared_mode: bool = False) -> Optional[str]:
    """
    Wrapper for configure_dft_grid that uses current grid as default.
    """
    import d12_constants
    
    # 3C methods have their own optimized grids
    if "-3C" in functional or functional.endswith("3C"):
        return None
    
    # Store the original get_user_input function
    original_get_user_input = d12_constants.get_user_input
    
    # Map grid names to choice numbers
    grid_map = {
        "OLDGRID": "1",
        "DEFAULT": "2", 
        "LGRID": "3",
        "XLGRID": "4",
        "XXLGRID": "5",
        "XXXLGRID": "6",
        "HUGEGRID": "7"
    }
    
    # Get the default choice based on current grid
    default_choice = grid_map.get(current_grid, "4")
    
    # Create a patched version that uses our default
    def patched_get_user_input(prompt, options, default):
        if "Select integration grid" in prompt:
            return default_choice
        return original_get_user_input(prompt, options, default)
    
    # Temporarily replace the function
    d12_constants.get_user_input = patched_get_user_input
    
    try:
        # Call the original configure_dft_grid with our patched input function
        result = configure_dft_grid(functional, shared_mode)
    finally:
        # Always restore the original function
        d12_constants.get_user_input = original_get_user_input
    
    return result


def configure_tolerances_with_defaults(current_tolerances: Dict[str, Any], 
                                      shared_mode: bool = False, 
                                      calculation_type: str = None) -> Dict[str, Any]:
    """Wrapper for configure_tolerances that shows current settings as defaults"""
    # Extract current values
    current_toldee = current_tolerances.get('TOLDEE', current_tolerances.get('toldee', 7))
    current_tolinteg = current_tolerances.get('TOLINTEG', current_tolerances.get('tolinteg', '7 7 7 7 14'))
    
    # Convert tolinteg to string if it's a list
    if isinstance(current_tolinteg, list):
        current_tolinteg = ' '.join(map(str, current_tolinteg))
    
    # Determine which preset matches current settings
    if current_tolinteg == "7 7 7 7 14" and current_toldee == 7:
        default_choice = "1"
    elif current_tolinteg == "8 8 8 9 24" and current_toldee == 9:
        default_choice = "2"
    elif current_tolinteg == "9 9 9 11 38" and current_toldee == 11:
        default_choice = "3"
    else:
        default_choice = "4"  # Custom
    
    # Print current settings
    print("\n=== SCF CONVERGENCE SETTINGS ===")
    print(f"Current settings: TOLINTEG: {current_tolinteg}, TOLDEE: {current_toldee}")
    
    # Menu-based selection
    if calculation_type == "FREQ":
        print("\nSelect SCF convergence level (FREQ calculations require tighter tolerances):")
        print("1: Standard - TOLINTEG: 7 7 7 7 14, TOLDEE: 7")
        print("2: Tight - TOLINTEG: 8 8 8 9 24, TOLDEE: 9 (recommended for FREQ)")
        print("3: Very tight - TOLINTEG: 9 9 9 11 38, TOLDEE: 11 (default for FREQ)")
        print("4: Custom (keep current or enter new values)")
        
        choice = input(f"Select tolerance level (1-4) [{default_choice}]: ").strip()
        if not choice:
            choice = default_choice
    else:
        # SP/OPT calculations
        print("\nSelect SCF convergence level:")
        print("1: Standard - TOLINTEG: 7 7 7 7 14, TOLDEE: 7 (default for OPT/SP)")
        print("2: Tight - TOLINTEG: 8 8 8 9 24, TOLDEE: 9 (higher precision)")
        print("3: Very tight - TOLINTEG: 9 9 9 11 38, TOLDEE: 11 (ultra-high precision)")
        print("4: Custom (keep current or enter new values)")
        
        choice = input(f"Select tolerance level (1-4) [{default_choice}]: ").strip()
        if not choice:
            choice = default_choice
    
    # Process the choice
    tolerances = {}
    if choice == "1":
        tolerances["TOLINTEG"] = "7 7 7 7 14"
        tolerances["TOLDEE"] = 7
    elif choice == "2":
        tolerances["TOLINTEG"] = "8 8 8 9 24"
        tolerances["TOLDEE"] = 9
    elif choice == "3":
        tolerances["TOLINTEG"] = "9 9 9 11 38"
        tolerances["TOLDEE"] = 11
    elif choice == "4":
        # Custom - show current values as defaults
        print("\nEnter custom tolerance values (press Enter to keep current):")
        
        # TOLINTEG
        tolinteg_input = input(f"TOLINTEG [{current_tolinteg}]: ").strip()
        if tolinteg_input:
            tolerances["TOLINTEG"] = tolinteg_input
        else:
            tolerances["TOLINTEG"] = current_tolinteg
        
        # TOLDEE
        toldee_input = input(f"TOLDEE [{current_toldee}]: ").strip()
        if toldee_input:
            try:
                tolerances["TOLDEE"] = int(toldee_input)
            except ValueError:
                print(f"Invalid input, keeping current value of {current_toldee}")
                tolerances["TOLDEE"] = current_toldee
        else:
            tolerances["TOLDEE"] = current_toldee
    
    return tolerances


def configure_scf_settings_with_defaults(current_scf: Dict[str, Any], 
                                       shared_mode: bool = False) -> Dict[str, Any]:
    """Wrapper for configure_scf_settings that shows current settings as defaults"""
    # Extract current values
    current_maxcycle = current_scf.get('maxcycle', 800)
    current_fmixing = current_scf.get('fmixing', 30)
    current_method = current_scf.get('method', 'DIIS')
    current_ppan = current_scf.get('ppan', True)
    current_biposize = current_scf.get('biposize', 110000000)
    current_exchsize = current_scf.get('exchsize', 110000000)
    
    scf_settings = {}
    
    print("\n=== SCF SETTINGS ===")
    print(f"Current settings: Method={current_method}, MAXCYCLE={current_maxcycle}, FMIXING={current_fmixing}%")
    
    # MAXCYCLE
    print("\nMaximum SCF cycles:")
    print("  - Default: 800 (recommended)")
    print("  - Increase for difficult convergence")
    
    maxcycle_input = input(f"MAXCYCLE [{current_maxcycle}]: ").strip()
    if maxcycle_input:
        try:
            scf_settings["maxcycle"] = int(maxcycle_input)
        except ValueError:
            print(f"Invalid input, using current value of {current_maxcycle}")
            scf_settings["maxcycle"] = current_maxcycle
    else:
        scf_settings["maxcycle"] = current_maxcycle
    
    # FMIXING
    print("\nFMIXING percentage:")
    print("  - Controls mixing of old and new density matrices")
    print("  - Default: 30 (30%)")
    print("  - Lower values = more stable but slower convergence")
    
    fmixing_input = input(f"FMIXING [{current_fmixing}]: ").strip()
    if fmixing_input:
        try:
            fmixing = int(fmixing_input)
            if 0 <= fmixing <= 100:
                scf_settings["fmixing"] = fmixing
            else:
                print(f"Value out of range, using current value of {current_fmixing}")
                scf_settings["fmixing"] = current_fmixing
        except ValueError:
            print(f"Invalid input, using current value of {current_fmixing}")
            scf_settings["fmixing"] = current_fmixing
    else:
        scf_settings["fmixing"] = current_fmixing
    
    # SCF method
    print("\nSCF method:")
    print("  1. DIIS (default - fastest convergence)")
    print("  2. Broyden (alternative for difficult cases)")
    
    current_method_choice = "1" if current_method == "DIIS" else "2"
    method_choice = input(f"Select method (1-2) [{current_method_choice}]: ").strip()
    if not method_choice:
        method_choice = current_method_choice
        
    if method_choice == "1":
        scf_settings["method"] = "DIIS"
    else:
        scf_settings["method"] = "BROYDEN"
    
    # PPAN
    current_ppan_str = "yes" if current_ppan else "no"
    ppan_input = yes_no_prompt(f"Enable PPAN (parallel diagonalization)?", current_ppan_str)
    scf_settings["ppan"] = ppan_input
    
    # BIPOSIZE/EXCHSIZE
    if yes_no_prompt("\nModify memory settings (BIPOSIZE/EXCHSIZE)?", "no"):
        print(f"\nCurrent: BIPOSIZE={current_biposize}, EXCHSIZE={current_exchsize}")
        print("Default: 110000000 (for large systems)")
        
        biposize_input = input(f"BIPOSIZE [{current_biposize}]: ").strip()
        if biposize_input:
            try:
                scf_settings["biposize"] = int(biposize_input)
            except ValueError:
                print(f"Invalid input, using current value of {current_biposize}")
                scf_settings["biposize"] = current_biposize
        else:
            scf_settings["biposize"] = current_biposize
        
        exchsize_input = input(f"EXCHSIZE [{current_exchsize}]: ").strip()
        if exchsize_input:
            try:
                scf_settings["exchsize"] = int(exchsize_input)
            except ValueError:
                print(f"Invalid input, using current value of {current_exchsize}")
                scf_settings["exchsize"] = current_exchsize
        else:
            scf_settings["exchsize"] = current_exchsize
    
    return scf_settings


def configure_spin_polarization_with_defaults(current_settings: Dict[str, Any], 
                                             shared_mode: bool = False) -> Dict[str, Any]:
    """Wrapper for configure_spin_polarization that shows current settings as defaults"""
    # Extract current values
    current_spin = current_settings.get('spin_polarized', current_settings.get('is_spin_polarized', True))
    current_spinlock = current_settings.get('spinlock', 0)
    
    spin_config = {}
    
    print("\n=== SPIN POLARIZATION ===")
    print(f"Current setting: Spin polarization = {'Yes' if current_spin else 'No'}")
    
    default_spin = "yes" if current_spin else "no"
    use_spin = yes_no_prompt(
        "Use spin-polarized calculation?",
        default_spin
    )
    
    spin_config["spin_polarized"] = use_spin
    
    if use_spin:
        print(f"\nCurrent SPINLOCK: {current_spinlock}")
        print("SPINLOCK options (number of unpaired electrons, nα-nβ):")
        print("  - Enter 0 for automatic spin optimization")
        print("  - Enter positive integer for fixed spin multiplicity (e.g., 2 for triplet)")
        print("  - Enter -1 for antiferromagnetic initial guess")
        
        spinlock_input = input(f"SPINLOCK value (nα-nβ) [{current_spinlock}]: ").strip()
        if spinlock_input:
            try:
                spin_config["spinlock"] = int(spinlock_input)
            except ValueError:
                print(f"Invalid input, using current value of {current_spinlock}")
                spin_config["spinlock"] = current_spinlock
        else:
            spin_config["spinlock"] = current_spinlock
    
    return spin_config


def configure_smearing_with_defaults(current_settings: Dict[str, Any], 
                                   temp_kelvin: float = 300, 
                                   shared_mode: bool = False) -> Dict[str, Any]:
    """Wrapper for configure_smearing that shows current settings as defaults"""
    # Extract current values
    current_use_smearing = current_settings.get('use_smearing', current_settings.get('smearing', False))
    current_width = current_settings.get('smearing_width', 0.01)
    
    smearing_config = {}
    
    print("\n=== FERMI SMEARING ===")
    print(f"Current setting: Smearing = {'Yes' if current_use_smearing else 'No'}")
    if current_use_smearing:
        print(f"Current smearing width: {current_width} Ha")
    
    default_smearing = "yes" if current_use_smearing else "no"
    use_smearing = yes_no_prompt(
        "Use Fermi smearing? (recommended for metals)",
        default_smearing
    )
    
    smearing_config["use_smearing"] = use_smearing
    
    if use_smearing:
        print("\nSmearing width options:")
        print("  1. Default (0.01 Ha)")
        print("  2. Temperature-based")
        print("  3. Custom value")
        
        # Determine current choice
        if abs(current_width - 0.01) < 1e-6:
            current_choice = "1"
        else:
            current_choice = "3"
        
        choice = input(f"Select option (1-3) [{current_choice}]: ").strip()
        if not choice:
            choice = current_choice
        
        if choice == "1":
            smearing_config["smearing_width"] = 0.01
        elif choice == "2":
            temp_input = input(f"Enter temperature in Kelvin [{temp_kelvin}]: ").strip()
            if temp_input:
                try:
                    temp = float(temp_input)
                    # Convert temperature to Hartree (kT)
                    # 1 Ha = 315775 K
                    smearing_config["smearing_width"] = temp / 315775.0
                except ValueError:
                    print(f"Invalid input, using {temp_kelvin} K")
                    smearing_config["smearing_width"] = temp_kelvin / 315775.0
            else:
                smearing_config["smearing_width"] = temp_kelvin / 315775.0
        else:
            width_input = input(f"Enter smearing width in Hartree [{current_width}]: ").strip()
            if width_input:
                try:
                    smearing_config["smearing_width"] = float(width_input)
                except ValueError:
                    print(f"Invalid input, using current value of {current_width}")
                    smearing_config["smearing_width"] = current_width
            else:
                smearing_config["smearing_width"] = current_width
    else:
        smearing_config["smearing_width"] = 0.0
    
    return smearing_config


def configure_method(options: Dict[str, Any]) -> Dict[str, Any]:
    """Configure method type and functional"""
    method_options = [
        ("1", "Hartree-Fock (HF)"),
        ("2", "Density Functional Theory (DFT)")
    ]
    
    # Determine default based on current settings
    current_method = options.get("method_type", options.get("method", "DFT"))
    current_functional = options.get("functional", "")
    
    # Set default based on current method
    if current_method == "HF" or current_functional in ["RHF", "UHF", "HF3C", "HFSOL3C"]:
        method_default = "1"
    else:
        method_default = "2"
    
    print("\nSelect method type:")
    for key, desc in method_options:
        print(f"{key}. {desc}")
    
    method_choice = get_user_choice("Select method type", method_options, method_default)
    
    if method_choice == "1":
        options["method"] = "HF"  # Added for compatibility
        options["method_type"] = "HF"
        # Select HF variant
        hf_methods = FUNCTIONAL_CATEGORIES["HF"]["functionals"]
        hf_options = []
        
        # Determine default based on current functional
        hf_default = "1"
        if current_functional in hf_methods:
            hf_default = str(hf_methods.index(current_functional) + 1)
        
        print("\nSelect Hartree-Fock method:")
        for i, method in enumerate(hf_methods, 1):
            desc = FUNCTIONAL_CATEGORIES["HF"]["descriptions"][method]
            print(f"{i}: {method} - {desc}")
            hf_options.append((str(i), f"{method} - {desc}"))
        
        hf_choice = get_user_choice("Select Hartree-Fock method", hf_options, hf_default)
        options["functional"] = hf_methods[int(hf_choice) - 1]
        options["hf_method"] = options["functional"]  # Added for compatibility
    else:
        options["method"] = "DFT"  # Added for compatibility
        options["method_type"] = "DFT"
        
        # Select functional category (excluding HF) - ordered to match NewCifToD12.py
        ordered_categories = ["LDA", "GGA", "HYBRID", "MGGA", "3C"]
        cat_options = []
        
        # Determine current functional's category
        cat_default = "3"  # Default to HYBRID
        if current_functional:
            # Strip dispersion suffix if present
            base_functional = current_functional.replace("-D3", "").replace("-D4", "")
            for i, cat in enumerate(ordered_categories, 1):
                if base_functional in FUNCTIONAL_CATEGORIES[cat]["functionals"]:
                    cat_default = str(i)
                    break
        
        print("\nAvailable functional categories:")
        for i, cat in enumerate(ordered_categories, 1):
            cat_info = FUNCTIONAL_CATEGORIES[cat]
            print(f"{i}. {cat_info['name']}")
            print(f"   {cat_info['description']}")
            # Show appropriate examples for each category
            if cat == "HYBRID":
                print(f"   Examples: B3LYP, PBE0, HSE06, LC-wPBE")
            elif cat == "3C":
                print(f"   Examples: PBEH3C, HSE3C, B973C")
            else:
                print(f"   Examples: {', '.join(cat_info['functionals'][:4])}")
            cat_options.append((str(i), cat))
        
        cat_choice = get_user_choice("Select functional category", cat_options, cat_default)
        
        category = ordered_categories[int(cat_choice) - 1]
        functionals = FUNCTIONAL_CATEGORIES[category]["functionals"]
        
        # Select specific functional
        print(f"\nAvailable {FUNCTIONAL_CATEGORIES[category]['name']}:")
        func_options = []
        default_func = None
        
        # Try to find current functional in this category
        base_functional = current_functional.replace("-D3", "").replace("-D4", "") if current_functional else ""
        
        for i, func in enumerate(functionals, 1):
            desc = FUNCTIONAL_CATEGORIES[category]["descriptions"].get(func, "")
            # Check if functional supports D3
            d3_marker = " [D3✓]" if func in D3_FUNCTIONALS else ""
            full_desc = f"{func} - {desc}{d3_marker}"
            print(f"{i}: {full_desc}")
            func_options.append((str(i), func))
            
            # Set default to current functional if it's in this category
            if func == base_functional:
                default_func = str(i)
            # Otherwise, set HSE06 as default for hybrid functionals
            elif default_func is None and func == "HSE06" and category == "HYBRID":
                default_func = str(i)
        
        if not default_func:
            default_func = "1"
            
        func_choice = get_user_choice(f"Select {FUNCTIONAL_CATEGORIES[category]['name'].split()[0]} functional", func_options, default_func)
        
        options["functional"] = functionals[int(func_choice) - 1]
        options["dft_functional"] = options["functional"]  # Added for compatibility
        
        # Ask about D3 dispersion if functional supports it
        if options["functional"] in D3_FUNCTIONALS:
            use_d3 = yes_no_prompt(f"\nAdd D3 dispersion correction to {options['functional']}?", "yes")
            if use_d3:
                options["dispersion"] = True
                options["use_dispersion"] = True  # Added for compatibility
                options["functional"] += "-D3"
            else:
                options["dispersion"] = False
                options["use_dispersion"] = False
    
    return options


def get_shared_calculation_options() -> Dict[str, Any]:
    """Get calculation options for shared settings mode"""
    options = {}
    
    # Symmetry configuration first
    symmetry_config = configure_symmetry_handling()
    options.update(symmetry_config)
    
    # Calculation type
    options["calculation_type"] = get_calculation_type()
    
    # Calculation-specific settings immediately after type selection
    if options["calculation_type"] == "OPT":
        options["optimization_settings"] = configure_optimization()
    elif options["calculation_type"] == "FREQ":
        freq_settings = get_frequency_configuration()
        options["frequency_settings"] = freq_settings
        if freq_settings.get("use_advanced"):
            options["advanced_freq_settings"] = get_advanced_frequency_settings()
    
    # Method configuration
    options = configure_method(options)
    
    # Basis set
    basis_config = select_basis_set_with_defaults(
        [], 
        options.get("method_type", "DFT"), 
        options.get("functional"),
        current_basis_type=options.get("basis_set_type", "INTERNAL"),
        current_basis=options.get("basis_set", "POB-TZVP-REV2")
    )
    options.update(basis_config)
    
    # DFT grid (right after functional and basis set for DFT calculations)
    if options["method_type"] == "DFT":
        options["dft_grid"] = configure_dft_grid(options.get("functional", ""))
    
    # Tolerances
    options["tolerances"] = configure_tolerances(calculation_type=options.get("calculation_type"))
    
    # Advanced electronic and convergence settings
    advanced_config = configure_advanced_electronic_settings(options)
    options.update(advanced_config)
    
    return options


def get_calculation_options_from_current(current_settings: Dict[str, Any], 
                                        shared_mode: bool = False,
                                        calc_type: str = None) -> Dict[str, Any]:
    """Get calculation options starting from current settings
    
    Args:
        current_settings: Current settings extracted from files
        shared_mode: If True, only ask for calculation settings to be shared
        calc_type: Pre-selected calculation type (bypasses selection menu)
        
    Returns:
        dict: Options for the calculation
    """
    options = current_settings.copy()
    
    # Determine if we're in expert mode (when calc_type is pre-selected)
    expert_mode = calc_type is not None
    
    if not shared_mode:
        # Display current settings
        print("\n" + "=" * 60)
        print("SETTINGS FROM PREVIOUS CALCULATION")
        print("=" * 60)
        print("The following settings were extracted from your optimized structure:")
        display_current_settings(current_settings, extracted=True, expert_mode=expert_mode)
        
        # Ask if user wants to keep current settings
        print("\nYou can either:")
        print("  - Use these exact settings for the new calculation")
        print("  - Modify settings (change functional, basis set, tolerances, etc.)")
        keep_settings = yes_no_prompt(
            "\nUse these exact settings for the new calculation?", "no"
        )
    else:
        keep_settings = False
    
    if not keep_settings or shared_mode:
        # In shared mode, first show current settings
        if shared_mode:
            display_current_settings(current_settings, extracted=True, expert_mode=expert_mode)
            use_extracted = yes_no_prompt(
                "\nUse these extracted settings as baseline for shared configuration?", "yes"
            )
            if not use_extracted:
                # Reset to defaults if user doesn't want extracted settings
                options = {
                    # Start with sensible defaults
                    "method": "DFT",
                    "method_type": "DFT", 
                    "functional": "HSE06-D3",
                    "basis_set": "POB-TZVP-REV2",
                    "basis_set_type": "INTERNAL",
                    "dft_grid": "XLGRID",
                    "spin_polarized": True,
                    "is_spin_polarized": True,
                    "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7},
                    "scf_settings": {
                        "method": "DIIS",
                        "maxcycle": 800,
                        "fmixing": 30,
                        "ppan": True,
                        "biposize": 110000000,
                        "exchsize": 110000000
                    }
                }
                
                # First configure symmetry when starting fresh
                print("\nConfiguring symmetry settings...")
                symmetry_config = configure_symmetry_handling(from_crystal_output=True)
                options.update(symmetry_config)
        else:
            # Using extracted settings, keep existing symmetry
            for key in ["spacegroup", "origin_setting", "dimensionality", "write_only_unique"]:
                if key in current_settings:
                    options[key] = current_settings[key]
        
        # Get calculation type (use pre-selected if provided)
        if calc_type:
            options["calculation_type"] = calc_type
            print(f"\n  Calculation type: {calc_type} (pre-selected)")
        else:
            options["calculation_type"] = get_calculation_type()
        
        # Calculation-specific settings - do this IMMEDIATELY after calc type selection
        print("\n" + "="*60)
        print(f"CALCULATION-SPECIFIC SETTINGS: {options['calculation_type']}")
        print("="*60)
        
        if options["calculation_type"] == "SP":
            # Single point calculations - remove redundant prompt
            # The user already selected SP, no need to ask if they want to configure it
            pass  # SP configuration happens with tolerances below
        elif options["calculation_type"] == "OPT":
            current_opt_type = options.get("optimization_type", "FULLOPTG")
            opt_desc = {
                "FULLOPTG": "Full geometry optimization (atoms + cell)",
                "ATOMONLY": "Atom positions only (fixed cell)",
                "CELLONLY": "Cell parameters only (fixed atoms)"
            }
            current_opt_desc = opt_desc.get(current_opt_type, current_opt_type)
            if not shared_mode or yes_no_prompt(f"\nChange optimization settings? (Current: {current_opt_desc})", "no"):
                opt_settings = configure_optimization()
                options["optimization_type"] = opt_settings.get("type", "FULLOPTG")
                options["optimization_settings"] = opt_settings
        elif options["calculation_type"] == "FREQ":
            # Check if there are existing freq settings
            current_freq_mode = "Not configured"
            if "freq_settings" in options and options["freq_settings"]:
                freq_mode = options["freq_settings"].get("freq_mode", "FREQCALC")
                if freq_mode == "ANHARM":
                    current_freq_mode = "ANHARM - Anharmonic X-H stretching"
                else:
                    # Check for template or custom settings
                    if options["freq_settings"].get("template"):
                        template_name = options["freq_settings"]["template"]
                        current_freq_mode = f"FREQCALC - {template_name} template"
                    else:
                        current_freq_mode = "FREQCALC - Harmonic frequencies"
            
            # Always configure frequency settings for FREQ calculations
            from d12_calc_freq import get_frequency_configuration
            freq_settings = get_frequency_configuration(current_settings=options, shared_mode=shared_mode)
            options["freq_settings"] = freq_settings
        
        # Ensure method_type is set from functional if not already present
        if "method_type" not in options and "functional" in options:
            functional = options.get("functional", "")
            if functional and (functional in ["RHF", "UHF", "HF3C", "HFSOL3C"] or functional.startswith("HF")):
                options["method_type"] = "HF"
            else:
                options["method_type"] = "DFT"
        
        # Method configuration
        current_functional = options.get("functional", "Not set")
        current_method = options.get("method_type", "DFT")
        if not shared_mode or yes_no_prompt(f"\nChange method/functional? (Current: {current_method}/{current_functional})", "no"):
            options = configure_method(options)
        
        # Basis set
        current_basis = options.get("basis_set", "Not set")
        current_basis_type = options.get("basis_set_type", "INTERNAL")
        if not shared_mode or yes_no_prompt(f"\nChange basis set? (Current: {current_basis} [{current_basis_type}])", "no"):
            basis_config = select_basis_set_with_defaults(
                [], 
                options.get("method_type", "DFT"), 
                options.get("functional"),
                current_basis_type=current_basis_type,
                current_basis=current_basis
            )
            options.update(basis_config)
        
        # DFT grid (right after functional and basis set for DFT calculations)
        if options.get("method_type", "DFT") == "DFT":
            current_grid = options.get("dft_grid", "XLGRID")
            if not shared_mode or yes_no_prompt(f"\nChange DFT grid? (Current: {current_grid})", "no"):
                options["dft_grid"] = configure_dft_grid_with_defaults(
                    options.get("functional", ""), 
                    current_grid=current_grid
                )
        
        # Tolerances - handle SP, OPT, and FREQ calculations with nice formatting
        if options["calculation_type"] in ["SP", "OPT", "FREQ"]:
            # For SP and OPT, give special tolerance options with convergence choices
            print("\n" + "="*60)
            if options["calculation_type"] == "SP":
                print("SCF CONVERGENCE SETTINGS FOR SP CALCULATION")
            elif options["calculation_type"] == "OPT":
                print("SCF CONVERGENCE SETTINGS FOR OPT CALCULATION")
            else:  # FREQ
                print("SCF CONVERGENCE SETTINGS FOR FREQ CALCULATION")
            print("="*60)
            
            print("\nSCF convergence tolerances control the electronic structure accuracy:")
            print("  TOLINTEG: Integral accuracy thresholds")
            print("  TOLDEE: Energy convergence threshold")
            
            print("\nChoose SCF convergence level:")
            
            if options["calculation_type"] == "FREQ":
                # For FREQ, default should be Very Tight (9s)
                print("1. Standard - TOLINTEG: 7 7 7 7 14, TOLDEE: 7")
                print("   Basic convergence (not recommended for frequencies)")
                
                print("\n2. Tight - TOLINTEG: 8 8 8 9 24, TOLDEE: 9")
                print("   Good convergence for preliminary frequency calculations")
                
                print("\n3. Very tight - TOLINTEG: 9 9 9 11 38, TOLDEE: 11")
                print("   Recommended for accurate frequencies and force constants")
                
                print("\n4. Custom - Set your own tolerances")
                
                default_choice = "3"  # Very tight for FREQ
            else:
                # SP and OPT calculations
                print("1. Standard - TOLINTEG: 7 7 7 7 14, TOLDEE: 7")
                if options["calculation_type"] == "SP":
                    print("   Good for most single point calculations")
                else:
                    print("   Good for most geometry optimizations")
                
                print("\n2. Tight - TOLINTEG: 8 8 8 9 24, TOLDEE: 9")
                if options["calculation_type"] == "SP":
                    print("   Recommended for accurate energies and properties")
                else:
                    print("   Recommended for accurate forces and final optimizations")
                
                print("\n3. Very tight - TOLINTEG: 9 9 9 11 38, TOLDEE: 11")
                print("   High precision for benchmarking or difficult cases")
                
                print("\n4. Custom - Set your own tolerances")
                
                default_choice = "1"  # Standard for SP/OPT
            
            convergence_choice = input(f"\nSelect SCF convergence level [1-4, default={default_choice}]: ").strip() or default_choice
            
            if convergence_choice == "1":
                options["tolerances"] = {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}
                print("Using standard SCF convergence")
            elif convergence_choice == "2":
                options["tolerances"] = {"TOLINTEG": "8 8 8 9 24", "TOLDEE": 9}
                if options["calculation_type"] == "SP":
                    options["use_tight_sp"] = True
                print("Using tight SCF convergence")
            elif convergence_choice == "3":
                options["tolerances"] = {"TOLINTEG": "9 9 9 11 38", "TOLDEE": 11}
                if options["calculation_type"] == "SP":
                    options["use_tight_sp"] = True
                print("Using very tight SCF convergence")
            else:
                # Custom tolerances
                options["tolerances"] = configure_tolerances(calculation_type=options["calculation_type"])
        else:
            # This else block should not be reached anymore since we handle FREQ above
            # But keep it for any future calculation types
            current_tol = options.get("tolerances", {})
            tol_info = f"TOLDEE={current_tol.get('toldee', 7)}"
            if current_tol.get("tolinteg"):
                tol_info = f"TOLINTEG={' '.join(map(str, current_tol['tolinteg']))}, {tol_info}"
            if not shared_mode or yes_no_prompt(f"\nChange tolerances? (Current: {tol_info})", "no"):
                options["tolerances"] = configure_tolerances(calculation_type=options.get("calculation_type"))
        
        # Advanced electronic and convergence settings
        # Show current advanced settings with (Default) markers where appropriate
        print("\n" + "="*60)
        print("CURRENT ADVANCED ELECTRONIC AND CONVERGENCE SETTINGS")
        print("="*60)
        
        # Check if we have actual settings or defaults
        has_spin_setting = "spin_polarized" in options or "is_spin_polarized" in options
        current_spin = "Yes" if options.get("spin_polarized", options.get("is_spin_polarized", True)) else "No"
        
        has_spinlock = "spinlock" in options
        current_spinlock = options.get("spinlock", 0)
        
        has_smearing = "use_smearing" in options or "smearing" in options
        current_smearing = "Yes" if options.get("use_smearing", False) else "No"
        
        # Get SCF settings or use defaults
        current_scf = options.get("scf_settings", {})
        has_scf = bool(current_scf)  # Check if we have any SCF settings
        
        # Extract individual SCF values with defaults
        current_scf_method = current_scf.get("method", "DIIS")
        current_maxcycle = current_scf.get("maxcycle", 800)
        current_fmixing = current_scf.get("fmixing", 30)
        current_ppan = "Yes" if current_scf.get("ppan", True) else "No"
        current_biposize = current_scf.get("biposize", 110000000)
        current_exchsize = current_scf.get("exchsize", 110000000)
        
        # Check for LEVSHIFT
        has_levshift = "levshift" in options
        levshift_info = "None"
        if has_levshift and options["levshift"]:
            levshift_info = f"{options['levshift']['shift']} Hartree for {options['levshift']['ncycles']} cycles"
        
        # Display with (Default) markers
        print(f"Spin polarization: {current_spin}{'' if has_spin_setting else ' (Default)'}")
        if current_spin == "Yes":
            spinlock_desc = '(automatic spin optimization)' if current_spinlock == 0 else ''
            print(f"SPINLOCK: {current_spinlock} {spinlock_desc}{'' if has_spinlock else ' (Default)'}")
        print(f"Fermi smearing: {current_smearing}{'' if has_smearing else ' (Default)'}")
        print(f"LEVSHIFT: {levshift_info}{'' if has_levshift else ' (Default)'}")
        print(f"SCF method: {current_scf_method}{'' if current_scf.get('method') else ' (Default)'}")
        print(f"SCF max cycles: {current_maxcycle}{'' if current_scf.get('maxcycle') else ' (Default)'}")
        print(f"FMIXING: {current_fmixing}%{'' if current_scf.get('fmixing') else ' (Default)'}")
        print(f"PPAN: {current_ppan} (parallel diagonalization){'' if current_scf.get('ppan') is not None else ' (Default)'}")
        
        # Check if BIPOSIZE/EXCHSIZE were explicitly set
        has_bipo_exch = current_scf.get('biposize') is not None or current_scf.get('exchsize') is not None
        print(f"BIPOSIZE/EXCHSIZE: {current_biposize}/{current_exchsize}{'' if has_bipo_exch else ' (Default)'}")
        print("="*60)
        
        if not shared_mode or yes_no_prompt(f"\nChange advanced electronic and convergence settings?", "no"):
            advanced_config = configure_advanced_electronic_settings(options, show_current=True)
            options.update(advanced_config)
        else:
            # Keep existing settings if not changing
            pass
    
    return options


def get_calculation_options(current_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get complete calculation options interactively (wrapper for compatibility).
    
    Args:
        current_settings: Optional current settings (not used for new configurations)
        
    Returns:
        Dictionary with all calculation settings
    """
    # For compatibility, just call get_calculation_options_new
    return get_calculation_options_new()


def get_calculation_options_new() -> Dict[str, Any]:
    """Get calculation options for new calculations (from CIF)"""
    options = {}
    
    # First, show default settings and ask if user wants to use them
    display_default_settings()
    use_defaults = yes_no_prompt("\nDo you want to use these default settings?", "yes")
    
    if use_defaults:
        # Use all default settings
        options = DEFAULT_SETTINGS.copy()
        options["optimization_settings"] = DEFAULT_OPT_SETTINGS.copy()
        options["tolerances"] = DEFAULT_TOLERANCES.copy()
        options["calculation_type"] = "OPT"  # Default to optimization
        options["optimization_type"] = "FULLOPTG"  # Default optimization type
        options["dimensionality"] = "CRYSTAL"  # Default to 3D periodic
        
        # Add default symmetry settings
        options["symmetry_handling"] = "CIF"
        options["write_full_cell"] = False
        options["validate_symmetry"] = True
        options["trigonal_axes"] = "AUTO"
        options["origin_choice"] = "AUTO"
        
        # Set additional compatibility keys
        options["method"] = "DFT"
        options["method_type"] = "DFT"
        options["dft_functional"] = options.get("functional", "HSE06")
        options["use_dispersion"] = True
        options["is_spin_polarized"] = options.get("spin_polarized", True)
        options["scf_method"] = "DIIS"
        options["scf_maxcycle"] = 800
        options["fmixing"] = 30
        
        # Set frequency settings if needed
        options["freq_settings"] = DEFAULT_FREQ_SETTINGS.copy()
        
        # Smearing is off by default
        options["smearing"] = None
        options["use_smearing"] = False
        options["smearing_width"] = 0.0
        
        # IMPORTANT: Even with defaults, always prompt for basis set selection
        # This ensures users can choose between internal/external basis sets
        print("\n" + "="*60)
        print("BASIS SET SELECTION")
        print("="*60)
        print("Even with default settings, you can choose your preferred basis set type.")
        
        # Get basis set selection with current defaults as fallbacks
        basis_config = select_basis_set_with_defaults(
            [], 
            options.get("method_type", "DFT"), 
            options.get("functional"),
            current_basis_type=options.get("basis_set_type", "INTERNAL"),
            current_basis=options.get("basis_set", "POB-TZVP-REV2")
        )
        options.update(basis_config)
    else:
        # Custom settings - go through each option in the correct order
        
        # 1. Symmetry configuration FIRST
        symmetry_config = configure_symmetry_handling()
        options.update(symmetry_config)
        
        # 2. Dimensionality selection (affects available calculations)
        dimensionality_options = [
            ("1", "CRYSTAL - 3D periodic system"),
            ("2", "SLAB - 2D periodic system (surface)"),
            ("3", "POLYMER - 1D periodic system"),
            ("4", "MOLECULE - 0D non-periodic system")
        ]
        
        dimensionality_map = {
            "1": "CRYSTAL",
            "2": "SLAB",
            "3": "POLYMER",
            "4": "MOLECULE"
        }
        
        print("\nSelect system dimensionality:")
        for key, desc in dimensionality_options:
            print(f"{key}. {desc}")
        
        dim_choice = get_user_choice("Select system dimensionality", dimensionality_options, "1")
        options["dimensionality"] = dimensionality_map[dim_choice]
        
        # 3. Calculation type
        options["calculation_type"] = get_calculation_type()
        
        # 3a. Immediately configure OPT/FREQ settings if needed
        if options["calculation_type"] == "OPT":
            # Get optimization configuration right after selecting OPT
            opt_config = configure_optimization()
            # Store both the optimization type and settings properly
            options["optimization_type"] = opt_config.get("type", "FULLOPTG")
            options["optimization_settings"] = opt_config
        elif options["calculation_type"] == "FREQ":
            freq_settings = get_frequency_configuration()
            options["frequency_settings"] = freq_settings
            if freq_settings.get("use_advanced"):
                options["advanced_freq_settings"] = get_advanced_frequency_settings()
        
        # 4. Method selection (HF/DFT)
        options = configure_method(options)
        
        # 5. Functional selection (if DFT) - already handled in configure_method
        
        # 6. Basis set selection
        basis_config = select_basis_set_with_defaults(
            [], 
            options.get("method_type", "DFT"), 
            options.get("functional"),
            current_basis_type=options.get("basis_set_type", "INTERNAL"),
            current_basis=options.get("basis_set", "POB-TZVP-REV2")
        )
        options.update(basis_config)
        
        # 7. DFT grid (right after functional and basis set for DFT calculations)
        if options["method_type"] == "DFT":
            options["dft_grid"] = configure_dft_grid(options.get("functional", ""))
        
        # 8. Tolerance settings (basic tolerances)
        options["tolerances"] = configure_tolerances(calculation_type=options.get("calculation_type"))
        
        # 9. Advanced electronic and convergence settings
        advanced_config = configure_advanced_electronic_settings(options)
        options.update(advanced_config)
    
    return options


def configure_symmetry_handling(from_crystal_output: bool = False) -> Dict[str, Any]:
    """Configure how to handle crystal symmetry
    
    Args:
        from_crystal_output: If True, we're configuring from CRYSTAL output (CRYSTALOptToD12 mode)
    """
    symmetry_config = {}
    
    # Show default symmetry settings
    print("\n" + "="*60)
    print("DEFAULT SYMMETRY SETTINGS")
    print("="*60)
    if from_crystal_output:
        print("Symmetry handling: Preserve from CRYSTAL calculation")
        print("Atom writing: Write unique atoms only (use symmetry operations)")
        print("Validate symmetry: No")
        print("Trigonal axes: AUTO - Use setting from calculation")
        print("Origin choice: ALTERNATE - Force alternate origin (ITA Origin 1)")
    else:
        print("Symmetry handling: CIF - Use symmetry as defined in the CIF file")
        print("Atom writing: Write unique atoms only (use symmetry operations)")
        print("Validate symmetry: No")
        print("Trigonal axes: AUTO - Use setting as detected in CIF")
        print("Origin choice: ALTERNATE - Force alternate origin (ITA Origin 1)")
    print("="*60)
    
    # Ask if user wants to use defaults
    use_defaults = yes_no_prompt("\nUse these default symmetry settings?", "yes")
    
    if use_defaults:
        # Set all defaults
        if from_crystal_output:
            symmetry_config["symmetry_handling"] = "PRESERVE"
            symmetry_config["write_full_cell"] = False
            symmetry_config["validate_symmetry"] = False
            symmetry_config["trigonal_axes"] = "AUTO"
            symmetry_config["origin_choice"] = "ALTERNATE"
        else:
            symmetry_config["symmetry_handling"] = "CIF"
            symmetry_config["write_full_cell"] = False
            symmetry_config["validate_symmetry"] = False
            symmetry_config["trigonal_axes"] = "AUTO"
            symmetry_config["origin_choice"] = "ALTERNATE"
    else:
        # Custom symmetry configuration
        if from_crystal_output:
            sym_options = [
                ("1", "PRESERVE - Keep symmetry from CRYSTAL calculation"),
                ("2", "SPGLIB - Re-analyze symmetry with spglib"),
                ("3", "P1 - Use P1 symmetry (all atoms explicit)")
            ]
            
            symmetry_map = {
                "1": "PRESERVE",
                "2": "SPGLIB",
                "3": "P1"
            }
        else:
            sym_options = [
                ("1", "CIF - Use symmetry as defined in the CIF file"),
                ("2", "SPGLIB - Use spglib to analyze symmetry"),
                ("3", "P1 - Use P1 symmetry (all atoms explicit)")
            ]
            
            symmetry_map = {
                "1": "CIF",
                "2": "SPGLIB",
                "3": "P1"
            }
        
        print("\nSelect symmetry handling:")
        for key, desc in sym_options:
            print(f"{key}. {desc}")
        
        choice = get_user_choice("Select symmetry handling", sym_options, "1")
        symmetry_config["symmetry_handling"] = symmetry_map[choice]
        
        # Additional symmetry questions if not using P1
        if symmetry_config["symmetry_handling"] != "P1":
            # Atom writing options
            atom_options = [
                ("1", "Write unique atoms only (use symmetry operations)"),
                ("2", "Write full unit cell (all atoms explicit)")
            ]
            
            print("\nAtom writing options:")
            for key, desc in atom_options:
                print(f"{key}. {desc}")
            
            atom_choice = get_user_choice("Atom writing options", atom_options, "1")
            symmetry_config["write_full_cell"] = (atom_choice == "2")
            
            # Validate symmetry operations
            symmetry_config["validate_symmetry"] = yes_no_prompt(
                "\nValidate symmetry operations?",
                "no"
            )
            
            # Trigonal axes preference
            if symmetry_config["symmetry_handling"] == "CIF":
                trig_options = [
                    ("1", "AUTO - Use setting as detected in CIF"),
                    ("2", "HEXAGONAL - Force hexagonal axes"),
                    ("3", "RHOMBOHEDRAL - Force rhombohedral axes")
                ]
                
                print("\nTrigonal axes preference (for rhombohedral space groups):")
                for key, desc in trig_options:
                    print(f"{key}. {desc}")
                
                trig_choice = get_user_choice("Trigonal axes preference", trig_options, "1")
                
                trig_map = {"1": "AUTO", "2": "HEXAGONAL", "3": "RHOMBOHEDRAL"}
                symmetry_config["trigonal_axes"] = trig_map[trig_choice]
            elif symmetry_config["symmetry_handling"] == "PRESERVE":
                # For PRESERVE mode, always use AUTO
                symmetry_config["trigonal_axes"] = "AUTO"
                print("\n(Trigonal axes will be preserved from the original calculation)")
            else:
                # Set default for SPGLIB method
                symmetry_config["trigonal_axes"] = "AUTO"
            
            # High symmetry space group origins
            print("\nFor space groups with multiple origins (e.g., 227-Fd-3m):")
            print("  Standard (ITA Origin 2, CRYSTAL '0 0 0'): Si at (1/8,1/8,1/8), 36 operators")
            print("  Alternate (ITA Origin 1, CRYSTAL '0 0 1'): Si at (0,0,0), 24 operators")
            
            origin_options = [
                ("1", "AUTO - Use origin as detected in CIF"),
                ("2", "STANDARD - Force standard origin (ITA Origin 2)"),
                ("3", "ALTERNATE - Force alternate origin (ITA Origin 1)")
            ]
            
            print("\nSelect origin choice:")
            for key, desc in origin_options:
                print(f"{key}. {desc}")
            
            origin_choice = get_user_choice("Select origin choice", origin_options, "1")
            
            origin_map = {"1": "AUTO", "2": "STANDARD", "3": "ALTERNATE"}
            symmetry_config["origin_choice"] = origin_map[origin_choice]
        else:
            # P1 doesn't need these settings
            symmetry_config["write_full_cell"] = True  # P1 always writes full cell
            symmetry_config["validate_symmetry"] = False
            symmetry_config["trigonal_axes"] = "AUTO"
            symmetry_config["origin_choice"] = "AUTO"
    
    return symmetry_config


def configure_print_options() -> List[str]:
    """Configure CRYSTAL print options"""
    print("\nSelect print options (space-separated numbers):")
    
    for key, desc in PRINT_OPTIONS.items():
        print(f"{key}. {desc}")
    
    choices = input("Enter choices (e.g., 1 3 5): ").strip().split()
    
    # Validate choices
    valid_choices = []
    for choice in choices:
        if choice in PRINT_OPTIONS:
            valid_choices.append(choice)
    
    return valid_choices


def save_options_to_file(options: Dict[str, Any], filename: str):
    """Save calculation options to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(options, f, indent=2)
        print(f"\nOptions saved to {filename}")
    except Exception as e:
        print(f"Error saving options: {e}")


def load_options_from_file(filename: str) -> Optional[Dict[str, Any]]:
    """Load calculation options from JSON file"""
    try:
        with open(filename, 'r') as f:
            options = json.load(f)
        print(f"\nOptions loaded from {filename}")
        return options
    except Exception as e:
        print(f"Error loading options: {e}")
        return None


def interactive_d12_configuration(mode: str = "new", 
                                current_settings: Optional[Dict[str, Any]] = None,
                                shared_mode: bool = False) -> Dict[str, Any]:
    """Main entry point for interactive D12 configuration
    
    Args:
        mode: Configuration mode - "new" for CIF->D12, "current" for OUT->D12
        current_settings: Current settings if mode is "current"
        shared_mode: If True, configure shared settings for batch processing
        
    Returns:
        dict: Complete configuration options
    """
    print("\n" + "="*60)
    print("CRYSTAL D12 INPUT FILE CONFIGURATION")
    print("="*60)
    
    if mode == "new":
        if shared_mode:
            print("\nConfiguring shared settings for batch processing...")
            options = get_shared_calculation_options()
        else:
            options = get_calculation_options_new()
    else:
        if current_settings is None:
            raise ValueError("Current settings required for 'current' mode")
        options = get_calculation_options_from_current(current_settings, shared_mode)
    
    # Add additional configuration that's common to both modes
    # Note: Symmetry handling is now configured within get_calculation_options_new()
    
    # Print options
    if yes_no_prompt("\nConfigure print options?", "no"):
        options["print_options"] = configure_print_options()
    
    # Ghost atoms (if needed)
    if yes_no_prompt("\nAdd ghost atoms?", "no"):
        options["ghost_atoms"] = configure_ghost_atoms()
    
    # External pressure
    if yes_no_prompt("\nApply external pressure?", "no"):
        options["external_pressure"] = configure_external_pressure()
    
    # Save options
    if yes_no_prompt("\nSave these options to file?", "no"):
        filename = input("Enter filename (default: d12_options.json): ").strip()
        if not filename:
            filename = "d12_options.json"
        save_options_to_file(options, filename)
    
    return options


def configure_ghost_atoms() -> Dict[str, Any]:
    """Configure ghost atom settings"""
    ghost_config = {}
    
    ghost_options = [
        ("1", "Add ghost atoms above surface"),
        ("2", "Add ghost atoms at specific positions"),
        ("3", "Convert specific atoms to ghosts")
    ]
    
    print("\nGhost atom configuration:")
    for key, desc in ghost_options:
        print(f"{key}. {desc}")
    
    choice = get_user_choice("Ghost atom configuration", ghost_options, "1")
    
    if choice == "1":
        ghost_config["type"] = "surface"
        distance = safe_float(
            input("Distance above surface (Angstrom, default: 2.0): ").strip(),
            2.0
        )
        ghost_config["distance"] = distance
        
    elif choice == "2":
        ghost_config["type"] = "positions"
        positions = []
        print("Enter ghost atom positions (empty line to finish):")
        while True:
            pos_str = input("Position (x y z): ").strip()
            if not pos_str:
                break
            try:
                x, y, z = map(float, pos_str.split())
                positions.append([x, y, z])
            except:
                print("Invalid format. Use: x y z")
        ghost_config["positions"] = positions
        
    else:  # choice == "3"
        ghost_config["type"] = "convert"
        atom_nums = input("Atom numbers to convert (space-separated): ").strip()
        try:
            ghost_config["atom_numbers"] = [int(x) for x in atom_nums.split()]
        except:
            print("Invalid atom numbers, skipping ghost atom configuration")
            return {}
    
    return ghost_config


def configure_external_pressure() -> Dict[str, float]:
    """Configure external pressure settings"""
    print("\nExternal pressure configuration:")
    
    pressure = safe_float(
        input("Pressure in GPa (default: 0.0): ").strip(),
        0.0
    )
    
    return {"pressure": pressure}


def configure_advanced_electronic_settings(options: Dict[str, Any], show_current: bool = False) -> Dict[str, Any]:
    """Configure advanced electronic and convergence settings
    
    Groups together:
    - Spin polarization and SPINLOCK
    - Fermi smearing
    - LEVSHIFT
    - SCF settings (MAXCYCLE, FMIXING, etc.)
    
    Args:
        options: Current configuration options
        show_current: If True, user already saw current settings, just ask to use them
        
    Returns:
        Dictionary with advanced settings
    """
    advanced_config = {}
    
    if show_current:
        # User already saw current settings, just ask if they want to use them
        # Don't show defaults or current again
        use_current = yes_no_prompt("\nUse these current advanced settings?", "yes")
    else:
        # Show defaults (old behavior)
        print("\n" + "="*60)
        print("DEFAULT ADVANCED ELECTRONIC AND CONVERGENCE SETTINGS")
        print("="*60)
        print("Spin polarization: Yes")
        print("SPINLOCK: 0 (automatic spin optimization)")
        print("Fermi smearing: No")
        print("LEVSHIFT: None")
        print("SCF method: DIIS")
        print("SCF max cycles: 800")
        print("FMIXING: 30%")
        print("PPAN: Yes (parallel diagonalization)")
        print("BIPOSIZE/EXCHSIZE: 110000000/110000000")
        print("="*60)
        
        use_current = yes_no_prompt("\nUse these default advanced settings?", "yes")
    
    if use_current and show_current:
        # Keep all current settings
        # Extract existing advanced settings from options
        advanced_config["spin_polarized"] = options.get("spin_polarized", True)
        advanced_config["is_spin_polarized"] = options.get("is_spin_polarized", True)
        advanced_config["spinlock"] = options.get("spinlock", 0)
        advanced_config["smearing"] = options.get("smearing")
        advanced_config["use_smearing"] = options.get("use_smearing", False)
        advanced_config["smearing_width"] = options.get("smearing_width", 0.0)
        
        # Get SCF settings
        current_scf = options.get("scf_settings", {})
        advanced_config["scf_settings"] = {
            "method": current_scf.get("method", "DIIS"),
            "maxcycle": current_scf.get("maxcycle", 800),
            "fmixing": current_scf.get("fmixing", 30),
            "ppan": current_scf.get("ppan", True),
            "biposize": current_scf.get("biposize", 110000000),
            "exchsize": current_scf.get("exchsize", 110000000)
        }
        
        # Copy any additional SCF settings
        if current_scf.get("histdiis"):
            advanced_config["scf_settings"]["histdiis"] = current_scf["histdiis"]
        if current_scf.get("broyden_w0"):
            advanced_config["scf_settings"]["broyden_w0"] = current_scf["broyden_w0"]
        
        # Copy LEVSHIFT if present
        if options.get("levshift"):
            advanced_config["levshift"] = options["levshift"]
        
        # Compatibility keys
        advanced_config["scf_method"] = advanced_config["scf_settings"]["method"]
        advanced_config["scf_maxcycle"] = advanced_config["scf_settings"]["maxcycle"]
        advanced_config["fmixing"] = advanced_config["scf_settings"]["fmixing"]
    elif use_current and not show_current:
        # Set all defaults
        advanced_config["spin_polarized"] = True
        advanced_config["is_spin_polarized"] = True
        advanced_config["spinlock"] = 0
        advanced_config["smearing"] = None
        advanced_config["use_smearing"] = False
        advanced_config["smearing_width"] = 0.0
        advanced_config["scf_settings"] = {
            "method": "DIIS",
            "maxcycle": 800,
            "fmixing": 30,
            "ppan": True,
            "biposize": 110000000,
            "exchsize": 110000000
        }
        advanced_config["scf_method"] = "DIIS"
        advanced_config["scf_maxcycle"] = 800
        advanced_config["fmixing"] = 30
    else:
        # Custom advanced settings
        # Use wrapper functions that show current settings as defaults
        spin_config = configure_spin_polarization_with_defaults(options)
        advanced_config.update(spin_config)
        advanced_config["is_spin_polarized"] = spin_config.get("spin_polarized", True)
        
        # Smearing (if metallic and periodic)
        # Get dimensionality from options or use default
        dimensionality = options.get("dimensionality", "CRYSTAL")
        if dimensionality in ["SLAB", "CRYSTAL", "POLYMER"]:
            smear_config = configure_smearing_with_defaults(options)
            advanced_config["smearing"] = smear_config.get("smearing_width", 0.0)
            advanced_config["use_smearing"] = smear_config.get("use_smearing", False)
            advanced_config["smearing_width"] = smear_config.get("smearing_width", 0.0)
        else:
            # Molecules don't need smearing
            advanced_config["smearing"] = None
            advanced_config["use_smearing"] = False
            advanced_config["smearing_width"] = 0.0
        
        # LEVSHIFT
        print("\n=== LEVEL SHIFTING ===")
        current_levshift = options.get("levshift", {})
        has_levshift = bool(current_levshift)
        default_levshift = "yes" if has_levshift else "no"
        
        use_levshift = yes_no_prompt("Use level shifting (for difficult SCF convergence)?", default_levshift)
        if use_levshift:
            current_shift = current_levshift.get("shift", 5.0) if has_levshift else 5.0
            current_ncycles = current_levshift.get("ncycles", 30) if has_levshift else 30
            
            levshift_value = safe_float(
                input(f"LEVSHIFT value (Hartree) [{current_shift}]: ").strip(),
                current_shift
            )
            ncycles = safe_int(
                input(f"Number of cycles to apply LEVSHIFT [{current_ncycles}]: ").strip(),
                current_ncycles
            )
            advanced_config["levshift"] = {
                "shift": levshift_value,
                "ncycles": ncycles
            }
        
        # SCF settings - use wrapper to show current settings
        scf_config = configure_scf_settings_with_defaults(options.get("scf_settings", {}))
        # Advanced SCF options if using DIIS
        if scf_config.get("method") == "DIIS":
            current_histdiis = options.get("scf_settings", {}).get("histdiis")
            if current_histdiis:
                default_hist = "yes"
                print(f"\nCurrent HISTDIIS: {current_histdiis}")
            else:
                default_hist = "no"
            
            if yes_no_prompt("\nUse HISTDIIS (keep more SCF history)?", default_hist):
                hist_default = current_histdiis if current_histdiis else 100
                hist_size = safe_int(
                    input(f"HISTDIIS size [{hist_default}]: ").strip(),
                    hist_default
                )
                scf_config["histdiis"] = hist_size
        
        # Additional SCF options - already handled by configure_scf_settings_with_defaults
        # The wrapper function already asked about PPAN and BIPOSIZE/EXCHSIZE
        
        advanced_config["scf_settings"] = scf_config
        # Add compatibility keys
        advanced_config["scf_method"] = scf_config.get("method", "DIIS")
        advanced_config["scf_maxcycle"] = scf_config.get("maxcycle", 800)
        advanced_config["fmixing"] = scf_config.get("fmixing", 30)
    
    return advanced_config


def display_configuration_summary(options: Dict[str, Any]):
    """Display a summary of the configured options"""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    # Basic settings
    print(f"\nCalculation Type: {options.get('calculation_type', 'Unknown')}")
    print(f"Method: {options.get('method_type', 'Unknown')}")
    if options.get('functional'):
        print(f"Functional: {options['functional']}")
    print(f"Basis Set: {options.get('basis_set', 'Unknown')}")
    
    # SCF settings
    if 'shrink' in options:
        print(f"\nSHRINK: {options['shrink'][0]} {options['shrink'][1]}")
    if options.get('method_type') == 'DFT' and 'dft_grid' in options:
        print(f"DFT Grid: {options['dft_grid']}")
    
    # Tolerances
    if 'tolerances' in options:
        tol = options['tolerances']
        print(f"\nTOLINTEG: {' '.join(map(str, tol.get('tolinteg', [])))}")
        print(f"TOLDEE: {tol.get('toldee', 7)}")
    
    # Additional settings
    if 'dispersion' in options and options['dispersion']:
        print(f"\nDispersion: {options['dispersion']}")
    if 'spin_polarized' in options and options['spin_polarized']:
        print(f"Spin Polarized: Yes")
    
    # Calculation-specific settings
    if options.get('calculation_type') == 'OPT' and 'optimization_settings' in options:
        opt = options['optimization_settings']
        print(f"\nOptimization Type: {opt.get('type', 'FULLOPTG')}")
        print(f"Max Cycles: {opt.get('maxcycle', 800)}")
        print(f"Convergence: {opt.get('convergence', 'TIGHTOPT')}")
    
    elif options.get('calculation_type') == 'FREQ' and 'frequency_settings' in options:
        freq = options['frequency_settings']
        print(f"\nFrequency Type: {freq.get('type', 'FREQCALC')}")
        if freq.get('type') == 'ANHARM':
            print(f"Anharmonic Mode: {freq.get('anharm_mode', 'AUTOMATIC')}")
    
    print("="*60)