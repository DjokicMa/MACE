#!/usr/bin/env python3
"""
Interactive configuration functions for D12 file creation.
Consolidates user interaction and configuration from NewCifToD12.py and CRYSTALOptToD12.py.
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
    print("\n" + "="*70)
    print("DEFAULT RECOMMENDED SETTINGS FOR CRYSTAL23")
    print("="*70)
    
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
    print("0.00003")
    print("TOLDEX")
    print("0.00012")
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


def display_current_settings(settings: Dict[str, Any], extracted: bool = False):
    """Display current calculation settings from parsed files
    
    Args:
        settings: Dictionary of settings to display
        extracted: If True, shows as "EXTRACTED CALCULATION SETTINGS"
    """
    print("\n" + "="*70)
    if extracted:
        print("EXTRACTED CALCULATION SETTINGS")
    else:
        print("CURRENT CALCULATION SETTINGS")
    print("="*70)
    
    # Structure information (only for extracted settings)
    if extracted:
        print(f"\nDimensionality: {settings.get('dimensionality', 'CRYSTAL')}")
        print(f"Space group: {settings.get('spacegroup', 'N/A')}")
        print(f"Origin setting: {settings.get('origin_setting', '0 0 0')}")
    
    # Method settings
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
    print(f"Spin polarized: {settings.get('spin_polarized', False)}")
    
    if settings.get("dft_grid"):
        print(f"DFT grid: {settings.get('dft_grid')}")
    elif settings.get("functional") and settings["functional"] not in ["HF", "RHF", "UHF"]:
        print(f"DFT grid: DEFAULT")
    
    if settings.get("tolerances"):
        tolinteg = settings['tolerances'].get('TOLINTEG')
        toldee = settings['tolerances'].get('TOLDEE')
        
        # Handle TOLINTEG as either string or list
        if isinstance(tolinteg, list):
            tolinteg_str = ' '.join(map(str, tolinteg))
        else:
            tolinteg_str = str(tolinteg) if tolinteg else 'N/A'
            
        print(f"Tolerances: TOLINTEG={tolinteg_str}, TOLDEE={toldee if toldee else 'N/A'}")
    
    if settings.get("scf_settings"):
        print(f"SCF method: {settings['scf_settings'].get('method', 'DIIS')}")
        print(f"SCF max cycles: {settings['scf_settings'].get('maxcycle', 800)}")
        print(f"FMIXING: {settings['scf_settings'].get('fmixing', 30)}%")
    
    if settings.get("k_points"):
        # Handle k_points as either string or formatted value
        k_points = settings['k_points']
        if isinstance(k_points, str) and k_points.startswith("SCFDIR"):
            print(f"K-points: {k_points}")
        else:
            print(f"K-points: {k_points}")
    
    if settings.get("smearing"):
        print(f"Fermi smearing: Yes (width={settings.get('smearing_width', 0.01)})")
    
    print("="*70)


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


def configure_method(options: Dict[str, Any]) -> Dict[str, Any]:
    """Configure method type and functional"""
    method_options = [
        ("1", "Hartree-Fock (HF)"),
        ("2", "Density Functional Theory (DFT)")
    ]
    
    print("\nSelect method type:")
    for key, desc in method_options:
        print(f"{key}. {desc}")
    
    method_choice = get_user_choice("Select method type", method_options, "2")
    
    if method_choice == "1":
        options["method"] = "HF"  # Added for compatibility
        options["method_type"] = "HF"
        # Select HF variant
        hf_methods = FUNCTIONAL_CATEGORIES["HF"]["functionals"]
        hf_options = []
        
        print("\nSelect Hartree-Fock method:")
        for i, method in enumerate(hf_methods, 1):
            desc = FUNCTIONAL_CATEGORIES["HF"]["descriptions"][method]
            print(f"{i}: {method} - {desc}")
            hf_options.append((str(i), f"{method} - {desc}"))
        
        hf_choice = get_user_choice("Select Hartree-Fock method", hf_options, "1")
        options["functional"] = hf_methods[int(hf_choice) - 1]
        options["hf_method"] = options["functional"]  # Added for compatibility
    else:
        options["method"] = "DFT"  # Added for compatibility
        options["method_type"] = "DFT"
        
        # Select functional category (excluding HF) - ordered to match NewCifToD12.py
        ordered_categories = ["LDA", "GGA", "HYBRID", "MGGA", "3C"]
        cat_options = []
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
        
        # Default to HYBRID (3)
        cat_choice = get_user_choice("Select functional category", cat_options, "3")
        
        category = ordered_categories[int(cat_choice) - 1]
        functionals = FUNCTIONAL_CATEGORIES[category]["functionals"]
        
        # Select specific functional
        print(f"\nAvailable {FUNCTIONAL_CATEGORIES[category]['name']}:")
        func_options = []
        default_func = None
        for i, func in enumerate(functionals, 1):
            desc = FUNCTIONAL_CATEGORIES[category]["descriptions"].get(func, "")
            # Check if functional supports D3
            d3_marker = " [D3✓]" if func in D3_FUNCTIONALS else ""
            full_desc = f"{func} - {desc}{d3_marker}"
            print(f"{i}: {full_desc}")
            func_options.append((str(i), func))
            # Set HSE06 as default for hybrid functionals
            if func == "HSE06" and category == "HYBRID":
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
    basis_config = select_basis_set([], options.get("method_type", "DFT"), options.get("functional"))
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
                                        shared_mode: bool = False) -> Dict[str, Any]:
    """Get calculation options starting from current settings
    
    Args:
        current_settings: Current settings extracted from files
        shared_mode: If True, only ask for calculation settings to be shared
        
    Returns:
        dict: Options for the calculation
    """
    options = current_settings.copy()
    
    if not shared_mode:
        # Display current settings
        display_current_settings(current_settings, extracted=True)
        
        # Ask if user wants to keep current settings
        keep_settings = yes_no_prompt(
            "\nKeep these settings for the new calculation?", "yes"
        )
    else:
        keep_settings = False
    
    if not keep_settings or shared_mode:
        # In shared mode, first show current settings
        if shared_mode:
            display_current_settings(current_settings, extracted=True)
            use_extracted = yes_no_prompt(
                "\nUse these extracted settings as baseline for shared configuration?", "yes"
            )
            if not use_extracted:
                # Reset to minimal settings if user doesn't want extracted settings
                options = {}
        
        # Get calculation type
        options["calculation_type"] = get_calculation_type()
        
        # Calculation-specific settings - do this IMMEDIATELY after calc type selection
        print("\n" + "="*50)
        print(f"CALCULATION-SPECIFIC SETTINGS: {options['calculation_type']}")
        print("="*50)
        
        if options["calculation_type"] == "SP":
            # Single point calculations have minimal specific settings
            current_sp = "Standard single point energy calculation"
            if not shared_mode or yes_no_prompt(f"\nConfigure single point settings? (Current: {current_sp})", "no"):
                sp_settings = configure_single_point()
                options.update(sp_settings)
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
                options["optimization_type"] = opt_settings.get("optimization_type", "FULLOPTG")
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
            if functional in ["RHF", "UHF", "HF3C", "HFSOL3C"] or functional.startswith("HF"):
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
            basis_config = select_basis_set([], options.get("method_type", "DFT"), options.get("functional"))
            options.update(basis_config)
        
        # DFT grid (right after functional and basis set for DFT calculations)
        if options.get("method_type", "DFT") == "DFT":
            current_grid = options.get("dft_grid", "XLGRID")
            if not shared_mode or yes_no_prompt(f"\nChange DFT grid? (Current: {current_grid})", "no"):
                options["dft_grid"] = configure_dft_grid(options.get("functional", ""))
        
        # SCF settings
        current_scf = options.get("scf_settings", {})
        scf_info = f"MAXCYCLE={current_scf.get('max_cycles', 800)}, FMIXING={current_scf.get('fmixing', 30)}"
        if not shared_mode or yes_no_prompt(f"\nChange SCF settings? (Current: {scf_info})", "no"):
            scf_config = configure_scf_settings()
            options["scf_settings"] = scf_config
        
        # Tolerances
        current_tol = options.get("tolerances", {})
        tol_info = f"TOLDEE={current_tol.get('toldee', 7)}"
        if current_tol.get("tolinteg"):
            tol_info = f"TOLINTEG={' '.join(map(str, current_tol['tolinteg']))}, {tol_info}"
        if not shared_mode or yes_no_prompt(f"\nChange tolerances? (Current: {tol_info})", "no"):
            options["tolerances"] = configure_tolerances(calculation_type=options.get("calculation_type"))
        
        # Dispersion correction - only ask if functional was not changed
        # (D3 is already handled when changing functional)
        if options.get("method_type", "DFT") == "DFT" and not shared_mode:
            # Don't ask about dispersion if we just configured the method
            pass
        
        # Additional options
        current_spin = "Enabled" if options.get("spin_polarized", False) else "Disabled"
        if not shared_mode or yes_no_prompt(f"\nChange spin polarization? (Current: {current_spin})", "no"):
            spin_config = configure_spin_polarization()
            options["spin_polarized"] = spin_config.get("spin_polarized", False)
            if "spinlock" in spin_config:
                options["spinlock"] = spin_config["spinlock"]
        else:
            # Keep existing spin settings if not changing
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
        basis_config = select_basis_set([], options.get("method_type", "DFT"), options.get("functional"))
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


def configure_symmetry_handling() -> Dict[str, Any]:
    """Configure how to handle crystal symmetry"""
    symmetry_config = {}
    
    # Show default symmetry settings
    print("\n" + "="*70)
    print("DEFAULT SYMMETRY SETTINGS")
    print("="*70)
    print("Symmetry handling: CIF - Use symmetry as defined in the CIF file")
    print("Atom writing: Write unique atoms only (use symmetry operations)")
    print("Validate symmetry: No")
    print("Trigonal axes: AUTO - Use setting as detected in CIF")
    print("Origin choice: ALTERNATE - Force alternate origin (ITA Origin 1)")
    print("="*70)
    
    # Ask if user wants to use defaults
    use_defaults = yes_no_prompt("\nUse these default symmetry settings?", "yes")
    
    if use_defaults:
        # Set all defaults
        symmetry_config["symmetry_handling"] = "CIF"
        symmetry_config["write_full_cell"] = False
        symmetry_config["validate_symmetry"] = False
        symmetry_config["trigonal_axes"] = "AUTO"
        symmetry_config["origin_choice"] = "ALTERNATE"
    else:
        # Custom symmetry configuration
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
            else:
                # Set default for non-CIF methods
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


def configure_advanced_electronic_settings(options: Dict[str, Any]) -> Dict[str, Any]:
    """Configure advanced electronic and convergence settings
    
    Groups together:
    - Spin polarization and SPINLOCK
    - Fermi smearing
    - LEVSHIFT
    - SCF settings (MAXCYCLE, FMIXING, etc.)
    
    Args:
        options: Current configuration options
        
    Returns:
        Dictionary with advanced settings
    """
    advanced_config = {}
    
    # Show default advanced settings
    print("\n" + "="*70)
    print("DEFAULT ADVANCED ELECTRONIC AND CONVERGENCE SETTINGS")
    print("="*70)
    print("Spin polarization: Yes")
    print("SPINLOCK: 0 (automatic spin optimization)")
    print("Fermi smearing: No")
    print("LEVSHIFT: None")
    print("SCF method: DIIS")
    print("SCF max cycles: 800")
    print("FMIXING: 30%")
    print("="*70)
    
    # Ask if user wants to use defaults
    use_defaults = yes_no_prompt("\nUse these default advanced settings?", "yes")
    
    if use_defaults:
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
            "fmixing": 30
        }
        advanced_config["scf_method"] = "DIIS"
        advanced_config["scf_maxcycle"] = 800
        advanced_config["fmixing"] = 30
    else:
        # Custom advanced settings
        print("\n=== SPIN POLARIZATION ===")
        use_spin = yes_no_prompt("Use spin-polarized calculation?", "yes")
        advanced_config["spin_polarized"] = use_spin
        advanced_config["is_spin_polarized"] = use_spin
        
        if use_spin:
            print("\nSPINLOCK options (number of unpaired electrons, nα-nβ):")
            print("  - Enter 0 for automatic spin optimization")
            print("  - Enter positive integer for fixed spin multiplicity (e.g., 2 for triplet)")
            print("  - Enter -1 for antiferromagnetic initial guess")
            spinlock_input = input("SPINLOCK value (nα-nβ) [0]: ").strip()
            if spinlock_input:
                try:
                    advanced_config["spinlock"] = int(spinlock_input)
                except ValueError:
                    print("Invalid input, using default value of 0")
                    advanced_config["spinlock"] = 0
            else:
                advanced_config["spinlock"] = 0
        
        # Smearing (if metallic and periodic)
        if options.get("dimensionality") in ["SLAB", "CRYSTAL", "POLYMER"]:
            print("\n=== FERMI SURFACE SMEARING ===")
            use_smearing = yes_no_prompt("Use Fermi surface smearing for metallic systems?", "no")
            if use_smearing:
                smear_config = configure_smearing("metal")
                advanced_config["smearing"] = smear_config
                advanced_config["use_smearing"] = smear_config.get("enabled", True)
                advanced_config["smearing_width"] = smear_config.get("width", 0.005)
            else:
                advanced_config["smearing"] = None
                advanced_config["use_smearing"] = False
                advanced_config["smearing_width"] = 0.0
        else:
            # Molecules don't need smearing
            advanced_config["smearing"] = None
            advanced_config["use_smearing"] = False
            advanced_config["smearing_width"] = 0.0
        
        # LEVSHIFT
        print("\n=== LEVEL SHIFTING ===")
        use_levshift = yes_no_prompt("Use level shifting (for difficult SCF convergence)?", "no")
        if use_levshift:
            levshift_value = safe_float(
                input("LEVSHIFT value (Hartree) [5.0]: ").strip(),
                5.0
            )
            ncycles = safe_int(
                input("Number of cycles to apply LEVSHIFT [30]: ").strip(),
                30
            )
            advanced_config["levshift"] = {
                "shift": levshift_value,
                "ncycles": ncycles
            }
        
        # SCF settings
        print("\n=== SCF CONVERGENCE SETTINGS ===")
        print("SCF method options:")
        print("1: DIIS (Direct Inversion in Iterative Subspace) - fastest")
        print("2: DIIS with HISTDIIS (keep more history)")
        print("3: BROYDEN (quasi-Newton method)")
        print("4: DIIS + BROYDEN (hybrid approach)")
        print("5: Simple mixing only (no acceleration)")
        
        scf_method_choice = input("Select SCF method (1-5) [1]: ").strip() or "1"
        
        scf_config = {}
        if scf_method_choice == "1":
            scf_config["method"] = "DIIS"
        elif scf_method_choice == "2":
            scf_config["method"] = "DIIS"
            hist_size = safe_int(
                input("HISTDIIS size [100]: ").strip(),
                100
            )
            scf_config["histdiis"] = hist_size
        elif scf_method_choice == "3":
            scf_config["method"] = "BROYDEN"
            scf_config["broyden_w0"] = safe_float(
                input("BROYDEN initial mixing [0.1]: ").strip(),
                0.1
            )
        elif scf_method_choice == "4":
            scf_config["method"] = "DIIS+BROYDEN"
        else:
            scf_config["method"] = "NONE"
        
        # MAXCYCLE
        maxcycle = safe_int(
            input("\nSCF max cycles [800]: ").strip(),
            800
        )
        scf_config["maxcycle"] = maxcycle
        
        # FMIXING
        fmixing = safe_int(
            input("FMIXING percentage (0-100) [30]: ").strip(),
            30
        )
        scf_config["fmixing"] = fmixing
        
        # Additional SCF options
        print("\nAdditional SCF options:")
        if yes_no_prompt("Use PPAN (parallel diagonalization)?", "yes"):
            scf_config["ppan"] = True
        
        if yes_no_prompt("Set BIPOSIZE/EXCHSIZE for large systems?", "no"):
            biposize = safe_int(
                input("BIPOSIZE value [110000000]: ").strip(),
                110000000
            )
            exchsize = safe_int(
                input("EXCHSIZE value [110000000]: ").strip(),
                110000000
            )
            scf_config["biposize"] = biposize
            scf_config["exchsize"] = exchsize
        
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
        print(f"Max Cycles: {opt.get('maxcycle', 200)}")
        print(f"Convergence: {opt.get('convergence', 'TIGHTOPT')}")
    
    elif options.get('calculation_type') == 'FREQ' and 'frequency_settings' in options:
        freq = options['frequency_settings']
        print(f"\nFrequency Type: {freq.get('type', 'FREQCALC')}")
        if freq.get('type') == 'ANHARM':
            print(f"Anharmonic Mode: {freq.get('anharm_mode', 'AUTOMATIC')}")
    
    print("="*60)