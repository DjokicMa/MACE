#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIF to D12 Converter for CRYSTAL23
----------------------------------
This script converts CIF files to D12 input files for CRYSTAL23 with multiple options
for calculation type, basis sets, functionals, and other computational parameters.

DESCRIPTION:
    This tool automates the process of converting CIF files to D12 input files for CRYSTAL23
    quantum chemical calculations. It allows customization of calculation types (single point,
    geometry optimization, frequency), basis sets, DFT functionals, and many other parameters.

REQUIRED PACKAGES:
    - numpy
    - ase (Atomic Simulation Environment)
    - spglib (Optional, for symmetry detection)

INSTALLATION:
    Using conda:
        conda install -c conda-forge numpy ase spglib

    Using pip:
        pip install numpy ase spglib

USAGE:
    1. Basic usage (interactive mode):
       python NewCifToD12.py --cif_dir /path/to/cif/files

    2. Save options for batch processing:
       python NewCifToD12.py --save_options --options_file my_settings.json

    3. Run in batch mode with saved options:
       python NewCifToD12.py --batch --options_file my_settings.json --cif_dir /path/to/cif/files

    4. Specify output directory:
       python NewCifToD12.py --cif_dir /path/to/cif/files --output_dir /path/to/output

CONFIGURATION:
    ** IMPORTANT: Before running, modify the path constants at the top of this script **

    DEFAULT_DZ_PATH = "./full.basis.doublezeta/"  # Path to DZVP-REV2 external basis set
    DEFAULT_TZ_PATH = "./full.basis.triplezeta/"  # Path to TZVP-REV2 external basis set

    Update these paths to point to your basis set directories on your system.

AUTHOR:
    Original script by Marcus Djokic
    Enhanced with comprehensive features by Marcus Djokic with AI assistance
"""

import os
import sys
import glob
import argparse
import numpy as np
from ase.io import read
import json

# Path constants for external basis sets - MODIFY THESE TO MATCH YOUR SYSTEM
# Note: These are the REV2 versions of the basis sets
DEFAULT_DZ_PATH = "./full.basis.doublezeta/"  # DZVP-REV2 external basis set directory
DEFAULT_TZ_PATH = "./full.basis.triplezeta/"  # TZVP-REV2 external basis set directory

# Import shared D12 creation utilities
try:
    from d12creation import *
except ImportError:
    print("Error: Could not import d12creation module.")
    print(
        "Please ensure d12creation.py is in the same directory or in your Python path."
    )
    sys.exit(1)

# Try to import spglib for symmetry operations
try:
    import spglib

    SPGLIB_AVAILABLE = True
except ImportError:
    SPGLIB_AVAILABLE = False
    print("Warning: spglib not found. Symmetry reduction features will be limited.")
    print("Install spglib for full symmetry functionality: pip install spglib")


def read_basis_file(basis_dir, atomic_number):
    """
    Read a basis set file for a given element

    Args:
        basis_dir (str): Directory containing basis set files
        atomic_number (int): Element atomic number

    Returns:
        str: Content of the basis set file
    """
    try:
        with open(os.path.join(basis_dir, str(atomic_number)), "r") as f:
            return f.read()
    except FileNotFoundError:
        print(
            f"Warning: Basis set file for element {atomic_number} not found in {basis_dir}"
        )
        return ""


def parse_cif(cif_file):
    """
    Parse a CIF file to extract crystallographic data

    Args:
        cif_file (str): Path to the CIF file

    Returns:
        dict: Extracted crystallographic data
    """
    try:
        # Try reading with ASE first
        atoms = read(cif_file, format="cif")
        cell_params = atoms.get_cell_lengths_and_angles()
        a, b, c = cell_params[:3]
        alpha, beta, gamma = cell_params[3:]

        # Get atomic positions and symbols
        positions = atoms.get_scaled_positions()
        symbols = atoms.get_chemical_symbols()

        # Get space group number if available
        spacegroup = None
        cif_symmetry_name = None

        # First try to get from ASE info
        if hasattr(atoms, "info") and "spacegroup" in atoms.info:
            spacegroup = atoms.info["spacegroup"].no

        # If not available, try to parse from CIF file directly
        if spacegroup is None:
            with open(cif_file, "r") as f:
                cif_content = f.read()

                # Look for International Tables number
                import re

                sg_match = re.search(
                    r"_symmetry_Int_Tables_number\s+(\d+)", cif_content
                )
                if sg_match:
                    spacegroup = int(sg_match.group(1))

                # Also get the H-M symbol if available for reference
                hm_match = re.search(
                    r'_symmetry_space_group_name_H-M\s+[\'"](.*?)[\'"]', cif_content
                )
                if hm_match:
                    cif_symmetry_name = hm_match.group(1)

        # If still not found, prompt user
        if spacegroup is None:
            print(f"Warning: Space group not found in {cif_file}")
            if cif_symmetry_name:
                print(f"Found Hermann-Mauguin symbol: {cif_symmetry_name}")
            spacegroup = int(input("Please enter the space group number: "))

        # Convert symbols to atomic numbers
        atomic_numbers = [ELEMENT_SYMBOLS.get(sym, 0) for sym in symbols]

        return {
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "spacegroup": spacegroup,
            "cif_symmetry_name": cif_symmetry_name,
            "atomic_numbers": atomic_numbers,
            "symbols": symbols,
            "positions": positions,
            "name": os.path.basename(cif_file).replace(".cif", ""),
        }

    except Exception as e:
        # If ASE fails, use manual parsing
        print(f"ASE parsing failed: {e}")
        print("Falling back to manual parsing...")

        with open(cif_file, "r") as f:
            contents = f.readlines()

        # Initialize variables
        data = {
            "a": None,
            "b": None,
            "c": None,
            "alpha": None,
            "beta": None,
            "gamma": None,
            "spacegroup": None,
            "atomic_numbers": [],
            "symbols": [],
            "positions": [],
            "name": os.path.basename(cif_file).replace(".cif", ""),
        }

        # Counters for parsing
        sym_counter = 0
        atom_counter = 0
        a_counter = b_counter = c_counter = 0
        alpha_counter = beta_counter = gamma_counter = 0
        atom_list = []

        # Parse CIF file
        for line in contents:
            words = line.split()
            for i, word in enumerate(words):
                # Get lattice parameters
                if word == "_cell_length_a":
                    a_counter = 1
                elif a_counter == 1:
                    data["a"] = float(word)
                    a_counter = 0

                if word == "_cell_length_b":
                    b_counter = 1
                elif b_counter == 1:
                    data["b"] = float(word)
                    b_counter = 0

                if word == "_cell_length_c":
                    c_counter = 1
                elif c_counter == 1:
                    data["c"] = float(word)
                    c_counter = 0

                # Get unit cell angles
                if word == "_cell_angle_alpha":
                    alpha_counter = 1
                elif alpha_counter == 1:
                    data["alpha"] = float(word)
                    alpha_counter = 0

                if word == "_cell_angle_beta":
                    beta_counter = 1
                elif beta_counter == 1:
                    data["beta"] = float(word)
                    beta_counter = 0

                if word == "_cell_angle_gamma":
                    gamma_counter = 1
                elif gamma_counter == 1:
                    data["gamma"] = float(word)
                    gamma_counter = 0

                # Get space group
                if (
                    word == "_symmetry_Int_Tables_number"
                    or word == "_space_group_IT_number"
                ):
                    sym_counter = 1
                elif sym_counter == 1:
                    data["spacegroup"] = int(word)
                    sym_counter = 0

                # Get atom data
                if word == "_atom_site_occupancy":
                    atom_counter = 1
                elif word == "loop_":
                    atom_counter = 0
                elif atom_counter == 1:
                    atom_list.append(word)

        # Process atom data
        true_index = 0
        index = 0
        atom_name = []
        h = []
        k = []
        l = []

        for i in atom_list:
            if index == 1:
                atom_name.append(i)
            if index == 2:
                h.append(float(i))
            if index == 3:
                k.append(float(i))
            if index == 4:
                l.append(float(i))

            true_index += 1
            index += 1
            if index == 8:
                index = 0

        # Convert atom names to atomic numbers
        atomic_numbers = []
        for name in atom_name:
            try:
                atom = getattr(Element, name)
                atomic_numbers.append(int(atom))
            except (AttributeError, ValueError):
                atomic_numbers.append(ELEMENT_SYMBOLS.get(name, 0))

        # Create fractional coordinates
        positions = []
        for i in range(len(h)):
            positions.append([h[i], k[i], l[i]])

        data["atomic_numbers"] = atomic_numbers
        data["symbols"] = atom_name
        data["positions"] = positions

        return data


def display_default_settings():
    """Display the default settings in a formatted way"""
    print("\n" + "=" * 70)
    print("DEFAULT RECOMMENDED SETTINGS FOR CRYSTAL23")
    print("=" * 70)

    print("\n### SYMMETRY SETTINGS ###")
    print(f"Symmetry handling: CIF (Use symmetry as defined in the CIF file)")
    print(f"Trigonal axes: AUTO (Use setting as detected in CIF)")
    print(f"High symmetry space groups: AUTO (Use origin as detected in CIF)")

    print("\n### CALCULATION SETTINGS ###")
    print(f"Dimensionality: CRYSTAL (3D periodic system)")
    print(f"Calculation type: OPT (Geometry optimization)")
    print(f"Optimization type: FULLOPTG (Full geometry optimization)")
    print(f"Optimization parameters:")
    print(f"  - TOLDEG: {DEFAULT_OPT_SETTINGS['TOLDEG']} (RMS of gradient)")
    print(f"  - TOLDEX: {DEFAULT_OPT_SETTINGS['TOLDEX']} (RMS of displacement)")
    print(f"  - TOLDEE: {DEFAULT_OPT_SETTINGS['TOLDEE']} (Energy convergence)")
    print(f"  - MAXCYCLE: {DEFAULT_OPT_SETTINGS['MAXCYCLE']} (Max optimization steps)")
    print(f"  - MAXTRADIUS: Not set (default)")

    print("\n### BASIS SET AND DFT SETTINGS ###")
    print(f"Method: DFT")
    print(f"Basis set type: INTERNAL")
    print(f"Basis set: POB-TZVP-REV2 (Triple-ζ + polarization, revised)")
    print(f"DFT functional: HSE06-D3 (Screened hybrid with D3 dispersion)")
    print(f"DFT grid: XLGRID (Extra large grid)")
    print(f"Spin polarized: Yes")
    print(f"Fermi smearing: No")

    print("\n### SCF SETTINGS ###")
    print(
        f"Tolerances: TOLINTEG={DEFAULT_TOLERANCES['TOLINTEG']}, TOLDEE={DEFAULT_TOLERANCES['TOLDEE']}"
    )
    print(f"SCF method: DIIS")
    print(f"SCF max cycles: 800")
    print(f"FMIXING: 30%")

    print("\n### EXAMPLE D12 OUTPUT ###")
    print("-" * 70)
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
    print("MAXCYCLE")
    print("800")
    print("TOLDEG")
    print("0.00003")
    print("TOLDEX")
    print("0.00012")
    print("TOLDEE")
    print("7")
    print("ENDOPT")
    print("END")
    print("BASISSET")
    print("POB-TZVP-REV2")
    print("DFT")
    print("SPIN")
    print("HSE06-D3")
    print("XLGRID")
    print("ENDDFT")
    print("TOLINTEG")
    print("7 7 7 7 14")
    print("TOLDEE")
    print("7")
    print("SHRINK")
    print("0 24")
    print("12 12 12")
    print("SCFDIR")
    print("BIPOSIZE")
    print("110000000")
    print("EXCHSIZE")
    print("110000000")
    print("MAXCYCLE")
    print("800")
    print("FMIXING")
    print("30")
    print("DIIS")
    print("HISTDIIS")
    print("100")
    print("PPAN")
    print("END")
    print("-" * 70)


def select_method():
    """
    Select calculation method (HF or DFT)

    Returns:
        str: 'HF' or 'DFT'
    """
    method_options = {"1": "DFT", "2": "HF"}

    print("\nSelect calculation method:")
    print("1: DFT - Density Functional Theory")
    print("2: HF - Hartree-Fock")

    method_choice = get_user_input("Select method", method_options, "1")
    return method_options[method_choice]


def select_dft_functional():
    """
    Select functional/method by category

    Returns:
        tuple: (functional, basis_set_requirement) or (functional, None)
    """
    # First, select DFT category
    dft_categories = {k: v for k, v in FUNCTIONAL_CATEGORIES.items() if k != "HF"}
    category_options = {}

    for i, (key, info) in enumerate(dft_categories.items(), 1):
        category_options[str(i)] = key
        print(f"\n{i}. {info['name']}")
        print(f"   {info['description']}")
        # Show appropriate examples for each category
        if key == "HYBRID":
            print(f"   Examples: B3LYP, PBE0, HSE06, LC-wPBE")
        elif key == "3C":
            print(f"   Examples: PBEh-3C, HSE-3C, B97-3C")
        else:
            print(f"   Examples: {', '.join(info['functionals'][:4])}")

    category_choice = get_user_input(
        "Select functional category", category_options, "3"
    )  # Default to HYBRID
    selected_category = category_options[category_choice]

    # Then select specific functional
    category_info = dft_categories[selected_category]
    functional_options = {
        str(i + 1): func for i, func in enumerate(category_info["functionals"])
    }

    print(f"\nAvailable {category_info['name']}:")
    for key, func in functional_options.items():
        # Build the description string
        desc_parts = []

        # Add functional description if available
        if "descriptions" in category_info and func in category_info["descriptions"]:
            desc_parts.append(category_info["descriptions"][func])

        # Add D3 support indicator
        if func in D3_FUNCTIONALS:
            desc_parts.append("[D3✓]")

        # Add basis requirement for 3C methods
        if selected_category == "3C" and "basis_requirements" in category_info:
            basis = category_info["basis_requirements"].get(func, "")
            desc_parts.append(f"(requires {basis})")

        # Print the functional with description
        if desc_parts:
            print(f"{key}: {func} - {' '.join(desc_parts)}")
        else:
            print(f"{key}: {func}")

    functional_choice = get_user_input(
        f"Select {category_info['name']}", functional_options, "1"
    )
    selected_functional = functional_options[functional_choice]

    # Check if functional has basis set requirement
    if selected_category == "3C" and "basis_requirements" in category_info:
        required_basis = category_info["basis_requirements"].get(selected_functional)
        return selected_functional, required_basis

    return selected_functional, None


def get_advanced_frequency_settings():
    """Get advanced frequency calculation settings from user"""
    freq_settings = {}
    
    print("\n=== FREQUENCY CALCULATION SETTINGS ===")
    
    # Mode selection
    print("\nFrequency calculation modes:")
    print("1: Gamma point only (default) - Thermodynamic properties")
    print("2: Phonon dispersion - Band structure and DOS")
    print("3: Custom k-points - Specific points in Brillouin zone")
    
    mode_choice = input("Select mode (1-3) [1]: ").strip() or "1"
    
    if mode_choice == "1":
        freq_settings["mode"] = "GAMMA"
    elif mode_choice == "2":
        freq_settings["mode"] = "DISPERSION"
        # Ask for dispersion details
        n_kpoints = int(input("Number of k-points for dispersion [20]: ") or 20)
        freq_settings["n_kpoints"] = n_kpoints
        
        # Ask if they want to specify a path or use automatic
        use_auto_path = yes_no_prompt("Use automatic k-path generation?", "yes")
        if not use_auto_path:
            print("Enter k-points path (e.g., 'G-X-M-G' or custom coordinates)")
            print("Note: Custom path specification will be added to TODO list")
            # TODO: Implement custom k-path specification
    else:
        freq_settings["mode"] = "CUSTOM"
        # TODO: Implement custom k-points input
        print("Custom k-points mode not yet fully implemented")
    
    # Numerical derivative method
    print("\nNumerical derivative method:")
    print("1: First derivatives only (faster, less accurate)")
    print("2: Second derivatives (default, recommended)")
    
    numderiv = input("Select method (1-2) [2]: ").strip() or "2"
    freq_settings["numderiv"] = int(numderiv)
    
    # IR intensities
    calc_ir = yes_no_prompt("\nCalculate IR intensities?", "no")
    if calc_ir:
        freq_settings["intensities"] = True
        
        print("\nIR intensity calculation method:")
        print("1: Berry phase (default, fast)")
        print("2: Wannier functions (localized, good for molecules)")
        print("3: CPHF (most accurate, expensive)")
        
        ir_method_choice = input("Select method (1-3) [1]: ").strip() or "1"
        ir_methods = {"1": "BERRY", "2": "WANNIER", "3": "CPHF"}
        freq_settings["ir_method"] = ir_methods[ir_method_choice]
        
        if freq_settings["ir_method"] == "CPHF":
            # CPHF specific options
            print("\nCPHF calculation options:")
            max_iter = int(input("Maximum CPHF iterations [30]: ") or 30)
            freq_settings["cphf_max_iter"] = max_iter
            
            tol = float(input("CPHF convergence tolerance (10^-x) [6]: ") or 6)
            freq_settings["cphf_tolerance"] = tol
    
    # Raman intensities (requires CPHF)
    calc_raman = yes_no_prompt("\nCalculate Raman intensities? (requires CPHF)", "no")
    if calc_raman:
        freq_settings["raman"] = True
        freq_settings["intensities"] = True  # IR is required for Raman
        freq_settings["ir_method"] = "CPHF"  # Force CPHF for Raman
        
        # Set CPHF parameters if not already set
        if "cphf_max_iter" not in freq_settings:
            freq_settings["cphf_max_iter"] = 30
        if "cphf_tolerance" not in freq_settings:
            freq_settings["cphf_tolerance"] = 6
    
    # Spectral generation
    if freq_settings.get("intensities") or freq_settings.get("raman"):
        gen_spectra = yes_no_prompt("\nGenerate IR/Raman spectra?", "yes")
        if gen_spectra:
            if freq_settings.get("intensities"):
                freq_settings["ir_spectrum"] = True
                width = float(input("IR peak width (cm^-1) [10]: ") or 10)
                freq_settings["ir_spectrum_width"] = width
                
            if freq_settings.get("raman"):
                freq_settings["raman_spectrum"] = True
                width = float(input("Raman peak width (cm^-1) [10]: ") or 10)
                freq_settings["raman_spectrum_width"] = width
    
    # Anharmonic calculations
    calc_anharm = yes_no_prompt("\nInclude anharmonic corrections (X-H stretches)?", "no")
    if calc_anharm:
        freq_settings["anharmonic"] = True
        
        print("\nAnharmonic calculation type:")
        print("1: ANHARM - Basic anharmonic for X-H")
        print("2: VSCF - Vibrational SCF")
        print("3: VCI - Vibrational CI (most accurate)")
        
        anharm_choice = input("Select type (1-3) [1]: ").strip() or "1"
        anharm_types = {"1": "ANHARM", "2": "VSCF", "3": "VCI"}
        freq_settings["anharm_type"] = anharm_types[anharm_choice]
        
        if freq_settings["anharm_type"] in ["VSCF", "VCI"]:
            # VSCF/VCI specific options
            max_quanta = int(input("Maximum quanta per mode [4]: ") or 4)
            freq_settings["vscf_max_quanta"] = max_quanta
            
            if freq_settings["anharm_type"] == "VCI":
                max_coupled = int(input("Maximum coupled modes [2]: ") or 2)
                freq_settings["vci_max_coupled"] = max_coupled
    
    # Temperature for thermodynamic properties
    if freq_settings.get("mode") == "GAMMA":
        print("\nThermodynamic properties calculation:")
        temp_list = input("Enter temperatures (K) separated by spaces [298.15]: ").strip()
        if temp_list:
            freq_settings["temperatures"] = [float(t) for t in temp_list.split()]
        else:
            freq_settings["temperatures"] = [298.15]
    
    # Additional options
    print("\nAdditional options:")
    
    # Restart capability
    use_restart = yes_no_prompt("Enable restart capability?", "yes")
    if use_restart:
        freq_settings["restart"] = True
    
    # Print level
    print("\nOutput verbosity:")
    print("0: Minimal output")
    print("1: Standard output (default)")
    print("2: Detailed output")
    print("3: Debug output")
    
    print_level = input("Select print level (0-3) [1]: ").strip() or "1"
    freq_settings["print_level"] = int(print_level)
    
    return freq_settings


def get_calculation_options():
    """Gather calculation options from user"""
    options = {}

    # First, show default settings and ask if user wants to use them
    display_default_settings()
    use_defaults = yes_no_prompt("\nDo you want to use these default settings?", "yes")

    if use_defaults:
        # Use all default settings
        options = DEFAULT_SETTINGS.copy()
        options["optimization_settings"] = DEFAULT_OPT_SETTINGS.copy()
        options["tolerances"] = DEFAULT_TOLERANCES.copy()
    else:
        # Custom settings - go through each option

        # Ask about symmetry handling
        symmetry_options = {
            "1": "CIF",  # Use symmetry as defined in the CIF file (trust the file)
            "2": "SPGLIB",  # Use spglib to analyze symmetry (may detect different symmetry)
            "3": "P1",  # Use P1 symmetry (all atoms explicit, no symmetry)
        }
        symmetry_choice = get_user_input(
            "Select symmetry handling method", symmetry_options, "1"
        )
        options["symmetry_handling"] = symmetry_options[symmetry_choice]

        # If using spglib symmetry analysis
        if options["symmetry_handling"] == "SPGLIB" and SPGLIB_AVAILABLE:
            # Ask about symmetry tolerance
            print("\nSymmetry tolerance options:")
            print("  Loose (1e-3): More forgiving, good for experimental structures")
            print("  Default (1e-5): Standard tolerance for most cases") 
            print("  Strict (1e-7): High precision, for theoretical/perfect structures")
            
            tolerance_options = {
                "1": 1e-3,  # Loose tolerance - more forgiving of deviations
                "2": 1e-5,  # Default tolerance
                "3": 1e-7,  # Strict tolerance - requires high precision
            }
            tolerance_choice = get_user_input(
                "Select symmetry detection tolerance", tolerance_options, "2"
            )
            options["symmetry_tolerance"] = tolerance_options[tolerance_choice]

            # Ask about asymmetric unit reduction
            reduce_atoms = yes_no_prompt(
                "Reduce structure to asymmetric unit using spglib?", "yes"
            )
            options["reduce_to_asymmetric"] = reduce_atoms
        elif options["symmetry_handling"] == "CIF":
            # For CIF mode, use default tolerance but allow user to know about it
            options["symmetry_tolerance"] = 1e-5

        # Ask about atom writing preference for symmetry handling (unified with CRYSTALOptToD12.py)
        if options["symmetry_handling"] in ["CIF", "SPGLIB"]:
            print("\nAtom writing options:")
            if not SPGLIB_AVAILABLE:
                print("Note: spglib not available - asymmetric unit reduction will be limited")
            
            atom_writing_options = {
                "1": "Write only unique atoms (asymmetric unit) - requires spglib for CIF mode",
                "2": "Write all atoms (full unit cell)",
            }
            atom_writing_choice = get_user_input(
                "How should atoms be written in the inputs?", 
                atom_writing_options, 
                "1"
            )
            options["write_only_unique"] = atom_writing_choice == "1"
            
            # For modes that use spglib, ask about validation (applies to both CIF and SPGLIB modes)
            if options["write_only_unique"] and SPGLIB_AVAILABLE:
                validate_symmetry = yes_no_prompt(
                    "Validate symmetry operations can reconstruct original structure?", "no"
                )
                options["validate_symmetry"] = validate_symmetry
            else:
                options["validate_symmetry"] = False
        else:
            # For P1 symmetry, all atoms must be written
            options["write_only_unique"] = False
            options["validate_symmetry"] = False

        # For trigonal space groups, ask about axis representation
        trigonal_axes_options = {
            "1": "AUTO",  # Use setting as detected in CIF
            "2": "HEXAGONAL_AXES",  # Force hexagonal axes
            "3": "RHOMBOHEDRAL_AXES",  # Force rhombohedral axes
        }
        trigonal_axes_choice = get_user_input(
            "For trigonal space groups (143-167), which axes do you prefer?",
            trigonal_axes_options,
            "1",
        )
        options["trigonal_axes"] = trigonal_axes_options[trigonal_axes_choice]

        # For space groups with multiple origins
        origin_options = {
            "1": "AUTO",  # Use origin as detected in CIF
            "2": "STANDARD",  # Force standard origin (ITA Origin 2) - CRYSTAL "0 0 0"
            "3": "ALTERNATE",  # Force alternate origin (ITA Origin 1) - CRYSTAL "0 0 1"
        }
        origin_choice = get_user_input(
            "For space groups with multiple origins (e.g., 227-Fd-3m):\n"
            "  Standard (ITA Origin 2, CRYSTAL '0 0 0'): Si at (1/8,1/8,1/8), 36 operators\n"
            "  Alternate (ITA Origin 1, CRYSTAL '0 0 1'): Si at (0,0,0), 24 operators",
            origin_options,
            "1",
        )
        options["origin_setting"] = origin_options[origin_choice]

        # Get dimensionality
        dimensionality_options = {
            "1": "CRYSTAL",
            "2": "SLAB",
            "3": "POLYMER",
            "4": "MOLECULE",
        }
        dimensionality_choice = get_user_input(
            "Select the dimensionality of the system", dimensionality_options, "1"
        )
        options["dimensionality"] = dimensionality_options[dimensionality_choice]

        # Get calculation type
        calc_options = {
            "1": "SP",  # Single Point
            "2": "OPT",  # Geometry Optimization
            "3": "FREQ",  # Frequency Calculation
        }
        calc_choice = get_user_input("Select calculation type", calc_options, "2")
        options["calculation_type"] = calc_options[calc_choice]

        # If geometry optimization, get optimization type
        if options["calculation_type"] == "OPT":
            opt_choice = get_user_input("Select optimization type", OPT_TYPES, "1")
            options["optimization_type"] = OPT_TYPES[opt_choice]

            # Ask for default or custom optimization settings
            use_default_opt = yes_no_prompt(
                "Use default optimization settings? (TOLDEG=0.00003, TOLDEX=0.00012, TOLDEE=7, MAXCYCLE=800)",
                "yes",
            )

            if not use_default_opt:
                custom_settings = {}
                custom_settings["TOLDEG"] = float(
                    input("Enter TOLDEG (RMS of gradient, default 0.00003): ")
                    or 0.00003
                )
                custom_settings["TOLDEX"] = float(
                    input("Enter TOLDEX (RMS of displacement, default 0.00012): ")
                    or 0.00012
                )
                custom_settings["TOLDEE"] = int(
                    input("Enter TOLDEE (energy difference exponent, default 7): ") or 7
                )
                custom_settings["MAXCYCLE"] = int(
                    input("Enter MAXCYCLE (max optimization steps, default 800): ")
                    or 800
                )
                options["optimization_settings"] = custom_settings
            else:
                options["optimization_settings"] = DEFAULT_OPT_SETTINGS.copy()

            # Ask about MAXTRADIUS
            use_maxtradius = yes_no_prompt(
                "Set maximum step size (MAXTRADIUS) for geometry optimization?", "no"
            )

            if use_maxtradius:
                if "optimization_settings" not in options:
                    options["optimization_settings"] = DEFAULT_OPT_SETTINGS.copy()
                maxtradius = float(
                    input("Enter MAXTRADIUS (max displacement, default 0.25): ") or 0.25
                )
                options["optimization_settings"]["MAXTRADIUS"] = maxtradius

        # If frequency calculation, ask about numerical derivative level
        if options["calculation_type"] == "FREQ":
            use_default_freq = yes_no_prompt(
                "Use default frequency calculation settings? (NUMDERIV=2, TOLINTEG=12 12 12 12 24, TOLDEE=12)",
                "yes",
            )

            if not use_default_freq:
                options["freq_settings"] = get_advanced_frequency_settings()
            else:
                options["freq_settings"] = DEFAULT_FREQ_SETTINGS.copy()

            # For frequency calculations, use tighter default tolerances
            if "tolerances" not in options:
                options["tolerances"] = {
                    "TOLINTEG": DEFAULT_FREQ_SETTINGS["TOLINTEG"],
                    "TOLDEE": DEFAULT_FREQ_SETTINGS["TOLDEE"],
                }

        # Select method (HF or DFT)
        options["method"] = select_method()

        # Select functional/method (before basis set)
        if options["method"] == "HF":
            hf_info = FUNCTIONAL_CATEGORIES["HF"]
            functional_options = {
                str(i + 1): func for i, func in enumerate(hf_info["functionals"])
            }

            print(f"\nAvailable {hf_info['name']}:")
            for key, func in functional_options.items():
                desc = hf_info["descriptions"].get(func, "")
                if func in ["HF3C", "HFSOL3C"]:
                    basis = hf_info["basis_requirements"][func]
                    print(f"{key}: {func} - {desc} (requires {basis} basis set)")
                else:
                    print(f"{key}: {func} - {desc}")

            functional_choice = get_user_input(
                "Select HF method", functional_options, "1"
            )
            selected_functional = functional_options[functional_choice]

            # Check if it has basis set requirement
            if selected_functional in hf_info.get("basis_requirements", {}):
                required_basis = hf_info["basis_requirements"][selected_functional]
                options["hf_method"] = selected_functional
                options["basis_set_type"] = "INTERNAL"
                options["basis_set"] = required_basis
                print(
                    f"\nNote: {selected_functional} requires {required_basis} basis set. Using it automatically."
                )
                # HF3C methods already include corrections
                options["use_dispersion"] = False
                # 3C methods have their own grids (if applicable)
                options["dft_grid"] = None
            else:
                options["hf_method"] = selected_functional
                # Regular HF methods don't use dispersion
                options["use_dispersion"] = False
        else:
            functional, required_basis = select_dft_functional()
            options["dft_functional"] = functional

            # If functional has required basis set, use it automatically
            if required_basis:
                print(
                    f"\nNote: {functional} requires {required_basis} basis set. Using it automatically."
                )
                options["basis_set_type"] = "INTERNAL"
                options["basis_set"] = required_basis
                # 3C methods already include dispersion and other corrections
                options["use_dispersion"] = False
                # 3C methods have their own optimized grids
                options["dft_grid"] = None
            else:
                # Check if dispersion correction is available immediately after functional selection
                if functional in D3_FUNCTIONALS:
                    use_dispersion = yes_no_prompt(
                        f"Add D3 dispersion correction to {functional}?", "yes"
                    )
                    options["use_dispersion"] = use_dispersion
                else:
                    # Check if functional already includes dispersion (3C methods)
                    if functional in [
                        "PBEh-3C",
                        "HSE-3C",
                        "B97-3C",
                        "PBEsol0-3C",
                        "HSEsol-3C",
                    ]:
                        print(
                            f"Note: {functional} already includes dispersion corrections."
                        )
                        options["use_dispersion"] = False
                    else:
                        print(
                            f"Note: D3 dispersion correction not available for {functional}"
                        )
                        options["use_dispersion"] = False

        # Get basis set type (if not already determined)
        if "basis_set" not in options:
            basis_options = {"1": "EXTERNAL", "2": "INTERNAL"}

            print("\nSelect basis set type:")
            print("1: EXTERNAL - Full-core and ECP basis sets (DZVP-REV2, TZVP-REV2)")
            print("   Note: Elements 37-99 (except Tc) use ECP, up to Es (Z=99)")
            print("2: INTERNAL - All-electron basis sets with limited element coverage")

            basis_choice = get_user_input("Select basis set type", basis_options, "2")
            options["basis_set_type"] = basis_options[basis_choice]

            # Get specific basis set
            if options["basis_set_type"] == "EXTERNAL":
                external_basis_options = {
                    "1": DEFAULT_DZ_PATH,  # DZVP-REV2
                    "2": DEFAULT_TZ_PATH,  # TZVP-REV2
                }

                print(f"\nSelect external basis set directory:")
                print(f"1: DZVP-REV2 ({DEFAULT_DZ_PATH})")
                print("   Full-core: H-Kr, Tc")
                print("   ECP: Rb-Es (except noble gases)")
                print(f"2: TZVP-REV2 ({DEFAULT_TZ_PATH})")
                print("   Full-core: H-Kr, Tc")
                print("   ECP: Rb-Es (except noble gases)")

                basis_dir_choice = get_user_input(
                    "Select external basis set", external_basis_options, "2"
                )

                # Allow user to override the default path if needed
                selected_path = external_basis_options[basis_dir_choice]
                custom_path = input(
                    f"Use this path ({selected_path}) or enter a custom path (press Enter to use default): "
                )

                if custom_path:
                    options["basis_set"] = custom_path
                else:
                    options["basis_set"] = selected_path
            else:
                # Internal basis sets
                internal_basis_options = {}
                print("\nAvailable internal basis sets:")

                # First show standard basis sets
                print("\n--- STANDARD BASIS SETS ---")
                option_num = 1
                standard_options = []
                for bs_name, bs_info in INTERNAL_BASIS_SETS.items():
                    if bs_info.get("standard", False):
                        internal_basis_options[str(option_num)] = bs_name
                        standard_options.append(option_num)
                        element_info = get_element_info_string(bs_name)
                        print(f"{option_num}: {bs_name} - {bs_info['description']}")
                        print(f"   {element_info}")
                        option_num += 1

                # Then show additional basis sets
                print("\n--- ADDITIONAL BASIS SETS ---")
                for bs_name, bs_info in INTERNAL_BASIS_SETS.items():
                    if not bs_info.get("standard", False):
                        internal_basis_options[str(option_num)] = bs_name
                        element_info = get_element_info_string(bs_name)
                        print(f"{option_num}: {bs_name} - {bs_info['description']}")
                        print(f"   {element_info}")
                        option_num += 1

                # Default to POB-TZVP-REV2 (should be option 7 in standard sets)
                default_option = "7"
                internal_basis_choice = get_user_input(
                    "Select internal basis set", internal_basis_options, default_option
                )
                options["basis_set"] = internal_basis_options[internal_basis_choice]

        # Get DFT grid (only for non-3C DFT methods)
        if options["method"] == "DFT":
            functional = options.get("dft_functional", "")
            # Check if it's a 3C method
            if "-3C" not in functional and "3C" not in functional:
                grid_choice = get_user_input(
                    "Select DFT integration grid", DFT_GRIDS, "4"
                )  # Default to XLGRID
                options["dft_grid"] = DFT_GRIDS[grid_choice]
            else:
                # 3C methods have their own optimized grids
                options["dft_grid"] = None

        # Ask about spin polarization
        is_spin_polarized = yes_no_prompt("Use spin-polarized calculation?", "yes")
        options["is_spin_polarized"] = is_spin_polarized

        # Ask about Fermi surface smearing for metals
        use_smearing = yes_no_prompt(
            "Use Fermi surface smearing for metallic systems?", "no"
        )
        options["use_smearing"] = use_smearing

        if use_smearing:
            smearing_width = float(
                input(
                    "Enter smearing width in hartree (recommended: 0.001-0.02, default 0.01): "
                )
                or 0.01
            )
            options["smearing_width"] = smearing_width

        # Get tolerance settings (if not already set by FREQ)
        if "tolerances" not in options:
            use_default_tol = yes_no_prompt(
                "Use default tolerance settings? (TOLINTEG=7 7 7 7 14, TOLDEE=7)", "yes"
            )

            if not use_default_tol:
                custom_tol = {}
                tolinteg = input(
                    "Enter TOLINTEG values (5 integers separated by spaces, default 7 7 7 7 14): "
                )
                tolinteg = tolinteg if tolinteg else "7 7 7 7 14"
                custom_tol["TOLINTEG"] = tolinteg

                toldee = input("Enter TOLDEE value (integer, default 7): ")
                toldee = int(toldee) if toldee else 7
                custom_tol["TOLDEE"] = toldee

                options["tolerances"] = custom_tol
            else:
                options["tolerances"] = DEFAULT_TOLERANCES.copy()
        else:
            # Ensure all required tolerance keys are present
            if "TOLINTEG" not in options["tolerances"]:
                options["tolerances"]["TOLINTEG"] = DEFAULT_TOLERANCES["TOLINTEG"]
            if "TOLDEE" not in options["tolerances"]:
                options["tolerances"]["TOLDEE"] = DEFAULT_TOLERANCES["TOLDEE"]

        # Get SCF convergence method
        scf_options = {str(i + 1): method for i, method in enumerate(SCF_METHODS)}
        scf_choice = get_user_input(
            "Select SCF convergence method", scf_options, "1"
        )  # Default to DIIS
        options["scf_method"] = scf_options[scf_choice]

        # Ask about SCF MAXCYCLE
        use_default_scf_maxcycle = yes_no_prompt(
            "Use default SCF MAXCYCLE (800)?", "yes"
        )

        if not use_default_scf_maxcycle:
            options["scf_maxcycle"] = int(input("Enter SCF MAXCYCLE value: ") or 800)
        else:
            options["scf_maxcycle"] = 800

        # Ask about FMIXING
        use_default_fmixing = yes_no_prompt("Use default FMIXING (30%)?", "yes")

        if not use_default_fmixing:
            options["fmixing"] = int(
                input("Enter FMIXING percentage (0-100, default 30): ") or 30
            )
        else:
            options["fmixing"] = 30

    return options


def is_trigonal(spacegroup):
    """
    Check if space group is trigonal

    Args:
        spacegroup (int): Space group number

    Returns:
        bool: True if trigonal, False otherwise
    """
    return 143 <= spacegroup <= 167


def detect_trigonal_setting(cif_data):
    """
    Detect whether a trigonal structure is in hexagonal or rhombohedral axes

    Args:
        cif_data (dict): Parsed CIF data

    Returns:
        str: 'hexagonal_axes' or 'rhombohedral_axes'
    """
    # Check if it's a trigonal space group
    if 143 <= cif_data["spacegroup"] <= 167:
        # Determine which setting based on cell parameters
        if (
            abs(cif_data["alpha"] - 90) < 1e-3
            and abs(cif_data["beta"] - 90) < 1e-3
            and abs(cif_data["gamma"] - 120) < 1e-3
        ):
            # Alpha ~ 90, beta ~ 90, gamma ~ 120 indicates hexagonal axes
            return "hexagonal_axes"
        elif (
            abs(cif_data["alpha"] - cif_data["beta"]) < 1e-3
            and abs(cif_data["beta"] - cif_data["gamma"]) < 1e-3
        ):
            # Alpha = beta = gamma != 90 indicates rhombohedral axes
            return "rhombohedral_axes"

    # Default to hexagonal axes
    return "hexagonal_axes"


def verify_and_reduce_to_asymmetric_unit(cif_data, tolerance=1e-5, validate_symmetry=False):
    """
    Verify spglib symmetry analysis matches CIF data and reduce to asymmetric unit

    Args:
        cif_data (dict): Parsed CIF data
        tolerance (float): Symmetry tolerance for spglib
        validate_symmetry (bool): Whether to validate that symmetry operations can reconstruct the original structure

    Returns:
        dict: Modified CIF data with only asymmetric unit atoms, or original if verification fails
    """
    if not SPGLIB_AVAILABLE:
        print("Warning: spglib not available, cannot reduce to asymmetric unit.")
        print("Using all atoms from the CIF file.")
        return cif_data

    try:
        # Create proper lattice matrix from CIF parameters
        a, b, c = cif_data["a"], cif_data["b"], cif_data["c"]
        alpha, beta, gamma = cif_data["alpha"], cif_data["beta"], cif_data["gamma"]
        
        # Convert to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        
        # Build proper lattice matrix
        lattice = np.zeros((3, 3))
        lattice[0, 0] = a
        lattice[1, 0] = b * np.cos(gamma_rad)
        lattice[1, 1] = b * np.sin(gamma_rad)
        lattice[2, 0] = c * np.cos(beta_rad)
        lattice[2, 1] = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        lattice[2, 2] = c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 + 2*np.cos(alpha_rad)*np.cos(beta_rad)*np.cos(gamma_rad)) / np.sin(gamma_rad)

        positions = np.array(cif_data["positions"])
        numbers = np.array(cif_data["atomic_numbers"])

        cell = (lattice, positions, numbers)

        # Get spacegroup data with the specified tolerance
        spacegroup_info = spglib.get_spacegroup(cell, symprec=tolerance)
        dataset = spglib.get_symmetry_dataset(cell, symprec=tolerance)
        
        if dataset is None:
            print("Warning: spglib could not analyze the structure symmetry.")
            print("Using all atoms from the CIF file.")
            return cif_data
        
        detected_spacegroup_num = dataset['number']
        original_spacegroup_num = cif_data["spacegroup"]
        
        print(f"\nSymmetry Analysis Results:")
        print(f"  CIF space group: {original_spacegroup_num}")
        print(f"  spglib detected: {detected_spacegroup_num} ({spacegroup_info})")
        
        # Check if space groups match
        spacegroup_match = detected_spacegroup_num == original_spacegroup_num
        
        if not spacegroup_match:
            print(f"\n⚠️  WARNING: Space group mismatch!")
            print(f"     CIF file specifies space group {original_spacegroup_num}")
            print(f"     spglib detects space group {detected_spacegroup_num}")
            print(f"     Current tolerance: {tolerance}")
            print(f"     This could indicate:")
            print(f"       - Tolerance issues (try different tolerance)")
            print(f"       - Incorrect CIF space group assignment")
            print(f"       - Non-standard atomic positions in CIF")
            
            # Offer options to the user
            print(f"\nOptions:")
            print(f"  1: Try different tolerance values")
            print(f"  2: Proceed with spglib space group {detected_spacegroup_num}")
            print(f"  3: Use original CIF space group {original_spacegroup_num} (no reduction)")
            
            choice = get_user_input("Select option", {"1": "tolerance", "2": "spglib", "3": "original"}, "3")
            
            if choice == "1":
                # Try different tolerances
                for test_tolerance in [1e-3, 1e-4, 1e-6, 1e-7]:
                    if test_tolerance != tolerance:
                        print(f"\nTrying tolerance {test_tolerance}...")
                        test_dataset = spglib.get_symmetry_dataset(cell, symprec=test_tolerance)
                        if test_dataset and test_dataset['number'] == original_spacegroup_num:
                            print(f"✓ Match found with tolerance {test_tolerance}!")
                            use_tolerance = yes_no_prompt(f"Use tolerance {test_tolerance}?", "yes")
                            if use_tolerance:
                                return verify_and_reduce_to_asymmetric_unit(cif_data, test_tolerance)
                
                print("\nNo tolerance found that matches CIF space group.")
                final_choice = get_user_input("Final choice", {"1": "spglib", "2": "original"}, "2")
                if final_choice == "2":
                    print("Using all atoms from the CIF file without reduction.")
                    return cif_data
                else:
                    print(f"Proceeding with spglib space group {detected_spacegroup_num}")
                    cif_data["spacegroup"] = detected_spacegroup_num
                    
            elif choice == "2":
                print(f"Proceeding with spglib space group {detected_spacegroup_num}")
                cif_data["spacegroup"] = detected_spacegroup_num
            else:
                print("Using all atoms from the CIF file without reduction.")
                return cif_data
        else:
            print(f"✓ Space group verification successful!")

        # Get unique atoms (asymmetric unit)
        equivalent_atoms = dataset['equivalent_atoms']
        unique_indices = []
        seen_representatives = set()
        
        for i, representative in enumerate(equivalent_atoms):
            if representative not in seen_representatives:
                unique_indices.append(i)
                seen_representatives.add(representative)

        # Verify atom count makes sense
        original_atom_count = len(numbers)
        unique_atom_count = len(unique_indices)
        symmetry_operations = len(dataset['rotations'])
        
        print(f"\nAsymmetric Unit Analysis:")
        print(f"  Original atoms: {original_atom_count}")
        print(f"  Unique atoms: {unique_atom_count}")
        print(f"  Symmetry operations: {symmetry_operations}")
        print(f"  Expected multiplicity: {original_atom_count / unique_atom_count:.1f}")
        
        # Check if the reduction makes sense
        if unique_atom_count >= original_atom_count:
            print("\n⚠️  WARNING: Asymmetric unit contains all or almost all atoms.")
            print("     This suggests the structure may already be in the asymmetric unit,")
            print("     or there's an issue with symmetry detection.")
            
            use_reduction = yes_no_prompt("Use the 'reduced' structure anyway?", "yes")
            if not use_reduction:
                print("Using all atoms from the CIF file.")
                return cif_data

        # Create new cif_data with only asymmetric unit atoms
        new_cif_data = cif_data.copy()
        new_cif_data["atomic_numbers"] = [numbers[i] for i in unique_indices]
        new_cif_data["symbols"] = [cif_data["symbols"][i] for i in unique_indices]
        new_cif_data["positions"] = [positions[i] for i in unique_indices]

        # Verify the positions are reasonable and show mapping
        print(f"\nAsymmetric unit atoms (with original indices):")
        for i, idx in enumerate(unique_indices):
            symbol = cif_data["symbols"][idx]
            pos = positions[idx]
            equiv_count = sum(1 for eq in equivalent_atoms if eq == equivalent_atoms[idx])
            print(f"  {i+1}: {symbol} at ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}) [orig #{idx+1}, {equiv_count} equivalent]")
        
        # Show which atoms are equivalent to which
        if len(unique_indices) < original_atom_count:
            print(f"\nEquivalence mapping:")
            for i, representative in enumerate(equivalent_atoms):
                if i not in unique_indices:
                    rep_idx = unique_indices.index(representative) if representative in unique_indices else -1
                    if rep_idx >= 0:
                        print(f"  Atom {i+1} ({cif_data['symbols'][i]}) → equivalent to asymmetric atom {rep_idx+1}")

        # Optional: Validate that symmetry operations can reconstruct original structure
        if validate_symmetry:
            print("\nPerforming symmetry validation...")
            try:
                # Apply symmetry operations to asymmetric unit
                rotations = dataset['rotations']
                translations = dataset['translations']
                
                reconstructed_positions = []
                reconstructed_numbers = []
                
                for i, asym_idx in enumerate(unique_indices):
                    asym_pos = positions[asym_idx]
                    asym_num = numbers[asym_idx]
                    
                    for rot, trans in zip(rotations, translations):
                        new_pos = np.dot(rot, asym_pos) + trans
                        # Wrap to unit cell
                        new_pos = new_pos % 1.0
                        reconstructed_positions.append(new_pos)
                        reconstructed_numbers.append(asym_num)
                
                # Check if we get the same number of atoms
                if len(reconstructed_positions) >= original_atom_count:
                    print(f"✓ Symmetry validation: Generated {len(reconstructed_positions)} positions from {len(unique_indices)} asymmetric atoms")
                    print(f"  Original structure had {original_atom_count} atoms")
                else:
                    print(f"⚠️  Symmetry validation: Only generated {len(reconstructed_positions)} positions, expected {original_atom_count}")
                    
            except Exception as e:
                print(f"⚠️  Symmetry validation failed: {e}")
        
        print(f"\n✓ Successfully reduced structure to asymmetric unit.")
        return new_cif_data

    except Exception as e:
        print(f"\n❌ Error during symmetry analysis: {e}")
        print("Using all atoms from the CIF file.")
        return cif_data


def reduce_to_asymmetric_unit(cif_data, validate_symmetry=False):
    """
    Legacy function - now calls the enhanced verification function
    """
    return verify_and_reduce_to_asymmetric_unit(cif_data, 1e-5, validate_symmetry)


def create_d12_file(cif_data, output_file, options):
    """
    Create a D12 input file for CRYSTAL23 from CIF data

    Args:
        cif_data (dict): Parsed CIF data
        output_file (str): Output file path
        options (dict): Calculation options

    Returns:
        None
    """
    # Extract CIF data
    a = cif_data["a"]
    b = cif_data["b"]
    c = cif_data["c"]
    alpha = cif_data["alpha"]
    beta = cif_data["beta"]
    gamma = cif_data["gamma"]
    spacegroup = cif_data["spacegroup"]
    atomic_numbers = cif_data["atomic_numbers"]
    symbols = cif_data["symbols"]
    positions = cif_data["positions"]

    # Check basis set compatibility
    is_compatible, missing_elements = check_basis_set_compatibility(
        options["basis_set"], atomic_numbers, options["basis_set_type"]
    )

    if not is_compatible:
        print(
            f"\nWARNING: The selected basis set '{options['basis_set']}' does not support all elements in your structure!"
        )
        print(
            f"Missing elements: {', '.join([f'{ATOMIC_NUMBER_TO_SYMBOL.get(z, z)} (Z={z})' for z in missing_elements])}"
        )
        if not yes_no_prompt("Continue anyway?", "no"):
            print("Aborting D12 file creation.")
            return

    # Extract options
    dimensionality = options["dimensionality"]
    calculation_type = options["calculation_type"]
    optimization_type = options.get("optimization_type", None)
    optimization_settings = options.get("optimization_settings", DEFAULT_OPT_SETTINGS)
    freq_settings = options.get("freq_settings", DEFAULT_FREQ_SETTINGS)
    basis_set_type = options["basis_set_type"]
    basis_set = options["basis_set"]
    method = options["method"]
    is_spin_polarized = options["is_spin_polarized"]
    tolerances = options["tolerances"]
    scf_method = options["scf_method"]
    scf_maxcycle = options.get("scf_maxcycle", 800)
    fmixing = options.get("fmixing", 30)
    use_smearing = options.get("use_smearing", False)
    smearing_width = options.get("smearing_width", 0.01)

    # Determine crystal settings
    trigonal_setting = cif_data.get("trigonal_setting", None)
    origin_setting = options.get("origin_setting", "AUTO")

    # Determine space group origin settings
    origin_directive = "0 0 0"  # Default value

    # Handle space groups with multiple origin choices
    if spacegroup in MULTI_ORIGIN_SPACEGROUPS:
        spg_info = MULTI_ORIGIN_SPACEGROUPS[spacegroup]

        # Handle origin setting
        if origin_setting == "STANDARD":
            origin_directive = spg_info["crystal_code"]
            print(
                f"Using standard origin setting ({spg_info['default']}) for space group {spacegroup} ({spg_info['name']})"
            )
            print(f"CRYSTAL directive: {origin_directive}")

        elif origin_setting == "ALTERNATE" and "alt_crystal_code" in spg_info:
            origin_directive = spg_info["alt_crystal_code"]
            print(
                f"Using alternate origin setting ({spg_info['alt']}) for space group {spacegroup} ({spg_info['name']})"
            )
            print(f"CRYSTAL directive: {origin_directive}")

        elif (
            origin_setting == "AUTO" and spacegroup == 227
        ):  # Special handling for Fd-3m
            # Try to detect based on atom positions
            std_pos = spg_info.get("default_pos", (0.125, 0.125, 0.125))
            alt_pos = spg_info.get("alt_pos", (0.0, 0.0, 0.0))

            # Check if any atoms are near the standard position
            std_detected = False
            alt_detected = False

            for pos in positions:
                # Check for atoms near standard position (1/8, 1/8, 1/8)
                if (
                    abs(pos[0] - std_pos[0]) < 0.01
                    and abs(pos[1] - std_pos[1]) < 0.01
                    and abs(pos[2] - std_pos[2]) < 0.01
                ):
                    std_detected = True

                # Check for atoms near alternate position (0, 0, 0)
                if (
                    abs(pos[0] - alt_pos[0]) < 0.01
                    and abs(pos[1] - alt_pos[1]) < 0.01
                    and abs(pos[2] - alt_pos[2]) < 0.01
                ):
                    alt_detected = True

            if alt_detected and not std_detected:
                # If only alternate position atoms found, use alternate origin
                origin_directive = spg_info["alt_crystal_code"]
                print(
                    f"Detected alternate origin ({spg_info['alt']}) for space group 227 (atoms at {alt_pos})"
                )
                print(
                    f"Using CRYSTAL directive: {origin_directive} (fewer symmetry operators with translational components)"
                )
            else:
                # Default to standard origin
                origin_directive = spg_info["crystal_code"]
                print(
                    f"Using standard origin ({spg_info['default']}) for space group 227"
                )
                print(f"CRYSTAL directive: {origin_directive}")

    # Handle trigonal space groups for rhombohedral axes directive
    use_rhombohedral_axes = False
    if is_trigonal(spacegroup):
        trigonal_axes = options.get("trigonal_axes", "AUTO")
        if trigonal_axes == "RHOMBOHEDRAL_AXES":
            use_rhombohedral_axes = True
            print(
                f"Using rhombohedral axes (0 1 0) for trigonal space group {spacegroup}"
            )
        elif trigonal_axes == "AUTO":
            # Try to detect from CIF
            trigonal_setting = detect_trigonal_setting(cif_data)
            if trigonal_setting == "rhombohedral_axes":
                use_rhombohedral_axes = True
                print(
                    f"Detected rhombohedral axes setting for space group {spacegroup}"
                )

    # Open output file
    with open(output_file, "w") as f:
        # Write title
        print(os.path.basename(output_file).replace(".d12", ""), file=f)

        # Write structure type and space group
        if dimensionality == "CRYSTAL":
            print("CRYSTAL", file=f)

            # Handle specific origin settings for space groups
            if spacegroup in MULTI_ORIGIN_SPACEGROUPS:
                print(origin_directive, file=f)
            # Handle rhombohedral axes for trigonal space groups
            elif is_trigonal(spacegroup) and use_rhombohedral_axes:
                print("0 1 0", file=f)  # Use rhombohedral axes setting
            else:
                print("0 0 0", file=f)  # Default: use standard setting

            print(spacegroup, file=f)

            # Generate and write unit cell line
            cell_params = [a, b, c, alpha, beta, gamma]
            cell_line = generate_unit_cell_line(spacegroup, cell_params, dimensionality)
            print(cell_line, file=f)
        elif dimensionality == "SLAB":
            print("SLAB", file=f)
            print(spacegroup, file=f)
            print(f"{a:.8f} {b:.8f} {gamma:.6f}", file=f)
        elif dimensionality == "POLYMER":
            print("POLYMER", file=f)
            print(spacegroup, file=f)
            print(f"{a:.8f}", file=f)
        elif dimensionality == "MOLECULE":
            print("MOLECULE", file=f)
            print("1", file=f)  # C1 symmetry for molecules

        # Write atomic positions
        print(str(len(atomic_numbers)), file=f)

        for i in range(len(atomic_numbers)):
            atomic_number = atomic_numbers[i]

            # Add 200 to atomic number ONLY if ECP is required for EXTERNAL basis sets
            if basis_set_type == "EXTERNAL" and atomic_number in ECP_ELEMENTS_EXTERNAL:
                atomic_number += 200
            # For internal basis sets, do NOT add 200 - they handle ECP internally

            # Write with different format depending on dimensionality (increased precision)
            print(
                f"{atomic_number} {positions[i][0]:.10f} {positions[i][1]:.10f} {positions[i][2]:.10f} Biso 1.000000 {symbols[i]}",
                file=f,
            )

        # Write calculation-type specific parameters
        if calculation_type == "OPT":
            # For geometry optimization
            write_optimization_section(f, optimization_type, optimization_settings)
        elif calculation_type == "FREQ":
            # For frequency calculation
            write_frequency_section(f, freq_settings)
        # For single point calculations, no additional sections needed

        # Handle HF and DFT methods differently
        if method == "HF":
            hf_method = options.get("hf_method", "RHF")

            # Handle 3C methods and basis sets
            if hf_method in ["HF3C", "HFSOL3C"]:
                # These are HF methods with corrections, write basis set
                write_basis_set_section(f, "INTERNAL", basis_set, atomic_numbers)

                # Add 3C corrections
                if hf_method == "HF3C":
                    print("HF3C", file=f)
                    print("END", file=f)
                elif hf_method == "HFSOL3C":
                    print("HFSOL3C", file=f)
                    print("END", file=f)
            else:
                # Standard HF methods (RHF, UHF)
                # Write basis set
                write_basis_set_section(
                    f, basis_set_type, basis_set, atomic_numbers, read_basis_file
                )

                # For UHF, add the UHF keyword
                if hf_method == "UHF":
                    print("UHF", file=f)

        else:  # DFT method
            dft_functional = options.get("dft_functional", "")
            use_dispersion = options.get("use_dispersion", False)
            dft_grid = options.get("dft_grid")  # Can be None for 3C methods

            # Write basis set
            write_basis_set_section(
                f, basis_set_type, basis_set, atomic_numbers, read_basis_file
            )

            # Write DFT section
            write_dft_section(
                f, dft_functional, use_dispersion, dft_grid, is_spin_polarized
            )

        # Write SCF parameters
        # Prepare k-points
        ka, kb, kc = generate_k_points(a, b, c, dimensionality, spacegroup)

        write_scf_section(
            f,
            tolerances,
            (ka, kb, kc),
            dimensionality,
            use_smearing,
            smearing_width,
            scf_method,
            scf_maxcycle,
            fmixing,
            len(atomic_numbers),
            spacegroup,
        )


def process_cifs(cif_directory, options, output_directory=None):
    """
    Process all CIF files in a directory

    Args:
        cif_directory (str): Directory containing CIF files
        options (dict): Calculation options
        output_directory (str, optional): Output directory for D12 files

    Returns:
        None
    """
    if output_directory is None:
        output_directory = cif_directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Find all CIF files
    cif_files = glob.glob(os.path.join(cif_directory, "*.cif"))

    if not cif_files:
        print(f"No CIF files found in {cif_directory}")
        return

    print(f"Found {len(cif_files)} CIF files to process")

    # Process each CIF file
    for cif_file in cif_files:
        base_name = os.path.basename(cif_file).replace(".cif", "")

        # Generate output filename
        dimensionality = options["dimensionality"]
        calc_type = options["calculation_type"]

        # Method identifier
        if options["method"] == "HF":
            method_name = options.get("hf_method", "RHF")
        else:
            method_name = options.get("dft_functional", "")
            # Don't add -D3 to 3C methods or if dispersion is already included in the name
            if (
                options.get("use_dispersion")
                and "-3C" not in method_name
                and "3C" not in method_name
            ):
                method_name += "-D3"

        symmetry_tag = "P1" if options["symmetry_handling"] == "P1" else "symm"

        if options["basis_set_type"] == "EXTERNAL":
            basis_name = os.path.basename(options["basis_set"].rstrip("/"))
        else:
            basis_name = options["basis_set"]

        output_name = f"{base_name}_{dimensionality}_{calc_type}_{symmetry_tag}_{method_name}_{basis_name}.d12"
        output_file = os.path.join(output_directory, output_name)

        try:
            print(f"Processing {cif_file}...")

            # Parse CIF file
            cif_data = parse_cif(cif_file)

            # Apply symmetry handling
            if options["symmetry_handling"] == "P1":
                # If P1 symmetry requested, override the spacegroup
                cif_data["spacegroup"] = 1
                print("Using P1 symmetry (no symmetry operations, all atoms explicit)")
            elif options["symmetry_handling"] == "SPGLIB":
                # If spglib symmetry requested and reduction is enabled
                if SPGLIB_AVAILABLE and options.get("reduce_to_asymmetric", True):
                    print("\nPerforming spglib symmetry analysis with verification...")
                    tolerance = options.get("symmetry_tolerance", 1e-5)
                    validate_symmetry = options.get("validate_symmetry", False)
                    cif_data = verify_and_reduce_to_asymmetric_unit(cif_data, tolerance, validate_symmetry)
            elif options["symmetry_handling"] == "CIF":
                # For CIF symmetry, optionally reduce to unique atoms based on user preference
                if options.get("write_only_unique", True):
                    if SPGLIB_AVAILABLE:
                        print("\nUsing CIF symmetry - verifying with spglib and reducing to asymmetric unit...")
                        tolerance = options.get("symmetry_tolerance", 1e-5)
                        validate_symmetry = options.get("validate_symmetry", False)
                        cif_data = verify_and_reduce_to_asymmetric_unit(cif_data, tolerance, validate_symmetry)
                    else:
                        print("Warning: Cannot identify unique atoms without spglib. Writing all atoms.")
                        print("Install spglib to enable asymmetric unit reduction: pip install spglib")
                else:
                    print("Using CIF symmetry but writing all atoms explicitly")

            # Create D12 file
            create_d12_file(cif_data, output_file, options)

            print(f"Created {output_file}")

        except Exception as e:
            print(f"Error processing {cif_file}: {e}")
            continue


def print_summary(options):
    """Print a summary of the selected options"""
    print("\n--- Selected Options Summary ---")

    # Method and functional
    if options.get("method") == "HF":
        print(f"Method: Hartree-Fock ({options.get('hf_method', 'RHF')})")
    else:
        functional = options.get("dft_functional", "")
        if options.get("use_dispersion"):
            functional += "-D3"
        print(f"Method: DFT")
        print(f"Functional: {functional}")

    # Basic settings
    print(f"Dimensionality: {options.get('dimensionality', 'CRYSTAL')}")
    print(f"Calculation type: {options.get('calculation_type', 'SP')}")
    if options.get("calculation_type") == "OPT":
        print(f"Optimization type: {options.get('optimization_type', 'FULLOPTG')}")

    # Symmetry settings
    print(f"Symmetry handling: {options.get('symmetry_handling', 'CIF')}")
    if options.get("symmetry_handling") == "SPGLIB":
        print(f"  - Tolerance: {options.get('symmetry_tolerance', 1e-5)}")
        print(f"  - Reduce to asymmetric: {options.get('reduce_to_asymmetric', False)}")

    # Basis set
    print(f"Basis set type: {options.get('basis_set_type', 'INTERNAL')}")
    print(f"Basis set: {options.get('basis_set', 'N/A')}")

    # Additional settings
    if options.get("method") == "DFT":
        print(f"DFT grid: {options.get('dft_grid', 'XLGRID')}")
    print(f"Spin polarized: {options.get('is_spin_polarized', False)}")
    if options.get("use_smearing"):
        print(f"Fermi smearing: Yes (width={options.get('smearing_width', 0.01)})")

    # Tolerances
    if "tolerances" in options:
        print(f"Tolerances:")
        for key, value in options["tolerances"].items():
            print(f"  - {key}: {value}")

    # SCF settings
    print(f"SCF method: {options.get('scf_method', 'DIIS')}")
    print(f"SCF maxcycle: {options.get('scf_maxcycle', 800)}")
    print(f"FMIXING: {options.get('fmixing', 30)}%")

    print("-------------------------------\n")


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(
        description="Convert CIF files to D12 input files for CRYSTAL23"
    )
    parser.add_argument(
        "--cif_dir", type=str, default="./", help="Directory containing CIF files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for D12 files (default: same as CIF directory)",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Run in batch mode using saved options"
    )
    parser.add_argument(
        "--save_options",
        action="store_true",
        help="Save options to file for batch mode",
    )
    parser.add_argument(
        "--options_file",
        type=str,
        default="cif2d12_options.json",
        help="File to save/load options for batch mode",
    )

    args = parser.parse_args()

    if args.batch:
        # Load options from file
        try:
            with open(args.options_file, "r") as f:
                options = json.load(f)
            print(f"Loaded options from {args.options_file}")
            print_summary(options)
        except Exception as e:
            print(f"Error loading options from {args.options_file}: {e}")
            print("Please run the script without --batch to create options file first")
            return
    else:
        # Get options interactively
        print("CIF to D12 Converter for CRYSTAL23")
        print("==================================")
        print("Enhanced by Marcus Djokic with AI assistance")
        print("")
        options = get_calculation_options()
        print_summary(options)

    # Always ask about saving options (unless in batch mode)
    if not args.batch:
        save_options = yes_no_prompt(
            "Save these options for future batch processing?", "yes"
        )
        if save_options or args.save_options:
            try:
                with open(args.options_file, "w") as f:
                    json.dump(options, f, indent=2)
                print(f"Saved options to {args.options_file}")
            except Exception as e:
                print(f"Error saving options to {args.options_file}: {e}")

    # Process CIF files
    process_cifs(args.cif_dir, options, args.output_dir)


if __name__ == "__main__":
    main()
