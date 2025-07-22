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
"""

import os
import sys
import glob
import argparse
import numpy as np
from ase.io import read
import json

# Import from new modular structure
from d12_constants import (
    # Constants
    ELEMENT_SYMBOLS,
    SYMBOL_TO_NUMBER,
    SPACEGROUP_SYMBOLS,
    DEFAULT_SETTINGS,
    DEFAULT_OPT_SETTINGS,
    DEFAULT_TOLERANCES,
    DEFAULT_FREQ_SETTINGS,
    FUNCTIONAL_CATEGORIES,
    COMMON_FUNCTIONALS,
    D3_FUNCTIONALS,
    DFT_GRID_OPTIONS,
    DISPERSION_OPTIONS,
    SMEARING_OPTIONS,
    PRINT_OPTIONS,
    MULTI_ORIGIN_SPACEGROUPS,
    ATOMIC_NUMBER_TO_SYMBOL,
    ECP_ELEMENTS_EXTERNAL,
    # Utility functions
    yes_no_prompt,
    get_valid_input,
    safe_float,
    safe_int,
    generate_unit_cell_line,
    read_basis_file,
    generate_k_points,
    check_basis_set_compatibility,
    get_user_input,
    # Configuration functions (from merged d12_config_common)
    configure_tolerances,
    configure_scf_settings,
    select_basis_set,
    configure_dft_grid,
    configure_dispersion,
    configure_spin_polarization,
    configure_smearing,
)
from d12_calc_freq import get_advanced_frequency_settings, write_frequency_section
from d12_calc_basic import write_optimization_section, configure_single_point
from d12_writer import (
    write_method_block,
    write_basis_block,
    write_scf_block,
    write_optimization_block,
    write_frequency_block,
    write_properties_block,
    write_print_options,
    write_k_points,
    write_spin_settings,
    write_smearing_settings,
    write_minimal_raman_section,
    write_dft_section,
    write_basis_set_section,
)
from d12_interactive import (
    display_default_settings,
    get_calculation_options_new,
    configure_symmetry_handling,
    configure_print_options,
    save_options_to_file,
    load_options_from_file,
)

# Import MACE configuration for paths
import sys
from pathlib import Path

# Try to import from mace_config first
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mace_config import BASIS_DOUBLEZETA_DIR, BASIS_TRIPLEZETA_DIR

    # Use absolute paths from config
    DEFAULT_DZ_PATH = str(BASIS_DOUBLEZETA_DIR) + "/"
    DEFAULT_TZ_PATH = str(BASIS_TRIPLEZETA_DIR) + "/"
except ImportError:
    # Fallback to local paths relative to this script
    DEFAULT_DZ_PATH = (
        "./basis_sets/full.basis.doublezeta/"  # DZVP-REV2 external basis set directory
    )
    DEFAULT_TZ_PATH = (
        "./basis_sets/full.basis.triplezeta/"  # TZVP-REV2 external basis set directory
    )

# Import write_scf_section from d12_writer
from d12_writer import write_scf_section

# Try to import spglib for symmetry operations
try:
    import spglib

    SPGLIB_AVAILABLE = True
except ImportError:
    SPGLIB_AVAILABLE = False
    print("Warning: spglib not found. Symmetry reduction features will be limited.")
    print("Install spglib for full symmetry functionality: pip install spglib")


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
        atomic_numbers = []
        for sym in symbols:
            atomic_num = SYMBOL_TO_NUMBER.get(sym)
            if atomic_num is None:
                raise ValueError(f"Unknown element symbol '{sym}' in CIF file")
            atomic_numbers.append(atomic_num)

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

        # Convert atom names to atomic numbers and filter out unknown elements
        atomic_numbers = []
        filtered_atom_names = []
        filtered_positions = []

        for i, name in enumerate(atom_name):
            try:
                atom = getattr(Element, name)
                atomic_num = int(atom)
            except (AttributeError, ValueError):
                atomic_num = SYMBOL_TO_NUMBER.get(name)
                if atomic_num is None:
                    print(
                        f"Warning: Unknown element symbol '{name}' at position {i} - skipping"
                    )
                    continue

            atomic_numbers.append(atomic_num)
            filtered_atom_names.append(name)
            filtered_positions.append([h[i], k[i], l[i]])

        data["atomic_numbers"] = atomic_numbers
        data["symbols"] = filtered_atom_names
        data["positions"] = filtered_positions

        return data


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


def verify_and_reduce_to_asymmetric_unit(
    cif_data, tolerance=1e-5, validate_symmetry=False
):
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
        lattice[2, 1] = (
            c
            * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad))
            / np.sin(gamma_rad)
        )
        lattice[2, 2] = (
            c
            * np.sqrt(
                1
                - np.cos(alpha_rad) ** 2
                - np.cos(beta_rad) ** 2
                - np.cos(gamma_rad) ** 2
                + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
            )
            / np.sin(gamma_rad)
        )

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

        detected_spacegroup_num = dataset["number"]
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
            print(
                f"  3: Use original CIF space group {original_spacegroup_num} (no reduction)"
            )

            choice = get_user_input(
                "Select option", {"1": "tolerance", "2": "spglib", "3": "original"}, "3"
            )

            if choice == "1":
                # Try different tolerances
                for test_tolerance in [1e-3, 1e-4, 1e-6, 1e-7]:
                    if test_tolerance != tolerance:
                        print(f"\nTrying tolerance {test_tolerance}...")
                        test_dataset = spglib.get_symmetry_dataset(
                            cell, symprec=test_tolerance
                        )
                        if (
                            test_dataset
                            and test_dataset["number"] == original_spacegroup_num
                        ):
                            print(f"✓ Match found with tolerance {test_tolerance}!")
                            use_tolerance = yes_no_prompt(
                                f"Use tolerance {test_tolerance}?", "yes"
                            )
                            if use_tolerance:
                                return verify_and_reduce_to_asymmetric_unit(
                                    cif_data, test_tolerance
                                )

                print("\nNo tolerance found that matches CIF space group.")
                final_choice = get_user_input(
                    "Final choice", {"1": "spglib", "2": "original"}, "2"
                )
                if final_choice == "2":
                    print("Using all atoms from the CIF file without reduction.")
                    return cif_data
                else:
                    print(
                        f"Proceeding with spglib space group {detected_spacegroup_num}"
                    )
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
        equivalent_atoms = dataset["equivalent_atoms"]
        unique_indices = []
        seen_representatives = set()

        for i, representative in enumerate(equivalent_atoms):
            if representative not in seen_representatives:
                unique_indices.append(i)
                seen_representatives.add(representative)

        # Verify atom count makes sense
        original_atom_count = len(numbers)
        unique_atom_count = len(unique_indices)
        symmetry_operations = len(dataset["rotations"])

        print(f"\nAsymmetric Unit Analysis:")
        print(f"  Original atoms: {original_atom_count}")
        print(f"  Unique atoms: {unique_atom_count}")
        print(f"  Symmetry operations: {symmetry_operations}")
        print(f"  Expected multiplicity: {original_atom_count / unique_atom_count:.1f}")

        # Check if the reduction makes sense
        if unique_atom_count >= original_atom_count:
            print("\n⚠️  WARNING: Asymmetric unit contains all or almost all atoms.")
            print(
                "     This suggests the structure may already be in the asymmetric unit,"
            )
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
            equiv_count = sum(
                1 for eq in equivalent_atoms if eq == equivalent_atoms[idx]
            )
            print(
                f"  {i + 1}: {symbol} at ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}) [orig #{idx + 1}, {equiv_count} equivalent]"
            )

        # Show which atoms are equivalent to which
        if len(unique_indices) < original_atom_count:
            print(f"\nEquivalence mapping:")
            for i, representative in enumerate(equivalent_atoms):
                if i not in unique_indices:
                    rep_idx = (
                        unique_indices.index(representative)
                        if representative in unique_indices
                        else -1
                    )
                    if rep_idx >= 0:
                        print(
                            f"  Atom {i + 1} ({cif_data['symbols'][i]}) → equivalent to asymmetric atom {rep_idx + 1}"
                        )

        # Optional: Validate that symmetry operations can reconstruct original structure
        if validate_symmetry:
            print("\nPerforming symmetry validation...")
            try:
                # Apply symmetry operations to asymmetric unit
                rotations = dataset["rotations"]
                translations = dataset["translations"]

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
                    print(
                        f"✓ Symmetry validation: Generated {len(reconstructed_positions)} positions from {len(unique_indices)} asymmetric atoms"
                    )
                    print(f"  Original structure had {original_atom_count} atoms")
                else:
                    print(
                        f"⚠️  Symmetry validation: Only generated {len(reconstructed_positions)} positions, expected {original_atom_count}"
                    )

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
    # Handle frequency settings - check for nested structure from JSON files
    freq_settings = options.get(
        "frequency_settings", options.get("freq_settings", DEFAULT_FREQ_SETTINGS)
    )

    # If freq_settings has a nested "freq_settings" key (from older JSON files), use the nested one
    if isinstance(freq_settings, dict) and "freq_settings" in freq_settings:
        # Extract the actual settings from the nested structure
        nested_settings = freq_settings["freq_settings"]
        # Preserve any top-level keys like tolerances
        for key in freq_settings:
            if key != "freq_settings" and key not in nested_settings:
                nested_settings[key] = freq_settings[key]
        freq_settings = nested_settings

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
            if dimensionality == "SLAB":
                # For SLAB: fractional a,b coordinates and Cartesian z coordinate
                z_cart = positions[i][2] * c  # Convert fractional z to Cartesian
                print(
                    f"{atomic_number} {positions[i][0]:.10f} {positions[i][1]:.10f} {z_cart:.6f} Biso 1.000000 {symbols[i]}",
                    file=f,
                )
            elif dimensionality == "POLYMER":
                # For POLYMER: fractional x, Cartesian y,z coordinates
                y_cart = positions[i][1] * b  # Convert fractional y to Cartesian
                z_cart = positions[i][2] * c  # Convert fractional z to Cartesian
                print(
                    f"{atomic_number} {positions[i][0]:.10f} {y_cart:.6f} {z_cart:.6f} Biso 1.000000 {symbols[i]}",
                    file=f,
                )
            elif dimensionality == "MOLECULE":
                # For MOLECULE: all Cartesian coordinates
                x_cart = positions[i][0] * a  # Convert fractional x to Cartesian
                y_cart = positions[i][1] * b  # Convert fractional y to Cartesian
                z_cart = positions[i][2] * c  # Convert fractional z to Cartesian
                print(
                    f"{atomic_number} {x_cart:.6f} {y_cart:.6f} {z_cart:.6f} Biso 1.000000 {symbols[i]}",
                    file=f,
                )
            else:
                # For CRYSTAL: all fractional coordinates
                print(
                    f"{atomic_number} {positions[i][0]:.10f} {positions[i][1]:.10f} {positions[i][2]:.10f} Biso 1.000000 {symbols[i]}",
                    file=f,
                )

        # Write calculation-type specific parameters BEFORE basis set
        if calculation_type == "OPT":
            # For geometry optimization - OPTGEOM block comes directly after coordinates
            write_optimization_section(f, optimization_type, optimization_settings)
        elif calculation_type == "FREQ":
            # For frequency calculation - FREQCALC block comes directly after coordinates
            # Determine crystal system from space group number
            crystal_system = None
            if spacegroup:
                if 1 <= spacegroup <= 2:
                    crystal_system = "triclinic"
                elif 3 <= spacegroup <= 15:
                    crystal_system = "monoclinic"
                elif 16 <= spacegroup <= 74:
                    crystal_system = "orthorhombic"
                elif 75 <= spacegroup <= 142:
                    crystal_system = "tetragonal"
                elif 143 <= spacegroup <= 167:
                    crystal_system = "trigonal"
                elif 168 <= spacegroup <= 194:
                    crystal_system = "hexagonal"
                elif 195 <= spacegroup <= 230:
                    crystal_system = "cubic"
            write_frequency_section(f, freq_settings, crystal_system, spacegroup)
        # For single point calculations, no additional sections needed

        # Handle HF and DFT methods differently
        if method == "HF":
            hf_method = options.get("hf_method", "RHF")

            # Handle 3C methods and basis sets
            if hf_method in ["HF3C", "HFSOL3C"]:
                # These are HF methods with corrections, write basis set
                print("BASISSET", file=f)
                print(basis_set, file=f)

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
                if basis_set_type == "INTERNAL":
                    print("BASISSET", file=f)
                    print(basis_set, file=f)
                else:
                    # External basis set handling
                    unique_atoms = set(atomic_numbers)
                    for atom_num in sorted(unique_atoms):
                        basis_content = read_basis_file(basis_set, atom_num)
                        if basis_content:
                            print(basis_content, file=f, end="")
                    print("99 0", file=f)
                    print("END", file=f)

                # For UHF, add the UHF keyword
                if hf_method == "UHF":
                    print("UHF", file=f)

        else:  # DFT method
            dft_functional = options.get("dft_functional", "")

            # Handle method_modifications
            if "method_modifications" in options:
                modifications = options["method_modifications"]
                if "functional" in modifications:
                    print(
                        f"Overriding functional from method_modifications: {modifications['functional']}"
                    )
                    dft_functional = modifications["functional"]
                    # Update 3C method flag if needed
                    if "3C" in dft_functional:
                        options["is_3c_method"] = True

            # Also check if functional is directly specified (for backward compatibility)
            if not dft_functional and "functional" in options:
                dft_functional = options["functional"]
                if "3C" in dft_functional:
                    options["is_3c_method"] = True

            use_dispersion = options.get("use_dispersion", False)
            dft_grid = options.get("dft_grid")  # Can be None for 3C methods

            # Write basis set
            if basis_set_type == "INTERNAL":
                print("BASISSET", file=f)
                print(basis_set, file=f)
            else:
                # External basis set handling
                unique_atoms = set(atomic_numbers)
                for atom_num in sorted(unique_atoms):
                    basis_content = read_basis_file(basis_set, atom_num)
                    if basis_content:
                        print(basis_content, file=f, end="")
                print("99 0", file=f)
                print("END", file=f)

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

        # Note: The single END at the very end is written by write_scf_section


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

            # Handle method_modifications for filename
            if "method_modifications" in options:
                modifications = options["method_modifications"]
                if "functional" in modifications:
                    method_name = modifications["functional"]

            # Also check if functional is directly specified
            if not method_name and "functional" in options:
                method_name = options["functional"]

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
                    cif_data = verify_and_reduce_to_asymmetric_unit(
                        cif_data, tolerance, validate_symmetry
                    )
            elif options["symmetry_handling"] == "CIF":
                # For CIF symmetry, optionally reduce to unique atoms based on user preference
                if options.get("write_only_unique", True):
                    if SPGLIB_AVAILABLE:
                        print(
                            "\nUsing CIF symmetry - verifying with spglib and reducing to asymmetric unit..."
                        )
                        tolerance = options.get("symmetry_tolerance", 1e-5)
                        validate_symmetry = options.get("validate_symmetry", False)
                        cif_data = verify_and_reduce_to_asymmetric_unit(
                            cif_data, tolerance, validate_symmetry
                        )
                    else:
                        print(
                            "Warning: Cannot identify unique atoms without spglib. Writing all atoms."
                        )
                        print(
                            "Install spglib to enable asymmetric unit reduction: pip install spglib"
                        )
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
        print("=" * 60)
        print("Enhanced by Marcus Djokic with AI assistance")
        print("")
        options = get_calculation_options_new()
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
