#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRYSTAL17/23 Optimization Output to D12 Converter
-------------------------------------------------
This script extracts optimized geometries from CRYSTAL17/23 output files
and creates new D12 input files for follow-up calculations.

DESCRIPTION:
    Takes the optimized geometry from CRYSTAL17/23 output files and creates
    new D12 input files with updated coordinates. The script attempts to
    preserve the original calculation settings and allows the user to
    modify them interactively.

USAGE:
    1. Single file processing:
       python CRYSTALOptToD12.py --out-file file.out --d12-file file.d12

    2. Process all files in a directory:
       python CRYSTALOptToD12.py --directory /path/to/files

    3. Batch processing with shared settings:
       python CRYSTALOptToD12.py --directory /path/to/files --shared-settings

    4. Specify output directory:
       python CRYSTALOptToD12.py --directory /path/to/files --output-dir /path/to/output

    5. Save/load settings:
       python CRYSTALOptToD12.py --save-options --options-file settings.json

AUTHOR:
    New entirely reworked script by Marcus Djokic
    Based on prior versions written by Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic
"""

import os
import sys
import re
import argparse
import json
from pathlib import Path

# Import from new modular structure
from d12_constants import (
    # Constants
    ELEMENT_SYMBOLS, SPACEGROUP_SYMBOLS, DEFAULT_SETTINGS,
    DEFAULT_OPT_SETTINGS, DEFAULT_TOLERANCES, DEFAULT_FREQ_SETTINGS,
    FUNCTIONAL_CATEGORIES, COMMON_FUNCTIONALS, D3_FUNCTIONALS,
    DFT_GRID_OPTIONS, DISPERSION_OPTIONS, SMEARING_OPTIONS,
    PRINT_OPTIONS, MULTI_ORIGIN_SPACEGROUPS, ATOMIC_NUMBER_TO_SYMBOL,
    ECP_ELEMENTS_EXTERNAL,
    # Utility functions
    yes_no_prompt, get_valid_input, safe_float, safe_int,
    generate_unit_cell_line, read_basis_file, generate_k_points,
    check_basis_set_compatibility,
    # Configuration functions (from merged d12_config_common)
    configure_tolerances, configure_scf_settings, select_basis_set,
    configure_dft_grid, configure_dispersion, configure_spin_polarization,
    configure_smearing
)
from d12_parsers import CrystalOutputParser, CrystalInputParser
from d12_calc_freq import get_advanced_frequency_settings, write_frequency_section
from d12_calc_basic import write_optimization_section, configure_single_point
from d12_writer import (
    write_method_block, write_basis_block, write_scf_block,
    write_optimization_block, write_frequency_block, write_properties_block,
    write_print_options, write_k_points, write_spin_settings,
    write_smearing_settings, write_minimal_raman_section,
    write_dft_section, write_basis_set_section
)
# Import write_scf_section from d12_writer  
from d12_writer import write_scf_section
from d12_interactive import (
    display_current_settings, interactive_d12_configuration,
    get_calculation_options_from_current, get_calculation_options,
    save_options_to_file, load_options_from_file
)



def _get_phonon_band_path_title(band_settings, geometry_data):
    """Generate path information string for phonon band calculations."""
    from d3_kpoints import get_band_path_from_symmetry, unicode_to_ascii_kpoint
    
    # For automatic paths, we need to determine the path based on space group
    if band_settings.get("auto_path", False) or band_settings.get("path") == "AUTO" or band_settings.get("path") == "auto":
        # Try to get path from space group
        space_group = geometry_data.get("spacegroup", 1)
        lattice_type = "P"  # Default
        
        # Try to extract lattice type from optimization content
        opt_content = geometry_data.get("optimization_content", "")
        if opt_content:
            import re
            sg_symbol_match = re.search(r'SPACE GROUP.*?:\s+([A-Z]\s*[\-/0-9\s]*[A-Z0-9]*)', opt_content)
            if sg_symbol_match:
                symbol = sg_symbol_match.group(1).strip()
                if symbol:
                    lattice_type = symbol[0]
        
        # Generate the path labels based on the format
        format_type = band_settings.get("format", "labels")
        path_method = band_settings.get("path_method", "labels")
        
        # For all formats, we get the appropriate labels
        if format_type == "seekpath" and band_settings.get("seekpath_full", False):
            # Try to get SeeK-path labels
            try:
                from d3_kpoints import get_seekpath_labels
                path_labels = get_seekpath_labels(space_group, lattice_type)
            except:
                # Fallback to standard labels
                path_labels = get_band_path_from_symmetry(space_group, lattice_type)
        else:
            # For all other formats (labels, vectors, literature), use standard labels
            path_labels = get_band_path_from_symmetry(space_group, lattice_type)
    elif "path_labels" in band_settings:
        path_labels = band_settings["path_labels"]
    else:
        # Try to extract from path if it's in label format
        path = band_settings.get("path", [])
        if path and isinstance(path[0], str) and " " in path[0]:
            # Extract labels from segments like "G X", "X M"
            path_labels = []
            for segment in path:
                parts = segment.split()
                if path_labels and parts[0] == path_labels[-1]:
                    # Continuous path
                    path_labels.append(parts[1])
                else:
                    # Discontinuous path
                    if path_labels:
                        path_labels.append("|")
                    path_labels.extend(parts)
        else:
            return None
    
    # Convert labels to ASCII and format
    path_str = []
    for label in path_labels:
        if label == "|":
            path_str.append("|")
        else:
            path_str.append(unicode_to_ascii_kpoint(label))
    
    # Determine the k-path source for the title
    kpath_source = band_settings.get("kpath_source", "default")
    if kpath_source == "seekpath_inv":
        source_info = " - SeeKPath (w.I)"
    elif kpath_source == "seekpath_noinv":
        source_info = " - SeeKPath (no.I)"
    elif kpath_source == "seekpath":
        source_info = " - SeeKPath"
    elif kpath_source == "literature":
        source_info = " - Literature"
    elif kpath_source == "manual":
        source_info = " - Manual"
    elif kpath_source == "template":
        source_info = " - Template"
    elif kpath_source == "fractional":
        source_info = " - Fractional"
    else:
        source_info = " - default"
    
    # Create title string
    return " - Phonon Band Structure" + source_info + " - " + "-".join(path_str)


def write_d12_file(output_file, geometry_data, settings, external_basis_data=None):
    """Write new D12 file with optimized geometry and settings"""

    with open(output_file, "w") as f:
        # Title
        title = output_file.replace(".d12", "")
        
        # Add phonon band path information if this is a FREQ calculation with bands
        calc_type = settings.get("calculation_type", settings.get("calc_type", "OPT"))
        if calc_type == "FREQ":
            freq_settings = settings.get("freq_settings", {})
            if freq_settings.get("dispersion", False) and "bands" in freq_settings:
                band_settings = freq_settings["bands"]
                # Debug output
                # print(f"DEBUG: band_settings = {band_settings}")
                path_info = _get_phonon_band_path_title(band_settings, geometry_data)
                if path_info:
                    title += path_info
        
        f.write(f"{title}\n")

        # Structure section
        dimensionality = settings.get("dimensionality", "CRYSTAL")
        f.write(f"{dimensionality}\n")

        if dimensionality == "CRYSTAL":
            # Handle space groups with multiple origins
            spacegroup = settings.get("spacegroup", 1)
            origin_setting = settings.get("origin_setting", "0 0 0")

            # Check if this space group has special origin settings
            if spacegroup in MULTI_ORIGIN_SPACEGROUPS:
                spg_info = MULTI_ORIGIN_SPACEGROUPS[spacegroup]
                # Preserve the original origin setting if it matches known alternatives
                if origin_setting == spg_info.get('alt_crystal_code', ''):
                    f.write(f"{origin_setting}\n")  # Use original alternate origin
                elif origin_setting == spg_info.get('crystal_code', '0 0 0'):
                    f.write(f"{origin_setting}\n")  # Use original default origin
                else:
                    # Fallback to extracted origin setting to preserve original
                    f.write(f"{origin_setting}\n")
            else:
                f.write(f"{origin_setting}\n")

            f.write(f"{spacegroup}\n")
        elif dimensionality in ["SLAB", "POLYMER"]:
            f.write(f"{settings.get('spacegroup', 1)}\n")
        elif dimensionality == "MOLECULE":
            f.write("1\n")  # C1 symmetry

        # Unit cell parameters (if not molecule)
        if dimensionality != "MOLECULE" and geometry_data.get("conventional_cell"):
            cell_line = generate_unit_cell_line(
                settings.get("spacegroup", 1),
                geometry_data["conventional_cell"],
                dimensionality,
            )
            if cell_line:
                f.write(f"{cell_line}\n")

        # Atomic coordinates - filter based on symmetry preference
        coords = geometry_data["coordinates"]

        # Filter coordinates if requested
        if settings.get("write_only_unique", False):
            coords_to_write = [c for c in coords if c.get("is_unique", True)]
        else:
            coords_to_write = coords

        f.write(f"{len(coords_to_write)}\n")

        for atom in coords_to_write:
            atom_num = int(atom["atom_number"])
            # Store original atomic number for symbol lookup
            original_atom_num = atom_num
            
            # Add 200 to atomic number ONLY if ECP is required for EXTERNAL basis sets
            if (
                settings.get("basis_set_type") == "EXTERNAL"
                and atom_num in ECP_ELEMENTS_EXTERNAL
            ):
                atom_num += 200
            # For internal basis sets, do NOT add 200 - they handle ECP internally

            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(original_atom_num, "X")
            
            # Handle different coordinate systems based on dimensionality
            if dimensionality == "SLAB":
                # For SLAB: fractional a,b coordinates and Cartesian z coordinate
                # In CRYSTAL output files for SLAB, z-coordinates from "ATOMS IN THE ASYMMETRIC UNIT"
                # are already in Angstroms (shown as Z(ANGSTROM) in header), not fractional
                z_cart = float(atom['z'])
                f.write(
                    f"{atom_num} {atom['x']} {atom['y']} {z_cart:.6f} Biso 1.000000 {symbol}\n"
                )
            elif dimensionality == "POLYMER":
                # For POLYMER: fractional x, Cartesian y,z coordinates
                # In CRYSTAL output files for POLYMER, y,z coordinates should already be in Angstroms
                y_cart = float(atom['y'])
                z_cart = float(atom['z'])
                f.write(
                    f"{atom_num} {atom['x']} {y_cart:.6f} {z_cart:.6f} Biso 1.000000 {symbol}\n"
                )
            elif dimensionality == "MOLECULE":
                # For MOLECULE: all Cartesian coordinates
                # In CRYSTAL output files for MOLECULE, all coordinates should be in Angstroms
                x_cart = float(atom['x'])
                y_cart = float(atom['y'])
                z_cart = float(atom['z'])
                f.write(
                    f"{atom_num} {x_cart:.6f} {y_cart:.6f} {z_cart:.6f} Biso 1.000000 {symbol}\n"
                )
            else:
                # For CRYSTAL: all fractional coordinates
                f.write(
                    f"{atom_num} {atom['x']} {atom['y']} {atom['z']} Biso 1.000000 {symbol}\n"
                )

        # Calculation-specific section - MUST come before basis set
        if settings["calculation_type"] == "OPT":
            # For OPT: OPTGEOM follows directly after coordinates, no END needed
            write_optimization_section(
                f,
                settings.get("optimization_type", "FULLOPTG"),
                settings.get("optimization_settings", DEFAULT_OPT_SETTINGS),
            )
        elif settings["calculation_type"] == "FREQ":
            # Check if this is ANHARM or FREQCALC
            if settings.get("freq_mode") == "ANHARM":
                # For ANHARM: Goes outside geometry block
                # First close the geometry section
                f.write("END\n")
                # Then write ANHARM section
                anharm_settings = settings.get("anharm_settings", {})
                # Convert format from UI to d12creation
                converted_anharm = {
                    "atom_label": anharm_settings.get("h_atom", 1),
                    "keepsymm": anharm_settings.get("keep_symmetry", False),
                    "points": 26 if anharm_settings.get("points26", False) else 7
                }
                # Handle isotopes format conversion
                if "isotopes" in anharm_settings:
                    isotope_dict = {}
                    for atom_label, mass in anharm_settings["isotopes"]:
                        isotope_dict[atom_label] = mass
                    converted_anharm["isotopes"] = isotope_dict
                    
                from d12_calc_freq import write_anharm_section
                write_anharm_section(f, converted_anharm)
            else:
                # For FREQCALC: FREQCALC follows directly after coordinates, no END needed
                # Determine crystal system from space group
                spacegroup = settings.get("spacegroup", 1)
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
                        
                # Get optimization section text if available
                optimization_section = geometry_data.get("optimization_content", None)
                    
                write_frequency_section(
                    f, settings.get("freq_settings", DEFAULT_FREQ_SETTINGS), 
                    crystal_system, spacegroup, optimization_section
                )
        else:
            # For SP: Plain SP calculations don't have OPTGEOM/FREQCALC blocks
            # Just continue to basis set section
            pass

        # Handle basis sets and method section
        functional = settings.get("functional", "")
        method = "HF" if functional in ["RHF", "UHF", "HF3C", "HFSOL3C"] else "DFT"

        # Handle HF 3C methods and regular HF methods
        if functional in ["HF3C", "HFSOL3C"]:
            # These are HF methods with corrections, write basis set but no DFT block
            # Write BASISSET keyword for internal basis sets
            f.write("BASISSET\n")
            f.write(f"{settings['basis_set']}\n")

            # Add 3C corrections
            if functional == "HF3C":
                f.write("HF3C\n")
                f.write("END\n")
            elif functional == "HFSOL3C":
                f.write("HFSOL3C\n")
                f.write("END\n")
        elif functional in ["RHF", "UHF"]:
            # Standard HF methods
            if settings.get("basis_set_type") == "EXTERNAL":
                # External basis set handling - need END to close geometry section
                f.write("END\n")
                # Handle external basis sets for HF methods
                if settings.get("use_original_external_basis") and external_basis_data:
                    # Use the external basis from the original file
                    for line in external_basis_data:
                        f.write(f"{line}\n")
                    f.write("99 0\n")
                    f.write("END\n")
                elif settings.get("basis_set_path"):
                    # Read basis sets from specified path
                    f.write(f"# External basis set from: {settings['basis_set_path']}\n")
                    unique_atoms = set()
                    for atom in coords_to_write:
                        unique_atoms.add(int(atom["atom_number"]))

                    # Read basis set files
                    for atom_num in sorted(unique_atoms):
                        basis_file = os.path.join(settings["basis_set_path"], str(atom_num))
                        if os.path.exists(basis_file):
                            with open(basis_file, "r") as bf:
                                f.write(bf.read())
                        else:
                            print(f"Warning: Basis set file not found for element {atom_num}")

                    f.write("99 0\n")
                    f.write("END\n")
            else:
                # Internal basis set
                f.write("BASISSET\n")
                f.write(f"{settings.get('basis_set', 'POB-TZVP-REV2')}\n")
            
            # For UHF, add the UHF keyword
            if functional == "UHF":
                f.write("UHF\n")
        elif functional in ["PBEH3C", "HSE3C", "B973C", "PBESOL03C", "HSESOL3C"]:
            # DFT 3C methods
            f.write("BASISSET\n")
            f.write(f"{settings['basis_set']}\n")

            # Use write_dft_section to properly handle XLGRID for HSESOL3C
            write_dft_section(
                f,
                functional,
                False,  # No dispersion for 3C methods
                settings.get("dft_grid", "XLGRID"),
                settings.get("spin_polarized"),
            )
        else:
            # Standard basis set and method handling
            if settings.get("basis_set_type") == "EXTERNAL":
                # External basis set handling - need END to close geometry section
                f.write("END\n")
                # Write external basis set data
                if settings.get("use_original_external_basis") and external_basis_data:
                    # Use the external basis from the original file
                    for line in external_basis_data:
                        f.write(f"{line}\n")
                    f.write("99 0\n")
                    f.write("END\n")
                elif settings.get("basis_set_path"):
                    # Read basis sets from specified path
                    f.write(
                        f"# External basis set from: {settings['basis_set_path']}\n"
                    )
                    unique_atoms = set()
                    for atom in coords_to_write:
                        unique_atoms.add(int(atom["atom_number"]))

                    # Read basis set files
                    for atom_num in sorted(unique_atoms):
                        basis_file = os.path.join(
                            settings["basis_set_path"], str(atom_num)
                        )
                        if os.path.exists(basis_file):
                            with open(basis_file, "r") as bf:
                                f.write(bf.read())
                        else:
                            print(
                                f"Warning: Basis set file not found for element {atom_num}"
                            )

                    f.write("99 0\n")
                    f.write("END\n")
            else:
                # Internal basis set
                f.write("BASISSET\n")
                f.write(f"{settings.get('basis_set', 'POB-TZVP-REV2')}\n")

            # Write method section
            if method == "HF":
                # Handle HF methods
                if functional == "UHF":
                    f.write("UHF\n")
                # RHF is default, no keyword needed
            else:
                # Write DFT section
                write_dft_section(
                    f,
                    functional,
                    settings.get("dispersion"),
                    settings.get("dft_grid", "XLGRID"),
                    settings.get("spin_polarized"),
                )

        # SCF parameters section
        atomic_numbers = [int(atom["atom_number"]) for atom in coords_to_write]

        # Check basis set compatibility
        is_compatible, missing_elements = check_basis_set_compatibility(
            settings.get("basis_set", "POB-TZVP-REV2"), 
            atomic_numbers, 
            settings.get("basis_set_type", "INTERNAL")
        )
        
        if not is_compatible:
            print(
                f"\nWARNING: The selected basis set '{settings.get('basis_set')}' does not support all elements in your structure!"
            )
            print(
                f"Missing elements: {', '.join([f'{ATOMIC_NUMBER_TO_SYMBOL.get(z, z)} (Z={z})' for z in missing_elements])}"
            )
            if not yes_no_prompt("\nDo you want to continue anyway?"):
                print("Aborting D12 file creation.")
                return

        # Prepare k-points with same logic as d12creation.py
        k_points_info = None
        if settings.get("k_points"):
            k_points_raw = settings["k_points"]
            
            # Handle different k-points formats from extraction
            if isinstance(k_points_raw, tuple) and len(k_points_raw) == 3:
                # Tuple format (ka, kb, kc) from extraction
                k_points_info = k_points_raw
            elif isinstance(k_points_raw, str):
                # String format like "12 12 12" from extraction
                try:
                    parts = k_points_raw.split()
                    if len(parts) == 3:
                        k_points_info = (int(parts[0]), int(parts[1]), int(parts[2]))
                    elif len(parts) == 1:
                        # Single value - apply to all directions
                        k = int(parts[0])
                        k_points_info = (k, k, k)
                except (ValueError, IndexError):
                    k_points_info = None
                    
        # Fallback: Generate k-points based on cell size if not extracted
        if k_points_info is None and geometry_data.get("conventional_cell"):
            a, b, c = [float(x) for x in geometry_data["conventional_cell"][:3]]
            k_points_info = generate_k_points(
                a, b, c, dimensionality, settings.get("spacegroup", 1)
            )

        # Enhanced k-points handling with symmetry consideration
        enhanced_k_points = k_points_info
        if k_points_info and dimensionality == "CRYSTAL":
            ka, kb, kc = k_points_info
            spacegroup = settings.get("spacegroup", 1)
            
            # For symmetrized structures (non-P1), prefer uniform k-points
            # This ensures compatibility with simplified SHRINK format
            if spacegroup != 1 and (ka != kb or kb != kc or ka != kc):
                # Use maximum k-point for uniform sampling in symmetrized structures
                k_max = max(ka, kb, kc)
                enhanced_k_points = (k_max, k_max, k_max)
                print(f"Note: Using uniform k-points ({k_max},{k_max},{k_max}) for symmetrized structure (space group {spacegroup})")

        write_scf_section(
            f,
            settings.get("tolerances", DEFAULT_TOLERANCES),
            enhanced_k_points,
            dimensionality,
            settings.get("smearing"),
            settings.get("smearing_width", 0.01),
            settings.get("scf_settings", {}).get("method", "DIIS"),
            settings.get("scf_settings", {}).get("maxcycle", 800),
            settings.get("scf_settings", {}).get("fmixing", 30),
            len(atomic_numbers),
            settings.get("spacegroup", 1),
        )
        
        # Note: The single END at the very end is written by write_scf_section


def process_files(output_file, input_file=None, shared_settings=None, config_file=None, non_interactive=False, calc_type=None, opt_type=None, origin_setting="auto"):
    """Process CRYSTAL output and input files

    Args:
        output_file: Path to .out file
        input_file: Path to .d12 file (optional)
        shared_settings: Pre-defined settings to use (optional)
        config_file: Path to JSON config file (optional)
        non_interactive: Run in non-interactive mode (optional)
        calc_type: Calculation type for non-interactive mode (optional)
        opt_type: Optimization type for non-interactive mode (optional)
        origin_setting: Origin setting for non-interactive mode (optional)

    Returns:
        tuple: (success, settings_used)
    """

    # Parse output file
    print(f"\nParsing output file: {output_file}")
    out_parser = CrystalOutputParser(output_file)
    try:
        out_data = out_parser.parse()
    except Exception as e:
        print(f"Error parsing output file: {e}")
        return False, None

    # Parse input file if provided
    settings = out_data.copy()
    external_basis_data = []

    if input_file and os.path.exists(input_file):
        print(f"Parsing input file: {input_file}")
        in_parser = CrystalInputParser(input_file)
        try:
            in_data = in_parser.parse()

            # Merge data, with special handling for DFT settings
            for key, value in in_data.items():
                if key not in settings or settings[key] is None:
                    settings[key] = value
                elif key in ["functional", "dispersion", "spin_polarized", "dft_grid", "method", 
                           "is_3c_method", "use_smearing", "smearing_width", 
                           "k_points", "scf_method", "scf_maxcycle", "fmixing", "scf_direct",
                           "mulliken_analysis", "diis_history", "calculation_type", 
                           "optimization_settings", "freq_settings", "origin_setting", 
                           "spacegroup", "dimensionality", "tolerances"]:
                    # For all calculation settings, prefer input file (.d12) over output file (.out)
                    # because .d12 contains the original user-specified settings
                    # INCLUDING tolerances - the output parser has issues extracting these correctly
                    if value is not None:
                        settings[key] = value
                        # Debug output for symmetry-related settings
                        if key in ["origin_setting", "spacegroup", "dimensionality"]:
                            print(f"  Preserving {key} from D12 file: {value} (was {settings.get(key, 'not set')} from output)")
                elif key == "scf_settings":
                    # Merge SCF settings
                    if "scf_settings" not in settings:
                        settings["scf_settings"] = {}
                    settings["scf_settings"].update(value)

            # Store external basis data
            external_basis_data = in_data.get("external_basis_data", [])
        except Exception as e:
            print(f"Warning: Error parsing input file: {e}")
            print("Continuing with output file data only")

    # Set defaults if not found
    if not settings.get("spacegroup"):
        if settings["dimensionality"] == "MOLECULE":
            settings["spacegroup"] = 1
        else:
            print("Warning: Space group not found. Defaulting to P1")
            settings["spacegroup"] = 1

    if not settings.get("basis_set"):
        settings["basis_set"] = "POB-TZVP-REV2"
        settings["basis_set_type"] = "INTERNAL"

    # Set default tolerances if not found
    if not settings.get("tolerances"):
        settings["tolerances"] = DEFAULT_TOLERANCES.copy()

    # Set default SCF settings if not found
    if not settings.get("scf_settings"):
        settings["scf_settings"] = {"method": "DIIS", "maxcycle": 800, "fmixing": 30}

    # Get user options or use shared settings
    if config_file:
        # Config file takes precedence - process it first
        # Load settings from config file
        print(f"\nLoading settings from config file: {config_file}")
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Show config summary
            print("\n" + "="*60)
            print("CONFIG FILE SETTINGS")
            print("="*60)
            print(f"Calculation type: {config_data.get('calculation_type', 'Not specified')}")
            
            # Check for method_modifications
            if 'method_modifications' in config_data:
                method_mods = config_data['method_modifications']
                if 'new_functional' in method_mods:
                    print(f"Functional: {method_mods['new_functional']} (via method_modifications)")
                elif 'functional' in method_mods:
                    print(f"Functional: {method_mods['functional']} (via method_modifications)")
                else:
                    print(f"Functional: {config_data.get('functional', 'Not specified')}")
            else:
                print(f"Method: {config_data.get('method', 'Not specified')}")
                print(f"Functional: {config_data.get('functional', 'Not specified')}")
            
            if config_data.get('dispersion'):
                print(f"Dispersion: Yes")
            print(f"Basis set: {config_data.get('basis_set', 'Not specified')}")
            print(f"DFT grid: {config_data.get('dft_grid', 'Not specified')}")
            
            # Show tolerance modifications if present
            if 'tolerance_modifications' in config_data:
                tol_mods = config_data['tolerance_modifications']
                if 'custom_tolerances' in tol_mods:
                    print(f"Custom tolerances: {tol_mods['custom_tolerances']}")
            
            print("="*60)
            
            # Ask user if they want to apply these settings (skip in non-interactive mode)
            if non_interactive:
                apply_config = True
                print("\nApplying config file settings (non-interactive mode).")
            else:
                apply_config = yes_no_prompt("\nApply these settings from config file?", default="yes")
            
            if apply_config:
                # Use settings from config file
                options = settings.copy()
                # Override with config file settings
                for key, value in config_data.items():
                    if key not in ["coordinates", "primitive_cell", "conventional_cell"]:
                        options[key] = value
                
                # Handle both "frequency_settings" (from workflow) and "freq_settings" (direct usage)
                freq_key = None
                if "frequency_settings" in options:
                    freq_key = "frequency_settings"
                elif "freq_settings" in options:
                    freq_key = "freq_settings"
                
                if freq_key and isinstance(options[freq_key], dict):
                    # Rename to freq_settings for consistency with rest of script
                    if freq_key == "frequency_settings":
                        options["freq_settings"] = options.pop("frequency_settings")
                    
                    # Convert temprange from dict to tuple if needed
                    if "temprange" in options["freq_settings"] and isinstance(options["freq_settings"]["temprange"], dict):
                        temprange_dict = options["freq_settings"]["temprange"]
                        options["freq_settings"]["temprange"] = (
                            temprange_dict.get("n_temps", 20),
                            temprange_dict.get("t_min", 0),
                            temprange_dict.get("t_max", 400)
                        )
                    # Also convert pressrange if it exists
                    if "pressrange" in options["freq_settings"] and isinstance(options["freq_settings"]["pressrange"], dict):
                        pressrange_dict = options["freq_settings"]["pressrange"]
                        options["freq_settings"]["pressrange"] = (
                            pressrange_dict.get("n_press", 20),
                            pressrange_dict.get("p_min", 0),
                            pressrange_dict.get("p_max", 10)
                        )
                
                # Set write_only_unique if not specified in config
                if "write_only_unique" not in options:
                    # Check if original input had space group > 1 (not P1)
                    if settings.get("spacegroup", 1) > 1:
                        # For symmetric structures, default to writing only unique atoms
                        options["write_only_unique"] = config_data.get("write_only_unique", True)
                    else:
                        # For P1 structures, write all atoms
                        options["write_only_unique"] = config_data.get("write_only_unique", False)
                
                # Handle method_modifications if present
                if "method_modifications" in config_data:
                    method_mods = config_data["method_modifications"]
                    if "new_functional" in method_mods:
                        options["functional"] = method_mods["new_functional"]
                        print(f"  Functional changed to: {method_mods['new_functional']}")
                    elif "functional" in method_mods:
                        options["functional"] = method_mods["functional"]
                        print(f"  Functional changed to: {method_mods['functional']}")
                    
                    # Check if the selected functional is a 3C method
                    functional_name = method_mods.get("new_functional") or method_mods.get("functional")
                    if functional_name:
                        # If it's a 3C method, update basis set
                        if functional_name in ["PBEH3C", "HSE3C", "B973C", "PBESOL03C", "HSESOL3C", "HF3C", "HFSOL3C"]:
                            options["is_3c_method"] = True
                            options["basis_set_type"] = "INTERNAL"
                            # Get the basis set for this 3C method
                            # FUNCTIONAL_CATEGORIES already imported from d12_constants
                            for category in FUNCTIONAL_CATEGORIES.values():
                                if "basis_requirements" in category:
                                    if functional_name in category["basis_requirements"]:
                                        options["basis_set"] = category["basis_requirements"][functional_name]
                                        print(f"  Basis set updated to: {options['basis_set']} (required for 3C method)")
                            # For HSESOL3C, ensure XLGRID is set
                            if functional_name == "HSESOL3C":
                                options["dft_grid"] = "XLGRID"
                                print(f"  DFT grid set to: XLGRID (required for HSESOL3C)")
                    if "keep_spin" in method_mods and not method_mods["keep_spin"]:
                        options["spin_polarized"] = False
                    if "keep_grid" in method_mods and not method_mods["keep_grid"]:
                        options["dft_grid"] = None
                
                # Handle tolerance_modifications if present
                if "tolerance_modifications" in config_data:
                    tol_mods = config_data["tolerance_modifications"]
                    if "custom_tolerances" in tol_mods:
                        options["tolerances"] = tol_mods["custom_tolerances"]
                        print(f"  Tolerances updated: {tol_mods['custom_tolerances']}")
                    
                print("Config file settings applied.")
            else:
                # Fall back to interactive mode
                options = get_calculation_options_from_current(settings)
                
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Falling back to interactive mode.")
            options = get_calculation_options_from_current(settings)
    elif non_interactive and not calc_type:
        # True non-interactive mode (no config file, no calc type specified)
        options = settings.copy()
        # Default to SP if not specified
        options["calculation_type"] = "SP"
        
        # Set optimization type if it's an OPT calculation
        if options["calculation_type"] == "OPT":
            if opt_type:
                options["optimization_type"] = opt_type
            else:
                # Default to FULLOPTG
                options["optimization_type"] = "FULLOPTG"
        
        # Handle origin setting
        if origin_setting == "auto":
            # Auto-detect based on space group
            spacegroup = settings.get('spacegroup', 1)
            if 143 <= spacegroup <= 194:
                # Hexagonal space groups
                options["origin_setting"] = "0 1 0"
            elif spacegroup in [146, 148, 155, 160, 161, 166, 167]:
                # Rhombohedral space groups that can use hexagonal setting
                options["origin_setting"] = "0 1 0"
            else:
                # Most other space groups use standard setting
                options["origin_setting"] = "0 0 1"
        else:
            options["origin_setting"] = origin_setting
        
        # Keep all other settings from the extracted data
        if "write_only_unique" not in options:
            # Check if original input had space group > 1 (not P1)
            if settings.get("spacegroup", 1) > 1:
                # For symmetric structures, default to writing only unique atoms
                options["write_only_unique"] = True
            else:
                # For P1 structures, write all atoms
                options["write_only_unique"] = False
            
        print("\nRunning in non-interactive mode with settings:")
        print(f"  Calculation type: {options['calculation_type']}")
        if options['calculation_type'] == 'OPT':
            print(f"  Optimization type: {options['optimization_type']}")
        print(f"  Origin setting: {options['origin_setting']}")
    elif non_interactive and calc_type:
        # When calc_type is provided but --non-interactive is set,
        # we still want interactive mode like the D3 scripts
        # This is used by the workflow manager for expert mode
        options = get_calculation_options_from_current(settings, calc_type=calc_type)
        
    elif shared_settings:
        # Merge shared settings with current settings
        options = settings.copy()
        # Override with shared settings (except geometry-specific data)
        for key, value in shared_settings.items():
            if key not in [
                "coordinates",
                "primitive_cell",
                "conventional_cell",
                "spacegroup",
                "dimensionality",
                "origin_setting",
            ]:
                options[key] = value

        # Ensure consistency for 3C methods
        if options.get("functional") in [
            "HF3C",
            "HFSOL3C",
            "PBEH3C",
            "HSE3C",
            "B973C",
            "PBESOL03C",
            "HSESOL3C",
        ]:
            options["dispersion"] = False
            options["is_3c_method"] = True
            options["dft_grid"] = None
    else:
        # Interactive mode (possibly with pre-selected calc_type)
        options = get_calculation_options_from_current(settings, calc_type=calc_type)
        
        # Ensure write_only_unique is set based on space group
        if "write_only_unique" not in options:
            if settings.get("spacegroup", 1) > 1:
                options["write_only_unique"] = True
            else:
                options["write_only_unique"] = False

    # Ensure symmetry settings are preserved from original input
    if "spacegroup" not in options and "spacegroup" in settings:
        options["spacegroup"] = settings["spacegroup"]
    if "origin_setting" not in options and "origin_setting" in settings:
        options["origin_setting"] = settings["origin_setting"]
    if "dimensionality" not in options and "dimensionality" in settings:
        options["dimensionality"] = settings["dimensionality"]
    
    # Always ensure write_only_unique is set based on space group
    # This applies to all modes (interactive, shared, non-interactive)
    if "write_only_unique" not in options:
        spacegroup = options.get("spacegroup", settings.get("spacegroup", 1))
        if spacegroup > 1:
            options["write_only_unique"] = True
        else:
            options["write_only_unique"] = False
    
    # Create output filename
    base_name = os.path.splitext(output_file)[0]
    calc_type = options["calculation_type"]
    functional = options.get("functional", "RHF")

    # Don't add -D3 to 3C methods or HF methods or if dispersion is already included in the name
    if (
        options.get("dispersion")
        and "-3C" not in functional
        and "3C" not in functional
        and functional not in ["RHF", "UHF", "HF3C", "HFSOL3C"]
    ):
        functional += "-D3"

    new_filename = f"{base_name}_{calc_type.lower()}_{functional}_optimized.d12"

    # Write new D12 file
    print(f"\nWriting new D12 file: {new_filename}")
    
    # Debug output for symmetry settings
    print(f"Symmetry settings:")
    print(f"  Space group: {options.get('spacegroup', 'Not set')}")
    print(f"  Origin setting: {options.get('origin_setting', 'Not set')}")
    print(f"  Dimensionality: {options.get('dimensionality', 'Not set')}")
    print(f"  Write only unique atoms: {options.get('write_only_unique', 'Not set')}")
    
    # Convert frequency settings if present
    if "freq_settings" in options:
        converted_options = options.copy()
        freq_settings = options["freq_settings"].copy()
        
        # Pass space group and Bravais lattice info for high-symmetry point generation
        if "spacegroup" in out_data:
            freq_settings["space_group"] = out_data["spacegroup"]
        
        # Try to determine Bravais lattice from the geometry
        # This is a simplified approach - could be enhanced with proper symmetry analysis
        if "spacegroup" in out_data and out_data["spacegroup"] is not None:
            sg = out_data["spacegroup"]
            # Determine Bravais lattice based on space group ranges
            # This is a simplified mapping - ideally would extract from symmetry operations
            if sg <= 2:
                bravais = "P"  # Triclinic
            elif sg <= 15:
                bravais = "P" if sg <= 9 else "C"  # Monoclinic
            elif sg <= 74:
                # Orthorhombic - needs more detailed analysis
                if sg in [20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68]:
                    bravais = "C"
                elif sg in [22, 42, 43, 69, 70]:
                    bravais = "F"
                elif sg in [23, 24, 44, 45, 46, 71, 72, 73, 74]:
                    bravais = "I"
                else:
                    bravais = "P"
            elif sg <= 142:
                # Tetragonal
                bravais = "I" if sg >= 79 else "P"
            elif sg <= 167:
                # Trigonal/Rhombohedral
                bravais = "R" if sg in [146, 148, 155, 160, 161, 166, 167] else "P"
            elif sg <= 194:
                # Hexagonal
                bravais = "P"
            else:
                # Cubic
                if sg in [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228]:
                    bravais = "F"
                elif sg in [197, 199, 204, 206, 211, 214, 217, 220, 229, 230]:
                    bravais = "I"
                else:
                    bravais = "P"
            
            freq_settings["bravais_lattice"] = bravais
        
        # Handle phonon bands conversion
        if freq_settings.get("phonon_bands", False):
            # Remove the phonon_bands flag and replace with proper bands dict
            freq_settings.pop("phonon_bands", None)
            
            # Create bands dictionary
            bands_dict = {
                "shrink": freq_settings.get("shrink", 16),
                "npoints": freq_settings.get("n_points_per_segment", 100),
                "path": "AUTO" if freq_settings.get("auto_kpath", False) else []
            }
            
            # Handle custom path if provided
            if "band_path" in freq_settings:
                bands_dict["path"] = freq_settings["band_path"]
            
            freq_settings["bands"] = bands_dict
            
            # Mark as dispersion calculation
            freq_settings["dispersion"] = True
        
        # Handle phonon DOS conversion
        if freq_settings.get("phonon_dos", False):
            freq_settings.pop("phonon_dos", None)
            
            # Get DOS settings if provided
            dos_settings = freq_settings.get("dos_settings", {})
            freq_settings["pdos"] = {
                "max_freq": dos_settings.get("max_freq", 2000),
                "nbins": dos_settings.get("n_bins", 200),
                "projected": dos_settings.get("projected", True)
            }
            
            # Mark as dispersion calculation
            freq_settings["dispersion"] = True
        
        # Handle INS conversion
        if freq_settings.get("calculate_ins", False):
            freq_settings.pop("calculate_ins", None)
            
            # Get INS settings if provided
            ins_settings = freq_settings.get("ins_settings", {})
            freq_settings["ins"] = {
                "max_freq": ins_settings.get("max_freq", 3000),
                "nbins": ins_settings.get("n_bins", 300),
                "neutron_type": ins_settings.get("neutron_type", 2)
            }
            
            # INS requires dispersion
            freq_settings["dispersion"] = True
        
        converted_options["freq_settings"] = freq_settings
    else:
        converted_options = options
    
    # Use optimized geometry from output but with preserved settings
    # The geometry_data (out_data) contains the optimized coordinates with is_unique flags
    # The settings (options) contains the preserved symmetry and other settings from D12
    write_d12_file(new_filename, out_data, converted_options, external_basis_data)

    print(f"\nSuccessfully created {new_filename}")

    return True, options


def find_file_pairs(directory):
    """Find matching .out and .d12 file pairs in a directory

    Returns:
        list: List of tuples (out_file, d12_file or None)
    """
    pairs = []

    # Find all .out files, excluding SLURM output files
    out_files = [f for f in os.listdir(directory) 
                 if f.endswith(".out") and not f.startswith("slurm-")]

    for out_file in out_files:
        base_name = out_file[:-4]  # Remove .out extension
        d12_file = f"{base_name}.d12"

        full_out_path = os.path.join(directory, out_file)
        full_d12_path = (
            os.path.join(directory, d12_file)
            if os.path.exists(os.path.join(directory, d12_file))
            else None
        )

        pairs.append((full_out_path, full_d12_path))

    return pairs


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert CRYSTAL17/23 optimization output to new D12 input files"
    )
    parser.add_argument("--out-file", type=str, help="CRYSTAL output file (.out)")
    parser.add_argument(
        "--d12-file", type=str, help="Original CRYSTAL input file (.d12)"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Directory containing files (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for new D12 files (default: same as input)",
    )
    parser.add_argument(
        "--shared-settings",
        action="store_true",
        help="Apply the same calculation settings to all files",
    )
    parser.add_argument(
        "--save-options", action="store_true", help="Save options to file"
    )
    parser.add_argument(
        "--options-file",
        type=str,
        default="crystal_opt_settings.json",
        help="File to save/load options",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="JSON config file to load calculation settings from (skips interactive prompts)",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (skip all prompts, use defaults or provided options)",
    )
    parser.add_argument(
        "--calc-type",
        choices=["SP", "OPT", "FREQ"],
        help="Calculation type for non-interactive mode",
    )
    parser.add_argument(
        "--opt-type",
        choices=["FULLOPTG", "ATOMONLY", "CELLONLY"],
        help="Optimization type for OPT calculations in non-interactive mode",
    )
    parser.add_argument(
        "--origin-setting",
        default="auto",
        help="Origin setting: 'auto', '0 0 1', '0 1 0', etc. (default: auto-detect)",
    )

    args = parser.parse_args()

    print("CRYSTAL17/23 Optimization Output to D12 Converter")
    print("=" * 60)
    print("Enhanced version matching NewCifToD12.py configurations")
    print("New entirely reworked script by Marcus Djokic")
    print(
        "Based on old versions by Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic"
    )
    print("")

    # Create output directory if specified
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Single file processing
    if args.out_file:
        if not os.path.exists(args.out_file):
            print(f"Error: Output file {args.out_file} not found")
            return

        success, options = process_files(
            args.out_file, 
            args.d12_file, 
            config_file=args.config_file,
            non_interactive=args.non_interactive,
            calc_type=args.calc_type,
            opt_type=args.opt_type,
            origin_setting=args.origin_setting
        )

        if success and args.save_options:
            with open(args.options_file, "w") as f:
                # Convert non-serializable items
                save_options = {}
                for k, v in options.items():
                    if k not in ["coordinates", "primitive_cell", "conventional_cell"]:
                        save_options[k] = v
                json.dump(save_options, f, indent=2)
            print(f"Settings saved to {args.options_file}")

    else:
        # Directory processing
        file_pairs = find_file_pairs(args.directory)

        if not file_pairs:
            print(f"No .out files found in {args.directory}")
            # Ask for output file path
            print()
            input_file = input("Enter CRYSTAL output file path: ").strip()
            
            if not input_file:
                print("No file path provided. Exiting.")
                return
                
            input_path = Path(input_file)
            
            if not input_path.exists():
                print(f"Error: File or directory not found: {input_file}")
                return
            
            # Check if it's a directory
            if input_path.is_dir():
                # Look for .out files in the directory
                file_pairs = find_file_pairs(input_path)
                if not file_pairs:
                    print(f"\nError: No CRYSTAL output files (.out) found in directory: {input_path}")
                    print("Please specify a CRYSTAL output file (e.g., material.out) or a directory containing .out files.")
                    return
                else:
                    print(f"\nFound {len(file_pairs)} output file(s) in {input_path.name}:")
                    for pair in sorted(file_pairs)[:10]:  # Show first 10
                        # pair[0] is a string path, so we need to convert it to Path to get the name
                        print(f"  - {Path(pair[0]).name}")
                    if len(file_pairs) > 10:
                        print(f"  ... and {len(file_pairs) - 10} more")
            elif input_path.is_file():
                # Convert single file mode - process as single file
                # Create a single file pair for processing
                file_pairs = [(input_path, None)]
            else:
                print(f"Error: {input_path} is not a valid file or directory.")
                return

        print(f"Found {len(file_pairs)} output file(s) to process")

        # Ask about shared settings mode if multiple files and not specified
        use_shared_settings = args.shared_settings
        if not args.non_interactive and len(file_pairs) > 1 and not args.config_file:
            print("\n" + "=" * 60)
            print("MULTIPLE FILE PROCESSING OPTIONS")
            print("=" * 60)
            print("You can either:")
            print("1. Use shared settings for all files (faster, applies same calculation settings)")
            print("2. Configure each file individually (more control per file)")
            print("\nNote: Geometry and symmetry are always preserved from each file.")
            
            use_shared = input("\nUse shared settings for all files? [Y/n]: ").strip().lower()
            use_shared_settings = use_shared != 'n'
        
        # If shared settings requested, get them once
        shared_settings = None
        if use_shared_settings and len(file_pairs) > 1:
            print("\n" + "=" * 60)
            print("SHARED SETTINGS MODE")
            print("=" * 60)
            print("Define calculation settings to apply to all files.")
            print(
                "Note: Geometry, symmetry, and space group info will be preserved from each file."
            )

            # Use first file as template for getting settings
            first_out, first_d12 = file_pairs[0]
            print(f"\nUsing {os.path.basename(first_out)} as template for settings...")

            # Parse first file to get baseline settings
            out_parser = CrystalOutputParser(first_out)
            try:
                out_data = out_parser.parse()
                settings = out_data.copy()

                if first_d12:
                    in_parser = CrystalInputParser(first_d12)
                    try:
                        in_data = in_parser.parse()
                        # Use the same merge logic as in process_files
                        for key, value in in_data.items():
                            if key not in settings or settings[key] is None:
                                settings[key] = value
                            elif key in ["functional", "dispersion", "spin_polarized", "dft_grid", "method", 
                                       "is_3c_method", "use_smearing", "smearing_width", 
                                       "k_points", "scf_method", "scf_maxcycle", "fmixing", "scf_direct",
                                       "mulliken_analysis", "diis_history", "calculation_type", 
                                       "optimization_settings", "freq_settings", "origin_setting", 
                                       "spacegroup", "dimensionality", "tolerances"]:
                                # For all calculation settings, prefer input file (.d12) over output file (.out)
                                # INCLUDING tolerances - the output parser has issues extracting these correctly
                                if value is not None:
                                    settings[key] = value
                            elif key == "scf_settings":
                                # Merge SCF settings
                                if "scf_settings" not in settings:
                                    settings["scf_settings"] = {}
                                settings["scf_settings"].update(value)
                    except:
                        pass

                # Get shared settings
                shared_settings = get_calculation_options_from_current(settings, shared_mode=True)

                print("\n" + "=" * 60)
                print("Shared settings defined. These will be applied to all files.")
                print("=" * 60)

            except Exception as e:
                print(f"Error getting shared settings: {e}")
                return

        # Process all file pairs
        success_count = 0
        for out_file, d12_file in file_pairs:
            print(f"\n{'=' * 70}")
            print(f"Processing: {os.path.basename(out_file)}")
            if d12_file:
                print(f"With input: {os.path.basename(d12_file)}")
            else:
                print("No corresponding .d12 file found")
            print("=" * 70)

            success, options = process_files(
                out_file, 
                d12_file, 
                shared_settings, 
                config_file=args.config_file,
                non_interactive=args.non_interactive,
                calc_type=args.calc_type,
                opt_type=args.opt_type,
                origin_setting=args.origin_setting
            )
            if success:
                success_count += 1

        print(f"\n{'=' * 70}")
        print(
            f"Processing complete: {success_count}/{len(file_pairs)} files processed successfully"
        )
        print("=" * 60)

        # Save options if requested
        if args.save_options and shared_settings:
            with open(args.options_file, "w") as f:
                save_options = {}
                for k, v in shared_settings.items():
                    if k not in ["coordinates", "primitive_cell", "conventional_cell"]:
                        save_options[k] = v
                json.dump(save_options, f, indent=2)
            print(f"\nShared settings saved to {args.options_file}")


if __name__ == "__main__":
    main()
