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

# Import shared D12 creation utilities
try:
    from d12creation import *
except ImportError:
    print("Error: Could not import d12creation module.")
    print(
        "Please ensure d12creation.py is in the same directory or in your Python path."
    )
    sys.exit(1)


def display_current_settings(settings):
    """Display current calculation settings"""
    print("\n" + "=" * 70)
    print("EXTRACTED CALCULATION SETTINGS")
    print("=" * 70)

    print(f"\nDimensionality: {settings.get('dimensionality', 'CRYSTAL')}")
    print(f"Space group: {settings.get('spacegroup', 'N/A')}")
    print(f"Origin setting: {settings.get('origin_setting', '0 0 0')}")

    if settings.get("functional"):
        func = settings["functional"]
        # Determine if this is a Hartree-Fock method or DFT
        if func in ["RHF", "UHF", "HF-3C", "HFsol-3C"]:
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

    print(
        f"Basis set: {settings.get('basis_set', 'N/A')} ({settings.get('basis_set_type', 'INTERNAL')})"
    )
    print(f"Spin polarized: {settings.get('spin_polarized', False)}")

    if settings.get("dft_grid"):
        print(f"DFT grid: {settings.get('dft_grid')}")
    elif settings.get("functional") and settings["functional"] not in [
        "HF",
        "RHF",
        "UHF",
    ]:
        print(f"DFT grid: DEFAULT")

    if settings.get("tolerances"):
        print(
            f"Tolerances: TOLINTEG={settings['tolerances'].get('TOLINTEG', 'N/A')}, "
            f"TOLDEE={settings['tolerances'].get('TOLDEE', 'N/A')}"
        )

    if settings.get("scf_settings"):
        print(f"SCF method: {settings['scf_settings'].get('method', 'DIIS')}")
        print(f"SCF max cycles: {settings['scf_settings'].get('maxcycle', 800)}")
        print(f"FMIXING: {settings['scf_settings'].get('fmixing', 30)}%")

    if settings.get("k_points"):
        print(f"K-points: {settings['k_points']}")

    if settings.get("smearing"):
        print(f"Fermi smearing: Yes (width={settings.get('smearing_width', 0.01)})")

    print("=" * 70)


class CrystalOutputParser:
    """Enhanced parser for CRYSTAL17/23 output files"""

    def __init__(self, output_file):
        self.output_file = output_file
        self.data = {
            "primitive_cell": [],
            "conventional_cell": [],
            "coordinates": [],
            "spacegroup": None,
            "dimensionality": "CRYSTAL",
            "calculation_type": None,
            "functional": None,
            "basis_set": None,
            "basis_set_type": "INTERNAL",
            "tolerances": {},
            "scf_settings": {},
            "spin_polarized": False,
            "dft_grid": None,
            "dispersion": False,
            "k_points": None,
            "origin_setting": "0 0 0",
            "is_3c_method": False,
            "smearing": False,
            "smearing_width": None,
        }

    def parse(self):
        """Parse the output file and extract all relevant data"""
        with open(self.output_file, "r") as f:
            content = f.read()
            lines = content.split("\n")

        # Extract dimensionality first
        self._extract_dimensionality(lines)

        # Extract optimized geometry
        self._extract_geometry(content)

        # Extract calculation settings
        self._extract_settings(lines)

        return self.data

    def _extract_dimensionality(self, lines):
        """Extract system dimensionality"""
        for i, line in enumerate(lines):
            if "CRYSTAL" in line and "CALCULATION" in line:
                self.data["dimensionality"] = "CRYSTAL"
            elif "SLAB CALCULATION" in line:
                self.data["dimensionality"] = "SLAB"
            elif "POLYMER CALCULATION" in line:
                self.data["dimensionality"] = "POLYMER"
            elif "MOLECULE CALCULATION" in line:
                self.data["dimensionality"] = "MOLECULE"

            # Also check input echo section
            if "*                               CRYSTAL" in line:
                # Look for the dimensionality in the next few lines
                for j in range(i, min(i + 10, len(lines))):
                    if "CRYSTAL - PROPERTIES OF THE CRYSTALLINE STATE" in lines[j]:
                        self.data["dimensionality"] = "CRYSTAL"
                        break
                    elif "SLAB" in lines[j]:
                        self.data["dimensionality"] = "SLAB"
                        break
                    elif "POLYMER" in lines[j]:
                        self.data["dimensionality"] = "POLYMER"
                        break
                    elif "MOLECULE" in lines[j]:
                        self.data["dimensionality"] = "MOLECULE"
                        break

    def _extract_geometry(self, content):
        """Extract optimized geometry from output"""
        lines = content.split("\n")

        # Find FINAL OPTIMIZED GEOMETRY section
        final_geom_idx = None
        for i, line in enumerate(lines):
            if "FINAL OPTIMIZED GEOMETRY" in line and "DIMENSIONALITY" in line:
                final_geom_idx = i
                break

        if final_geom_idx is None:
            # Try alternative patterns
            for i, line in enumerate(lines):
                if "OPT END - CONVERGED" in line:
                    # Look for the geometry after this
                    for j in range(i, min(i + 50, len(lines))):
                        if (
                            "LATTICE PARAMETERS" in lines[j]
                            and "PRIMITIVE CELL" in lines[j]
                        ):
                            final_geom_idx = j - 2
                            break
                    break

        if final_geom_idx is None:
            # For single point calculations, look for initial geometry
            for i, line in enumerate(lines):
                if "GEOMETRY FOR WAVE FUNCTION" in line:
                    final_geom_idx = i
                    break

        if final_geom_idx is None:
            raise ValueError("Could not find geometry in output file")

        # Extract space group
        self._extract_spacegroup(lines)

        # Extract cell parameters
        self._extract_cell_parameters(lines, final_geom_idx)

        # Extract atomic coordinates
        self._extract_coordinates(lines, final_geom_idx)

    def _extract_spacegroup(self, lines):
        """Extract space group number from Hermann-Mauguin symbol"""
        for i, line in enumerate(lines):
            if "SPACE GROUP" in line and ":" in line:
                # Extract the space group symbol after the colon
                parts = line.split(":")
                if len(parts) >= 2:
                    sg_symbol = parts[1].strip()

                    # Try to match in our symbol dictionary
                    if sg_symbol in SPACEGROUP_SYMBOLS:
                        self.data["spacegroup"] = SPACEGROUP_SYMBOLS[sg_symbol]
                        return

                    # Try alternative spellings
                    if sg_symbol in SPACEGROUP_ALTERNATIVES:
                        self.data["spacegroup"] = SPACEGROUP_ALTERNATIVES[sg_symbol]
                        return

                    # Try with spaces removed
                    sg_no_space = sg_symbol.replace(" ", "")
                    if sg_no_space in SPACEGROUP_ALTERNATIVES:
                        self.data["spacegroup"] = SPACEGROUP_ALTERNATIVES[sg_no_space]
                        return

                    print(
                        f"Warning: Could not find space group number for symbol '{sg_symbol}'"
                    )

    def _extract_cell_parameters(self, lines, start_idx):
        """Extract unit cell parameters"""
        # Handle different dimensionalities
        if self.data["dimensionality"] == "MOLECULE":
            # No cell parameters for molecules
            return

        # Look for PRIMITIVE CELL
        for i in range(start_idx, min(start_idx + 100, len(lines))):
            if "PRIMITIVE CELL" in lines[i] and "LATTICE PARAMETERS" in lines[i]:
                # Skip to parameter line
                for j in range(i + 1, i + 10):
                    if (
                        "ALPHA" in lines[j]
                        and "BETA" in lines[j]
                        and "GAMMA" in lines[j]
                    ):
                        params = lines[j + 1].split()
                        if len(params) >= 6:
                            self.data["primitive_cell"] = params[:6]
                        break
                break

        # Look for CRYSTALLOGRAPHIC CELL
        for i in range(start_idx, min(start_idx + 100, len(lines))):
            if "CRYSTALLOGRAPHIC CELL" in lines[i] and "VOLUME" in lines[i]:
                # Skip to parameter line
                for j in range(i + 1, i + 10):
                    if (
                        "ALPHA" in lines[j]
                        and "BETA" in lines[j]
                        and "GAMMA" in lines[j]
                    ):
                        params = lines[j + 1].split()
                        if len(params) >= 6:
                            self.data["conventional_cell"] = params[:6]
                        break
                break

        # If no conventional cell found, use primitive
        if not self.data["conventional_cell"] and self.data["primitive_cell"]:
            self.data["conventional_cell"] = self.data["primitive_cell"]

    def _extract_coordinates(self, lines, start_idx):
        """Extract atomic coordinates with symmetry information"""
        coordinates = []

        # Look for coordinates based on dimensionality
        if self.data["dimensionality"] == "MOLECULE":
            # Look for Cartesian coordinates
            for i in range(start_idx, min(start_idx + 200, len(lines))):
                if "CARTESIAN COORDINATES" in lines[i] and "PRIMITIVE CELL" in lines[i]:
                    # Skip the header lines
                    j = i + 3
                    while j < len(lines):
                        parts = lines[j].split()
                        if len(parts) >= 6 and parts[1].isdigit():
                            try:
                                coord = {
                                    "atom_number": parts[2],
                                    "x": parts[3],
                                    "y": parts[4],
                                    "z": parts[5],
                                    "is_unique": True,  # Molecules typically have all atoms unique
                                }
                                coordinates.append(coord)
                            except:
                                break
                        else:
                            break
                        j += 1
                    break
        else:
            # Look for coordinates with T/F symmetry markers
            found_coords = False

            # First try to find ATOMS IN THE ASYMMETRIC UNIT section
            for i in range(start_idx, min(start_idx + 200, len(lines))):
                if "ATOMS IN THE ASYMMETRIC UNIT" in lines[i]:
                    # Skip header lines
                    j = i + 3
                    while j < len(lines):
                        parts = lines[j].split()
                        if len(parts) >= 7:
                            # Check for T/F marker
                            if parts[1] in ["T", "F"]:
                                try:
                                    coord = {
                                        "atom_number": parts[2],
                                        "x": parts[4],
                                        "y": parts[5],
                                        "z": parts[6],
                                        "is_unique": parts[1]
                                        == "T",  # True if T, False if F
                                    }
                                    coordinates.append(coord)
                                except:
                                    break
                            else:
                                break
                        elif len(parts) < 6 or not lines[j].strip():
                            break
                        j += 1
                    found_coords = True
                    break

            # If not found, try COORDINATES IN THE CRYSTALLOGRAPHIC CELL
            if not found_coords:
                for i in range(start_idx, min(start_idx + 200, len(lines))):
                    if "COORDINATES IN THE CRYSTALLOGRAPHIC CELL" in lines[i]:
                        # Skip header lines
                        j = i + 3
                        while j < len(lines):
                            parts = lines[j].split()
                            if len(parts) >= 7 and (parts[1] == "T" or parts[1] == "F"):
                                coord = {
                                    "atom_number": parts[2],
                                    "x": parts[4],
                                    "y": parts[5],
                                    "z": parts[6],
                                    "is_unique": parts[1] == "T",
                                }
                                coordinates.append(coord)
                            elif len(parts) < 6 or not lines[j].strip():
                                break
                            j += 1
                        break

        self.data["coordinates"] = coordinates

    def _extract_settings(self, lines):
        """Extract calculation settings from output"""
        # Extract DFT functional and check for 3C methods
        self._extract_functional(lines)

        # Extract basis set
        self._extract_basis_set(lines)

        # Extract tolerances
        self._extract_tolerances(lines)

        # Extract SCF settings
        self._extract_scf_settings(lines)

        # Extract k-points
        self._extract_kpoints(lines)

        # Check for spin polarization
        self.data["spin_polarized"] = any(
            "SPIN POLARIZED" in line or "UNRESTRICTED" in line or "SPIN" in line
            for line in lines
        )

        # Extract DFT grid
        self._extract_dft_grid(lines)

        # Check for dispersion correction
        self._extract_dispersion(lines)

        # Check for smearing
        self._extract_smearing(lines)

    def _extract_functional(self, lines):
        """Extract DFT functional including 3C methods"""

        # Check for Hartree-Fock methods first
        for i, line in enumerate(lines):
            if "TYPE OF CALCULATION" in line:
                # Check current line and next few lines for KOHN-SHAM
                is_dft = False
                for j in range(i, min(i + 5, len(lines))):
                    if "KOHN-SHAM" in lines[j]:
                        is_dft = True
                        break
                
                # Only assign HF if it's definitely not DFT
                if not is_dft:
                    if "RESTRICTED CLOSED SHELL" in line:
                        self.data["functional"] = "RHF"
                        return
                    elif "UNRESTRICTED OPEN SHELL" in line:
                        self.data["functional"] = "UHF"
                        return

        # Check for specific functional patterns in ENERGY EXPRESSION line
        for line in lines:
            if "ENERGY EXPRESSION" in line:
                # Parse the functional from this line
                if "HARTREE+FOCK" in line and "EXCH*0.20000" in line:
                    if "BECKE" in line and "LYP" in line:
                        self.data["functional"] = "B3LYP"
                        return
                    elif "BECKE" in line and "PW91" in line:
                        self.data["functional"] = "B3PW"
                        return
                elif "HARTREE+FOCK" in line and "EXCH*0.25000" in line:
                    if "PBE" in line:
                        self.data["functional"] = "PBE0"
                        return
                elif "HARTREE+FOCK" in line and not "EXCH*" in line:
                    self.data["functional"] = "RHF"
                    return

        # Check for 3C methods
        for line in lines:
            # Check for 3C methods
            if "HF-3C" in line or "HF3C" in line:
                self.data["functional"] = "HF-3C"
                self.data["is_3c_method"] = True
                return
            elif "PBEH-3C" in line or "PBEH3C" in line:
                self.data["functional"] = "PBEh-3C"
                self.data["is_3c_method"] = True
                return
            elif "HSE-3C" in line or "HSE3C" in line:
                self.data["functional"] = "HSE-3C"
                self.data["is_3c_method"] = True
                return
            elif "B97-3C" in line or "B973C" in line:
                self.data["functional"] = "B97-3C"
                self.data["is_3c_method"] = True
                return
            elif "HFSOL-3C" in line or "HFSOL3C" in line:
                self.data["functional"] = "HFsol-3C"
                self.data["is_3c_method"] = True
                return
            elif "PBESOL0-3C" in line or "PBESOL03C" in line:
                self.data["functional"] = "PBEsol0-3C"
                self.data["is_3c_method"] = True
                return
            elif "HSESOL-3C" in line or "HSESOL3C" in line:
                self.data["functional"] = "HSEsol-3C"
                self.data["is_3c_method"] = True
                return

        # Check for functional in DFT PARAMETERS section
        functional_found = False
        for i, line in enumerate(lines):
            if "KOHN-SHAM HAMILTONIAN" in line:
                # Look at the next few lines for functional info
                for j in range(i + 1, min(i + 10, len(lines))):
                    if "(EXCHANGE)" in lines[j] and "[CORRELATION]" in lines[j]:
                        # Parse functional info
                        if "BECKE 88" in lines[j] and "LEE-YANG-PARR" in lines[j]:
                            self.data["functional"] = "B3LYP"
                            functional_found = True
                            break
                        elif "PBE" in lines[j]:
                            # Check if it's a hybrid
                            for k in range(j, min(j + 5, len(lines))):
                                if "HYBRID EXCHANGE" in lines[k]:
                                    self.data["functional"] = "PBE0"
                                    functional_found = True
                                    break
                            if not functional_found:
                                self.data["functional"] = "PBE"
                                functional_found = True
                            break

                if functional_found:
                    break

        # If no DFT functional found, check if it's Hartree-Fock (already handled above)
        # This section is now redundant as HF detection moved to the beginning

    def _extract_basis_set(self, lines):
        """Extract basis set information"""
        # Look for "Loading internal basis set:" pattern
        for line in lines:
            if "Loading internal basis set:" in line:
                # Extract basis set name after the colon
                parts = line.split(":")
                if len(parts) >= 2:
                    basis_name = parts[1].strip()
                    self.data["basis_set"] = basis_name
                    self.data["basis_set_type"] = "INTERNAL"
                    return

        # For 3C methods, basis set is determined by the method
        if self.data["is_3c_method"] and self.data["functional"]:
            for category in FUNCTIONAL_CATEGORIES.values():
                if "basis_requirements" in category:
                    if self.data["functional"] in category["basis_requirements"]:
                        self.data["basis_set"] = category["basis_requirements"][
                            self.data["functional"]
                        ]
                        self.data["basis_set_type"] = "INTERNAL"
                        return

        # Look for basis set in output
        for i, line in enumerate(lines):
            # Check for internal basis sets
            for basis_name in INTERNAL_BASIS_SETS.keys():
                if basis_name in line and "BASIS SET" in line:
                    self.data["basis_set"] = basis_name
                    self.data["basis_set_type"] = "INTERNAL"
                    return

            # Check for external basis indication
            if "LOCAL ATOMIC FUNCTIONS BASIS SET" in line:
                self.data["basis_set_type"] = "EXTERNAL"
                # External basis set - will need to extract from d12 file
                self.data["basis_set"] = "EXTERNAL"
                return

    def _extract_tolerances(self, lines):
        """Extract tolerance settings"""
        # Initialize with default values to prevent KeyError
        if "tolerances" not in self.data:
            self.data["tolerances"] = {}
        
        for i, line in enumerate(lines):
            # Look for TOLINTEG pattern
            if "INFORMATION **** TOLINTEG ****" in line:
                # Look for the values in subsequent lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    if "COULOMB AND EXCHANGE SERIES TOLERANCES MODIFIED" in lines[j]:
                        # Parse the next lines for actual values
                        for k in range(j + 1, min(j + 10, len(lines))):
                            if "COULOMB OVERLAP TOL" in lines[k]:
                                # Extract T1 value
                                match = re.search(
                                    r"\(T1\)\s*10\*\*\s*(-?\d+)", lines[k]
                                )
                                if match:
                                    t1 = int(match.group(1))
                                    # Look for other values
                                    t2 = t3 = t4 = t5 = t1  # Default to same as T1

                                    # Extract other tolerances
                                    if (
                                        k + 1 < len(lines)
                                        and "COULOMB PENETRATION TOL" in lines[k + 1]
                                    ):
                                        match = re.search(
                                            r"\(T2\)\s*10\*\*\s*(-?\d+)", lines[k + 1]
                                        )
                                        if match:
                                            t2 = int(match.group(1))

                                    if (
                                        k + 2 < len(lines)
                                        and "EXCHANGE OVERLAP TOL" in lines[k + 2]
                                    ):
                                        match = re.search(
                                            r"\(T3\)\s*10\*\*\s*(-?\d+)", lines[k + 2]
                                        )
                                        if match:
                                            t3 = int(match.group(1))

                                    if (
                                        k + 3 < len(lines)
                                        and "EXCHANGE PSEUDO OVP (F(G))" in lines[k + 3]
                                    ):
                                        match = re.search(
                                            r"\(T4\)\s*10\*\*\s*(-?\d+)", lines[k + 3]
                                        )
                                        if match:
                                            t4 = int(match.group(1))

                                    if (
                                        k + 4 < len(lines)
                                        and "EXCHANGE PSEUDO OVP (P(G))" in lines[k + 4]
                                    ):
                                        match = re.search(
                                            r"\(T5\)\s*10\*\*\s*(-?\d+)", lines[k + 4]
                                        )
                                        if match:
                                            t5 = int(match.group(1))

                                    # Format as string
                                    self.data["tolerances"]["TOLINTEG"] = (
                                        f"{-t1} {-t2} {-t3} {-t4} {-t5}"
                                    )
                                    break
                        break

            # Look for TOLDEE
            elif "INFORMATION **** TOLDEE ****" in line:
                # Extract TOLDEE value
                match = re.search(r"SCF TOL ON TOTAL ENERGY SET TO\s*(\d+)", line)
                if match:
                    self.data["tolerances"]["TOLDEE"] = int(match.group(1))
        
        # Ensure default values are set if not found
        if "TOLINTEG" not in self.data["tolerances"] or self.data["tolerances"]["TOLINTEG"] is None:
            self.data["tolerances"]["TOLINTEG"] = "7 7 7 7 14"
        if "TOLDEE" not in self.data["tolerances"] or self.data["tolerances"]["TOLDEE"] is None:
            self.data["tolerances"]["TOLDEE"] = 7

    def _extract_scf_settings(self, lines):
        """Extract SCF settings"""
        for i, line in enumerate(lines):
            # SCF method - check for DIIS explicitly
            if "INFORMATION **** DIIS ****" in line and "DIIS FOR SCF ACTIVE" in line:
                self.data["scf_settings"]["method"] = "DIIS"
            elif "ANDERSON" in line and "MIXING" in line:
                self.data["scf_settings"]["method"] = "ANDERSON"
            elif "BROYDEN" in line and "MIXING" in line:
                self.data["scf_settings"]["method"] = "BROYDEN"

            # MAXCYCLE
            if "INFORMATION **** MAXCYCLE ****" in line:
                match = re.search(r"MAX NUMBER OF SCF CYCLES SET TO\s*(\d+)", line)
                if match:
                    self.data["scf_settings"]["maxcycle"] = int(match.group(1))

            # FMIXING
            if "FMIXING" in line and "SET TO" in line:
                match = re.search(r"FOCK/KS MATRIX MIXING SET TO\s*(\d+)\s*%", line)
                if match:
                    self.data["scf_settings"]["fmixing"] = int(match.group(1))

    def _extract_kpoints(self, lines):
        """Extract k-point information"""
        for i, line in enumerate(lines):
            if "SHRINK. FACT.(MONKH.)" in line:
                # Extract k-point mesh from the same line
                parts = line.split()
                for j, part in enumerate(parts):
                    if "SHRINK." in part:
                        # k-points should be the next 3 numbers
                        if j + 3 < len(parts):
                            try:
                                k1 = int(parts[j + 2])
                                k2 = int(parts[j + 3])
                                k3 = int(parts[j + 4])
                                self.data["k_points"] = f"{k1} {k2} {k3}"
                                return
                            except:
                                pass

    def _extract_dft_grid(self, lines):
        """Extract DFT grid information"""
        for line in lines:
            # Check for grid setting in information lines
            if "NEW DEFAULT: DFT INTEGRATION GRID INCREASED TO" in line:
                for grid_key, grid_name in DFT_GRIDS.items():
                    if grid_name in line:
                        self.data["dft_grid"] = grid_name
                        return

            # Also check for SIZE OF GRID
            if "SIZE OF GRID" in line:
                match = re.search(r"SIZE OF GRID\s*=\s*(\d+)", line)
                if match:
                    grid_size = int(match.group(1))
                    # Map grid sizes to grid names (approximate)
                    if grid_size < 1000:
                        self.data["dft_grid"] = "OLDGRID"
                    elif grid_size < 2000:
                        self.data["dft_grid"] = "LGRID"
                    elif grid_size < 3000:
                        self.data["dft_grid"] = "XLGRID"
                    else:
                        self.data["dft_grid"] = "XXLGRID"
                    return

        # If no grid found, set to None (not DEFAULT)
        # This is important for proper output
        self.data["dft_grid"] = None

    def _extract_dispersion(self, lines):
        """Extract dispersion correction information"""
        for line in lines:
            if "PERFORM LATEST DISPERSION CORRECTION DFT-D3" in line:
                self.data["dispersion"] = True
                return
            elif "D3 DISPERSION ENERGY" in line:
                self.data["dispersion"] = True
                return
            elif "GRIMME D3" in line or "DFT-D3" in line:
                self.data["dispersion"] = True
                return

    def _extract_smearing(self, lines):
        """Extract Fermi smearing information"""
        for line in lines:
            if "SMEARING" in line or "FERMI" in line:
                if "WIDTH" in line:
                    match = re.search(r"WIDTH.*?([\d.]+)", line)
                    if match:
                        self.data["smearing"] = True
                        self.data["smearing_width"] = float(match.group(1))


class CrystalInputParser:
    """Enhanced parser for CRYSTAL17/23 input files"""

    def __init__(self, input_file):
        self.input_file = input_file
        self.data = {
            "spacegroup": None,
            "dimensionality": "CRYSTAL",
            "basis_set_type": "INTERNAL",
            "basis_set": None,
            "basis_set_path": None,
            "scf_method": "DIIS",
            "scf_maxcycle": 800,
            "fmixing": 30,
            "k_points": None,
            "origin_setting": "0 0 0",
            "optimization_settings": {},
            "freq_settings": {},
            "external_basis_info": [],
            "external_basis_data": [],  # Store the actual basis set data
        }

    def parse(self):
        """Parse the input file"""
        with open(self.input_file, "r") as f:
            lines = f.readlines()

        # Extract dimensionality and space group
        for i, line in enumerate(lines):
            if line.strip() in ["CRYSTAL", "SLAB", "POLYMER", "MOLECULE"]:
                self.data["dimensionality"] = line.strip()
                # Next lines should have origin and space group for CRYSTAL
                if self.data["dimensionality"] == "CRYSTAL" and i + 2 < len(lines):
                    self.data["origin_setting"] = lines[i + 1].strip()
                    try:
                        self.data["spacegroup"] = int(lines[i + 2].strip())
                    except:
                        pass
                elif self.data["dimensionality"] in ["SLAB", "POLYMER"] and i + 1 < len(
                    lines
                ):
                    try:
                        self.data["spacegroup"] = int(lines[i + 1].strip())
                    except:
                        pass
                break

        # Extract basis set
        self._extract_basis_set(lines)

        # Extract optimization settings
        self._extract_optimization_settings(lines)

        # Extract DFT settings
        self._extract_dft_settings(lines)

        # Extract SCF settings
        for i, line in enumerate(lines):
            if "MAXCYCLE" in line and "OPTGEOM" not in lines[max(0, i - 5) : i]:
                if i + 1 < len(lines):
                    try:
                        self.data["scf_maxcycle"] = int(lines[i + 1].strip())
                    except:
                        pass
            elif "FMIXING" in line:
                if i + 1 < len(lines):
                    try:
                        self.data["fmixing"] = int(lines[i + 1].strip())
                    except:
                        pass
            elif line.strip() in ["DIIS", "ANDERSON", "BROYDEN"]:
                self.data["scf_method"] = line.strip()

        # Extract k-points
        for i, line in enumerate(lines):
            if "SHRINK" in line:
                if i + 2 < len(lines) and self.data["dimensionality"] == "CRYSTAL":
                    self.data["k_points"] = lines[i + 2].strip()
                break

        return self.data

    def _extract_basis_set(self, lines):
        """Extract basis set information"""
        # First find END of geometry section
        geom_end = None
        for i, line in enumerate(lines):
            if line.strip() == "END" and i > 0:
                # This should be the END after geometry
                geom_end = i
                break

        if geom_end is None:
            return

        # Check what comes after geometry END
        for i in range(geom_end + 1, len(lines)):
            if lines[i].strip() == "BASISSET":
                self.data["basis_set_type"] = "INTERNAL"
                if i + 1 < len(lines):
                    self.data["basis_set"] = lines[i + 1].strip()
                return
            elif "99 0" in lines[i]:
                # External basis set
                self.data["basis_set_type"] = "EXTERNAL"
                # Extract all basis set data between geometry END and 99 0
                for j in range(geom_end + 1, i):
                    line = lines[j].strip()
                    if line and not line.startswith("#"):  # Skip comments
                        self.data["external_basis_data"].append(line)
                return

    def _extract_optimization_settings(self, lines):
        """Extract optimization settings if present"""
        in_optgeom = False
        opt_settings = {}
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if "OPTGEOM" in line:
                in_optgeom = True
                self.data["calculation_type"] = "OPT"
            elif "ENDOPT" in line:
                in_optgeom = False
            elif in_optgeom:
                # Extract all optimization types from d12creation.py
                if stripped in ["FULLOPTG", "CVOLOPT", "CELLONLY", "ATOMONLY"]:
                    opt_settings["type"] = stripped
                    
                # Optimization tolerances
                elif stripped == "MAXCYCLE" and i + 1 < len(lines):
                    try:
                        opt_settings["MAXCYCLE"] = int(lines[i + 1].strip())
                    except (ValueError, IndexError):
                        pass
                elif stripped == "TOLDEG" and i + 1 < len(lines):
                    try:
                        opt_settings["TOLDEG"] = float(lines[i + 1].strip())
                    except (ValueError, IndexError):
                        pass
                elif stripped == "TOLDEX" and i + 1 < len(lines):
                    try:
                        opt_settings["TOLDEX"] = float(lines[i + 1].strip())
                    except (ValueError, IndexError):
                        pass
                elif stripped == "TOLDEE" and i + 1 < len(lines):
                    try:
                        opt_settings["TOLDEE"] = int(lines[i + 1].strip())
                    except (ValueError, IndexError):
                        pass
                elif stripped == "MAXTRADIUS" and i + 1 < len(lines):
                    try:
                        opt_settings["MAXTRADIUS"] = float(lines[i + 1].strip())
                    except (ValueError, IndexError):
                        pass
                        
        # Also check for frequency calculations
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "FREQCALC":
                self.data["calculation_type"] = "FREQ"
                in_freq = True
                freq_settings = {}
                
                # Look for NUMDERIV setting
                for j in range(i + 1, min(i + 10, len(lines))):
                    if lines[j].strip() == "NUMDERIV" and j + 1 < len(lines):
                        try:
                            freq_settings["NUMDERIV"] = int(lines[j + 1].strip())
                        except (ValueError, IndexError):
                            freq_settings["NUMDERIV"] = 2  # Default
                        break
                    elif lines[j].strip() == "END":
                        break
                        
                if freq_settings:
                    self.data["freq_settings"] = freq_settings
                break
                
        # Store optimization settings if any were found
        if opt_settings:
            self.data["optimization_settings"] = opt_settings

    def _extract_dft_settings(self, lines):
        """Extract DFT functional and settings from input file"""
        in_dft_block = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for DFT block
            if stripped == "DFT":
                in_dft_block = True
                self.data["method"] = "DFT"
                continue
            elif stripped == "ENDDFT":
                in_dft_block = False
                continue
            elif stripped in ["UHF", "RHF"] and not in_dft_block:
                # Hartree-Fock methods outside DFT block
                self.data["functional"] = stripped
                self.data["method"] = "HF"
                continue
                
            if in_dft_block:
                # Check for all functional categories from d12creation.py
                
                # LDA functionals
                if stripped in ["SVWN", "LDA", "VBH"]:
                    self.data["functional"] = stripped
                    
                # GGA functionals
                elif stripped in ["BLYP", "PBE", "PBESOL", "PWGGA", "SOGGA", "WCGGA", "B97"]:
                    self.data["functional"] = stripped
                    
                # Hybrid functionals
                elif stripped in ["B3LYP", "B3PW", "CAM-B3LYP", "PBE0", "PBESOL0", "PBE0-13", 
                                "HSE06", "HSEsol", "mPW1PW91", "mPW1K", "B1WC", "WC1LYP", 
                                "B97H", "wB97", "wB97X", "SOGGA11X", "SC-BLYP", "HISS", 
                                "RSHXLDA", "LC-wPBE", "LC-wPBEsol", "LC-wBLYP", "LC-BLYP", "LC-PBE"]:
                    self.data["functional"] = stripped
                    
                # meta-GGA functionals
                elif stripped in ["SCAN", "r2SCAN", "SCAN0", "r2SCANh", "r2SCAN0", "r2SCAN50",
                                "M05", "M052X", "M06", "M062X", "M06HF", "M06L", "revM06", 
                                "revM06L", "MN15", "MN15L", "B1B95", "mPW1B95", "mPW1B1K", 
                                "PW6B95", "PWB6K"]:
                    self.data["functional"] = stripped
                    
                # 3C composite methods (remove hyphens for CRYSTAL format)
                elif stripped in ["PBEh3C", "HSE3C", "B973C", "PBEsol03C", "HSEsol3C"]:
                    # Convert back to standard naming with hyphens
                    if stripped == "PBEh3C":
                        self.data["functional"] = "PBEh-3C"
                    elif stripped == "HSE3C":
                        self.data["functional"] = "HSE-3C"  
                    elif stripped == "B973C":
                        self.data["functional"] = "B97-3C"
                    elif stripped == "PBEsol03C":
                        self.data["functional"] = "PBEsol0-3C"
                    elif stripped == "HSEsol3C":
                        self.data["functional"] = "HSEsol-3C"
                    self.data["is_3c_method"] = True
                    
                # Check for D3 dispersion (both explicit and in functional name)
                elif stripped.endswith("-D3"):
                    base_functional = stripped[:-3]  # Remove -D3 suffix
                    self.data["functional"] = base_functional
                    self.data["dispersion"] = True
                    
                # Special cases like PW1PW-D3
                elif stripped == "PW1PW-D3":
                    self.data["functional"] = "mPW1PW91"
                    self.data["dispersion"] = True
                    
                # Grid settings
                elif stripped in ["OLDGRID", "DEFAULT", "LGRID", "XLGRID", "XXLGRID", "XXXLGRID", "HUGEGRID"]:
                    self.data["dft_grid"] = stripped
                    
                # Spin polarization
                elif stripped == "SPIN":
                    self.data["spin_polarized"] = True
                    
        # Extract smearing settings
        self._extract_smearing_settings(lines)
        
        # Extract tolerance settings  
        self._extract_tolerance_settings(lines)
        
        # Extract k-point settings
        self._extract_kpoint_settings(lines)
        
        # Extract SCF settings
        self._extract_scf_settings(lines)
        
    def _extract_smearing_settings(self, lines):
        """Extract Fermi smearing settings"""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "SMEAR":
                self.data["use_smearing"] = True
                # Check next line for smearing width
                if i + 1 < len(lines):
                    try:
                        self.data["smearing_width"] = float(lines[i + 1].strip())
                    except (ValueError, IndexError):
                        self.data["smearing_width"] = 0.005  # Default
                        
    def _extract_tolerance_settings(self, lines):
        """Extract computational tolerance settings"""
        tolerances = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "TOLINTEG":
                # Next line contains tolerance values
                if i + 1 < len(lines):
                    tolerances["TOLINTEG"] = lines[i + 1].strip()
            elif stripped == "TOLDEE":
                # Next line contains tolerance value
                if i + 1 < len(lines):
                    try:
                        tolerances["TOLDEE"] = int(lines[i + 1].strip())
                    except (ValueError, IndexError):
                        tolerances["TOLDEE"] = 7  # Default
                        
        if tolerances:
            self.data["tolerances"] = tolerances
            
    def _extract_kpoint_settings(self, lines):
        """Extract k-point grid settings"""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "SHRINK":
                # Next lines contain k-point specification
                if i + 2 < len(lines):
                    # Check if it's simplified format (k n_shrink) or directional (0 n_shrink, then ka kb kc)
                    shrink_line = lines[i + 1].strip().split()
                    if len(shrink_line) == 2 and shrink_line[0] != "0":
                        # Simplified format: SHRINK k n_shrink
                        self.data["k_points"] = (int(shrink_line[0]), int(shrink_line[0]), int(shrink_line[0]))
                    elif len(shrink_line) == 2 and shrink_line[0] == "0":
                        # Directional format: SHRINK 0 n_shrink, then ka kb kc
                        kpoint_line = lines[i + 2].strip().split()
                        if len(kpoint_line) >= 3:
                            self.data["k_points"] = (int(kpoint_line[0]), int(kpoint_line[1]), int(kpoint_line[2]))
                        elif len(kpoint_line) == 1:
                            # Single k-point value for all directions
                            k = int(kpoint_line[0])
                            self.data["k_points"] = (k, k, k)
                            
    def _extract_scf_settings(self, lines):
        """Extract SCF convergence settings"""
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # SCF method
            if stripped in ["DIIS", "ANDERSON", "BROYDEN"]:
                self.data["scf_method"] = stripped
                
            # SCF max cycles
            elif stripped == "MAXCYCLE" and "OPTGEOM" not in " ".join(lines[max(0, i-5):i]):
                if i + 1 < len(lines):
                    try:
                        self.data["scf_maxcycle"] = int(lines[i + 1].strip())
                    except (ValueError, IndexError):
                        self.data["scf_maxcycle"] = 800  # Default
                        
            # FMIXING percentage
            elif stripped == "FMIXING":
                if i + 1 < len(lines):
                    try:
                        self.data["fmixing"] = int(lines[i + 1].strip())
                    except (ValueError, IndexError):
                        self.data["fmixing"] = 30  # Default
                        
            # Additional SCF options
            elif stripped == "SCFDIR":
                self.data["scf_direct"] = True
            elif stripped == "PPAN":
                self.data["mulliken_analysis"] = True
            elif stripped == "HISTDIIS":
                if i + 1 < len(lines):
                    try:
                        self.data["diis_history"] = int(lines[i + 1].strip())
                    except (ValueError, IndexError):
                        self.data["diis_history"] = 20  # Default


def get_calculation_options(current_settings, shared_mode=False):
    """Get calculation options from user

    Args:
        current_settings: Current settings extracted from files
        shared_mode: If True, only ask for calculation settings to be shared

    Returns:
        dict: Options for the calculation
    """
    options = current_settings.copy()

    if not shared_mode:
        # Display current settings
        display_current_settings(current_settings)

        # Ask if user wants to keep current settings
        keep_settings = yes_no_prompt(
            "\nKeep these settings for the new calculation?", "yes"
        )
    else:
        # In shared mode, we'll modify calculation settings
        keep_settings = False

    if keep_settings:
        # Just ask for calculation type
        calc_options = {
            "1": "SP",  # Single Point
            "2": "OPT",  # Geometry Optimization
            "3": "FREQ",  # Frequency Calculation
        }
        calc_choice = get_user_input(
            "Select calculation type for the new input", calc_options, "1"
        )
        options["calculation_type"] = calc_options[calc_choice]

        # If OPT selected, warn that geometry is already optimized
        if options["calculation_type"] == "OPT":
            print(
                "\nWarning: The geometry is already optimized. Are you sure you want to run another optimization?"
            )
            if not yes_no_prompt("Continue with geometry optimization?", "no"):
                options["calculation_type"] = "SP"

            # If still OPT, get optimization settings
            if options["calculation_type"] == "OPT":
                opt_choice = get_user_input("Select optimization type", OPT_TYPES, "1")
                options["optimization_type"] = OPT_TYPES[opt_choice]

                # Use default optimization settings
                options["optimization_settings"] = DEFAULT_OPT_SETTINGS.copy()

        # If FREQ, set frequency settings
        if options["calculation_type"] == "FREQ":
            options["freq_settings"] = DEFAULT_FREQ_SETTINGS.copy()
            # Override tolerances for frequency calculations
            options["tolerances"] = {
                "TOLINTEG": DEFAULT_FREQ_SETTINGS["TOLINTEG"],
                "TOLDEE": DEFAULT_FREQ_SETTINGS["TOLDEE"],
            }
    else:
        # Allow full customization
        if shared_mode:
            print("\nDefine shared calculation settings for all files:")
        else:
            print("\nCustomize calculation settings:")

        # Calculation type
        calc_options = {
            "1": "SP",  # Single Point
            "2": "OPT",  # Geometry Optimization
            "3": "FREQ",  # Frequency Calculation
        }
        calc_choice = get_user_input("Select calculation type", calc_options, "1")
        options["calculation_type"] = calc_options[calc_choice]

        # Optimization settings if OPT
        if options["calculation_type"] == "OPT":
            opt_choice = get_user_input("Select optimization type", OPT_TYPES, "1")
            options["optimization_type"] = OPT_TYPES[opt_choice]

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
                maxtradius = float(
                    input("Enter MAXTRADIUS (max displacement, default 0.25): ") or 0.25
                )
                options["optimization_settings"]["MAXTRADIUS"] = maxtradius

        # Frequency settings if FREQ
        if options["calculation_type"] == "FREQ":
            use_default_freq = yes_no_prompt(
                "Use default frequency calculation settings? (NUMDERIV=2, TOLINTEG=12 12 12 12 24, TOLDEE=12)",
                "yes",
            )

            if not use_default_freq:
                custom_settings = {}
                custom_settings["NUMDERIV"] = int(
                    input("Enter NUMDERIV (numerical derivative level, default 2): ")
                    or 2
                )
                options["freq_settings"] = custom_settings
            else:
                options["freq_settings"] = DEFAULT_FREQ_SETTINGS.copy()

            # For frequency calculations, use tighter default tolerances
            options["tolerances"] = {
                "TOLINTEG": DEFAULT_FREQ_SETTINGS["TOLINTEG"],
                "TOLDEE": DEFAULT_FREQ_SETTINGS["TOLDEE"],
            }

        # Method selection
        method_options = {"1": "DFT", "2": "HF"}
        print("\nSelect calculation method:")
        print("1: DFT - Density Functional Theory")
        print("2: HF - Hartree-Fock")

        # Default to DFT if current functional is DFT, otherwise HF
        default_method = (
            "1" if options.get("functional") not in ["HF", "UHF", None] else "2"
        )
        method_choice = get_user_input("Select method", method_options, default_method)

        if method_options[method_choice] == "HF":
            # HF method
            change_functional = yes_no_prompt(
                f"Change HF method (current: {options.get('functional', 'RHF')})?",
                "no",  # Changed default to "no" for both modes
            )
            if change_functional:
                functional, required_basis = select_functional()
                options["functional"] = functional

                # If functional has required basis set, use it automatically
                if required_basis:
                    print(
                        f"\nNote: {functional} requires {required_basis} basis set. Using it automatically."
                    )
                    options["basis_set_type"] = "INTERNAL"
                    options["basis_set"] = required_basis
                    options["is_3c_method"] = True
                    # 3C methods already include corrections - no additional dispersion
                    options["dispersion"] = False
                    # HF 3C methods don't need DFT grids
                    options["dft_grid"] = None
                else:
                    options["is_3c_method"] = False
                    # HF methods don't use dispersion
                    options["dispersion"] = False
        else:
            # DFT method
            change_functional = yes_no_prompt(
                f"Change DFT functional (current: {options.get('functional', 'N/A')})?",
                "no",  # Changed default to "no" for both modes
            )
            if change_functional:
                functional, required_basis = select_functional()
                options["functional"] = functional

                # If functional has required basis set, use it automatically
                if required_basis:
                    print(
                        f"\nNote: {functional} requires {required_basis} basis set. Using it automatically."
                    )
                    options["basis_set_type"] = "INTERNAL"
                    options["basis_set"] = required_basis
                    options["is_3c_method"] = True
                    # 3C methods already include dispersion
                    options["dispersion"] = False
                    # 3C methods have their own grids
                    options["dft_grid"] = None
                else:
                    options["is_3c_method"] = False

                    # Check if dispersion correction is available
                    if functional in D3_FUNCTIONALS:
                        options["dispersion"] = yes_no_prompt(
                            f"Add D3 dispersion correction to {functional}?",
                            "yes" if options.get("dispersion") else "no",
                        )
                    else:
                        options["dispersion"] = False

        # Basis set (if not determined by functional)
        if not (options.get("is_3c_method") and options.get("basis_set")):
            change_basis = yes_no_prompt(
                f"Change basis set (current: {options.get('basis_set', 'N/A')})?",
                "no",  # Changed default to "no" for both modes
            )
            if change_basis:
                basis_type = get_user_input(
                    "Basis set type", {"1": "EXTERNAL", "2": "INTERNAL"}, "2"
                )

                if basis_type == "2":
                    internal_basis_options = {}
                    print("\nAvailable internal basis sets:")

                    # First show standard basis sets
                    print("\n--- STANDARD BASIS SETS ---")
                    option_num = 1
                    for bs_name, bs_info in INTERNAL_BASIS_SETS.items():
                        if bs_info.get("standard", False):
                            internal_basis_options[str(option_num)] = bs_name
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

                    basis_choice = get_user_input(
                        "Select internal basis set", internal_basis_options, "7"
                    )
                    options["basis_set"] = internal_basis_options[basis_choice]
                    options["basis_set_type"] = "INTERNAL"
                else:
                    options["basis_set_type"] = "EXTERNAL"
                    print("\nExternal basis set options:")
                    print("1: DZVP-REV2 (./full.basis.doublezeta/)")
                    print("2: TZVP-REV2 (./full.basis.triplezeta/)")
                    print("3: Custom path")
                    print("4: Use basis from original file")

                    external_choice = get_user_input(
                        "Select external basis set",
                        {"1": "DZ", "2": "TZ", "3": "Custom", "4": "Original"},
                        "4" if not shared_mode else "2",
                    )

                    if external_choice == "1":
                        options["basis_set_path"] = "./full.basis.doublezeta/"
                    elif external_choice == "2":
                        options["basis_set_path"] = "./full.basis.triplezeta/"
                    elif external_choice == "3":
                        options["basis_set_path"] = input(
                            "Enter path to external basis set directory: "
                        ).strip()
                    else:
                        # Use external basis from original file
                        options["use_original_external_basis"] = True

        # DFT grid (only for non-3C DFT methods)
        if (
            options.get("functional")
            and options["functional"] not in ["HF", "RHF", "UHF"]
            and not options.get("is_3c_method")
            and "-3C" not in options.get("functional", "")
            and "3C" not in options.get("functional", "")
        ):
            current_grid = (
                options.get("dft_grid", "XLGRID") or "XLGRID"
            )  # Default to XLGRID if None
            change_grid = yes_no_prompt(
                f"Change DFT integration grid (current: {current_grid})?", "no"
            )
            if change_grid:
                grid_choice = get_user_input(
                    "Select DFT integration grid", DFT_GRIDS, "4"
                )
                options["dft_grid"] = DFT_GRIDS[grid_choice]
            else:
                # If not changing, ensure a valid grid is set
                if not options.get("dft_grid"):
                    options["dft_grid"] = "XLGRID"
        elif options.get("is_3c_method") or "-3C" in options.get("functional", ""):
            # 3C methods have their own grids
            options["dft_grid"] = None

        # Spin polarization - UPDATED DEFAULT TO YES
        options["spin_polarized"] = yes_no_prompt(
            "Use spin-polarized calculation?",
            "yes",  # Changed default to "yes"
        )

        # Fermi smearing
        options["smearing"] = yes_no_prompt(
            "Use Fermi surface smearing for metallic systems?",
            "yes" if options.get("smearing") else "no",
        )

        if options["smearing"]:
            default_width = options.get("smearing_width", 0.01)
            width = input(
                f"Enter smearing width in hartree (recommended: 0.001-0.02, default {default_width}): "
            ).strip()
            options["smearing_width"] = float(width) if width else default_width

        # Tolerances (if not already set by FREQ)
        if options["calculation_type"] != "FREQ":
            change_tol = yes_no_prompt("Change tolerance settings?", "no")
            if change_tol:
                if "tolerances" not in options:
                    options["tolerances"] = {}

                tolinteg = input(
                    "Enter TOLINTEG values (5 integers, default '7 7 7 7 14'): "
                ).strip()
                options["tolerances"]["TOLINTEG"] = (
                    tolinteg if tolinteg else "7 7 7 7 14"
                )

                toldee = input("Enter TOLDEE value (integer, default 7): ").strip()
                options["tolerances"]["TOLDEE"] = int(toldee) if toldee else 7
            else:
                # If not changing tolerances, use current settings or defaults
                if "tolerances" not in options:
                    options["tolerances"] = DEFAULT_TOLERANCES.copy()
                else:
                    # Ensure all required tolerance keys are present
                    if "TOLINTEG" not in options["tolerances"]:
                        options["tolerances"]["TOLINTEG"] = DEFAULT_TOLERANCES[
                            "TOLINTEG"
                        ]
                    if "TOLDEE" not in options["tolerances"]:
                        options["tolerances"]["TOLDEE"] = DEFAULT_TOLERANCES["TOLDEE"]

        # SCF settings
        change_scf = yes_no_prompt("Change SCF settings?", "no")
        if change_scf:
            scf_methods = {"1": "DIIS", "2": "ANDERSON", "3": "BROYDEN"}
            scf_choice = get_user_input("Select SCF method", scf_methods, "1")

            if "scf_settings" not in options:
                options["scf_settings"] = {}

            options["scf_settings"]["method"] = scf_methods[scf_choice]

            maxcycle = input("Enter SCF MAXCYCLE (default 800): ").strip()
            options["scf_settings"]["maxcycle"] = int(maxcycle) if maxcycle else 800

            fmixing = input("Enter FMIXING percentage (default 30): ").strip()
            options["scf_settings"]["fmixing"] = int(fmixing) if fmixing else 30
        else:
            # If not changing SCF settings, ensure defaults are set
            if "scf_settings" not in options:
                options["scf_settings"] = {
                    "method": "DIIS",
                    "maxcycle": 800,
                    "fmixing": 30,
                }
            else:
                # Ensure all required SCF keys are present
                if "method" not in options["scf_settings"]:
                    options["scf_settings"]["method"] = "DIIS"
                if "maxcycle" not in options["scf_settings"]:
                    options["scf_settings"]["maxcycle"] = 800
                if "fmixing" not in options["scf_settings"]:
                    options["scf_settings"]["fmixing"] = 30

    # Symmetry handling section - unified with NewCifToD12.py functionality
    if options.get("dimensionality") != "MOLECULE":
        if shared_mode:
            # In shared mode, ask generically without specific counts
            print("\nSymmetry handling for all files:")
            sym_options = {
                "1": "Write only unique atoms (asymmetric unit) when available",
                "2": "Write all atoms",
            }

            sym_choice = get_user_input(
                "How should atoms be written in the new inputs?",
                sym_options,
                "1",  # Default to writing only unique atoms
            )

            options["write_only_unique"] = sym_choice == "1"
        else:
            # In single file mode, show specific information
            has_symmetry_info = any(
                coord.get("is_unique") is not None
                for coord in options.get("coordinates", [])
            )

            if has_symmetry_info:
                unique_count = sum(
                    1
                    for coord in options["coordinates"]
                    if coord.get("is_unique", True)
                )
                total_count = len(options["coordinates"])

                print(f"\nSymmetry information detected:")
                print(f"  Unique atoms (T): {unique_count}")
                print(f"  Total atoms: {total_count}")

                sym_options = {
                    "1": "Write only unique atoms (asymmetric unit)",
                    "2": "Write all atoms",
                }

                sym_choice = get_user_input(
                    "How should atoms be written in the new input?",
                    sym_options,
                    "1",  # Default to writing only unique atoms
                )

                options["write_only_unique"] = sym_choice == "1"
                
                if options["write_only_unique"]:
                    print(f"Will write {unique_count} unique atoms (asymmetric unit)")
                else:
                    print(f"Will write all {total_count} atoms")
            else:
                # If no symmetry info detected, assume all atoms should be written
                options["write_only_unique"] = False
                print("\nNo symmetry information detected. All atoms will be written.")
    else:
        options["write_only_unique"] = False

    return options


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


def write_d12_file(output_file, geometry_data, settings, external_basis_data=None):
    """Write new D12 file with optimized geometry and settings"""

    with open(output_file, "w") as f:
        # Title
        title = output_file.replace(".d12", "")
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
                f.write(f"{spg_info['crystal_code']}\n")
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
            # Add 200 to atomic number ONLY if ECP is required for EXTERNAL basis sets
            if (
                settings.get("basis_set_type") == "EXTERNAL"
                and atom_num in ECP_ELEMENTS_EXTERNAL
            ):
                atom_num += 200
            # For internal basis sets, do NOT add 200 - they handle ECP internally

            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(atom["atom_number"]), "X")
            f.write(
                f"{atom_num} {atom['x']} {atom['y']} {atom['z']} Biso 1.000000 {symbol}\n"
            )

        # Calculation-specific section
        if settings["calculation_type"] == "OPT":
            # For OPT: No END after coordinates, OPTGEOM follows directly
            write_optimization_section(
                f,
                settings.get("optimization_type", "FULLOPTG"),
                settings.get("optimization_settings", DEFAULT_OPT_SETTINGS),
            )
        elif settings["calculation_type"] == "FREQ":
            # For FREQ: No END after coordinates, FREQCALC follows directly  
            write_frequency_section(
                f, settings.get("freq_settings", DEFAULT_FREQ_SETTINGS)
            )
        else:
            # For SP: Write END after coordinates, then proceed to basis set
            f.write("END\n")

        # Handle basis sets and method section
        functional = settings.get("functional", "")
        method = "HF" if functional in ["RHF", "UHF", "HF-3C", "HFsol-3C"] else "DFT"

        # Handle HF 3C methods and regular HF methods
        if functional in ["HF-3C", "HFsol-3C"]:
            # These are HF methods with corrections, write basis set but no DFT block
            write_basis_set_section(
                f, "INTERNAL", settings["basis_set"], coords_to_write
            )

            # Add 3C corrections
            if functional == "HF-3C":
                f.write("HF3C\n")
                f.write("END\n")
            elif functional == "HFsol-3C":
                f.write("HFSOL3C\n")
                f.write("END\n")
        elif functional in ["RHF", "UHF"]:
            # Standard HF methods
            if settings.get("basis_set_type") == "EXTERNAL":
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
                write_basis_set_section(
                    f, "INTERNAL", 
                    settings.get("basis_set", "POB-TZVP-REV2"), 
                    coords_to_write
                )
            
            # For UHF, add the UHF keyword
            if functional == "UHF":
                f.write("UHF\n")
        elif functional in ["PBEh-3C", "HSE-3C", "B97-3C", "PBEsol0-3C", "HSEsol-3C"]:
            # DFT 3C methods
            write_basis_set_section(
                f, "INTERNAL", settings["basis_set"], coords_to_write
            )

            f.write("DFT\n")
            if settings.get("spin_polarized"):
                f.write("SPIN\n")

            # Write 3C method (remove hyphen for CRYSTAL23)
            f.write(f"{functional.replace('-', '')}\n")
            f.write("ENDDFT\n")
        else:
            # Standard basis set and method handling
            if settings.get("basis_set_type") == "EXTERNAL":
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
                write_basis_set_section(
                    f,
                    "INTERNAL",
                    settings.get("basis_set", "POB-TZVP-REV2"),
                    coords_to_write,
                )

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

        # Prepare k-points
        if settings.get("k_points"):
            k_points_info = settings["k_points"]
        elif geometry_data.get("conventional_cell"):
            # Generate k-points based on cell size
            a, b, c = [float(x) for x in geometry_data["conventional_cell"][:3]]
            k_points_info = generate_k_points(
                a, b, c, dimensionality, settings.get("spacegroup", 1)
            )
        else:
            k_points_info = None

        write_scf_section(
            f,
            settings.get("tolerances", DEFAULT_TOLERANCES),
            k_points_info,
            dimensionality,
            settings.get("smearing"),
            settings.get("smearing_width", 0.01),
            settings.get("scf_settings", {}).get("method", "DIIS"),
            settings.get("scf_settings", {}).get("maxcycle", 800),
            settings.get("scf_settings", {}).get("fmixing", 30),
            len(atomic_numbers),
        )


def process_files(output_file, input_file=None, shared_settings=None):
    """Process CRYSTAL output and input files

    Args:
        output_file: Path to .out file
        input_file: Path to .d12 file (optional)
        shared_settings: Pre-defined settings to use (optional)

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
                           "is_3c_method", "use_smearing", "smearing_width", "tolerances", 
                           "k_points", "scf_method", "scf_maxcycle", "fmixing", "scf_direct",
                           "mulliken_analysis", "diis_history", "calculation_type", 
                           "optimization_settings", "freq_settings"]:
                    # For all calculation settings, prefer input file (.d12) over output file (.out)
                    # because .d12 contains the original user-specified settings
                    if value is not None:
                        settings[key] = value
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
    if shared_settings:
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
            "HF-3C",
            "HFsol-3C",
            "PBEh-3C",
            "HSE-3C",
            "B97-3C",
            "PBEsol0-3C",
            "HSEsol-3C",
        ]:
            options["dispersion"] = False
            options["is_3c_method"] = True
            options["dft_grid"] = None
    else:
        options = get_calculation_options(settings)

    # Create output filename
    base_name = os.path.splitext(output_file)[0]
    calc_type = options["calculation_type"]
    functional = options.get("functional", "RHF")

    # Don't add -D3 to 3C methods or HF methods or if dispersion is already included in the name
    if (
        options.get("dispersion")
        and "-3C" not in functional
        and "3C" not in functional
        and functional not in ["RHF", "UHF", "HF-3C", "HFsol-3C"]
    ):
        functional += "-D3"

    new_filename = f"{base_name}_{calc_type.lower()}_{functional}_optimized.d12"

    # Write new D12 file
    print(f"\nWriting new D12 file: {new_filename}")
    write_d12_file(new_filename, out_data, options, external_basis_data)

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

    args = parser.parse_args()

    print("CRYSTAL17/23 Optimization Output to D12 Converter")
    print("=" * 55)
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

        success, options = process_files(args.out_file, args.d12_file)

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
            return

        print(f"Found {len(file_pairs)} output file(s) to process")

        # If shared settings requested, get them once
        shared_settings = None
        if args.shared_settings and len(file_pairs) > 1:
            print("\n" + "=" * 70)
            print("SHARED SETTINGS MODE")
            print("=" * 70)
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
                        for key, value in in_data.items():
                            if key not in settings or settings[key] is None:
                                settings[key] = value
                    except:
                        pass

                # Get shared settings
                shared_settings = get_calculation_options(settings, shared_mode=True)

                print("\n" + "=" * 70)
                print("Shared settings defined. These will be applied to all files.")
                print("=" * 70)

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

            success, options = process_files(out_file, d12_file, shared_settings)
            if success:
                success_count += 1

        print(f"\n{'=' * 70}")
        print(
            f"Processing complete: {success_count}/{len(file_pairs)} files processed successfully"
        )
        print("=" * 70)

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
