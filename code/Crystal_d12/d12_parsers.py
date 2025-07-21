#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D12 Parser Module for CRYSTAL23
-------------------------------
This module contains parser classes for CRYSTAL input and output files.
It consolidates the parsing logic from CRYSTALOptToD12.py to reduce
code size and improve maintainability.

Classes:
- CrystalOutputParser: Parses CRYSTAL17/23 output files (.out)
- CrystalInputParser: Parses CRYSTAL17/23 input files (.d12)
"""

import re
from typing import Dict, List, Any, Optional
from d12_constants import (
    SPACEGROUP_SYMBOLS,
    SPACEGROUP_SYMBOL_TO_NUMBER,
    SPACEGROUP_ALTERNATIVES,
    MULTI_ORIGIN_SPACEGROUPS,
    RHOMBOHEDRAL_SPACEGROUPS
)


class CrystalOutputParser:
    """Enhanced parser for CRYSTAL17/23 output files"""

    def __init__(self, output_file: str):
        self.output_file = output_file
        self.data = {
            "primitive_cell": [],
            "conventional_cell": [],
            "coordinates": [],
            "spacegroup": None,
            "dimensionality": None,
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

    def parse(self) -> Dict[str, Any]:
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

    def _extract_dimensionality(self, lines: List[str]) -> None:
        """Extract system dimensionality"""
        # First pass - look for explicit calculation type declarations
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if "SLAB CALCULATION" in line_upper:
                self.data["dimensionality"] = "SLAB"
                return  # Found it, no need to continue
            elif "POLYMER CALCULATION" in line_upper:
                self.data["dimensionality"] = "POLYMER"
                return
            elif "MOLECULE CALCULATION" in line_upper:
                self.data["dimensionality"] = "MOLECULE"
                return
            elif "CRYSTAL CALCULATION" in line_upper:
                self.data["dimensionality"] = "CRYSTAL"
                return

        # Second pass - check for other indicators
        for i, line in enumerate(lines):
            # Check for CRYSTAL - SCF lines
            if "CRYSTAL - SCF" in line:
                # Default to CRYSTAL if not already set
                if self.data["dimensionality"] is None:
                    self.data["dimensionality"] = "CRYSTAL"
                    
            # Also check input echo section
            if "*                               CRYSTAL" in line:
                # Look for the dimensionality in the next few lines
                for j in range(i, min(i + 10, len(lines))):
                    if "CRYSTAL - PROPERTIES OF THE CRYSTALLINE STATE" in lines[j]:
                        if self.data["dimensionality"] is None:
                            self.data["dimensionality"] = "CRYSTAL"
                        break
                    elif "SLAB" in lines[j] and "CALCULATION" not in lines[j]:
                        if self.data["dimensionality"] is None:
                            self.data["dimensionality"] = "SLAB"
                        break
                    elif "POLYMER" in lines[j]:
                        if self.data["dimensionality"] is None:
                            self.data["dimensionality"] = "POLYMER"
                        break
                    elif "MOLECULE" in lines[j]:
                        if self.data["dimensionality"] is None:
                            self.data["dimensionality"] = "MOLECULE"
                        break
                        
        # Default to CRYSTAL if nothing found
        if self.data["dimensionality"] is None:
            self.data["dimensionality"] = "CRYSTAL"

    def _extract_geometry(self, content: str) -> None:
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

    def _extract_spacegroup(self, lines: List[str]) -> None:
        """Extract space group number from Hermann-Mauguin symbol"""
        for i, line in enumerate(lines):
            if "SPACE GROUP" in line and ":" in line:
                # Extract the space group symbol after the colon
                parts = line.split(":")
                if len(parts) >= 2:
                    sg_symbol = parts[1].strip()

                    # Try to match in our symbol-to-number dictionary
                    if sg_symbol in SPACEGROUP_SYMBOL_TO_NUMBER:
                        self.data["spacegroup"] = SPACEGROUP_SYMBOL_TO_NUMBER[sg_symbol]
                        return

                    # Try alternative spellings (including CRYSTAL output format with spaces)
                    if sg_symbol in SPACEGROUP_ALTERNATIVES:
                        self.data["spacegroup"] = SPACEGROUP_ALTERNATIVES[sg_symbol]
                        return

                    # Try with spaces removed
                    sg_no_space = sg_symbol.replace(" ", "")
                    if sg_no_space in SPACEGROUP_SYMBOL_TO_NUMBER:
                        self.data["spacegroup"] = SPACEGROUP_SYMBOL_TO_NUMBER[sg_no_space]
                        return
                    
                    if sg_no_space in SPACEGROUP_ALTERNATIVES:
                        self.data["spacegroup"] = SPACEGROUP_ALTERNATIVES[sg_no_space]
                        return

                    print(
                        f"Warning: Could not find space group number for symbol '{sg_symbol}'"
                    )

    def _extract_cell_parameters(self, lines: List[str], start_idx: int) -> None:
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
        if not self.data.get("conventional_cell") and self.data.get("primitive_cell"):
            self.data["conventional_cell"] = self.data["primitive_cell"]
            
        # Also try to extract lattice parameters from the optimized geometry section
        if not self.data.get("conventional_cell"):
            self._extract_optimized_cell_parameters(lines, start_idx)
    
    def _extract_optimized_cell_parameters(self, lines: List[str], start_idx: int) -> None:
        """Extract optimized lattice parameters from final geometry section"""
        # Look for optimized cell parameters near the final geometry
        for i in range(max(0, start_idx - 50), min(start_idx + 100, len(lines))):
            line = lines[i].strip()
            # Look for patterns like "LATTICE PARAMETERS (ANGSTROMS AND DEGREES)"
            if "LATTICE PARAMETERS" in line and ("ANGSTROM" in line or "DEGREE" in line):
                # Look for the parameter values in the next few lines
                for j in range(i + 1, min(i + 10, len(lines))):
                    parts = lines[j].strip().split()
                    if len(parts) >= 6:
                        try:
                            # Try to parse as 6 floating point numbers (a, b, c, alpha, beta, gamma)
                            params = [float(p) for p in parts[:6]]
                            self.data["conventional_cell"] = [str(p) for p in params]
                            return
                        except ValueError:
                            continue
            
            # Also look for simpler patterns with just lattice constants
            if ("A = " in line or "A=" in line) and ("B = " in line or "B=" in line):
                # Try to extract a, b, c values
                try:
                    a_match = re.search(r'A\s*=\s*([0-9.]+)', line)
                    b_match = re.search(r'B\s*=\s*([0-9.]+)', line)
                    c_match = re.search(r'C\s*=\s*([0-9.]+)', line)
                    if a_match and b_match and c_match:
                        a, b, c = a_match.group(1), b_match.group(1), c_match.group(1)
                        # Use default angles for simple cases
                        self.data["conventional_cell"] = [a, b, c, "90.0", "90.0", "90.0"]
                        return
                except:
                    continue

    def _extract_coordinates(self, lines: List[str], start_idx: int) -> None:
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

            # PRIORITY: First try to find COORDINATES IN THE CRYSTALLOGRAPHIC CELL
            # This is essential for maintaining consistency with lattice parameters
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
                    found_coords = True
                    break

            # Fallback: If crystallographic coordinates not found, try ATOMS IN THE ASYMMETRIC UNIT
            if not found_coords:
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

        self.data["coordinates"] = coordinates

    def _extract_settings(self, lines: List[str]) -> None:
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

    def _extract_functional(self, lines: List[str]) -> None:
        """Extract DFT functional including 3C methods - IMPROVED VERSION"""
        
        # First check if this is a DFT calculation by looking for KOHN-SHAM
        is_dft = False
        for line in lines:
            if "KOHN-SHAM HAMILTONIAN" in line:
                is_dft = True
                break
        
        # If not DFT, check for Hartree-Fock methods
        if not is_dft:
            for i, line in enumerate(lines):
                if "TYPE OF CALCULATION" in line:
                    if "RESTRICTED CLOSED SHELL" in line:
                        self.data["functional"] = "RHF"
                        return
                    elif "UNRESTRICTED OPEN SHELL" in line:
                        self.data["functional"] = "UHF"
                        return
                    elif "RESTRICTED OPEN SHELL" in line:
                        self.data["functional"] = "ROHF"
                        return
                    elif "HF-3c" in line:
                        self.data["functional"] = "HF3C"
                        self.data["is_3c_method"] = True
                        return
                    elif "HFSOL-3c" in line:
                        self.data["functional"] = "HFSOL3C"
                        self.data["is_3c_method"] = True
                        return
        
        # Look for DFT functional information
        functional_found = False
        
        for i, line in enumerate(lines):
            # Check for CRYSTAL23 format: (EXCHANGE)[CORRELATION] FUNCTIONAL:
            if "(EXCHANGE)[CORRELATION] FUNCTIONAL:" in line:
                # Check if functional name is on the same line
                if "PERDEW-BURKE-ERNZERHOF" in line:
                    self.data["functional"] = "PBE"
                    functional_found = True
                elif "BECKE 88" in line and "LEE-YANG-PARR" in line:
                    self.data["functional"] = "BLYP"
                    functional_found = True
                elif "B3LYP" in line:
                    self.data["functional"] = "B3LYP"
                    functional_found = True
                elif "PBE0" in line:
                    self.data["functional"] = "PBE0"
                    functional_found = True
                elif "HSE06" in line:
                    self.data["functional"] = "HSE06"
                    functional_found = True
                # If not found on same line, check the next line
                elif i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if "PERDEW-BURKE-ERNZERHOF" in next_line:
                        self.data["functional"] = "PBE"
                        functional_found = True
                    elif "BECKE 88" in next_line and i + 2 < len(lines) and "LEE-YANG-PARR" in lines[i + 2]:
                        self.data["functional"] = "BLYP"
                        functional_found = True
                    elif "B3LYP" in next_line:
                        self.data["functional"] = "B3LYP"
                        functional_found = True
                    elif "PBE0" in next_line:
                        self.data["functional"] = "PBE0"
                        functional_found = True
                    elif "HSE06" in next_line:
                        self.data["functional"] = "HSE06"
                        functional_found = True
                    
            # Check for exchange-correlation functional section (older format)
            elif "EXCHANGE-CORRELATION FUNCTIONAL" in line:
                # Look for the functional name in the next few lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    if "PERDEW-BURKE-ERNZERHOF(PBE)" in lines[j]:
                        self.data["functional"] = "PBE"
                        functional_found = True
                        break
                    elif "BECKE 88" in lines[j] and "LEE-YANG-PARR" in lines[j + 1] if j + 1 < len(lines) else False:
                        self.data["functional"] = "BLYP"
                        functional_found = True
                        break
                    elif "PERDEW-WANG" in lines[j]:
                        self.data["functional"] = "PW91"
                        functional_found = True
                        break
                    elif "B3LYP" in lines[j]:
                        self.data["functional"] = "B3LYP"
                        functional_found = True
                        break
                    elif "PBE0" in lines[j]:
                        self.data["functional"] = "PBE0"
                        functional_found = True
                        break
                    elif "HSE06" in lines[j]:
                        self.data["functional"] = "HSE06"
                        functional_found = True
                        break
                        
            # Check for 3C methods
            elif "PBEh-3c" in line or "PBEH3C" in line:
                self.data["functional"] = "PBEh-3C"
                self.data["is_3c_method"] = True
                return
            elif "HSE-3c" in line or "HSE3C" in line:
                self.data["functional"] = "HSE-3C"
                self.data["is_3c_method"] = True
                return
            elif "B97-3c" in line or "B973C" in line:
                self.data["functional"] = "B97-3C"
                self.data["is_3c_method"] = True
                return
            elif "PBEsol0-3c" in line or "PBESOL03C" in line:
                self.data["functional"] = "PBEsol0-3C"
                self.data["is_3c_method"] = True
                return
            elif "HSEsol-3c" in line or "HSESOL3C" in line:
                self.data["functional"] = "HSEsol-3C"
                self.data["is_3c_method"] = True
                return
            elif "r2SCAN-3c" in line or "R2SCAN3C" in line:
                self.data["functional"] = "r2SCAN-3C"
                self.data["is_3c_method"] = True
                return
        
        # After extracting functional, check if D3 dispersion is used
        # and append -D3 to the functional name
        if functional_found and self.data.get("functional"):
            # Check for dispersion correction after we have the functional
            for line in lines:
                if "GRIMME DISPERSION CORRECTION (VERSION D3)" in line or "GRIMME D3" in line:
                    self.data["dispersion"] = True
                    # Append -D3 to functional name if not already present
                    if not self.data["functional"].endswith("-D3"):
                        self.data["functional"] = self.data["functional"] + "-D3"
                    break

    def _extract_basis_set(self, lines: List[str]) -> None:
        """Extract basis set information"""
        # Check for "Loading internal basis set:" pattern first (CRYSTAL23 format)
        for i, line in enumerate(lines):
            if "Loading internal basis set:" in line:
                # Extract basis set name after colon
                parts = line.split(":", 1)
                if len(parts) > 1:
                    basis_name = parts[1].strip()
                    if basis_name:
                        self.data["basis_set"] = basis_name
                        self.data["basis_set_type"] = "INTERNAL"
                        return
        
        # First check in the input echo section (most reliable)
        in_input_echo = False
        for i, line in enumerate(lines):
            # Look for input echo markers
            if "*                               CRYSTAL" in line:
                in_input_echo = True
            elif "HAMILTONIAN" in line and "INFORMATION" in line:
                in_input_echo = False
                
            if in_input_echo and "BASISSET" in line and i + 1 < len(lines):
                # The basis set name is typically on the next line
                basis_name = lines[i + 1].strip()
                if basis_name and not basis_name.startswith("*"):
                    self.data["basis_set"] = basis_name
                    self.data["basis_set_type"] = "INTERNAL"
                    return
        
        # Check for TYPE OF BASIS SET line
        for i, line in enumerate(lines):
            if "TYPE OF BASIS SET:" in line:
                if "MOLECULAR ORBITALS FROM OTHER CALCULATION" in line or "GUESSP" in line:
                    # This is a SP calculation using wavefunctions from previous calc
                    # Try to extract the original basis set from input echo
                    self.data["basis_set_type"] = "GUESSP"
                    # The actual basis set might still be in the input echo
                    continue
                elif ":" in line:
                    # Extract basis set name after colon
                    basis_name = line.split(":", 1)[1].strip()
                    if basis_name:
                        self.data["basis_set"] = basis_name
                        self.data["basis_set_type"] = "INTERNAL"
                        return
        
        # If not found in input echo, look in BASIS SET INFORMATION section
        for i, line in enumerate(lines):
            if "BASIS SET INFORMATION" in line:
                # Look for internal or external basis set
                for j in range(i, min(i + 20, len(lines))):
                    if "ADOPTED BASIS SET" in lines[j]:
                        basis_name = lines[j].split(":")[-1].strip()
                        if basis_name:
                            self.data["basis_set"] = basis_name
                            self.data["basis_set_type"] = "INTERNAL"
                            return
                    elif "EXTERNAL" in lines[j] and "BASIS" in lines[j]:
                        self.data["basis_set_type"] = "EXTERNAL"
                        # Try to extract path
                        for k in range(j, min(j + 10, len(lines))):
                            if "DIRECTORY" in lines[k] or "PATH" in lines[k]:
                                path = lines[k].split(":")[-1].strip()
                                if path:
                                    self.data["basis_set"] = path
                                return

    def _extract_tolerances(self, lines: List[str]) -> None:
        """Extract tolerance settings"""
        # Initialize with default values to prevent KeyError
        if "tolerances" not in self.data:
            self.data["tolerances"] = {}
        
        # Look for the actual TOLINTEG values in the output format
        # They appear as:
        # N. OF ATOMS PER CELL         2  COULOMB OVERLAP TOL         (T1) 10**   -7
        # NUMBER OF SHELLS            16  COULOMB PENETRATION TOL     (T2) 10**   -7
        # NUMBER OF AO                36  EXCHANGE OVERLAP TOL        (T3) 10**   20
        # N. OF ELECTRONS PER CELL    12  EXCHANGE PSEUDO OVP (F(G))  (T4) 10**   20
        # CORE ELECTRONS PER CELL      4  EXCHANGE PSEUDO OVP (P(G))  (T5) 10**   20
        
        tolinteg_values = []
        for i, line in enumerate(lines):
            # Skip INFORMATION lines that contain TOLINTEG
            if line.startswith(" INFORMATION") and "TOLINTEG" in line:
                continue
                
            # Extract T1-T5 values
            if "COULOMB OVERLAP TOL" in line and "(T1)" in line:
                match = re.search(r"\(T1\)\s*10\*\*\s*(-?\d+)", line)
                if match:
                    tolinteg_values.append(match.group(1))
            elif "COULOMB PENETRATION TOL" in line and "(T2)" in line:
                match = re.search(r"\(T2\)\s*10\*\*\s*(-?\d+)", line)
                if match:
                    tolinteg_values.append(match.group(1))
            elif "EXCHANGE OVERLAP TOL" in line and "(T3)" in line:
                match = re.search(r"\(T3\)\s*10\*\*\s*(-?\d+)", line)
                if match:
                    tolinteg_values.append(match.group(1))
            elif "EXCHANGE PSEUDO OVP (F(G))" in line and "(T4)" in line:
                match = re.search(r"\(T4\)\s*10\*\*\s*(-?\d+)", line)
                if match:
                    tolinteg_values.append(match.group(1))
            elif "EXCHANGE PSEUDO OVP (P(G))" in line and "(T5)" in line:
                match = re.search(r"\(T5\)\s*10\*\*\s*(-?\d+)", line)
                if match:
                    tolinteg_values.append(match.group(1))
                    
            # Also look for TOLDEE
            elif "TOLDEE" in line and "SCF TOL ON TOTAL ENERGY SET TO" in line:
                match = re.search(r"SCF TOL ON TOTAL ENERGY SET TO\s*(\d+)", line)
                if match:
                    self.data["tolerances"]["TOLDEE"] = int(match.group(1))
        
        # If we found all 5 TOLINTEG values, store them
        if len(tolinteg_values) == 5:
            # Convert negative values to positive (e.g., -7 -> 7)
            # In CRYSTAL output, "10** -7" means tolerance of 10^-7
            # But in input files, we write just "7" for 10^-7
            positive_values = []
            for val in tolinteg_values:
                try:
                    num = int(val)
                    positive_values.append(str(abs(num)))
                except ValueError:
                    positive_values.append(val)
            self.data["tolerances"]["TOLINTEG"] = " ".join(positive_values)

    def _extract_scf_settings(self, lines: List[str]) -> None:
        """Extract SCF settings"""
        for i, line in enumerate(lines):
            if "MIXING SCHEME" in line:
                if "DIIS" in line:
                    self.data["scf_settings"]["method"] = "DIIS"
                elif "BROYDEN" in line:
                    self.data["scf_settings"]["method"] = "BROYDEN"
                elif "ANDERSON" in line:
                    self.data["scf_settings"]["method"] = "ANDERSON"
            elif "FMIXING" in line:
                match = re.search(r"FMIXING\s*=?\s*(\d+)", line)
                if match:
                    self.data["scf_settings"]["fmixing"] = int(match.group(1))
            elif "MAXCYCLE" in line and "SCF" in line:
                match = re.search(r"MAXCYCLE\s*=?\s*(\d+)", line)
                if match:
                    self.data["scf_settings"]["maxcycle"] = int(match.group(1))

    def _extract_kpoints(self, lines: List[str]) -> None:
        """Extract k-point information"""
        for i, line in enumerate(lines):
            if "SHRINK FACTORS" in line:
                # Next line typically has the k-point info
                if i + 1 < len(lines):
                    parts = lines[i + 1].split()
                    if parts and parts[0].isdigit():
                        self.data["k_points"] = int(parts[0])

    def _extract_dft_grid(self, lines: List[str]) -> None:
        """Extract DFT integration grid"""
        for i, line in enumerate(lines):
            if "INTEGRATION GRID" in line:
                if "XLGRID" in line:
                    self.data["dft_grid"] = "XLGRID"
                elif "LGRID" in line:
                    self.data["dft_grid"] = "LGRID"
                elif "DEFAULT" in line or "STANDARD" in line:
                    self.data["dft_grid"] = "DEFAULT"

    def _extract_dispersion(self, lines: List[str]) -> None:
        """Check for dispersion correction"""
        for line in lines:
            if "GRIMME D3" in line or "DFT-D3" in line or "DISPERSION CORRECTION DFT-D3" in line:
                self.data["dispersion"] = True
                # If we have a functional, append -D3 to it
                if self.data.get("functional") and not self.data["functional"].endswith("-D3"):
                    self.data["functional"] = self.data["functional"] + "-D3"
                return

    def _extract_smearing(self, lines: List[str]) -> None:
        """Extract Fermi smearing information"""
        for i, line in enumerate(lines):
            if "FERMI SMEARING" in line:
                self.data["smearing"] = True
                # Try to extract smearing width
                match = re.search(r"WIDTH\s*=?\s*([0-9.]+)", line)
                if match:
                    self.data["smearing_width"] = float(match.group(1))


class CrystalInputParser:
    """Enhanced parser for CRYSTAL17/23 input files"""

    def __init__(self, input_file: str):
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

    def parse(self) -> Dict[str, Any]:
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
                if i + 1 < len(lines):
                    # The k-point values are on the line immediately after SHRINK
                    # Format can be either "k n_shrink" or "0 n_shrink" (followed by ka kb kc on next line)
                    shrink_line = lines[i + 1].strip()
                    if i + 2 < len(lines) and shrink_line.startswith("0 "):
                        # Directional format: next line has ka kb kc
                        self.data["k_points"] = lines[i + 2].strip()
                    else:
                        # Simple format: this line has the k-point info
                        self.data["k_points"] = shrink_line
                break

        return self.data

    def _extract_basis_set(self, lines: List[str]) -> None:
        """Extract basis set information"""
        # Look for BASISSET keyword anywhere in the file
        for i, line in enumerate(lines):
            if line.strip() == "BASISSET":
                self.data["basis_set_type"] = "INTERNAL"
                if i + 1 < len(lines):
                    self.data["basis_set"] = lines[i + 1].strip()
                return
        
        # If BASISSET not found, look for external basis set indicators
        # First find where geometry section ends (either END or where basis data starts)
        geom_end = None
        found_atoms = False
        
        for i, line in enumerate(lines):
            # Check if we've seen atomic positions (lines with atomic numbers and coordinates)
            if len(line.strip().split()) >= 4:
                try:
                    # Try to parse as atomic number followed by coordinates
                    parts = line.strip().split()
                    int(parts[0])  # Should be atomic number
                    float(parts[1])  # Should be x coordinate
                    found_atoms = True
                except (ValueError, IndexError):
                    pass
            
            # Look for END after we've seen atoms, or "99 0" which indicates external basis
            if (line.strip() == "END" and found_atoms) or "99 0" in line:
                if "99 0" in line:
                    # External basis set found
                    self.data["basis_set_type"] = "EXTERNAL"
                    # Find where geometry ends (last atom line)
                    for j in range(i-1, -1, -1):
                        if len(lines[j].strip().split()) >= 4:
                            try:
                                parts = lines[j].strip().split()
                                int(parts[0])  # Should be atomic number
                                float(parts[1])  # Should be x coordinate
                                geom_end = j
                                break
                            except (ValueError, IndexError):
                                pass
                    
                    # Extract all basis set data between geometry end and 99 0
                    if geom_end is not None:
                        for j in range(geom_end + 1, i):
                            line_content = lines[j].strip()
                            if line_content and not line_content.startswith("#"):  # Skip comments
                                self.data["external_basis_data"].append(line_content)
                    return
                else:
                    geom_end = i
                break

    def _extract_optimization_settings(self, lines: List[str]) -> None:
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

    def _extract_dft_settings(self, lines: List[str]) -> None:
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
                        self.data["functional"] = "PBEH3C"
                    elif stripped == "HSE3C":
                        self.data["functional"] = "HSE3C"  
                    elif stripped == "B973C":
                        self.data["functional"] = "B973C"
                    elif stripped == "PBEsol03C":
                        self.data["functional"] = "PBESOL03C"
                    elif stripped == "HSEsol3C":
                        self.data["functional"] = "HSESOL3C"
                    self.data["is_3c_method"] = True
                    
                # Check for D3 dispersion (both explicit and in functional name)
                elif stripped.endswith("-D3") or "-D3" in stripped:
                    # Handle functionals with -D3 suffix
                    base_functional = stripped.replace("-D3", "")
                    self.data["functional"] = stripped  # Keep the full name with -D3
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
        
    def _extract_smearing_settings(self, lines: List[str]) -> None:
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
                        
    def _extract_tolerance_settings(self, lines: List[str]) -> None:
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
            
    def _extract_kpoint_settings(self, lines: List[str]) -> None:
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
                            
    def _extract_scf_settings(self, lines: List[str]) -> None:
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