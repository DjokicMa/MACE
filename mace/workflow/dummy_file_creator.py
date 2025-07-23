#!/usr/bin/env python3
"""
Dummy File Creator for CRYSTAL Workflow
========================================
Creates properly formatted dummy D12 and OUT files for CRYSTALOptToD12.py
and other scripts that require input/output file pairs.

This module extracts real geometry and settings from D12 files and creates
dummy output files that match CRYSTAL's output format exactly.

Author: Marcus Djokic
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Try to import constants from d12_constants
try:
    # Add the Crystal_d12 directory to path
    parent_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(parent_dir / "Crystal_d12"))
    from d12_constants import (
        ATOMIC_NUMBER_TO_SYMBOL, 
        SPACEGROUP_SYMBOLS,
        SPACEGROUP_ALTERNATIVES,
        SPACEGROUP_SYMBOL_TO_NUMBER,
        FUNCTIONAL_CATEGORIES,
        FUNCTIONAL_KEYWORD_MAP,
        DEFAULT_SETTINGS,
        DEFAULT_TOLERANCES,
        DEFAULT_OPT_SETTINGS,
        COMMON_FUNCTIONALS,
        D3_FUNCTIONALS,
        generate_k_points
    )
    from d12_parsers import CrystalInputParser
except ImportError as e:
    print(f"ERROR: Could not import required d12_constants: {e}")
    print("The dummy_file_creator requires access to Crystal_d12 constants.")
    print("Please ensure the Crystal_d12 directory is in the correct location.")
    raise ImportError("Required d12_constants module not found")


class DummyFileCreator:
    """Creates dummy D12 and OUT files for workflow configuration."""
    
    def __init__(self):
        """Initialize the dummy file creator."""
        pass
    
    def extract_d12_settings(self, d12_file: Path) -> Dict[str, Any]:
        """Extract comprehensive settings from a D12 file.
        
        Args:
            d12_file: Path to the D12 input file
            
        Returns:
            Dictionary containing all extracted settings
        """
        if not d12_file.exists():
            return self._get_default_settings()
        
        # Always use manual extraction for now
        # The CrystalInputParser isn't parsing cell_parameters correctly
        return self._extract_d12_settings_manual(d12_file)
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Return default settings when no D12 file is available."""
        return {
            "title": "CRYSTAL CALCULATION",
            "basis_set": "POB-TZVP-REV2",
            "functional": "PBE-D3",
            "dft_grid": "XLGRID",
            "spin": False,
            "spacegroup": 1,
            "origin_setting": "0 0 0",
            "dimensionality": "CRYSTAL",
            "cell_parameters": None,
            "k_points": None,
            "n_atoms": 0,
            "atoms": [],
            "tolinteg": None,
            "toldee": None,
            "maxcycle": None,
            "fmixing": None,
            "shrink_values": None,
            "smearing": False,
            "smearing_width": None,
            "spinlock": None,
            "external_basis": False,
            "is_3c_method": False,
        }
    
    def _extract_d12_settings_manual(self, d12_file: Path) -> Dict[str, Any]:
        """Manual extraction of D12 settings (fallback method)."""
        settings = self._get_default_settings()
        
        try:
            with open(d12_file, "r") as f:
                lines = f.readlines()
            
            # Extract title (first line)
            if lines:
                settings["title"] = lines[0].strip()
            
            # Extract structure information
            for i in range(1, len(lines)):
                line = lines[i].strip()
                
                # Check for dimensionality
                if line in ["CRYSTAL", "SLAB", "POLYMER", "MOLECULE"]:
                    settings["dimensionality"] = line
                    
                    # For CRYSTAL, next lines are origin and space group
                    if line == "CRYSTAL" and i + 2 < len(lines):
                        settings["origin_setting"] = lines[i + 1].strip()
                        try:
                            settings["spacegroup"] = int(lines[i + 2].strip())
                            
                            # Extract cell parameters
                            if i + 3 < len(lines):
                                cell_line = lines[i + 3].strip()
                                cell_params = cell_line.split()
                                if len(cell_params) >= 1:
                                    settings["cell_parameters"] = self._parse_cell_parameters(
                                        settings["spacegroup"], cell_params
                                    )
                                    
                                    # Calculate k-points if we have cell parameters
                                    if settings["cell_parameters"]:
                                        cp = settings["cell_parameters"]
                                        # Use generate_k_points from d12_constants
                                        ka, kb, kc = generate_k_points(
                                            float(cp["a"]), float(cp["b"]), float(cp["c"]),
                                            settings.get("dimensionality", "CRYSTAL"),
                                            settings.get("spacegroup", 1)
                                        )
                                        settings["k_points"] = f"{ka} {kb} {kc}"
                        except ValueError:
                            pass
                    break
            
            # Extract atomic positions
            settings = self._extract_atoms(lines, settings)
            
            # Extract basis set
            settings = self._extract_basis_set(lines, settings)
            
            # Extract DFT settings
            settings = self._extract_dft_settings(lines, settings)
            
            # Extract SCF settings
            settings = self._extract_scf_settings(lines, settings)
            
            # Extract SHRINK if present
            settings = self._extract_shrink(lines, settings)
            
        except Exception as e:
            print(f"Warning: Could not extract settings from D12: {e}")
        
        return settings
    
    def _parse_cell_parameters(self, spacegroup: int, cell_params: List[str]) -> Optional[Dict[str, float]]:
        """Parse cell parameters based on space group.
        
        Args:
            spacegroup: Space group number
            cell_params: List of cell parameter values
            
        Returns:
            Dictionary with a, b, c, alpha, beta, gamma or None
        """
        try:
            # Cubic: only a
            if 195 <= spacegroup <= 230:
                a = float(cell_params[0])
                return {
                    "a": a, "b": a, "c": a,
                    "alpha": 90.0, "beta": 90.0, "gamma": 90.0
                }
            # Tetragonal: a and c
            elif 75 <= spacegroup <= 142:
                a = float(cell_params[0])
                c = float(cell_params[1]) if len(cell_params) > 1 else a
                return {
                    "a": a, "b": a, "c": c,
                    "alpha": 90.0, "beta": 90.0, "gamma": 90.0
                }
            # Hexagonal/Trigonal: a and c
            elif (143 <= spacegroup <= 194) or (spacegroup in [146, 148, 155, 160, 161, 166, 167]):
                a = float(cell_params[0])
                c = float(cell_params[1]) if len(cell_params) > 1 else a
                return {
                    "a": a, "b": a, "c": c,
                    "alpha": 90.0, "beta": 90.0, "gamma": 120.0
                }
            # Orthorhombic: a, b, c
            elif 16 <= spacegroup <= 74:
                if len(cell_params) >= 3:
                    return {
                        "a": float(cell_params[0]),
                        "b": float(cell_params[1]),
                        "c": float(cell_params[2]),
                        "alpha": 90.0, "beta": 90.0, "gamma": 90.0
                    }
            # Monoclinic: a, b, c, beta
            elif 3 <= spacegroup <= 15:
                if len(cell_params) >= 4:
                    return {
                        "a": float(cell_params[0]),
                        "b": float(cell_params[1]),
                        "c": float(cell_params[2]),
                        "alpha": 90.0,
                        "beta": float(cell_params[3]),
                        "gamma": 90.0
                    }
            # Triclinic: a, b, c, alpha, beta, gamma
            else:
                if len(cell_params) >= 6:
                    return {
                        "a": float(cell_params[0]),
                        "b": float(cell_params[1]),
                        "c": float(cell_params[2]),
                        "alpha": float(cell_params[3]),
                        "beta": float(cell_params[4]),
                        "gamma": float(cell_params[5])
                    }
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _extract_atoms(self, lines: List[str], settings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract atomic positions from D12 lines."""
        # Look for atomic positions regardless of whether cell_parameters were parsed
        # This fixes the circular dependency issue
        for i, line in enumerate(lines):
            # Look for a line that's just a number (the atom count)
            # It typically appears after the cell parameters
            if line.strip().isdigit() and i >= 4:
                try:
                    n_atoms = int(line.strip())
                    if 0 < n_atoms < 1000:  # Reasonable number
                        settings["n_atoms"] = n_atoms
                        settings["atoms"] = []
                        
                        # Read atom lines
                        for j in range(i + 1, min(i + 1 + n_atoms, len(lines))):
                            atom_line = lines[j].strip().split()
                            if len(atom_line) >= 4:
                                settings["atoms"].append({
                                    "atomic_number": atom_line[0],
                                    "x": atom_line[1],
                                    "y": atom_line[2],
                                    "z": atom_line[3]
                                })
                        break
                except ValueError:
                    continue
        
        return settings
    
    def _extract_basis_set(self, lines: List[str], settings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basis set information."""
        content = "".join(lines)
        if "BASISSET" in content:
            for i, line in enumerate(lines):
                if line.strip() == "BASISSET":
                    if i + 1 < len(lines):
                        settings["basis_set"] = lines[i + 1].strip()
                        break
        
        return settings
    
    def _extract_dft_settings(self, lines: List[str], settings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DFT functional and settings."""
        content = "".join(lines)
        if "DFT" in content and "ENDDFT" in content:
            dft_start = content.find("DFT")
            dft_end = content.find("ENDDFT")
            if dft_start != -1 and dft_end != -1:
                dft_section = content[dft_start:dft_end]
                dft_lines = dft_section.split("\n")
                
                for line in dft_lines[1:]:  # Skip 'DFT' line
                    line = line.strip()
                    if line and not line.startswith("END"):
                        if line == "SPIN":
                            settings["spin"] = True
                        elif "GRID" in line:
                            settings["dft_grid"] = line
                        elif line and "GRID" not in line and line != "SPIN":
                            settings["functional"] = line
        
        return settings
    
    def _extract_scf_settings(self, lines: List[str], settings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract SCF convergence settings."""
        for i, line in enumerate(lines):
            if line.strip() == "TOLINTEG":
                if i + 1 < len(lines):
                    settings["tolinteg"] = lines[i + 1].strip()
            elif line.strip() == "TOLDEE":
                if i + 1 < len(lines):
                    settings["toldee"] = lines[i + 1].strip()
            elif line.strip().startswith("MAXCYCLE"):
                parts = line.strip().split()
                if len(parts) > 1:
                    settings["maxcycle"] = parts[1]
            elif line.strip().startswith("FMIXING"):
                parts = line.strip().split()
                if len(parts) > 1:
                    settings["fmixing"] = parts[1]
        
        return settings
    
    def _extract_shrink(self, lines: List[str], settings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract SHRINK values (k-points)."""
        for i, line in enumerate(lines):
            if line.strip() == "SHRINK":
                if i + 1 < len(lines):
                    shrink_line = lines[i + 1].strip().split()
                    if len(shrink_line) >= 2:
                        try:
                            is1 = int(shrink_line[0])
                            is2 = int(shrink_line[1])
                            settings["k_points"] = f"{is1} {is1} {is1}"
                            settings["shrink_values"] = f"{is1} {is2}"
                        except ValueError:
                            pass
                    if i + 2 < len(lines) and len(shrink_line) == 1:
                        # For anisotropic k-points
                        gilat_line = lines[i + 2].strip().split()
                        if len(gilat_line) >= 3:
                            try:
                                settings["k_points"] = f"{gilat_line[0]} {gilat_line[1]} {gilat_line[2]}"
                            except:
                                pass
                break
        
        return settings
    
    def create_dummy_out(self, out_file: Path, d12_settings: Dict[str, Any]) -> None:
        """Create a dummy OUT file that matches CRYSTAL output format exactly.
        
        Args:
            out_file: Path to write the dummy output file
            d12_settings: Settings extracted from D12 file
        """
        with open(out_file, "w") as f:
            # CRYSTAL header section
            f.write(" *******************************************************************************\n")
            f.write(" *                                                                             *\n")
            f.write(" *                               CRYSTAL23                                     *\n")
            f.write(" *                      public : 1.0.1 - October 2023                         *\n")
            f.write(" *                                                                             *\n")
            f.write(" *******************************************************************************\n")
            f.write("\n")
            
            # Add title from D12
            f.write(f" {d12_settings.get('title', 'CRYSTAL CALCULATION')}\n")
            f.write("\n")
            
            # Write dimensionality info in CRYSTAL format
            dim = d12_settings.get("dimensionality", "CRYSTAL")
            if dim == "CRYSTAL":
                f.write(" CRYSTAL CALCULATION\n")
            elif dim == "SLAB":
                f.write(" SLAB CALCULATION\n")
            elif dim == "POLYMER":
                f.write(" POLYMER CALCULATION\n")
            else:
                f.write(" MOLECULE CALCULATION\n")
            f.write("\n")
            
            # Add space group information - parser looks for "SPACE GROUP : <symbol>"
            sg_num = d12_settings.get('spacegroup', 1)
            sg_symbol = SPACEGROUP_SYMBOLS.get(sg_num, 'P1')
            
            # The parser expects space groups without extra spaces between characters
            # The parser will try to match against SPACEGROUP_SYMBOL_TO_NUMBER which has
            # symbols like "Fm-3m", "P-4m2", etc. without spaces
            # So we should output them in that format
            f.write(f" SPACE GROUP : {sg_symbol}\n")
            f.write("\n")
            
            # Add basis set information for parser (CRYSTAL23 format)
            basis_set = d12_settings.get('basis_set', 'POB-TZVP-REV2')
            f.write(f" Loading internal basis set: {basis_set}\n")
            f.write("\n")
            
            # Write cell parameters - parser looks for CRYSTALLOGRAPHIC CELL or PRIMITIVE CELL
            if d12_settings.get("cell_parameters") and dim != "MOLECULE":
                cell = d12_settings["cell_parameters"]
                # Parser looks for "CRYSTALLOGRAPHIC CELL (VOLUME="
                volume = float(cell['a']) * float(cell['b']) * float(cell['c'])  # Simplified
                f.write(f" CRYSTALLOGRAPHIC CELL (VOLUME= {volume:.5f})\n")
                f.write("         A              B              C           ALPHA      BETA       GAMMA\n")
                f.write(f"     {float(cell['a']):14.8f} {float(cell['b']):14.8f} {float(cell['c']):14.8f}  "
                       f"{float(cell['alpha']):9.5f}  {float(cell['beta']):9.5f}  {float(cell['gamma']):9.5f}\n")
                f.write("\n")
            
            # Skip to geometry optimization section
            f.write(" INFORMATION **** OPTGEOM **** OPTIMIZES BOTH ATOMIC COORDINATES AND CELL PARAMETERS\n")
            f.write("\n")
            
            # Add geometry section
            f.write(" GEOMETRY FOR WAVE FUNCTION - DIMENSIONALITY OF THE SYSTEM    3\n")
            f.write(" (NON PERIODIC DIRECTION: LATTICE PARAMETER FORMALLY SET TO 500)\n")
            f.write(" *******************************************************************************\n")
            
            # Write primitive cell parameters - parser looks for "PRIMITIVE CELL" and "LATTICE PARAMETERS"
            if d12_settings.get("cell_parameters") and dim != "MOLECULE":
                cell = d12_settings["cell_parameters"]
                f.write(" PRIMITIVE CELL - LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM\n")
                # Parser looks for line with ALPHA BETA GAMMA
                f.write("         A              B              C           ALPHA      BETA       GAMMA\n")
                f.write(f"     {float(cell['a']):14.8f} {float(cell['b']):14.8f} {float(cell['c']):14.8f}  "
                       f"{float(cell['alpha']):9.5f}  {float(cell['beta']):9.5f}  {float(cell['gamma']):9.5f}\n")
            elif dim != "MOLECULE":
                # Default if no cell parameters found
                f.write(" PRIMITIVE CELL - LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM\n")
                f.write("         A              B              C           ALPHA      BETA       GAMMA\n")
                f.write("     5.00000000     5.00000000     5.00000000    90.000000  90.000000  90.000000\n")
            f.write(" *******************************************************************************\n")
            
            # Write atomic positions - parser looks for "COORDINATES IN THE CRYSTALLOGRAPHIC CELL"
            if d12_settings.get("atoms") and d12_settings.get("n_atoms"):
                n_atoms = d12_settings["n_atoms"]
                f.write(" COORDINATES IN THE CRYSTALLOGRAPHIC CELL\n")
                f.write(" *******************************************************************************\n")
                f.write("   ATOM          X(FRAC)          Y(FRAC)          Z(FRAC)\n")
                
                for i, atom in enumerate(d12_settings["atoms"]):
                    # Parser expects format: index T/F atom_number symbol x y z
                    is_unique = "T" if i < n_atoms else "F"
                    atom_symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(atom['atomic_number']), 'X')
                    x_frac = float(atom['x'])
                    y_frac = float(atom['y'])
                    z_frac = float(atom['z'])
                    # Format: "1 T 6 C 0.0000 0.0000 0.0000"
                    # Parser expects exactly 7 parts: index T/F atomic_number symbol x y z
                    f.write(f"   {i+1:3d} {is_unique}  {int(atom['atomic_number']):3d} {atom_symbol:<2s}  "
                           f"{x_frac:17.14f}  {y_frac:17.14f}  {z_frac:17.14f}\n")
                f.write("\n")
            
            # Add calculation type section
            if d12_settings.get("spin"):
                f.write(" TYPE OF CALCULATION :  UNRESTRICTED OPEN SHELL\n")
            else:
                f.write(" TYPE OF CALCULATION :  RESTRICTED CLOSED SHELL\n")
            
            # Check if this is a DFT calculation by looking at the functional
            functional = d12_settings.get("functional", "HF")
            # If functional is not pure HF, it's a DFT calculation
            if functional != "HF" and functional != "HARTREE-FOCK":
                f.write(" KOHN-SHAM HAMILTONIAN\n")
                f.write("\n")
                
                # Check for 3C composite methods first
                if functional in ["HF3C", "HF-3C", "HFSOL3C", "HFSOL-3C", 
                                 "PBEH3C", "PBEh-3C", "HSE3C", "HSE-3C", 
                                 "B973C", "B97-3C", "PBESOL03C", "PBEsol0-3C",
                                 "HSESOL3C", "HSEsol-3C", "R2SCAN3C", "r2SCAN-3C"]:
                    # 3C methods have their own special output format
                    f.write(f" COMPOSITE METHOD: {functional}\n")
                    f.write(" INCLUDES: GEOMETRICAL COUNTERPOISE CORRECTION (gCP)\n")
                    f.write("           GRIMME D3 DISPERSION CORRECTION\n")
                    f.write("           MODIFIED BASIS SET\n")
                    f.write("\n")
                else:
                    # Regular DFT functionals - based on exact parser logic
                    # Strip dispersion corrections for functional matching
                    base_functional = functional.replace("-D3", "").strip()
                    
                    # Write functional info EXACTLY as CRYSTAL outputs it (based on parser expectations)
                    # The parser looks for these exact strings to identify functionals
                    
                    # For new CRYSTAL23 format: (EXCHANGE)[CORRELATION] FUNCTIONAL:
                    if base_functional in ["PBE", "BLYP", "B3LYP", "PBE0", "HSE06"]:
                        f.write(" (EXCHANGE)[CORRELATION] FUNCTIONAL:")
                        
                        if base_functional == "PBE":
                            f.write("(PERDEW-BURKE-ERNZERHOF)\n")
                        elif base_functional == "BLYP":
                            # Parser looks for "BECKE 88" and "LEE-YANG-PARR" on same line
                            f.write("(BECKE 88)[LEE-YANG-PARR]\n")
                        elif base_functional == "B3LYP":
                            # Parser looks for "B3LYP" directly in line
                            f.write("B3LYP\n")
                        elif base_functional == "PBE0":
                            # Parser looks for "PBE0" directly in line
                            f.write("PBE0\n")
                        elif base_functional == "HSE06":
                            # Parser looks for "HSE06" directly in line
                            f.write("HSE06\n")
                    
                    # For older format: EXCHANGE-CORRELATION FUNCTIONAL
                    elif base_functional == "PW91":
                        f.write(" EXCHANGE-CORRELATION FUNCTIONAL\n")
                        f.write(" PERDEW-WANG\n")
                    
                    # Note: The parser doesn't look for other functional details like
                    # NON-LOCAL WEIGHTING FACTOR or HYBRID EXCHANGE percentage
                    # So we don't need to output those for the parser to work
                    
                    # Add dispersion correction info if present (ONLY D3 is supported in CRYSTAL)
                    if "-D3" in functional:
                        f.write("\n")
                        # Parser looks for this exact string
                        f.write(" GRIMME DISPERSION CORRECTION (VERSION D3)\n")
                        
                    f.write("\n")
            f.write("\n")
            
            # Add tolerances section - parser looks for T1-T5 values
            tolinteg_values = d12_settings.get("tolinteg")
            if tolinteg_values is None:
                tolinteg_values = "7 7 7 7 14"
            if tolinteg_values:
                f.write(" COULOMB AND EXCHANGE INTEGRALS EVALUATION\n")
                f.write("\n")
                tolinteg = tolinteg_values.split()
                if len(tolinteg) >= 5:
                    f.write(f" COULOMB OVERLAP TOL         (T1) 10**   -{tolinteg[0]}\n")
                    f.write(f" COULOMB PENETRATION TOL     (T2) 10**   -{tolinteg[1]}\n")
                    f.write(f" EXCHANGE OVERLAP TOL        (T3) 10**   {tolinteg[2]}\n")
                    f.write(f" EXCHANGE PSEUDO OVP (F(G))  (T4) 10**   {tolinteg[3]}\n")
                    f.write(f" EXCHANGE PSEUDO OVP (P(G))  (T5) 10**   {tolinteg[4]}\n")
                f.write("\n")
            
            # Add SCF settings - parser looks for these specific formats
            if d12_settings.get("toldee"):
                f.write(f" TOLDEE - SCF TOL ON TOTAL ENERGY SET TO {d12_settings['toldee']}\n")
            if d12_settings.get("maxcycle"):
                # Parser looks for MAXCYCLE with SCF in the line
                f.write(f" SCF MAXCYCLE = {d12_settings['maxcycle']}\n")
            if d12_settings.get("fmixing"):
                # Parser looks for specific FMIXING format
                f.write(f" FMIXING = {d12_settings['fmixing']}\n")
            
            # Add k-points section - parser looks for "SHRINK FACTORS"
            shrink_values = d12_settings.get("shrink_values")
            k_points = d12_settings.get("k_points")
            
            # If neither shrink_values nor k_points are provided, calculate them
            if shrink_values is None and k_points is None:
                # Calculate k-points based on cell parameters
                cell_params = d12_settings.get("cell_parameters")
                if cell_params:
                    a = float(cell_params.get('a', 10.0))
                    b = float(cell_params.get('b', 10.0))
                    c = float(cell_params.get('c', 10.0))
                    dimensionality = d12_settings.get('dimensionality', 'CRYSTAL')
                    spacegroup = d12_settings.get('spacegroup', 1)
                    
                    # Use the same logic as CRYSTALOptToD12.py
                    ka, kb, kc = generate_k_points(a, b, c, dimensionality, spacegroup)
                    # For SHRINK format, use the maximum k value as IS1 and IS2
                    k_max = max(ka, kb, kc)
                    shrink_values = f"{k_max} {k_max}"
                else:
                    # Fallback if no cell parameters
                    shrink_values = "8 8"
            
            if shrink_values:
                shrink = shrink_values.split()
                if len(shrink) >= 2:
                    f.write(" SHRINK FACTORS(MONKH.)\n")
                    f.write(f" {shrink[0]} {shrink[1]}\n")
                    f.write("\n")
            f.write("\n")
            
            # Add convergence section
            f.write(" == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -1.0000000000000E+02 CYCLES   1\n")
            f.write("\n")
            
            # Add final optimized geometry section (critical for parser)
            f.write(" FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n")
            f.write(" (NON PERIODIC DIRECTION: LATTICE PARAMETER FORMALLY SET TO 500)\n")
            f.write(" *******************************************************************************\n")
            
            # Final lattice parameters - ensure parser can find them
            if d12_settings.get("cell_parameters") and dim != "MOLECULE":
                cell = d12_settings["cell_parameters"]
                # Use same format as before for consistency
                f.write(" PRIMITIVE CELL - LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM\n")
                f.write("         A              B              C           ALPHA      BETA       GAMMA\n")
                f.write(f"     {float(cell['a']):14.8f} {float(cell['b']):14.8f} {float(cell['c']):14.8f}  "
                       f"{float(cell['alpha']):9.5f}  {float(cell['beta']):9.5f}  {float(cell['gamma']):9.5f}\n")
            f.write(" *******************************************************************************\n")
            
            # Add COORDINATES IN THE CRYSTALLOGRAPHIC CELL section first (parser prefers this)
            if d12_settings.get("atoms") and d12_settings.get("n_atoms"):
                n_atoms = d12_settings["n_atoms"]
                f.write(" COORDINATES IN THE CRYSTALLOGRAPHIC CELL\n")
                f.write(" *******************************************************************************\n")
                f.write("   ATOM          X(FRAC)          Y(FRAC)          Z(FRAC)\n")
                
                for i, atom in enumerate(d12_settings["atoms"]):
                    # Parser expects format: index T/F atomic_number symbol x y z
                    is_unique = "T" if i < n_atoms else "F"
                    atom_symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(atom['atomic_number']), 'X')
                    x_frac = float(atom['x'])
                    y_frac = float(atom['y'])
                    z_frac = float(atom['z'])
                    # Format: "1 T 6 C 0.0000 0.0000 0.0000"
                    f.write(f"   {i+1:3d} {is_unique}  {int(atom['atomic_number']):3d} {atom_symbol:<2s}  "
                           f"{x_frac:17.14f}  {y_frac:17.14f}  {z_frac:17.14f}\n")
                
                f.write("\n")
                
                # Also add ATOMS IN THE ASYMMETRIC UNIT section
                f.write(f" ATOMS IN THE ASYMMETRIC UNIT    {n_atoms} - ATOMS IN THE UNIT CELL:    {n_atoms}\n")
                f.write(" *******************************************************************************\n")
                f.write("     ATOM              X/A                 Y/B                 Z/C\n")
                f.write(" *******************************************************************************\n")
                
                for i, atom in enumerate(d12_settings["atoms"]):
                    is_unique = "T"
                    atom_symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(atom['atomic_number']), 'X')
                    x_frac = float(atom['x'])
                    y_frac = float(atom['y'])
                    z_frac = float(atom['z'])
                    # Same format as before
                    f.write(f"   {i+1:3d} {is_unique}  {int(atom['atomic_number']):3d} {atom_symbol:<2s}  "
                           f"{x_frac:17.14f}  {y_frac:17.14f}  {z_frac:17.14f}\n")
            
            f.write("\n")
            f.write(" T = ATOM BELONGING TO THE ASYMMETRIC UNIT\n")
            f.write("\n")
            
            # Add cartesian coordinates section (also used by parser)
            if d12_settings.get("atoms") and d12_settings.get("n_atoms") and dim != "MOLECULE":
                f.write("\n")
                f.write(" CARTESIAN COORDINATES - PRIMITIVE CELL\n")
                f.write(" *******************************************************************************\n")
                f.write(" *      ATOM          X(ANGSTROM)         Y(ANGSTROM)         Z(ANGSTROM)\n")
                f.write(" *******************************************************************************\n")
                
                # For now, use dummy cartesian coords
                for i, atom in enumerate(d12_settings["atoms"][:4]):
                    atom_symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(atom['atomic_number']), 'X')
                    # These would normally be calculated from fractional coords and cell
                    f.write(f"   {i+1:3d}    {atom['atomic_number']:2s} {atom_symbol:<2s}   "
                           f"-1.0000000000000E+00 -1.0000000000000E+00 -1.0000000000000E+00\n")
            
            f.write("\n")
            f.write(" TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT END         TELAPSE       59.36 TCPU       57.56\n")
    
    def create_minimal_dummy_out(self, out_file: Path) -> None:
        """Create a minimal dummy OUT file when no D12 is available.
        
        Args:
            out_file: Path to write the dummy output file
        """
        with open(out_file, "w") as f:
            f.write(" *******************************************************************************\n")
            f.write(" *                                                                             *\n")
            f.write(" *                               CRYSTAL23                                     *\n")
            f.write(" *                      public : 1.0.1 - October 2023                         *\n")
            f.write(" *                                                                             *\n")
            f.write(" *******************************************************************************\n")
            f.write("\n")
            f.write(" CRYSTAL CALCULATION\n")
            f.write("\n")
            # Space group - parser looks for this format (without extra spaces)
            f.write(" SPACE GROUP : P1\n")
            f.write("\n")
            # Lattice parameters in format parser expects
            f.write(" CRYSTALLOGRAPHIC CELL (VOLUME=  125.00000)\n")
            f.write("         A              B              C           ALPHA      BETA       GAMMA\n")
            f.write("      5.00000000      5.00000000      5.00000000    90.00000   90.00000   90.00000\n")
            f.write("\n")
            f.write(" FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n")
            f.write(" (NON PERIODIC DIRECTION: LATTICE PARAMETER FORMALLY SET TO 500)\n")
            f.write(" *******************************************************************************\n")
            f.write(" PRIMITIVE CELL - LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM\n")
            f.write("         A              B              C           ALPHA      BETA       GAMMA\n")
            f.write("      5.00000000      5.00000000      5.00000000    90.00000   90.00000   90.00000\n")
            f.write(" *******************************************************************************\n")
            # Minimal atomic positions
            f.write(" ATOMS IN THE ASYMMETRIC UNIT    1 - ATOMS IN THE UNIT CELL:    1\n")
            f.write(" *******************************************************************************\n")
            f.write("     ATOM              X/A                 Y/B                 Z/C\n")
            f.write(" *******************************************************************************\n")
            f.write("     1 T    6 C     0.00000000000000  0.00000000000000  0.00000000000000\n")
            f.write("\n")
            f.write(" TYPE OF CALCULATION :  RESTRICTED CLOSED SHELL\n")
            f.write(" KOHN-SHAM HAMILTONIAN\n")
            f.write("\n")
            f.write(" (EXCHANGE)[CORRELATION] FUNCTIONAL:(BECKE 88)[LEE-YANG-PARR]\n")
            f.write("\n")
            f.write(" == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -1.0000000000000E+02 CYCLES   1\n")
            f.write("\n")
            f.write(" TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT END         TELAPSE       59.36 TCPU       57.56\n")
    
    def create_dummy_d12(self, d12_file: Path, title: str = "DUMMY CRYSTAL CALCULATION") -> None:
        """Create a minimal dummy D12 file.
        
        Args:
            d12_file: Path to write the dummy D12 file
            title: Title for the calculation
        """
        with open(d12_file, "w") as f:
            f.write(f"{title}\n")
            f.write("CRYSTAL\n")
            f.write("0 0 0\n")
            f.write("225\n")  # Fm-3m (cubic)
            f.write("5.0\n")  # Lattice parameter
            f.write("2\n")    # Number of atoms
            f.write("14 0.0 0.0 0.0\n")  # Si at origin
            f.write("14 0.25 0.25 0.25\n")  # Si at 1/4,1/4,1/4
            f.write("OPTGEOM\n")
            f.write("FULLOPTG\n")
            f.write("ENDOPT\n")
            f.write("BASISSET\n")
            f.write("POB-TZVP-REV2\n")
            f.write("DFT\n")
            f.write("PBE-D3\n")
            f.write("XLGRID\n")
            f.write("ENDDFT\n")
            f.write("SHRINK\n")
            f.write("0 30\n")
            f.write("9 9 9\n")
            f.write("TOLINTEG\n")
            f.write("9 9 9 9 18\n")
            f.write("TOLDEE\n")
            f.write("9\n")
            f.write("END\n")
    
    def create_dummy_fort9(self, f9_file: Path) -> None:
        """Create a dummy fort.9 wavefunction file.
        
        Args:
            f9_file: Path to write the dummy fort.9 file
        """
        with open(f9_file, "w") as f:
            f.write("DUMMY WAVEFUNCTION FILE\n")
    
    # Alias for backwards compatibility
    def create_dummy_f9(self, f9_file: Path) -> None:
        """Alias for create_dummy_fort9 for backwards compatibility."""
        self.create_dummy_fort9(f9_file)


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create dummy D12 and OUT files for CRYSTAL workflows"
    )
    parser.add_argument(
        "--d12-input", 
        help="Input D12 file to extract settings from"
    )
    parser.add_argument(
        "--out-file", 
        required=True,
        help="Output file path for dummy OUT"
    )
    parser.add_argument(
        "--d12-file",
        help="Output file path for dummy D12 (optional)"
    )
    parser.add_argument(
        "--f9-file",
        help="Output file path for dummy fort.9 (optional)"
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Create minimal dummy files without extracting from D12"
    )
    
    args = parser.parse_args()
    
    creator = DummyFileCreator()
    
    if args.minimal or not args.d12_input:
        # Create minimal dummy files
        creator.create_minimal_dummy_out(Path(args.out_file))
        if args.d12_file:
            creator.create_dummy_d12(Path(args.d12_file))
        if args.f9_file:
            creator.create_dummy_fort9(Path(args.f9_file))
    else:
        # Extract settings and create realistic dummy OUT
        d12_path = Path(args.d12_input)
        if d12_path.exists():
            settings = creator.extract_d12_settings(d12_path)
            creator.create_dummy_out(Path(args.out_file), settings)
            if args.f9_file:
                creator.create_dummy_fort9(Path(args.f9_file))
        else:
            print(f"Error: D12 file not found: {d12_path}")
            sys.exit(1)
    
    print(f"Created dummy files successfully")