#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D12 File Writer Module for CRYSTAL23
------------------------------------
This module contains shared functions for writing D12 input files.
It consolidates the common D12 writing logic from NewCifToD12.py 
and CRYSTALOptToD12.py to reduce code duplication.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

from typing import Dict, Any, List, Optional, TextIO


def write_method_block(f: TextIO, method: str, options: Dict[str, Any]) -> None:
    """
    Write method (DFT/HF) configuration block to D12 file.
    
    Args:
        f: File handle to write to
        method: Method type ('DFT' or 'HF')
        options: Dictionary containing method configuration
    """
    if method == "DFT":
        # Write DFT method
        functional = options.get("dft_functional", "PBE")
        
        # Check if it's a 3C functional
        is_3c_functional = "-3C" in functional or functional.endswith("3C")
        
        # Write DFT keyword
        f.write("DFT\n")
        
        # Map functionals to CRYSTAL keywords
        functional_map = {
            # LDA
            "SVWN": "SVWN",
            "SVWN5": "SVWN5", 
            "VWN": "VWN",
            "PWLDA": "PWLDA",
            # GGA
            "PBE": "PBE",
            "PBESOL": "PBESOL",
            "BLYP": "BLYP",
            "BP86": "BP",
            "PW91": "PW91",
            "mPWPW": "PWPW",
            "SOGGA": "SOGGA",
            "SOGGA11": "SOGGA11",
            # Hybrid GGA
            "B3LYP": "B3LYP",
            "B3PW": "B3PW",
            "PBE0": "PBE0",
            "PBESOL0": "PBESOL0",
            "HSE06": "HSE06",
            "HSESOL": "HSESOL",
            "LC-wPBE": "WCGGA-PBE",
            "CAM-B3LYP": "CAM-B3LYP",
            "wB97": "WB97",
            "LC-BLYP": "WCGGA",
            # meta-GGA
            "M06L": "M06L",
            "TPSS": "TPSS",
            "SCAN": "SCAN",
            "MN15L": "MN15L",
            # Hybrid meta-GGA
            "M06": "M06",
            "M062X": "M062X",
            "MN15": "MN15",
            "TPSSh": "TPSSH",
            # 3C functionals
            "PBEh-3C": "PBEH3C",
            "HSE-3C": "HSE3C",
            "B97-3C": "B973C",
            "PBEsol0-3C": "PBESOL03C",
            "HSEsol-3C": "HSESOL3C",
            "r2SCAN-3C": "R2SCAN3C",
        }
        
        # Write functional
        crystal_functional = functional_map.get(functional, functional)
        
        # Handle special cases
        if functional == "LC-wPBE":
            f.write(f"WCGGA\nPBE\n")
        elif functional == "CAM-B3LYP":
            f.write(f"CAM-B3LYP\n")
        elif functional == "LC-BLYP":
            f.write(f"WCGGA\n")
        else:
            f.write(f"{crystal_functional}\n")
        
        # Add NONLOCAL for specific functionals
        nonlocal_functionals = [
            "BLYP", "BP86", "mPWPW", "PW91", "SVWN", "SVWN5", "VWN", 
            "PWLDA", "B97-3C", "LC-BLYP"
        ]
        if functional in nonlocal_functionals:
            f.write("NONLOCAL\n")
        
        # Write exchange if specified
        if "exchange" in options:
            f.write(f"EXCHANGE\n{options['exchange']}\n")
        
        # Write correlation if specified
        if "correlat" in options:
            f.write(f"CORRELAT\n{options['correlat']}\n")
        
        # Dispersion correction (if not 3C method)
        if options.get("use_dispersion") and not is_3c_functional:
            # Check which version of D3
            if options.get("d3_version") == "D3BJ":
                f.write("DFTD3BJ\n")
            else:
                f.write("DFTD3\n")  # Original D3 with zero damping
        
        # DFT grid (if not 3C method - they have their own grids)
        if not is_3c_functional:
            grid = options.get("dft_grid", "XLGRID")
            if grid:
                f.write(f"{grid}\n")
        
        # Write ENDDFT for DFT block
        f.write("ENDDFT\n")
        
    else:  # HF method
        hf_type = options.get("hf_method", "RHF")
        
        # Map HF methods
        hf_map = {
            "RHF": "",  # RHF is default, no keyword needed
            "UHF": "UHF",
            "ROHF": "ROHF",
            "HF3C": "HF3C",
            "HFSOL3C": "HFSOL3C",
        }
        
        hf_keyword = hf_map.get(hf_type, "")
        if hf_keyword:
            f.write(f"{hf_keyword}\n")


def write_basis_block(f: TextIO, basis_config: Dict[str, Any], 
                     geometry_data: Dict[str, Any]) -> None:
    """
    Write basis set section to D12 file.
    
    Args:
        f: File handle to write to
        basis_config: Basis set configuration
        geometry_data: Geometry information including elements
    """
    basis_type = basis_config.get("basis_set_type", "INTERNAL")
    
    if basis_type == "EXTERNAL":
        # External basis set
        basis_path = basis_config.get("basis_set", "./full.basis.triplezeta/")
        elements = geometry_data.get("elements", [])
        
        # Import read_basis_file from d12_constants
        from d12_constants import read_basis_file
        
        # Write basis sets for each unique element
        unique_elements = sorted(set(elements))
        for element in unique_elements:
            basis_content = read_basis_file(basis_path, element)
            if basis_content:
                f.write(basis_content)
        
        # End basis input
        f.write("99 0\n")
        
    else:
        # Internal basis set
        basis_name = basis_config.get("basis_set", "POB-TZVP")
        f.write("BASISSET\n")
        f.write(f"{basis_name}\n")


def write_scf_block(f: TextIO, tolerances: Dict[str, Any], 
                   scf_settings: Dict[str, Any]) -> None:
    """
    Write SCF configuration block including tolerances.
    
    Args:
        f: File handle to write to
        tolerances: Tolerance settings (TOLINTEG, TOLDEE)
        scf_settings: SCF settings (method, maxcycle, fmixing)
    """
    # Tolerances
    tolinteg = tolerances.get("TOLINTEG", "7 7 7 7 14")
    toldee = tolerances.get("TOLDEE", 7)
    
    f.write(f"TOLINTEG\n{tolinteg}\n")
    f.write(f"TOLDEE\n{toldee}\n")
    
    # SCF settings
    scf_method = scf_settings.get("method", "DIIS")
    if scf_method != "DIIS":  # DIIS is default
        f.write(f"{scf_method}\n")
    
    maxcycle = scf_settings.get("maxcycle", 800)
    if maxcycle != 800:  # Only write if not default
        f.write(f"MAXCYCLE\n{maxcycle}\n")
    
    fmixing = scf_settings.get("fmixing", 30)
    if fmixing != 30:  # Only write if not default
        f.write(f"FMIXING\n{fmixing}\n")
    
    # Additional SCF options
    if scf_settings.get("levshift"):
        levshift = scf_settings["levshift"]
        if isinstance(levshift, tuple):
            f.write(f"LEVSHIFT\n{levshift[0]} {levshift[1]}\n")
        else:
            f.write(f"LEVSHIFT\n{levshift} 0\n")
    
    if scf_settings.get("smear"):
        f.write(f"SMEAR\n{scf_settings['smear']}\n")
    
    if scf_settings.get("nodiis"):
        f.write("NODIIS\n")


def write_optimization_block(f: TextIO, opt_settings: Dict[str, Any]) -> None:
    """
    Write optimization (OPTGEOM) block to D12 file.
    
    Args:
        f: File handle to write to
        opt_settings: Optimization settings dictionary
    """
    # Import from d12_calc_opt
    from d12_calc_opt import write_optimization_section
    
    # Use the existing function
    optimization_type = opt_settings.get('type', 'FULLOPTG')
    write_optimization_section(f, optimization_type, opt_settings)


def write_frequency_block(f: TextIO, freq_settings: Dict[str, Any],
                         geometry_ended: bool = False) -> None:
    """
    Write frequency calculation block to D12 file.
    
    Args:
        f: File handle to write to  
        freq_settings: Frequency calculation settings
        geometry_ended: Whether geometry section has been closed with END
    """
    # Import from d12_calc_freq
    from d12_calc_freq import write_frequency_calculation
    
    # Use the existing function
    write_frequency_calculation(f, freq_settings, geometry_ended)


def write_properties_block(f: TextIO, calc_type: str, options: Dict[str, Any]) -> None:
    """
    Write properties calculation block based on calculation type.
    
    Args:
        f: File handle to write to
        calc_type: Calculation type (SP, OPT, FREQ)
        options: Calculation options
    """
    if calc_type == "SP":
        # For single point, we might want properties like band structure
        if options.get("calculate_bands"):
            f.write("BAND\n")
            # Band calculation parameters would go here
        
        if options.get("calculate_dos"):
            f.write("DOSS\n") 
            # DOS parameters would go here
    
    # Properties like PPAN can be added for all calculation types
    if options.get("mulliken_analysis"):
        f.write("PPAN\n")


def write_print_options(f: TextIO, print_level: int = 1) -> None:
    """
    Write print level options to control output verbosity.
    
    Args:
        f: File handle to write to
        print_level: Print level (0=minimal, 1=standard, 2=detailed, 3=debug)
    """
    if print_level == 0:
        f.write("PRINTOUT\nPRINT\n0\n")
    elif print_level == 2:
        f.write("PRINTOUT\nPRINT\n2\n")
    elif print_level == 3:
        f.write("PRINTOUT\nPRINT\n3\n")
    # Level 1 is default, no need to write anything


def write_k_points(f: TextIO, k_points: Any, dimensionality: str = "CRYSTAL") -> None:
    """
    Write k-point sampling information.
    
    Args:
        f: File handle to write to
        k_points: K-point specification (can be int or list)
        dimensionality: System dimensionality
    """
    if dimensionality == "MOLECULE":
        # Molecules don't need k-points
        return
    
    if isinstance(k_points, int):
        # Simple Monkhorst-Pack grid
        if dimensionality == "CRYSTAL":
            f.write(f"SHRINK\n{k_points} {k_points}\n")
        elif dimensionality == "SLAB":
            f.write(f"SHRINK\n0 {k_points}\n{k_points} {k_points} 1\n")
        elif dimensionality == "POLYMER":
            f.write(f"SHRINK\n{k_points}\n")
    elif isinstance(k_points, list):
        # Custom k-point specification
        if dimensionality == "CRYSTAL" and len(k_points) == 2:
            f.write(f"SHRINK\n{k_points[0]} {k_points[1]}\n")
        elif dimensionality == "SLAB" and len(k_points) == 3:
            f.write(f"SHRINK\n0 {k_points[0]}\n{k_points[1]} {k_points[2]} 1\n")
        elif dimensionality == "POLYMER" and len(k_points) == 1:
            f.write(f"SHRINK\n{k_points[0]}\n")


def write_spin_settings(f: TextIO, spin_polarized: bool, 
                       spinlock: Optional[int] = None) -> None:
    """
    Write spin polarization settings.
    
    Args:
        f: File handle to write to
        spin_polarized: Whether to use spin polarization
        spinlock: Fixed spin state (optional)
    """
    if spin_polarized:
        f.write("UHF\n")
        
        if spinlock is not None:
            f.write(f"SPINLOCK\n{spinlock}\n")


def write_smearing_settings(f: TextIO, smearing_config: Dict[str, Any]) -> None:
    """
    Write Fermi surface smearing settings for metallic systems.
    
    Args:
        f: File handle to write to
        smearing_config: Smearing configuration dictionary
    """
    if smearing_config.get("enabled", False):
        width = smearing_config.get("width", 0.01)
        f.write(f"SMEAR\n{width}\n")


def write_minimal_raman_section(f: TextIO) -> None:
    """Write a minimal Raman calculation section with CPHF
    
    This produces exactly:
    FREQCALC
    INTENS
    INTRAMAN
    INTCPHF
    ENDCPHF
    ENDFREQ
    """
    print("FREQCALC", file=f)
    print("INTENS", file=f)
    print("INTRAMAN", file=f)
    print("INTCPHF", file=f)
    print("ENDCPHF", file=f)
    print("ENDFREQ", file=f)


def write_dft_section(f: TextIO, functional: str, use_dispersion: bool, 
                     dft_grid: str, is_spin_polarized: bool) -> None:
    """Write the DFT section of the D12 file
    
    Args:
        f: File handle
        functional: DFT functional name
        use_dispersion: Whether to use dispersion correction
        dft_grid: DFT grid size
        is_spin_polarized: Whether calculation is spin polarized
    """
    from d12_constants import D3_FUNCTIONALS
    
    print("DFT", file=f)
    
    if is_spin_polarized:
        print("SPIN", file=f)
    
    # Map functionals to their correct CRYSTAL keywords
    functional_keyword_map = {
        "PBESOL": "PBESOLXC",
        "SOGGA": "SOGGAXC",
        "VBH": "VBHLYP",
        "PWGGA": "PW91GGA",
        "WCGGA": "WCGGAPBE",
    }
    
    # Handle special functional keywords
    if functional in ["PBEH3C", "HSE3C", "B973C", "PBESOL03C", "HSESOL3C"]:
        # These are standalone keywords in CRYSTAL23
        print(f"{functional}", file=f)
        # For HSESOL3C, we need XLGRID (even if not specified or DEFAULT)
        if functional == "HSESOL3C":
            # Always use XLGRID for HSESOL3C
            print("XLGRID", file=f)
    elif functional == "mPW1PW91" and use_dispersion:
        print("PW1PW-D3", file=f)
        # Add DFT grid size only if not default and not None
        if dft_grid and dft_grid != "DEFAULT":
            print(dft_grid, file=f)
    elif functional in functional_keyword_map:
        # Use the mapped keyword for functionals that need special syntax
        mapped_functional = functional_keyword_map[functional]
        if use_dispersion and functional in D3_FUNCTIONALS:
            print(f"{mapped_functional}-D3", file=f)
        else:
            print(f"{mapped_functional}", file=f)
        
        # Add DFT grid size only if not default and not None
        if dft_grid and dft_grid != "DEFAULT":
            print(dft_grid, file=f)
    else:
        # Standard functional
        if use_dispersion and functional in D3_FUNCTIONALS:
            print(f"{functional}-D3", file=f)
        else:
            print(f"{functional}", file=f)
        
        # Add DFT grid size only if not default and not None
        if dft_grid and dft_grid != "DEFAULT":
            print(dft_grid, file=f)
    
    print("ENDDFT", file=f)


def write_basis_set_section(f: TextIO, basis_type: str, dimensionality: str, 
                           atom_list: List[Dict[str, Any]], basis_path: Optional[str] = None) -> None:
    """Write the basis set section for the D12 file
    
    Args:
        f: File handle
        basis_type: Type of basis set (e.g., "POB-TZVP", "STO-3G", etc.)
        dimensionality: System dimensionality (CRYSTAL, SLAB, POLYMER, MOLECULE)
        atom_list: List of atoms with atomic numbers
        basis_path: Optional path to basis set files
    """
    # Handle internal basis sets
    if basis_type in ["STO-3G", "STO-6G", "3-21G", "6-21G", "6-31G"]:
        print(f"BASISSET", file=f)
        print(f"{basis_type}", file=f)
    else:
        # External basis sets - need to write atom-by-atom
        unique_atoms = {}
        for atom in atom_list:
            atomic_num = atom['atomic_number']
            if atomic_num not in unique_atoms:
                unique_atoms[atomic_num] = True
        
        # Write basis for each unique atom type
        for atomic_num in sorted(unique_atoms.keys()):
            # Here you would read the actual basis set from file
            # This is a placeholder - actual implementation would read from basis_path
            print(f"{atomic_num} {len(unique_atoms)}", file=f)
            # Basis set data would go here
    
    print("99 0", file=f)


def write_scf_section(f: TextIO, tolerances: Dict[str, Any], k_points: Any,
                     dimensionality: str, use_smearing: bool, smearing_width: float,
                     scf_method: str, scf_maxcycle: int, fmixing: int,
                     num_atoms: int, spacegroup: int = 1) -> None:
    """Write the SCF parameters section of the D12 file
    
    Args:
        f: File handle
        tolerances: Dictionary with TOLINTEG and TOLDEE
        k_points: K-point specification (string or tuple)
        dimensionality: System dimensionality
        use_smearing: Whether to use Fermi smearing
        smearing_width: Smearing width in Hartree
        scf_method: SCF method (DIIS, BROYDEN, etc.)
        scf_maxcycle: Maximum SCF cycles
        fmixing: FMIXING percentage
        num_atoms: Number of atoms in system
        spacegroup: Space group number
    """
    # Tolerance settings with proper fallback handling
    print("TOLINTEG", file=f)
    tolinteg_value = tolerances.get("TOLINTEG", "7 7 7 7 14")  # Default fallback
    if tolinteg_value is None:
        tolinteg_value = "7 7 7 7 14"
    print(tolinteg_value, file=f)
    
    print("TOLDEE", file=f)
    toldee_value = tolerances.get("TOLDEE", 7)  # Default fallback
    if toldee_value is None:
        toldee_value = 7
    print(toldee_value, file=f)

    # K-points
    if k_points and dimensionality != "MOLECULE":
        if isinstance(k_points, str):
            # Pre-formatted k-points string
            print("SHRINK", file=f)
            print("0 24", file=f)
            print(k_points, file=f)
        else:
            # Tuple of (ka, kb, kc)
            ka, kb, kc = k_points
            
            # Symmetry and dimensionality-aware SHRINK format selection
            # P1 (spacegroup 1): Always use two-line format (0 n_shrink)
            # Non-P1: Use one-line format if k-points are uniform
            
            if spacegroup == 1:
                # P1: Always use directional format
                n_shrink = max(ka, kb, kc) * 2
                print("SHRINK", file=f)
                print(f"0 {n_shrink}", file=f)
                
                if dimensionality == "CRYSTAL":
                    print(f"{ka} {kb} {kc}", file=f)
                elif dimensionality == "SLAB":
                    print(f"{ka} {kb} 1", file=f)
                elif dimensionality == "POLYMER":
                    print(f"{ka} 1 1", file=f)
            else:
                # Non-P1: Check if we can use simplified format
                if dimensionality == "CRYSTAL" and ka == kb == kc:
                    # Use simplified format for uniform k-points
                    n_shrink = ka * 2
                    print("SHRINK", file=f)
                    print(f"{ka} {n_shrink}", file=f)
                else:
                    # Non-uniform k-points or lower dimensionality
                    # For non-P1 with non-uniform k-points, make them uniform
                    if dimensionality == "CRYSTAL" and not (ka == kb == kc):
                        k_max = max(ka, kb, kc)
                        print(f"Note: Converting non-uniform k-points ({ka},{kb},{kc}) to uniform ({k_max},{k_max},{k_max}) for space group {spacegroup}")
                        ka = kb = kc = k_max
                        n_shrink = ka * 2
                        print("SHRINK", file=f)
                        print(f"{ka} {n_shrink}", file=f)
                    else:
                        # Lower dimensionality - use directional format
                        n_shrink = max(ka, kb, kc) * 2
                        print("SHRINK", file=f)
                        print(f"0 {n_shrink}", file=f)
                        
                        if dimensionality == "SLAB":
                            print(f"{ka} {kb} 1", file=f)
                        elif dimensionality == "POLYMER":
                            print(f"{ka} 1 1", file=f)

    # Fermi smearing
    if use_smearing:
        print("SMEAR", file=f)
        print(f"{smearing_width:.6f}", file=f)

    # SCF settings
    print("SCFDIR", file=f)

    # Add BIPOSIZE and EXCHSIZE for large systems
    if num_atoms > 5:
        print("BIPOSIZE", file=f)
        print("110000000", file=f)
        print("EXCHSIZE", file=f)
        print("110000000", file=f)

    # SCF convergence
    print("MAXCYCLE", file=f)
    print(scf_maxcycle, file=f)

    print("FMIXING", file=f)
    print(fmixing, file=f)

    print(scf_method, file=f)

    if scf_method == "DIIS":
        print("HISTDIIS", file=f)
        print("100", file=f)

    # Print options
    print("PPAN", file=f)  # Print Mulliken population analysis

    # End of input
    print("END", file=f)