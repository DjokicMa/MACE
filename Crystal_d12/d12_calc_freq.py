#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frequency Calculation Configuration Module for CRYSTAL23
--------------------------------------------------------
This module handles all frequency calculation configurations including:
- FREQCALC (harmonic frequencies)
- ANHARM (anharmonic X-H stretching)
- ANHAPES (anharmonic potential energy surface)
- VSCF/VCI (vibrational SCF/CI)

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

This is part of the refactored D12 creation system where calculation-specific
logic is separated into dedicated modules for better maintainability.
"""

from typing import Dict, Any, Tuple, Optional, List
import sys
from pathlib import Path

# Add Crystal_d3 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d3"))
try:
    from d3_kpoints import (get_band_path_from_symmetry, get_kpoint_coordinates_from_labels,
                           extract_and_process_shrink, scale_kpoint_segments, get_seekpath_labels,
                           get_seekpath_full_kpath, get_literature_kpath_vectors)
except ImportError:
    print("Warning: Could not import d3_kpoints module - some k-path features may be unavailable")
    get_band_path_from_symmetry = None
    get_kpoint_coordinates_from_labels = None
    extract_and_process_shrink = None
    scale_kpoint_segments = None
    get_seekpath_labels = None
    get_seekpath_full_kpath = None
    get_literature_kpath_vectors = None

# Default frequency settings
DEFAULT_FREQ_SETTINGS = {
    "NUMDERIV": 2,
    "TOLINTEG": "9 9 9 11 38",
    "TOLDEE": 11,
}

# Frequency calculation templates
FREQ_TEMPLATES = {
    "basic": {
        "numderiv": 2,
        "mode": "GAMMA",
        "intensities": False,
        "raman": False,
    },
    "ir_spectrum": {
        "numderiv": 2,
        "mode": "GAMMA",
        "intensities": True,
        "ir_method": "CPHF",
        "irspec": True,
        "spec_range": [0, 4000],
        "resolution": 16,
        "lorentz_width": 8,
    },
    "raman_spectrum": {
        "numderiv": 2,
        "mode": "GAMMA",
        "intensities": True,
        "ir_method": "CPHF",
        "raman": True,
        "cphf_max_iter": 30,
        "cphf_tolerance": 6,
        "ramspec": True,
        "spec_range": [0, 4000],
        "resolution": 16,
        "lorentz_width": 8,
        "laser_wavelength": 532,
        "temperature": 298.15,
    },
    "ir_raman": {
        "numderiv": 2,
        "mode": "GAMMA",
        "intensities": True,
        "ir_method": "CPHF",
        "raman": True,
        "cphf_max_iter": 30,
        "cphf_tolerance": 6,
        "irspec": True,
        "ramspec": True,
        "spec_range": [0, 4000],
        "resolution": 16,
        "lorentz_width": 8,
        "laser_wavelength": 532,
        "temperature": 298.15,
    },
    "thermodynamics": {
        "numderiv": 2,
        "mode": "GAMMA",
        "intensities": False,
        "thermo": True,
        "temprange": (20, 0, 400),
    },
    "phonon_bands": {
        "numderiv": 2,
        "mode": "DISPERSION",
        "dispersion": True,
        "scelphono": [2, 2, 2],
        "bands": {
            "shrink": 16,
            "npoints": 100,
            "path": "AUTO",
            "auto_path": True,
            "path_method": "labels",
            "format": "labels",
        },
    },
    "phonon_dos": {
        "numderiv": 2,
        "mode": "DISPERSION",
        "dispersion": True,
        "scelphono": [2, 2, 2],
        "pdos": {
            "max_freq": 2000,
            "nbins": 200,
            "projected": True,
        },
    },
}


def get_advanced_frequency_settings():
    """Get advanced frequency calculation settings from user"""
    from d12_constants import yes_no_prompt, get_user_input
    
    freq_settings = {}
    
    print("\n=== FREQUENCY CALCULATION SETTINGS ===")
    
    # First, ask if they want to use a template
    print("\nFrequency calculation templates:")
    print("1: Basic frequencies only")
    print("   - Gamma point frequencies, no intensities")
    print("   - Use for: ZPE, thermal corrections, stability check")
    print("   - Time: ~1-3x optimization")
    print("2: IR spectrum")
    print("   - IR intensities + broadened spectrum")
    print("   - Use for: Molecular IR spectroscopy")
    print("   - Time: +20-50% over basic (method dependent)")
    print("3: Raman spectrum")
    print("   - Raman activities + broadened spectrum (CPHF)")
    print("   - Use for: Raman spectroscopy")
    print("   - Time: +100-200% over basic")
    print("4: IR + Raman spectra")
    print("   - Both IR and Raman with CPHF")
    print("   - Use for: Complete vibrational spectroscopy")
    print("   - Time: +100-200% over basic")
    print("5: Thermodynamic properties")
    print("   - Frequencies + thermal analysis at multiple T")
    print("   - Use for: Gibbs energy, entropy, heat capacity")
    print("   - Time: Similar to basic frequencies")
    print("6: Phonon band structure")
    print("   - Full phonon dispersion with supercell")
    print("   - Use for: Solid-state phonon properties")
    print("   - Time: ~4-20x optimization (supercell dependent)")
    print("7: Phonon density of states")
    print("   - Phonon DOS from supercell calculation")
    print("   - Use for: Thermal properties, phonon analysis")
    print("   - Time: ~4-20x optimization (supercell dependent)")
    print("8: Custom settings")
    print("   - Full control over all parameters")
    
    template_choice = input("Select template (1-8) [1]: ").strip() or "1"
    
    template_map = {
        "1": "basic",
        "2": "ir_spectrum",
        "3": "raman_spectrum",
        "4": "ir_raman",
        "5": "thermodynamics",
        "6": "phonon_bands",
        "7": "phonon_dos",
    }
    
    if template_choice in template_map:
        # Use template as base
        template_name = template_map[template_choice]
        freq_settings = FREQ_TEMPLATES[template_name].copy()
        
        print(f"\nUsing '{template_name}' template as base.")
        
        # Ask about NUMDERIV for all templates
        print("\nNumerical derivative method (NUMDERIV):")
        print("  1: One displacement per atom (faster)")
        print("     Forward difference: (g(x+t)-g(x))/t where t=0.001 Å")
        print("  2: Two displacements per atom (more accurate)")
        print("     Central difference: (g(x+t)-g(x-t))/2t where t=0.001 Å")
        numderiv_choice = input("\nSelect method (1-2) [2]: ").strip() or "2"
        freq_settings["numderiv"] = int(numderiv_choice)
        
        # Allow customization based on template
        if template_choice == "1":  # Basic template
            # Ask about IR intensities for basic template
            calc_ir = yes_no_prompt("\nCalculate IR intensities?", "no")
            if calc_ir:
                freq_settings["intensities"] = True
                # Ask about IR method
                print("\nIR intensity calculation method:")
                print("1: Berry phase (INTPOL - default)")
                print("   - Best for: Periodic solids, semiconductors, insulators")
                print("   - Works well: Covalent materials, MOFs, zeolites, 2D materials")
                print("   - Limitations: Requires insulating state")
                print("   - Speed: Fast (+10-20% over base frequency)")
                print("   - Accuracy depends on k-point density")
                print("2: Wannier functions (INTLOC)")
                print("   - Best for: Molecular crystals, ionic solids")
                print("   - Works well: Systems with localized bonds/charges")
                print("   - Limitations: Requires insulating state, higher memory")
                print("   - Speed: Moderate (+20-30% over base frequency)")
                print("   - Can relocalize at each displaced geometry")
                print("3: CPHF/CPKS (INTCPHF - most accurate)")
                print("   - Best for: Any material (metals, semiconductors, insulators)")
                print("   - Works well: Small unit cells, high accuracy needed")
                print("   - Benefits: Analytical Born charges, enables Raman")
                print("   - Speed: Slowest (+50-100% over base frequency)")
                print("   - Memory: ~2x base requirement")
                print("\nNote: CPHF (3) is the default due to its broad applicability")
                print("      and highest accuracy for all material types")
                ir_method_choice = input("\nSelect method (1-3) [3]: ").strip() or "3"
                ir_methods = {"1": "BERRY", "2": "WANNIER", "3": "CPHF"}
                freq_settings["ir_method"] = ir_methods[ir_method_choice]
        elif template_choice in ["2", "3", "4"]:
            # Spectral templates - clarify behavior and ask about spectrum plots
            print("\nSpectrum generation options:")
            print("Note: When using IR/Raman templates:")
            print("  - Intensities/activities are ALWAYS calculated")
            print("  - Spectrum plots are OPTIONAL")
            print("  - Spectrum generation adds <1% computational time")
            
            # Ask about minimal mode
            if template_choice in ["2", "4"]:  # IR or IR+Raman
                # For spectral templates, IR method is already needed
                if "ir_method" not in freq_settings:
                    # Ask about IR method
                    print("\nIR intensity calculation method:")
                    print("1: Berry phase (INTPOL - default)")
                    print("   - Best for: Periodic solids, semiconductors, insulators")
                    print("   - Works well: Covalent materials, MOFs, zeolites, 2D materials")
                    print("   - Limitations: Requires insulating state")
                    print("   - Speed: Fast (+10-20% over base frequency)")
                    print("   - Accuracy depends on k-point density")
                    print("2: Wannier functions (INTLOC)")
                    print("   - Best for: Molecular crystals, ionic solids")
                    print("   - Works well: Systems with localized bonds/charges")
                    print("   - Limitations: Requires insulating state, higher memory")
                    print("   - Speed: Moderate (+20-30% over base frequency)")
                    print("   - Can relocalize at each displaced geometry")
                    print("3: CPHF/CPKS (INTCPHF - most accurate)")
                    print("   - Best for: Any material (metals, semiconductors, insulators)")
                    print("   - Works well: Small unit cells, high accuracy needed")
                    print("   - Benefits: Analytical Born charges, enables Raman")
                    print("   - Speed: Slowest (+50-100% over base frequency)")
                    print("   - Memory: ~2x base requirement")
                    print("\nNote: CPHF (3) is the default due to its broad applicability")
                    print("      and highest accuracy for all material types")
                    ir_method_choice = input("\nSelect method (1-3) [3]: ").strip() or "3"
                    ir_methods = {"1": "BERRY", "2": "WANNIER", "3": "CPHF"}
                    freq_settings["ir_method"] = ir_methods[ir_method_choice]
                
                generate_ir_plot = yes_no_prompt("\nGenerate IR spectrum plot?", "yes")
                if not generate_ir_plot:
                    freq_settings["minimal_ir"] = True
                    freq_settings["irspec"] = False
                    
            if template_choice in ["3", "4"]:  # Raman or IR+Raman
                generate_raman_plot = yes_no_prompt("\nGenerate Raman spectrum plot?", "yes")
                if not generate_raman_plot:
                    freq_settings["minimal_raman"] = True
                    freq_settings["ramspec"] = False
            
            # Ask about spectral range only if not minimal
            if not (freq_settings.get("minimal_ir", False) and freq_settings.get("minimal_raman", False)):
                print("\nSpectral range settings:")
                custom_range = yes_no_prompt("Customize spectral range?", "no")
                if custom_range:
                    min_freq = float(input("Minimum frequency (cm⁻¹) [0]: ") or 0)
                    max_freq = float(input("Maximum frequency (cm⁻¹) [4000]: ") or 4000)
                    freq_settings["spec_range"] = [min_freq, max_freq]
                
        elif template_choice == "5":
            # Thermodynamics - ask about temperature range
            print("\nThermodynamic settings:")
            print("Temperature range for thermal analysis:")
            print("  Default: 20 points from 0 K to 400 K")
            custom_temp = yes_no_prompt("\nCustomize temperature range?", "no")
            if custom_temp:
                n_temps = int(input("Number of temperature points [20]: ") or 20)
                t_min = float(input("Minimum temperature (K) [0]: ") or 0)
                t_max = float(input("Maximum temperature (K) [400]: ") or 400)
                freq_settings["temprange"] = (n_temps, t_min, t_max)
            else:
                print("Using default: 20 points from 0 K to 400 K")
        
        elif template_choice == "6":
            # Phonon bands - ask about path configuration
            print("\nPhonon band structure settings:")
            
            # Supercell configuration
            print("\nSupercell size (for phonon calculations):")
            print("  - Larger supercells give more accurate dispersion")
            print("  - Cost scales as (N₁×N₂×N₃)³")
            custom_supercell = yes_no_prompt("Customize supercell size?", "no")
            if custom_supercell:
                n1 = int(input("Supercell N₁ [2]: ") or 2)
                n2 = int(input("Supercell N₂ [2]: ") or 2)
                n3 = int(input("Supercell N₃ [2]: ") or 2)
                freq_settings["scelphono"] = [n1, n2, n3]
            
            # Band path configuration - Enhanced like electronic bands
            print("\nBand path definition:")
            print("1: Automatic - Use standard path based on crystal symmetry")
            print("2: Template selection - Choose from common band paths")
            print("3: Custom labels - Specify path using labels (G, X, M, etc.)")
            print("4: Fractional coordinates - Specify path using k-point vectors")
            path_choice = input("Select method (1-4) [1]: ").strip() or "1"
            
            if path_choice == "1":
                # Automatic path with format options
                freq_settings["bands"]["auto_path"] = True
                
                print("\nAutomatic path format:")
                print("1: High-symmetry labels (CRYSTAL-compatible subset)")
                print("2: K-point vectors (fractional coordinates)")
                print("3: Literature path with vectors (comprehensive)")
                print("4: SeeK-path full paths (extended Bravais lattice notation)")
                format_choice = input("Select format (1-4) [1]: ").strip() or "1"
                
                if format_choice == "1":
                    freq_settings["bands"]["path_method"] = "labels"
                    freq_settings["bands"]["shrink"] = 0  # Use labels mode
                    freq_settings["bands"]["format"] = "labels"
                    print("\n✓ Will use automatic path with high-symmetry labels (G, X, M, etc.)")
                elif format_choice == "2":
                    freq_settings["bands"]["path_method"] = "coordinates"
                    freq_settings["bands"]["path"] = "auto"  # Trigger auto-detection
                    freq_settings["bands"]["format"] = "vectors"
                    print("\n✓ Will use automatic path with k-point vectors")
                elif format_choice == "3":
                    freq_settings["bands"]["path_method"] = "coordinates"
                    freq_settings["bands"]["path"] = "auto"  # Trigger auto-detection
                    freq_settings["bands"]["literature_path"] = True
                    freq_settings["bands"]["format"] = "literature"
                    print("\n✓ Will use comprehensive literature k-path (Setyawan & Curtarolo 2010)")
                else:
                    freq_settings["bands"]["path_method"] = "coordinates"
                    freq_settings["bands"]["path"] = "auto"  # Trigger auto-detection
                    freq_settings["bands"]["seekpath_full"] = True
                    freq_settings["bands"]["format"] = "seekpath"
                    print("\n✓ Will use SeeK-path full k-path with extended Bravais lattice notation")
                    
            elif path_choice == "2":
                # Template selection
                print("\nSelect crystal system:")
                print("1: Cubic")
                print("2: Hexagonal")
                print("3: Tetragonal")
                print("4: Orthorhombic")
                print("5: Monoclinic")
                print("6: Triclinic")
                
                system_choice = input("Select system (1-6) [1]: ").strip() or "1"
                
                if system_choice == "1":
                    print("\nCubic templates:")
                    print("1: Simple cubic (sc)")
                    print("2: Face-centered cubic (fcc)")
                    print("3: Body-centered cubic (bcc)")
                    template_idx = input("Select template (1-3) [1]: ").strip() or "1"
                    template_map = {"1": "cubic_simple", "2": "cubic_fc", "3": "cubic_bc"}
                    template_name = template_map.get(template_idx, "cubic_simple")
                elif system_choice == "2":
                    template_name = "hexagonal"
                elif system_choice == "3":
                    print("\nTetragonal templates:")
                    print("1: Simple tetragonal")
                    print("2: Body-centered tetragonal")
                    template_idx = input("Select template (1-2) [1]: ").strip() or "1"
                    template_map = {"1": "tetragonal_simple", "2": "tetragonal_bc"}
                    template_name = template_map.get(template_idx, "tetragonal_simple")
                elif system_choice == "4":
                    print("\nOrthorhombic templates:")
                    print("1: Simple orthorhombic")
                    print("2: Base-centered orthorhombic (ab)")
                    print("3: Base-centered orthorhombic (bc)")
                    print("4: Face-centered orthorhombic")
                    template_idx = input("Select template (1-4) [1]: ").strip() or "1"
                    template_map = {"1": "orthorhombic_simple", "2": "orthorhombic_ab", 
                                  "3": "orthorhombic_bc", "4": "orthorhombic_fc"}
                    template_name = template_map.get(template_idx, "orthorhombic_simple")
                elif system_choice == "5":
                    print("\nMonoclinic templates:")
                    print("1: Simple monoclinic")
                    print("2: Base-centered monoclinic")
                    template_idx = input("Select template (1-2) [1]: ").strip() or "1"
                    template_map = {"1": "monoclinic_simple", "2": "monoclinic_ac"}
                    template_name = template_map.get(template_idx, "monoclinic_simple")
                else:
                    template_name = "triclinic"
                
                # Import BAND_PATHS from d3_kpoints
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d3"))
                try:
                    from d3_kpoints import BAND_PATHS
                    if template_name in BAND_PATHS:
                        path_labels = BAND_PATHS[template_name]
                        # Convert to segment format
                        path = []
                        for i in range(len(path_labels) - 1):
                            path.append(f"{path_labels[i]} {path_labels[i+1]}")
                        freq_settings["bands"]["path"] = path
                        freq_settings["bands"]["shrink"] = 0  # Labels mode
                        freq_settings["bands"]["template"] = template_name
                        freq_settings["bands"]["kpath_source"] = "template"
                        print(f"\n✓ Using {template_name} template path")
                    else:
                        print(f"\nTemplate {template_name} not found, using default")
                        freq_settings["bands"]["path"] = ["G X", "X M", "M G"]
                        freq_settings["bands"]["shrink"] = 0
                except ImportError:
                    print("\nCould not import templates, using default path")
                    freq_settings["bands"]["path"] = ["G X", "X M", "M G"]
                    freq_settings["bands"]["shrink"] = 0
                    
            elif path_choice == "3":
                # Custom labels
                freq_settings["bands"]["shrink"] = 0  # Use labels mode
                freq_settings["bands"]["kpath_source"] = "manual"
                
                # Display CRYSTAL-supported labels for the crystal system
                print("\nCRYSTAL-supported k-point labels:")
                print("=" * 60)
                print("  G: Γ (0, 0, 0) - Available in all systems")
                print("  Common labels by crystal system:")
                print("  - Cubic: M, R, X (P); X, L, W (F); H, P, N (I)")
                print("  - Hexagonal: M, K, A, L, H")
                print("  - Tetragonal: M, R, A, X, Z (P); M, P, X (I)")
                print("  - Orthorhombic: S, T, U, R, X, Y, Z")
                print("  - Monoclinic: A, B, C, D, E, Y, Z")
                print("  - Triclinic: (Use default labels)")
                print("=" * 60)
                
                print("\nEnter band path as space-separated labels (e.g., G X M G):")
                print("Note: Use G for Gamma point. Labels are case-sensitive.")
                print("      Use | for discontinuous paths (e.g., G X M G | R X)")
                print("      This creates separate segments without connecting G to R.")
                path_str = input("Path: ").strip()
                labels = path_str.split() if path_str else ["G", "X", "M", "G"]
                
                # Convert to segments - handle discontinuous paths correctly
                path = []
                current_segment = []
                
                for label in labels:
                    if label == "|":
                        # End current continuous segment
                        if len(current_segment) >= 2:
                            # Create segments from consecutive points
                            for i in range(len(current_segment) - 1):
                                path.append(f"{current_segment[i]} {current_segment[i+1]}")
                        current_segment = []
                    else:
                        current_segment.append(label)
                
                # Handle final segment
                if len(current_segment) >= 2:
                    for i in range(len(current_segment) - 1):
                        path.append(f"{current_segment[i]} {current_segment[i+1]}")
                
                freq_settings["bands"]["path"] = path
                # Create display string with | markers preserved
                display_path = []
                for label in labels:
                    if label == "|":
                        display_path.append("|")
                    else:
                        display_path.append(label)
                print(f"\n✓ Custom path set: {' '.join(display_path)}")
                
            else:
                # Manual k-point entry with mixed mode option
                freq_settings["bands"]["kpath_source"] = "fractional"
                use_mixed = yes_no_prompt("\nUse mixed label/coordinate segments?", "no")
                
                if use_mixed:
                    # Mixed path - each segment can be labels or coordinates
                    n_segments = int(input("Number of path segments: "))
                    path = []
                    freq_settings["bands"]["shrink"] = 16  # Default, will be adjusted
                    
                    for i in range(n_segments):
                        print(f"\nSegment {i+1}:")
                        print("1: Use high-symmetry labels (e.g., G X)")
                        print("2: Use fractional coordinates")
                        seg_type = input("Select type (1-2) [1]: ").strip() or "1"
                        
                        if seg_type == "1":
                            segment = input("Enter label pair (e.g., 'G X'): ").strip().upper()
                            if len(segment.split()) == 2:
                                path.append(segment)
                                freq_settings["bands"]["shrink"] = 0  # Force labels mode
                            else:
                                print("Invalid format, using 'G X'")
                                path.append("G X")
                        else:
                            print("Enter fractional coordinates:")
                            start_coords = [float(x) for x in input("Start point (3 values): ").split()]
                            end_coords = [float(x) for x in input("End point (3 values): ").split()]
                            path.append([start_coords, end_coords])
                    
                    freq_settings["bands"]["path"] = path
                    freq_settings["bands"]["mixed_path"] = True
                else:
                    # All coordinates
                    n_segments = int(input("Number of path segments: "))
                    path = []
                    print("\nEnter fractional coordinates for each segment:")
                    for i in range(n_segments):
                        print(f"\nSegment {i+1}:")
                        start_coords = [float(x) for x in input("Start point (3 values): ").split()]
                        end_coords = [float(x) for x in input("End point (3 values): ").split()]
                        path.append([start_coords, end_coords])
                    freq_settings["bands"]["path"] = path
            
            # K-point density (for non-label modes)
            if freq_settings["bands"].get("shrink", 16) != 0:
                custom_kpoints = yes_no_prompt("\nCustomize k-point density?", "no")
                if custom_kpoints:
                    shrink = int(input("Shrink factor for k-points [16]: ") or 16)
                    npoints = int(input("Points per band segment [100]: ") or 100)
                    freq_settings["bands"]["shrink"] = shrink
                    freq_settings["bands"]["npoints"] = npoints
        
        elif template_choice == "7":
            # Phonon DOS - first ask about DOS type
            print("\nPhonon DOS settings:")
            
            # DOS type selection
            print("\nSelect DOS calculation type:")
            print("1: PDOS - Phonon density of states")
            print("   Standard phonon DOS calculation")
            print("   Can be atom-projected for partial DOS analysis")
            print("\n2: INS - Inelastic neutron scattering spectrum")
            print("   Neutron-weighted phonon DOS")
            print("   For comparison with INS experiments")
            
            dos_type = input("\nSelect type (1-2) [1]: ").strip() or "1"
            
            if dos_type == "2":
                # INS specific settings
                print("\nINS neutron weighting:")
                print("0: Coherent cross-section only")
                print("1: Incoherent cross-section only")
                print("2: Coherent + Incoherent cross-section")
                neutron_type = int(input("Select neutron weighting (0-2) [2]: ") or 2)
                
                # Supercell configuration
                print("\nSupercell size (for phonon calculations):")
                print("  - Larger supercells give more accurate spectra")
                print("  - Cost scales as (N₁×N₂×N₃)³")
                custom_supercell = yes_no_prompt("Customize supercell size?", "no")
                if custom_supercell:
                    n1 = int(input("Supercell N₁ [2]: ") or 2)
                    n2 = int(input("Supercell N₂ [2]: ") or 2)
                    n3 = int(input("Supercell N₃ [2]: ") or 2)
                    freq_settings["scelphono"] = [n1, n2, n3]
                
                # INS parameters
                print("\nINS spectrum parameters:")
                max_freq = float(input("Maximum frequency for INS (cm⁻¹) [3000]: ") or 3000)
                n_bins = int(input("Number of spectrum bins [300]: ") or 300)
                
                freq_settings["ins"] = {
                    "max_freq": max_freq,
                    "nbins": n_bins,
                    "neutron_type": neutron_type
                }
            else:
                # PDOS settings
                # Supercell configuration
                print("\nSupercell size (for phonon calculations):")
                print("  - Larger supercells give more accurate DOS")
                print("  - Cost scales as (N₁×N₂×N₃)³")
                custom_supercell = yes_no_prompt("Customize supercell size?", "no")
                if custom_supercell:
                    n1 = int(input("Supercell N₁ [2]: ") or 2)
                    n2 = int(input("Supercell N₂ [2]: ") or 2)
                    n3 = int(input("Supercell N₃ [2]: ") or 2)
                    freq_settings["scelphono"] = [n1, n2, n3]
                
                # DOS parameters
                print("\nPhonon DOS parameters:")
                max_freq = float(input("Maximum frequency for DOS (cm⁻¹) [2000]: ") or 2000)
                n_bins = int(input("Number of DOS bins [200]: ") or 200)
                projected = yes_no_prompt("Calculate atom-projected DOS?", "yes")
                
                freq_settings["pdos"] = {
                    "max_freq": max_freq,
                    "nbins": n_bins,
                    "projected": projected
                }
                
    else:
        # Custom settings
        print("\n=== CUSTOM FREQUENCY SETTINGS ===")
        
        # Mode selection
        print("\nFrequency calculation modes:")
        print("1: Gamma point only (default)")
        print("   - Molecular/cluster frequencies at q=0")
        print("   - Can calculate IR/Raman intensities")
        print("   - Time: ~1-3x optimization")
        print("2: Phonon dispersion")
        print("   - Full phonon band structure")
        print("   - Requires supercell calculation")
        print("   - Cannot calculate IR/Raman intensities")
        print("   - Time: ~4-20x optimization (supercell dependent)")
        
        mode_choice = input("Select mode (1-2) [1]: ").strip() or "1"
        
        if mode_choice == "1":
            freq_settings["mode"] = "GAMMA"
        else:
            freq_settings["mode"] = "DISPERSION"
            freq_settings["dispersion"] = True
            
            # Supercell settings
            print("\nSupercell for phonon calculation:")
            print("1: Automatic (2x2x2)")
            print("2: Custom expansion factors")
            print("3: Custom transformation matrix")
            supercell_choice = input("Select supercell method (1-3) [1]: ").strip() or "1"
            
            if supercell_choice == "1":
                freq_settings["scelphono"] = [2, 2, 2]
            elif supercell_choice == "2":
                nx = int(input("Expansion in x [2]: ") or 2)
                ny = int(input("Expansion in y [2]: ") or 2)
                nz = int(input("Expansion in z [2]: ") or 2)
                freq_settings["scelphono"] = [nx, ny, nz]
            else:
                print("Enter 3x3 transformation matrix (9 integers):")
                matrix = []
                for i in range(3):
                    row = input(f"Row {i+1} (3 integers): ").split()
                    matrix.extend([int(x) for x in row])
                freq_settings["scelphono"] = matrix
            
            # Fourier interpolation
            use_interphess = yes_no_prompt("\nUse Fourier interpolation (INTERPHESS)?", "yes")
            if use_interphess:
                print("\nINTERPHESS settings:")
                l1 = int(input("L1 expansion factor [2]: ") or 2)
                l2 = int(input("L2 expansion factor [2]: ") or 2)
                l3 = int(input("L3 expansion factor [2]: ") or 2)
                print_level = int(input("Print level (0-2) [0]: ") or 0)
                freq_settings["interphess"] = {
                    "expand": [l1, l2, l3],
                    "print": print_level
                }
            
            # Wang correction for polar materials
            is_polar = yes_no_prompt("\nIs this a polar material (needs Wang correction)?", "no")
            if is_polar:
                print("\nWang correction requires high-frequency dielectric tensor.")
                print("Enter 9 tensor components (xx xy xz yx yy yz zx zy zz):")
                tensor_str = input("Tensor values: ").strip()
                if tensor_str:
                    tensor = [float(x) for x in tensor_str.split()]
                    if len(tensor) == 9:
                        freq_settings["wang"] = tensor
                    else:
                        print("Warning: Expected 9 values, Wang correction not applied.")
            
            # Ask for band structure calculation
            calc_bands = yes_no_prompt("\nCalculate phonon band structure?", "yes")
            if calc_bands:
                # Ask about calculation type
                print("\nPhonon band structure type:")
                print("1: Full band structure along paths")
                print("2: High-symmetry points only (quick check)")
                band_type = input("Select type (1-2) [1]: ").strip() or "1"
                
                if band_type == "2":
                    # High-symmetry points only
                    freq_settings["high_symmetry_only"] = True
                    freq_settings["scelphono"] = [1, 1, 1]  # Minimal supercell
                    freq_settings["bands"] = {
                        "shrink": 1,
                        "npoints": 1,  # Only endpoints
                        "path": "AUTO"
                    }
                    print("\n✓ High-symmetry points mode selected")
                    print("  Will calculate at Γ, X, M, K, etc. based on crystal symmetry")
                else:
                    # Full band structure
                    freq_settings["bands"] = {
                        "shrink": int(input("Shrink factor for k-points [16]: ") or 16),
                        "npoints": int(input("Points per band segment [100]: ") or 100)
                    }
                    
                    # Band path
                    print("\nBand path definition:")
                    print("1: Automatic - Use standard path based on crystal symmetry")
                    print("2: Template selection - Choose from common band paths")
                    print("3: Custom labels - Specify path using labels (G, X, M, etc.)")
                    print("4: Fractional coordinates - Specify path using k-point vectors")
                    path_choice = input("Select path method (1-4) [1]: ").strip() or "1"
                    
                    if path_choice == "1":
                        # Automatic path
                        freq_settings["bands"]["auto_path"] = True
                        
                        print("\nAutomatic path format:")
                        print("1: High-symmetry labels (CRYSTAL-compatible subset)")
                        print("2: K-point vectors (fractional coordinates)")
                        print("3: Literature path with vectors (comprehensive)")
                        print("4: SeeK-path full paths (extended Bravais lattice notation)")
                        format_choice = input("Select format (1-4) [1]: ").strip() or "1"
                        
                        if format_choice == "1":
                            # Labels with CRYSTAL subset
                            freq_settings["bands"]["path_method"] = "labels"
                            freq_settings["bands"]["shrink"] = 0  # Use labels mode
                            freq_settings["bands"]["format"] = "labels"
                            print("\n✓ Will use automatic path with CRYSTAL-compatible labels")
                        elif format_choice == "2":
                            # K-point vectors
                            freq_settings["bands"]["path_method"] = "coordinates"
                            freq_settings["bands"]["path"] = "auto"
                            freq_settings["bands"]["format"] = "vectors"
                            print("\n✓ Will use automatic path with k-point vectors")
                        elif format_choice == "3":
                            # Literature path
                            freq_settings["bands"]["path_method"] = "coordinates"
                            freq_settings["bands"]["path"] = "auto"
                            freq_settings["bands"]["literature_path"] = True
                            freq_settings["bands"]["format"] = "literature"
                            print("\n✓ Will use comprehensive literature k-path")
                        else:
                            # SeeK-path full paths
                            freq_settings["bands"]["path_method"] = "coordinates"
                            freq_settings["bands"]["path"] = "auto"
                            freq_settings["bands"]["seekpath_full"] = True
                            freq_settings["bands"]["format"] = "seekpath"
                            print("\n✓ Will use SeeK-path full k-path with extended Bravais lattice notation")
                    
                    elif path_choice == "2":
                        # Template selection
                        print("\nSelect crystal system:")
                        print("1: Cubic")
                        print("2: Hexagonal")
                        print("3: Tetragonal")
                        print("4: Orthorhombic")
                        print("5: Monoclinic")
                        print("6: Triclinic")
                        
                        system_choice = input("Select system (1-6) [1]: ").strip() or "1"
                        
                        if system_choice == "1":
                            print("\nCubic templates:")
                            print("1: Simple cubic (sc)")
                            print("2: Face-centered cubic (fcc)")
                            print("3: Body-centered cubic (bcc)")
                            template_idx = input("Select template (1-3) [1]: ").strip() or "1"
                            template_map = {"1": "cubic_simple", "2": "cubic_fc", "3": "cubic_bc"}
                            template_name = template_map.get(template_idx, "cubic_simple")
                        elif system_choice == "2":
                            template_name = "hexagonal"
                        elif system_choice == "3":
                            print("\nTetragonal templates:")
                            print("1: Simple tetragonal")
                            print("2: Body-centered tetragonal")
                            template_idx = input("Select template (1-2) [1]: ").strip() or "1"
                            template_map = {"1": "tetragonal_simple", "2": "tetragonal_bc"}
                            template_name = template_map.get(template_idx, "tetragonal_simple")
                        elif system_choice == "4":
                            print("\nOrthorhombic templates:")
                            print("1: Simple orthorhombic")
                            print("2: Base-centered orthorhombic (ab)")
                            print("3: Base-centered orthorhombic (bc)")
                            print("4: Face-centered orthorhombic")
                            template_idx = input("Select template (1-4) [1]: ").strip() or "1"
                            template_map = {"1": "orthorhombic_simple", "2": "orthorhombic_ab", 
                                          "3": "orthorhombic_bc", "4": "orthorhombic_fc"}
                            template_name = template_map.get(template_idx, "orthorhombic_simple")
                        elif system_choice == "5":
                            print("\nMonoclinic templates:")
                            print("1: Simple monoclinic")
                            print("2: Base-centered monoclinic")
                            template_idx = input("Select template (1-2) [1]: ").strip() or "1"
                            template_map = {"1": "monoclinic_simple", "2": "monoclinic_ac"}
                            template_name = template_map.get(template_idx, "monoclinic_simple")
                        else:
                            template_name = "triclinic"
                        
                        # Import BAND_PATHS from d3_kpoints
                        import sys
                        from pathlib import Path
                        sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d3"))
                        try:
                            from d3_kpoints import BAND_PATHS
                            if template_name in BAND_PATHS:
                                path_labels = BAND_PATHS[template_name]
                                # Convert to segment format
                                path = []
                                for i in range(len(path_labels) - 1):
                                    path.append(f"{path_labels[i]} {path_labels[i+1]}")
                                freq_settings["bands"]["path"] = path
                                freq_settings["bands"]["shrink"] = 0  # Labels mode
                                freq_settings["bands"]["template"] = template_name
                                print(f"\n✓ Using {template_name} template path")
                            else:
                                print(f"\nTemplate {template_name} not found, using default")
                                freq_settings["bands"]["path"] = ["G X", "X M", "M G"]
                                freq_settings["bands"]["shrink"] = 0
                        except ImportError:
                            print("\nCould not import templates, using default path")
                            freq_settings["bands"]["path"] = ["G X", "X M", "M G"]
                            freq_settings["bands"]["shrink"] = 0
                    
                    elif path_choice == "3":
                        # Custom labels
                        freq_settings["bands"]["shrink"] = 0  # Use labels mode
                        print("\nEnter band path as space-separated labels (e.g., G X M G):")
                        print("Note: Use G for Gamma point")
                        path_str = input("Path: ").strip()
                        labels = path_str.split() if path_str else ["G", "X", "M", "G"]
                        
                        # Convert to segments
                        path = []
                        for i in range(len(labels) - 1):
                            path.append(f"{labels[i]} {labels[i+1]}")
                        
                        freq_settings["bands"]["path"] = path
                        print(f"\n✓ Custom path set: {' → '.join(labels)}")
                    
                    else:
                        # Manual k-point entry with mixed mode option
                        use_mixed = yes_no_prompt("\nUse mixed label/coordinate segments?", "no")
                        
                        if use_mixed:
                            # Mixed path - each segment can be labels or coordinates
                            n_segments = int(input("Number of path segments: "))
                            path = []
                            freq_settings["bands"]["shrink"] = 16  # Default, will be adjusted
                            
                            for i in range(n_segments):
                                print(f"\nSegment {i+1}:")
                                print("1: Use high-symmetry labels (e.g., G X)")
                                print("2: Use fractional coordinates")
                                seg_type = input("Select type (1-2) [1]: ").strip() or "1"
                                
                                if seg_type == "1":
                                    segment = input("Enter label pair (e.g., 'G X'): ").strip().upper()
                                    if len(segment.split()) == 2:
                                        path.append(segment)
                                        freq_settings["bands"]["shrink"] = 0  # Force labels mode
                                    else:
                                        print("Invalid format, using 'G X'")
                                        path.append("G X")
                                else:
                                    print("Enter fractional coordinates:")
                                    start_coords = [float(x) for x in input("Start point (3 values): ").split()]
                                    end_coords = [float(x) for x in input("End point (3 values): ").split()]
                                    path.append([start_coords, end_coords])
                            
                            freq_settings["bands"]["path"] = path
                            freq_settings["bands"]["mixed_path"] = True
                        else:
                            # All coordinates
                            n_segments = int(input("Number of path segments: "))
                            path = []
                            print("\nEnter fractional coordinates for each segment:")
                            for i in range(n_segments):
                                print(f"\nSegment {i+1}:")
                                start_coords = [float(x) for x in input("Start point (3 values): ").split()]
                                end_coords = [float(x) for x in input("End point (3 values): ").split()]
                                path.append([start_coords, end_coords])
                            freq_settings["bands"]["path"] = path
            
            # Ask for DOS calculation
            calc_dos = yes_no_prompt("\nCalculate phonon density of states?", "yes")
            if calc_dos:
                max_freq = float(input("Maximum frequency for DOS (cm⁻¹) [2000]: ") or 2000)
                n_bins = int(input("Number of DOS bins [200]: ") or 200)
                projected = yes_no_prompt("Calculate atom-projected DOS?", "yes")
                freq_settings["pdos"] = {
                    "max_freq": max_freq,
                    "nbins": n_bins,
                    "projected": projected
                }
            
            # Ask for INS calculation
            calc_ins = yes_no_prompt("\nCalculate INS (Inelastic Neutron Scattering) spectrum?", "no")
            if calc_ins:
                ins_max_freq = float(input("Maximum frequency for INS (cm⁻¹) [3000]: ") or 3000)
                ins_n_bins = int(input("Number of INS bins [300]: ") or 300)
                
                print("\nNeutron scattering type:")
                print("0: Coherent scattering only")
                print("1: Incoherent scattering only")
                print("2: Both coherent and incoherent")
                neutron_type = int(input("Select type (0-2) [2]: ") or 2)
                
                freq_settings["ins"] = {
                    "max_freq": ins_max_freq,
                    "nbins": ins_n_bins,
                    "neutron_type": neutron_type
                }
        
        # Numerical derivative method
        print("\nNumerical derivative method:")
        print("1: One displacement per atom (faster, less accurate)")
        print("   Uses forward difference: (g(x+t)-g(x))/t where t=0.001 Å")
        print("2: Two displacements per atom (default, recommended)")
        print("   Uses central difference: (g(x+t)-g(x-t))/2t where t=0.001 Å")
        
        numderiv = input("Select method (1-2) [2]: ").strip() or "2"
        freq_settings["numderiv"] = int(numderiv)
        
        # Skip IR/Raman section entirely for phonon dispersion calculations
        if freq_settings.get("mode") != "DISPERSION":
            # IR intensities (only ask if not already set by template)
            if "intensities" not in freq_settings:
                calc_ir = yes_no_prompt("\nCalculate IR intensities?", "no")
                if calc_ir:
                    freq_settings["intensities"] = True
            
            # Only ask about IR method if intensities are enabled and method not already set
            if freq_settings.get("intensities") and "ir_method" not in freq_settings:
                print("\nIR intensity calculation method:")
                print("1: Berry phase (INTPOL - default)")
                print("   - Best for: Periodic solids, semiconductors, insulators")
                print("   - Works well: Covalent materials, MOFs, zeolites, 2D materials")
                print("   - Limitations: Requires insulating state")
                print("   - Speed: Fast (+10-20% over base frequency)")
                print("   - Accuracy depends on k-point density")
                print("2: Wannier functions (INTLOC)")
                print("   - Best for: Molecular crystals, ionic solids")
                print("   - Works well: Systems with localized bonds/charges")
                print("   - Limitations: Requires insulating state, higher memory")
                print("   - Speed: Moderate (+20-30% over base frequency)")
                print("   - Can relocalize at each displaced geometry")
                print("3: CPHF/CPKS (INTCPHF - most accurate)")
                print("   - Best for: Any material (metals, semiconductors, insulators)")
                print("   - Works well: Small unit cells, high accuracy needed")
                print("   - Benefits: Analytical Born charges, enables Raman")
                print("   - Speed: Slowest (+50-100% over base frequency)")
                print("   - Memory: ~2x base requirement")
                
                print("\nNote: CPHF (3) is the default due to its broad applicability")
                print("      and highest accuracy for all material types")
                ir_method_choice = input("\nSelect method (1-3) [3]: ").strip() or "3"
                ir_methods = {"1": "BERRY", "2": "WANNIER", "3": "CPHF"}
                freq_settings["ir_method"] = ir_methods[ir_method_choice]
                
                if freq_settings["ir_method"] == "WANNIER":
                    # Wannier function specific options
                    print("\nWannier function options:")
                    relocal = yes_no_prompt("Relocalize at each displaced geometry (RELOCAL)?", "no")
                    if relocal:
                        freq_settings["relocalize_wannier"] = True
                    
                elif freq_settings["ir_method"] == "CPHF":
                    # CPHF specific options
                    print("\nCPHF calculation options:")
                    try:
                        max_iter_input = input("Maximum CPHF iterations [30]: ").strip()
                        max_iter = int(max_iter_input) if max_iter_input else 30
                    except ValueError:
                        print("Invalid input, using default value of 30")
                        max_iter = 30
                    freq_settings["cphf_max_iter"] = max_iter
                    
                    try:
                        tol_input = input("CPHF convergence tolerance (10^-x) [6]: ").strip()
                        tol = float(tol_input) if tol_input else 6
                    except ValueError:
                        print("Invalid input, using default value of 6")
                        tol = 6
                    freq_settings["cphf_tolerance"] = tol
            
            # Skip Raman prompt if already configured by template
            if "raman" not in freq_settings:
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
                print("\n=== SPECTRAL GENERATION OPTIONS ===")
                
                # Check if user wants minimal calculations
                if freq_settings.get("intensities") and not freq_settings.get("raman"):
                    print("\nIR spectrum plot generation:")
                    print("  Note: IR intensities are ALWAYS calculated when IR is selected")
                    print("  You're choosing whether to also generate a spectrum plot")
                    print("\n  Options:")
                    print("  - Minimal (no plot): Only intensities in .out file")
                    print("  - Full (with plot): Intensities + broadened spectrum (IRSPEC)")
                    print("\n  Computational cost:")
                    print("  - Spectrum generation adds <1% to total calculation time")
                    print("  - Can be generated later from .out file if needed")
                    generate_ir_plot = yes_no_prompt("Generate IR spectrum plot?", "yes")
                    if not generate_ir_plot:
                        freq_settings["minimal_ir"] = True
                        freq_settings["irspec"] = False
                
                if freq_settings.get("raman"):
                    print("\nRaman spectrum plot generation:")
                    print("  Note: Raman activities are ALWAYS calculated when Raman is selected")
                    print("  You're choosing whether to also generate a spectrum plot")
                    print("\n  Options:")
                    print("  - Minimal (no plot): Only activities in .out file")
                    print("  - Full (with plot): Activities + broadened spectrum (RAMSPEC)")
                    print("\n  Computational cost:")
                    print("  - Spectrum generation adds <1% to total calculation time")
                    print("  - Can be generated later from .out file if needed")
                    generate_raman_plot = yes_no_prompt("Generate Raman spectrum plot?", "yes")
                    if not generate_raman_plot:
                        freq_settings["minimal_raman"] = True
                        freq_settings["ramspec"] = False
                
                # If not fully minimal, ask about spectrum parameters
                if not (freq_settings.get("minimal_ir", True) and freq_settings.get("minimal_raman", True)):
                    if "spec_range" not in freq_settings:
                        print("\nSpectrum plotting settings:")
                        custom_range = yes_no_prompt("Customize spectral range?", "no")
                        if custom_range:
                            min_freq = float(input("Minimum frequency (cm⁻¹) [0]: ") or 0)
                            max_freq = float(input("Maximum frequency (cm⁻¹) [4000]: ") or 4000)
                            freq_settings["spec_range"] = [min_freq, max_freq]
                    
                    if "resolution" not in freq_settings:
                        resolution = int(input("Resolution for spectrum (cm⁻¹) [16]: ") or 16)
                        freq_settings["resolution"] = resolution
                    
                    # Broadening
                    print("\nLine broadening type:")
                    print("1: Lorentzian (default)")
                    print("2: Gaussian")
                    broadening = input("Select broadening (1-2) [1]: ").strip() or "1"
                    freq_settings["broadening"] = "LORENTZ" if broadening == "1" else "GAUSS"
                    
                    if "lorentz_width" not in freq_settings and broadening == "1":
                        width = float(input("Lorentzian FWHM (cm⁻¹) [8]: ") or 8)
                        freq_settings["lorentz_width"] = width
                    elif "gauss_width" not in freq_settings and broadening == "2":
                        width = float(input("Gaussian FWHM (cm⁻¹) [10]: ") or 10)
                        freq_settings["gauss_width"] = width
                    
                    # Raman-specific settings
                    if freq_settings.get("raman") and not freq_settings.get("minimal_raman"):
                        if "laser_wavelength" not in freq_settings:
                            print("\nRaman excitation wavelength (nm):")
                            print("Common values: 488, 514.5, 532, 633, 785, 1064")
                            wavelength = float(input("Wavelength [532]: ") or 532)
                            freq_settings["laser_wavelength"] = wavelength
                        
                        if "temperature" not in freq_settings:
                            temp = float(input("Temperature for Raman (K) [298.15]: ") or 298.15)
                            freq_settings["temperature"] = temp
                        
                        # RAMANEXP option
                        use_ramanexp = yes_no_prompt("\nUse RAMANEXP for experimental conditions?", "no")
                        if use_ramanexp:
                            print("\nRAMANEXP takes into account experimental conditions")
                            print("(temperature, incoming laser) in the calculation of Raman intensities")
                            ramanexp_laser = float(input(f"Laser wavelength (nm) [{freq_settings.get('laser_wavelength', 532)}]: ") or freq_settings.get('laser_wavelength', 532))
                            ramanexp_temp = float(input(f"Temperature (K) [{freq_settings.get('temperature', 298.15)}]: ") or freq_settings.get('temperature', 298.15))
                            freq_settings["ramanexp"] = [ramanexp_temp, ramanexp_laser]
                        
                        # Raman units
                        print("\nRaman intensity units:")
                        print("1: Arbitrary units (default)")
                        print("2: Absolute differential cross section")
                        raman_units = input("Select units (1-2) [1]: ").strip() or "1"
                        if raman_units == "2":
                            freq_settings["raman_absolute"] = True
                
                # Spectral output format
                generate_spec = not (freq_settings.get("minimal_ir", True) and freq_settings.get("minimal_raman", True))
                if generate_spec:
                    print("\nSpectrum output formats:")
                    print("1: DAT file only (2-column ASCII)")
                    print("2: DAT + CSV")
                    print("3: DAT + CSV + JCAMP-DX")
                    output_format = input("Select output format (1-3) [1]: ").strip() or "1"
                    
                    if output_format in ["2", "3"]:
                        freq_settings["output_csv"] = True
                    if output_format == "3":
                        freq_settings["output_jcampdx"] = True
        
        # Thermodynamic properties
        if freq_settings.get("mode") == "GAMMA":
            calc_thermo = yes_no_prompt("\nCalculate thermodynamic properties?", "no")
            if calc_thermo:
                freq_settings["thermo"] = True
                print("\nThermodynamic calculation settings:")
                n_temps = int(input("Number of temperature points [20]: ") or 20)
                t_min = float(input("Minimum temperature (K) [0]: ") or 0)
                t_max = float(input("Maximum temperature (K) [400]: ") or 400)
                freq_settings["temprange"] = (n_temps, t_min, t_max)
        
        # Atomic projection
        print("\n=== VIBRATIONAL ANALYSIS OPTIONS ===")
        atom_proj = yes_no_prompt("Calculate atom-projected vibrational modes?", "yes")
        if atom_proj:
            freq_settings["atom_proj"] = True
        
        # Zero-point energy
        if freq_settings.get("mode") == "GAMMA":
            calc_zpe = yes_no_prompt("Calculate zero-point energy?", "yes")
            if calc_zpe:
                freq_settings["calc_zpe"] = True
        
        # Mode tracking
        if freq_settings.get("mode") == "GAMMA":
            track_modes = yes_no_prompt("Enable mode tracking (useful for optimization)?", "no")
            if track_modes:
                freq_settings["mode_tracking"] = True
        
        # Restart options
        print("\n=== RESTART AND PARALLELIZATION ===")
        ask_restart = yes_no_prompt("Configure restart capability?", "no")
        if ask_restart:
            freq_settings["restart"] = True
            checkpoint_interval = int(input("Checkpoint interval (calculations) [10]: ") or 10)
            freq_settings["checkpoint_interval"] = checkpoint_interval
        
        # Parallelization
        parallel_disp = yes_no_prompt("Enable parallel displacement calculations?", "yes")
        if parallel_disp:
            freq_settings["parallel_disp"] = True
    
    # Ask if user wants to configure advanced settings
    configure_advanced = yes_no_prompt("\nConfigure advanced frequency settings?", "no")
    
    if configure_advanced:
        # Special features for specific systems
        print("\n=== SPECIAL FEATURES ===")
        
        # Eckart conditions
        apply_eckart = yes_no_prompt("Apply Eckart conditions (remove rot/trans)?", "yes")
        if apply_eckart:
            freq_settings["eckart"] = True
        
        # Anharmonic corrections
        if freq_settings.get("mode") == "GAMMA" and not freq_settings.get("anharm"):
            calc_anharm = yes_no_prompt("Calculate anharmonic corrections?", "no")
            if calc_anharm:
                print("\nAnharmonic correction methods:")
                print("1: ANHARM - X-H stretching modes only")
                print("2: ANHAPES - Selected modes with PES scan")
                print("3: VSCF - Vibrational SCF (all modes)")
                print("4: VCI - Vibrational CI (most accurate)")
                
                anharm_choice = input("Select method (1-4): ").strip()
                
                if anharm_choice == "1":
                    freq_settings["anharm"] = True
                    # ANHARM specific settings handled elsewhere
                elif anharm_choice == "2":
                    freq_settings["anhapes"] = True
                    n_modes = int(input("Number of modes to scan [3]: ") or 3)
                    freq_settings["anhapes_modes"] = []
                    for i in range(n_modes):
                        mode = int(input(f"Mode number {i+1}: "))
                        freq_settings["anhapes_modes"].append(mode)
                elif anharm_choice == "3":
                    freq_settings["vscf"] = True
                    print("\nVSCF settings:")
                    vscf_level = int(input("VSCF level (1-4) [2]: ") or 2)
                    freq_settings["vscf_level"] = vscf_level
                elif anharm_choice == "4":
                    freq_settings["vci"] = True
                    print("\nVCI settings:")
                    vci_level = int(input("VCI level (1-4) [2]: ") or 2)
                    freq_settings["vci_level"] = vci_level
                    max_quanta = int(input("Maximum quanta sum [4]: ") or 4)
                    freq_settings["vci_max_quanta"] = max_quanta
        
        # Print level
        print("\n=== OUTPUT CONTROL ===")
        print("Print level for frequency calculation:")
        print("0: Minimal output")
        print("1: Standard output (default)")
        print("2: Detailed output")
        print("3: Debug output")
        print_level = int(input("Select print level (0-3) [1]: ") or 1)
        freq_settings["print_level"] = print_level
        
        # Mode following for difficult cases
        if freq_settings.get("mode") == "GAMMA":
            mode_follow = yes_no_prompt("\nEnable mode following (for nearly degenerate modes)?", "no")
            if mode_follow:
                freq_settings["mode_following"] = True
        
        # Symmetry handling
        print("\nSymmetry handling:")
        print("1: Use full symmetry (default)")
        print("2: Reduce symmetry for mode analysis")
        print("3: No symmetry")
        sym_choice = input("Select symmetry option (1-3) [1]: ").strip() or "1"
        
        if sym_choice == "2":
            freq_settings["reduce_symmetry"] = True
        elif sym_choice == "3":
            freq_settings["no_symmetry"] = True
    
    return freq_settings


def get_frequency_configuration(current_settings: Optional[Dict[str, Any]] = None,
                              shared_mode: bool = False) -> Dict[str, Any]:
    """
    Get frequency calculation configuration from user.
    
    This function handles all frequency calculation types:
    - FREQCALC (harmonic)
    - ANHARM (anharmonic X-H)
    - ANHAPES (anharmonic PES)
    - VSCF/VCI (anharmonic vibrational methods)
    
    Args:
        current_settings: Current settings (for CRYSTALOptToD12.py compatibility)
        shared_mode: If True, configuration will be used for multiple files
        
    Returns:
        Dict containing frequency calculation settings
    """
    from d12_constants import yes_no_prompt, get_user_input
    
    freq_config = {}
    
    # First determine the frequency calculation type
    print("\nFrequency calculation type:")
    print("1: FREQCALC - Harmonic frequencies (standard)")
    print("   - Calculates all normal modes")
    print("   - Can include IR/Raman intensities")
    print("   - Phonon dispersion for solids")
    print("2: ANHARM - Anharmonic X-H stretching")
    print("   - For X-H or X-D bonds only")
    print("   - More accurate frequencies for hydrogen stretching")
    print("   - Calculates anharmonic corrections")
    
    freq_type_choice = input("Select type (1-2) [1]: ").strip() or "1"
    
    if freq_type_choice == "2":
        # ANHARM calculation
        freq_config["freq_mode"] = "ANHARM"
        freq_config.update(_get_anharm_configuration())
    else:
        # FREQCALC calculation (default)
        freq_config["freq_mode"] = "FREQCALC"
        freq_config.update(_get_freqcalc_configuration())
    
    return freq_config


def _get_freqcalc_configuration() -> Dict[str, Any]:
    """Get FREQCALC (harmonic frequency) configuration"""
    from d12_constants import yes_no_prompt
    
    # Get the advanced frequency settings directly
    freq_settings = get_advanced_frequency_settings()
    
    # Build the configuration with settings at the top level
    freq_config = {
        **freq_settings,  # Unpack all frequency settings at top level
        "tolerances": {
            "TOLINTEG": DEFAULT_FREQ_SETTINGS["TOLINTEG"],
            "TOLDEE": DEFAULT_FREQ_SETTINGS["TOLDEE"],
        }
    }
    
    return freq_config


def _get_anharm_configuration() -> Dict[str, Any]:
    """Get ANHARM configuration for anharmonic X-H stretching"""
    from d12_constants import yes_no_prompt
    
    anharm_config = {"anharm_settings": {}}
    settings = anharm_config["anharm_settings"]
    
    print("\nAnharmonic X-H stretching calculation settings:")
    print("This calculates anharmonic frequencies for X-H or X-D bonds")
    print("The hydrogen atom is displaced along the X-H direction")
    
    # Get H atom label
    h_atom = input("\nEnter label (atom number) of H/D atom to displace: ").strip()
    try:
        settings["h_atom"] = int(h_atom)
    except ValueError:
        print("Invalid atom label, using atom 1")
        settings["h_atom"] = 1
    
    # Ask about symmetry
    keep_sym = yes_no_prompt(
        "\nMaintain symmetry (move all equivalent H atoms)?",
        "no"
    )
    if keep_sym:
        settings["keep_symmetry"] = True
    
    # Ask about points
    use_26 = yes_no_prompt(
        "\nUse extended 26-point grid (more accurate but slower)?",
        "no"
    )
    if use_26:
        settings["points26"] = True
    
    # Ask about isotopes
    calc_isotopes = yes_no_prompt(
        "\nCalculate isotope effects (H→D substitution)?",
        "no"
    )
    if calc_isotopes:
        n_isotopes = int(input("Number of isotopic substitutions: ") or 1)
        settings["isotopes"] = []
        for i in range(n_isotopes):
            label = int(input(f"  Atom label for substitution {i+1}: "))
            mass = float(input(f"  New mass (amu) [2.014 for D]: ") or 2.014)
            settings["isotopes"].append((label, mass))
    
    # ANHARM uses tight tolerances
    anharm_config["tolerances"] = {
        "TOLINTEG": "10 10 10 11 40",
        "TOLDEE": 12
    }
    
    return anharm_config




def write_frequency_calculation(f, calc_settings: Dict[str, Any], geometry_ended: bool = False):
    """
    Write frequency calculation section to D12 file.
    
    Args:
        f: File handle
        calc_settings: Dictionary containing all calculation settings
        geometry_ended: Whether the geometry section has been closed with END
    """
    freq_mode = calc_settings.get("freq_mode", "FREQCALC")
    
    if freq_mode == "ANHARM":
        # ANHARM goes outside geometry block
        if not geometry_ended:
            f.write("END\n")
        
        # Convert format from UI to d12creation
        anharm_settings = calc_settings.get("anharm_settings", {})
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
        
        write_anharm_section(f, converted_anharm)
        
    else:  # freq_mode == "FREQCALC" (default)
        # FREQCALC goes inside geometry block
        # Extract crystal system and space group if available
        crystal_system = calc_settings.get("crystal_system", None)
        space_group = calc_settings.get("space_group", None)
        optimization_section = calc_settings.get("optimization_section", None)
        write_frequency_section(f, calc_settings.get("freq_settings", {}), 
                              crystal_system, space_group, optimization_section)


def write_minimal_raman_section(f):
    """
    Write a minimal Raman calculation section with CPHF
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


def get_auto_phonon_path(crystal_system: str = None, space_group: int = None, 
                        shrink: int = 16, format_type: str = "labels",
                        lattice_type: str = "P", band_settings: Dict[str, Any] = None,
                        optimization_section: str = None) -> List:
    """
    Generate automatic phonon band path based on crystal system and space group.
    
    Args:
        crystal_system: Crystal system (cubic, hexagonal, tetragonal, etc.)
        space_group: Space group number (1-230)
        shrink: Shrink factor for k-points (0 for labels, >0 for coordinates)
        format_type: "labels", "vectors", "literature", or "seekpath"
        lattice_type: Lattice centering type (P, F, I, C, etc.)
        band_settings: Band configuration settings dict
        optimization_section: Optimization output text for structure extraction
        
    Returns:
        List of path segments
    """
    from d12_constants import HIGH_SYMMETRY_PATHS, SPACEGROUP_TO_PATH
    
    # Import the d3_config functions for consistency
    if get_band_path_from_symmetry is None:
        # Fallback if imports at module level failed
        return ["G X", "X M", "M G"]
    
    # For format_type == "seekpath", get SeeK-path full path
    if format_type == "seekpath" and get_seekpath_full_kpath:
        result = get_seekpath_full_kpath(space_group, lattice_type, optimization_section)
        if result:
            coord_segments, kpath_info = result
            # Store k-path source info if band_settings provided
            if band_settings:
                if kpath_info.get("source") == "literature":
                    band_settings["kpath_source"] = "literature"
                elif kpath_info.get("source") == "default":
                    band_settings["kpath_source"] = "default"
                elif kpath_info.get("has_inversion"):
                    band_settings["kpath_source"] = "seekpath_inv"
                else:
                    band_settings["kpath_source"] = "seekpath_noinv"
            # Convert to shrink-scaled integer coordinates for DISPERSION
            int_segments = []
            for seg in coord_segments:
                int_seg = [
                    int(seg[0] * shrink), int(seg[1] * shrink), int(seg[2] * shrink),
                    int(seg[3] * shrink), int(seg[4] * shrink), int(seg[5] * shrink)
                ]
                int_segments.append(int_seg)
            return int_segments
    
    # For format_type == "literature", get comprehensive path
    if format_type == "literature" and get_literature_kpath_vectors:
        coord_segments = get_literature_kpath_vectors(space_group, lattice_type)
        if coord_segments:
            # Store k-path source info if band_settings provided
            if band_settings:
                band_settings["kpath_source"] = "literature"
            # Convert to shrink-scaled integer coordinates for DISPERSION
            int_segments = []
            for seg in coord_segments:
                int_seg = [
                    int(seg[0] * shrink), int(seg[1] * shrink), int(seg[2] * shrink),
                    int(seg[3] * shrink), int(seg[4] * shrink), int(seg[5] * shrink)
                ]
                int_segments.append(int_seg)
            return int_segments
    
    # Get appropriate band path based on symmetry
    if get_band_path_from_symmetry and get_crystal_system_from_space_group:
        # Use d3_config functions for consistency
        path_labels = get_band_path_from_symmetry(space_group, lattice_type)
        crystal_sys = get_crystal_system_from_space_group(space_group, lattice_type)
        
        if format_type == "labels" or shrink == 0:
            # Return label-based path
            if band_settings:
                band_settings["kpath_source"] = "default"
            segments = []
            for i in range(len(path_labels) - 1):
                segments.append(f"{path_labels[i]} {path_labels[i+1]}")
            return segments
        
        elif format_type == "vectors" and get_kpoint_coordinates_from_labels:
            # Return coordinate-based path
            if band_settings:
                band_settings["kpath_source"] = "default"
            coord_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
            if coord_segments:
                # Convert to shrink-scaled integer coordinates
                int_segments = []
                for seg in coord_segments:
                    int_seg = [
                        int(seg[0] * shrink), int(seg[1] * shrink), int(seg[2] * shrink),
                        int(seg[3] * shrink), int(seg[4] * shrink), int(seg[5] * shrink)
                    ]
                    int_segments.append(int_seg)
                return int_segments
    
    # Original fallback code for backward compatibility
    if space_group and space_group in SPACEGROUP_TO_PATH:
        path_key = SPACEGROUP_TO_PATH[space_group]
        if path_key in HIGH_SYMMETRY_PATHS:
            path_data = HIGH_SYMMETRY_PATHS[path_key]
            if format_type == "labels" or shrink == 0:
                # Return label-based path
                labels = path_data.get("labels", ["G", "X", "M", "G"])
                # Convert list of points to list of segments
                if labels and isinstance(labels, list) and len(labels) > 1:
                    if " " not in labels[0]:  # Individual points
                        segments = []
                        for i in range(len(labels) - 1):
                            segments.append(f"{labels[i]} {labels[i+1]}")
                        return segments
                    else:  # Already in segment format
                        return labels
                return ["G X", "X M", "M G"]
            else:
                # Return coordinate-based path
                coords = path_data.get("coordinates", {})
                segments = []
                labels = path_data.get("labels", [])
                
                # Handle two formats: list of label pairs ["G X", "X M"] or list of points ["G", "X", "M"]
                if labels and isinstance(labels[0], str) and " " in labels[0]:
                    # Format 1: List of label pairs
                    for label_pair in labels:
                        start_label, end_label = label_pair.split()
                        if start_label in coords and end_label in coords:
                            start = coords[start_label]
                            end = coords[end_label]
                            # Convert to shrink-scaled integer coordinates
                            segment = [
                                int(start[0] * shrink), int(start[1] * shrink), int(start[2] * shrink),
                                int(end[0] * shrink), int(end[1] * shrink), int(end[2] * shrink)
                            ]
                            segments.append(segment)
                elif labels and len(labels) > 1:
                    # Format 2: List of individual points - create segments
                    for i in range(len(labels) - 1):
                        start_label = labels[i]
                        end_label = labels[i + 1]
                        if start_label in coords and end_label in coords:
                            start = coords[start_label]
                            end = coords[end_label]
                            # Convert to shrink-scaled integer coordinates
                            segment = [
                                int(start[0] * shrink), int(start[1] * shrink), int(start[2] * shrink),
                                int(end[0] * shrink), int(end[1] * shrink), int(end[2] * shrink)
                            ]
                            segments.append(segment)
                            
                return segments if segments else [[0, 0, 0, shrink//2, 0, 0]]
    
    # Fallback to simple cubic path if no specific path found
    if shrink == 0:
        return ["G X", "X M", "M G", "G R"]
    else:
        # Simple cubic path in fractional coordinates scaled by shrink
        return [
            [0, 0, 0, shrink//2, 0, 0],      # G to X
            [shrink//2, 0, 0, shrink//2, shrink//2, 0],  # X to M
            [shrink//2, shrink//2, 0, 0, 0, 0],          # M to G
            [0, 0, 0, shrink//2, shrink//2, shrink//2]   # G to R
        ]


def write_frequency_section(f, freq_settings, crystal_system: str = None, 
                          space_group: int = None, optimization_section: str = None):
    """
    Write the frequency calculation section of the D12 file
    
    Args:
        f: File handle
        freq_settings: Dictionary with frequency calculation settings
        crystal_system: Crystal system for AUTO path generation
        space_group: Space group number for AUTO path generation
        optimization_section: Optimization output text for structure extraction
    """
    # Check for minimal Raman mode
    if freq_settings.get("minimal_raman", False):
        write_minimal_raman_section(f)
        return
    
    print("FREQCALC", file=f)
    
    # Handle restart
    if freq_settings.get("restart", False):
        print("RESTART", file=f)
    
    # Pre-optimization if requested
    if freq_settings.get("preoptgeom", False):
        print("PREOPTGEOM", file=f)
        opt_settings = freq_settings.get("optgeom_settings", {})
        if opt_settings.get("fulloptg", False):
            print("FULLOPTG", file=f)
        # Optimization tolerances
        if "toldeg" in opt_settings:
            print("TOLDEG", file=f)
            print(format_crystal_float(opt_settings["toldeg"]), file=f)
        if "toldex" in opt_settings:
            print("TOLDEX", file=f)
            print(format_crystal_float(opt_settings["toldex"]), file=f)
        if "finalrun" in opt_settings:
            print("FINALRUN", file=f)
            print(opt_settings["finalrun"], file=f)
        print("END", file=f)
    
    # Mode analysis
    if freq_settings.get("analysis", False):
        print("ANALYSIS", file=f)
    elif freq_settings.get("noanalysis", False):
        print("NOANALYSIS", file=f)
    
    # Eckart conditions
    if not freq_settings.get("eckart", True):
        print("NOECKART", file=f)
    
    # Fragment calculation
    if "fragment" in freq_settings:
        fragment_atoms = freq_settings["fragment"]
        print("FRAGMENT", file=f)
        print(len(fragment_atoms), file=f)
        print(" ".join(str(a) for a in fragment_atoms), file=f)
    
    # Modified isotopes
    if "isotopes" in freq_settings:
        isotopes = freq_settings["isotopes"]
        print("ISOTOPES", file=f)
        print(len(isotopes), file=f)
        for atom_label, mass in isotopes.items():
            print(f"{atom_label} {mass}", file=f)
    
    # Numerical derivative method (skip if minimal_raman is True)
    if not freq_settings.get("minimal_raman", False):
        numderiv = freq_settings.get("numderiv", 2)
        print("NUMDERIV", file=f)
        print(numderiv, file=f)
    
    # Step size for numerical derivatives
    if "stepsize" in freq_settings:
        print("STEPSIZE", file=f)
        print(format_crystal_float(freq_settings["stepsize"]), file=f)
    
    # Temperature range
    if "temprange" in freq_settings:
        temprange = freq_settings["temprange"]
        # Handle both tuple and dict formats
        if isinstance(temprange, dict):
            n_temps = temprange.get("n_temps", 20)
            t_min = temprange.get("t_min", 0)
            t_max = temprange.get("t_max", 400)
        else:
            # Assume it's a tuple/list
            try:
                n_temps, t_min, t_max = temprange
            except (ValueError, TypeError):
                # Fallback to defaults if unpacking fails
                n_temps, t_min, t_max = 20, 0, 400
        print("TEMPERAT", file=f)
        print(f"{n_temps} {t_min} {t_max}", file=f)
    
    # Pressure range
    if "pressrange" in freq_settings:
        pressrange = freq_settings["pressrange"]
        # Handle both tuple and dict formats
        if isinstance(pressrange, dict):
            n_press = pressrange.get("n_press", 20)
            p_min = pressrange.get("p_min", 0)
            p_max = pressrange.get("p_max", 10)
        else:
            # Assume it's a tuple/list
            try:
                n_press, p_min, p_max = pressrange
            except (ValueError, TypeError):
                # Fallback to defaults if unpacking fails
                n_press, p_min, p_max = 20, 0, 10
        print("PRESSURE", file=f)
        print(f"{n_press} {p_min} {p_max}", file=f)
    
    # Neglect lowest frequencies
    if "neglectfreq" in freq_settings:
        print("NEGLEFRE", file=f)
        print(freq_settings["neglectfreq"], file=f)
    
    # Multitask for HPC
    if "multitask" in freq_settings:
        print("MULTITASK", file=f)
        print(freq_settings["multitask"], file=f)
    
    # Mode printing
    if not freq_settings.get("print_modes", True):
        print("NOMODES", file=f)
    
    # Frequency scaling
    if "freqscale" in freq_settings:
        print("FREQSCAL", file=f)
        print(format_crystal_float(freq_settings["freqscale"]), file=f)
    
    # Dielectric tensor or constant for LO/TO splitting
    if "dielectric_tensor" in freq_settings:
        tensor = freq_settings["dielectric_tensor"]
        print("DIELTENS", file=f)
        # Print 3x3 tensor (9 values)
        tensor_str = " ".join(format_crystal_float(v) for v in tensor[:9])
        print(tensor_str, file=f)
    elif "dielectric_constant" in freq_settings:
        print("DIELISO", file=f)
        print(format_crystal_float(freq_settings["dielectric_constant"]), file=f)
    
    # Chi2 tensor for Raman LO modes
    if "chi2tensor" in freq_settings:
        tensor = freq_settings["chi2tensor"]
        print("CHI2TENS", file=f)
        # Print 3x9 tensor (27 values)
        for i in range(0, 27, 9):
            tensor_str = " ".join(format_crystal_float(v) for v in tensor[i:i+9])
            print(tensor_str, file=f)
    
    # IR intensities
    if freq_settings.get("intensities", False):
        print("INTENS", file=f)
        
        ir_method = freq_settings.get("ir_method", "BERRY").upper()
        if ir_method == "WANNIER":
            print("INTLOC", file=f)
            
            # Wannier function settings
            if freq_settings.get("relocalize_wannier", False):
                print("DIPOMOME", file=f)
                print("RELOCAL", file=f)
                print("END", file=f)
        elif ir_method == "CPHF":
            print("INTCPHF", file=f)
            # CPHF settings block
            cphf_settings = freq_settings.get("cphf_settings", {})
            if "fmixing" in cphf_settings:
                print("FMIXING", file=f)
                print(cphf_settings["fmixing"], file=f)
            if "tolalpha" in cphf_settings:
                print("TOLALPHA", file=f)
                print(cphf_settings["tolalpha"], file=f)
            if "maxcycle" in cphf_settings:
                print("MAXCYCLE", file=f)
                print(cphf_settings["maxcycle"], file=f)
            # Print ENDCPHF to close the INTCPHF block
            print("ENDCPHF", file=f)
        # else: default is Berry phase (INTPOL), no keyword needed
        
        # Born tensor normalization
        if freq_settings.get("born_tensor_norm", False):
            print("NORMBORN", file=f)
    else:
        print("NOINTENS", file=f)
    
    # Raman intensities
    if freq_settings.get("raman", False):
        print("INTRAMAN", file=f)
        
        # Always need INTCPHF for Raman
        if not (freq_settings.get("intensities", False) and 
                freq_settings.get("ir_method", "BERRY").upper() == "CPHF"):
            print("INTCPHF", file=f)
            # CPHF settings for Raman
            cphf_settings = freq_settings.get("cphf_settings", {})
            if "fmixing2" in cphf_settings:
                print("FMIXING2", file=f)
                print(cphf_settings["fmixing2"], file=f)
            if "tolgamma" in cphf_settings:
                print("TOLGAMMA", file=f)
                print(cphf_settings["tolgamma"], file=f)
            if "maxcycle2" in cphf_settings:
                print("MAXCYCLE2", file=f)
                print(cphf_settings["maxcycle2"], file=f)
            print("ENDCPHF", file=f)
        
        # Experimental conditions
        if "ramanexp" in freq_settings:
            temp, laser_nm = freq_settings["ramanexp"]
            print("RAMANEXP", file=f)
            print(f"{temp} {laser_nm}", file=f)
        
        # Other Raman options
        if freq_settings.get("norenorm", False):
            print("NORENORM", file=f)
        if freq_settings.get("tensonly", False):
            print("TENSONLY", file=f)
    
    # Combination modes
    if "combmode" in freq_settings:
        print("COMBMODE", file=f)
        mode_type = freq_settings["combmode"].upper()
        if mode_type in ["IR", "RAMAN", "ALL"]:
            print(mode_type, file=f)
        # IRRAMAN is default, no keyword needed
        
        # Frequency range for combination modes
        if "combmode_range" in freq_settings:
            print("FREQRANGE", file=f)
            fmin, fmax = freq_settings["combmode_range"]
            print(f"{fmin} {fmax}", file=f)
        print("END", file=f)
    
    # Mode scanning
    if "scanmode" in freq_settings:
        scan_settings = freq_settings["scanmode"]
        modes = scan_settings.get("modes", [])
        if modes:
            print("SCANMODE", file=f)
            scan_range = scan_settings.get("range", (-10, 10, 0.2))
            print(f"{len(modes)} {scan_range[0]} {scan_range[1]} {scan_range[2]}", file=f)
            print(" ".join(str(m) for m in modes), file=f)
    
    # Phonon dispersion
    if freq_settings.get("dispersion", False):
        print("DISPERSION", file=f)
        
        # Supercell for phonon calculation
        if "scelphono" in freq_settings:
            supercell = freq_settings["scelphono"]
            print("SCELPHONO", file=f)
            # Can be either expansion factors or full transformation matrix
            if isinstance(supercell, list) and len(supercell) == 3:
                # Simple expansion factors [nx, ny, nz]
                print(" ".join(str(n) for n in supercell), file=f)
            elif isinstance(supercell, list) and len(supercell) == 9:
                # Full transformation matrix (3x3)
                for i in range(0, 9, 3):
                    print(" ".join(str(supercell[j]) for j in range(i, i+3)), file=f)
        
        # Fourier interpolation
        if "interphess" in freq_settings:
            interp = freq_settings["interphess"]
            print("INTERPHESS", file=f)
            l_params = interp.get("expand", [2, 2, 2])
            print(" ".join(str(l) for l in l_params[:3]), file=f)
            print(interp.get("print", 0), file=f)
        
        # Wang correction for polar materials
        if "wang" in freq_settings:
            wang_tensor = freq_settings["wang"]
            print("WANG", file=f)
            tensor_str = " ".join(format_crystal_float(v) for v in wang_tensor[:9])
            print(tensor_str, file=f)
        
        # Band structure
        if "bands" in freq_settings:
            band_settings = freq_settings["bands"]
            print("BANDS", file=f)
            
            # First line: ISS NSUB (shrinking factor and number of k-points)
            shrink = band_settings.get("shrink", 16)
            npoints = band_settings.get("npoints", 100)
            print(f"{shrink} {npoints}", file=f)
            
            # Get path
            path = band_settings.get("path", [])
            
            # Handle AUTO path generation or path == "auto"
            if path == "AUTO" or path == "auto" or band_settings.get("auto_path", False):
                # Initialize default lattice type
                lattice_type = "P"  # Default
                
                # Extract space group and lattice type from optimization section if available
                import re
                if optimization_section:
                    # Find space group number
                    sg_match = re.search(r'SPACE GROUP.*?NUMBER:\s*(\d+)', optimization_section)
                    if sg_match:
                        space_group = int(sg_match.group(1))
                    
                    # Find lattice type from space group symbol
                    sg_symbol_match = re.search(r'SPACE GROUP.*?:\s+([A-Z]\s*[\-/0-9\s]*[A-Z0-9]*)', optimization_section)
                    if sg_symbol_match:
                        symbol = sg_symbol_match.group(1).strip()
                        if symbol:
                            lattice_type = symbol[0]
                else:
                    # Try to extract from crystal_system if available
                    if crystal_system and "-" in crystal_system:
                        # Extract lattice type from crystal_system like "cubic-F"
                        lattice_type = crystal_system.split("-")[1]
                
                # Get format type from band settings
                format_type = band_settings.get("format", "labels")
                path_method = band_settings.get("path_method", "labels")
                
                # For vector-based paths, we need to handle them differently
                if format_type == "vectors" or path_method == "coordinates":
                    # Get the fractional k-point segments
                    if band_settings.get("seekpath_full", False) and get_seekpath_full_kpath:
                        # SeeK-path full path
                        result = get_seekpath_full_kpath(space_group, lattice_type, optimization_section)
                        if result:
                            frac_segments, kpath_info = result
                            # Store k-path source info
                            if kpath_info.get("source") == "literature":
                                band_settings["kpath_source"] = "literature"
                            elif kpath_info.get("source") == "default":
                                band_settings["kpath_source"] = "default"
                            elif kpath_info.get("has_inversion"):
                                band_settings["kpath_source"] = "seekpath_inv"
                            else:
                                band_settings["kpath_source"] = "seekpath_noinv"
                        # Get seekpath labels if available
                        if get_seekpath_labels:
                            path_labels = get_seekpath_labels(space_group, lattice_type, optimization_section)
                            band_settings["path_labels"] = path_labels
                    elif band_settings.get("literature_path", False) and get_literature_kpath_vectors:
                        # Literature path vectors
                        frac_segments = get_literature_kpath_vectors(space_group, lattice_type)
                        band_settings["kpath_source"] = "literature"
                        # Get standard labels for literature path
                        path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                        band_settings["path_labels"] = path_labels
                    else:
                        # Standard path vectors
                        band_settings["kpath_source"] = "default"
                        path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                        frac_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                        band_settings["path_labels"] = path_labels
                    
                    if frac_segments and scale_kpoint_segments:
                        # Scale fractional coordinates by shrink factor
                        path = scale_kpoint_segments(frac_segments, shrink)
                    else:
                        # Fallback to manual scaling
                        path = []
                        for seg in frac_segments:
                            int_seg = [
                                int(seg[0] * shrink), int(seg[1] * shrink), int(seg[2] * shrink),
                                int(seg[3] * shrink), int(seg[4] * shrink), int(seg[5] * shrink)
                            ]
                            path.append(int_seg)
                else:
                    # Label-based path
                    band_settings["kpath_source"] = "default"
                    path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                    path = []
                    for i in range(len(path_labels) - 1):
                        path.append(f"{path_labels[i]} {path_labels[i+1]}")
                    shrink = 0  # Force labels mode
                    # Store path labels for title generation
                    band_settings["path_labels"] = path_labels
            
            # Second line: NLINE (number of lines/segments)
            if isinstance(path, list):
                print(len(path), file=f)
            else:
                # Fallback for unexpected path type
                print(1, file=f)
                path = [[0, 0, 0, 1, 0, 0]]  # Default segment
            
            # Then the path segments
            if band_settings.get("mixed_path", False):
                # Handle mixed label/coordinate segments
                for segment in path:
                    if isinstance(segment, str):
                        # Label segment
                        print(segment, file=f)
                    elif isinstance(segment, (list, tuple)) and len(segment) == 2:
                        # Coordinate segment - scale by shrink if needed
                        if all(isinstance(coord, (list, tuple)) for coord in segment):
                            # Format: [[x1,y1,z1], [x2,y2,z2]]
                            start, end = segment
                            if shrink > 0:
                                # Scale fractional to integer coordinates
                                print(f"{int(start[0]*shrink)} {int(start[1]*shrink)} {int(start[2]*shrink)} "
                                      f"{int(end[0]*shrink)} {int(end[1]*shrink)} {int(end[2]*shrink)}", file=f)
                            else:
                                # Keep as fractional
                                print(f"{start[0]} {start[1]} {start[2]} {end[0]} {end[1]} {end[2]}", file=f)
                        else:
                            # Direct coordinate list
                            print(" ".join(str(coord) for coord in segment), file=f)
                    else:
                        # Fallback
                        print("G X", file=f)
            elif shrink == 0:
                # ISS=0: Use high-symmetry labels
                for segment in path:
                    if isinstance(segment, str) and len(segment) >= 2:
                        # Labels like "G X" or "GM"
                        print(segment, file=f)
                    else:
                        print("G X", file=f)  # Default fallback
            else:
                # ISS>0: Use integer coordinates
                for segment in path:
                    if isinstance(segment, (list, tuple)) and len(segment) == 6:
                        # Direct format: 6 coordinates on one line
                        print(" ".join(str(coord) for coord in segment), file=f)
                    elif isinstance(segment, (list, tuple)) and len(segment) == 2:
                        # Tuple format: ((x1,y1,z1), (x2,y2,z2))
                        start, end = segment
                        print(f"{start[0]} {start[1]} {start[2]} {end[0]} {end[1]} {end[2]}", file=f)
                    else:
                        # Default path segment
                        print("0 0 0 1 0 0", file=f)
        
        # Phonon DOS
        if "pdos" in freq_settings:
            pdos_settings = freq_settings["pdos"]
            print("PDOS", file=f)
            # PDOS format: NUMA NBIN (max frequency and number of bins)
            print(f"{pdos_settings.get('max_freq', 2000)} {pdos_settings.get('nbins', 200)}", file=f)
            # LPRO (0 or 1 for projected DOS)
            print(1 if pdos_settings.get("projected", True) else 0, file=f)
        
        # INS spectrum
        if "ins" in freq_settings:
            ins_settings = freq_settings["ins"]
            print("INS", file=f)
            # INS format: NUMA NBIN (max frequency and number of bins)
            print(f"{ins_settings.get('max_freq', 3000)} {ins_settings.get('nbins', 300)}", file=f)
            # NWTYPE (0, 1, or 2 for neutron cross-section type)
            print(ins_settings.get("neutron_type", 2), file=f)  # 0=coherent, 1=incoherent, 2=both
    
    # Anisotropic displacement parameters
    if "adp" in freq_settings:
        adp_settings = freq_settings["adp"]
        print("ADP", file=f)
        print(f"{adp_settings.get('type', 0)} {adp_settings.get('neglect', 0)}", file=f)
    
    # IR spectrum generation
    if freq_settings.get("irspec", False):
        print("IRSPEC", file=f)
        
        # Spectrum settings
        if "spec_range" in freq_settings:
            print("RANGE", file=f)
            print(f"{freq_settings['spec_range'][0]} {freq_settings['spec_range'][1]}", file=f)
        if "spec_step" in freq_settings:
            print("LENSTEP", file=f)
            print(format_crystal_float(freq_settings["spec_step"]), file=f)
        if "spec_dampfac" in freq_settings:
            print("DAMPFAC", file=f)
            print(format_crystal_float(freq_settings["spec_dampfac"]), file=f)
        if freq_settings.get("spec_gaussian", False):
            print("GAUSS", file=f)
        if "spec_angle" in freq_settings:
            print("ANGLE", file=f)
            print(format_crystal_float(freq_settings["spec_angle"]), file=f)
        if freq_settings.get("spec_refrind", False):
            print("REFRIND", file=f)
        if freq_settings.get("spec_dielfun", False):
            print("DIELFUN", file=f)
        
        print("END", file=f)
    
    # Raman spectrum generation
    if freq_settings.get("ramspec", False):
        print("RAMSPEC", file=f)
        
        # Spectrum settings
        if "spec_range" in freq_settings:
            print("RANGE", file=f)
            print(f"{freq_settings['spec_range'][0]} {freq_settings['spec_range'][1]}", file=f)
        if "spec_step" in freq_settings:
            print("LENSTEP", file=f)
            print(format_crystal_float(freq_settings["spec_step"]), file=f)
        if "raman_voigt" in freq_settings:
            print("VOIGT", file=f)
            print(format_crystal_float(freq_settings["raman_voigt"]), file=f)
        elif "spec_voigt" in freq_settings:
            print("VOIGT", file=f)
            print(format_crystal_float(freq_settings["spec_voigt"]), file=f)
        if "raman_dampfac" in freq_settings:
            print("DAMPFAC", file=f)
            print(format_crystal_float(freq_settings["raman_dampfac"]), file=f)
        elif "spec_dampfac" in freq_settings:
            print("DAMPFAC", file=f)
            print(format_crystal_float(freq_settings["spec_dampfac"]), file=f)
        
        print("END", file=f)
    
    # Thermodynamic calculations
    if freq_settings.get("thermo", False):
        print("THERMO", file=f)
        
        # Thermodynamic settings
        if "thermo_type" in freq_settings:
            # Options: MOLECULE, LINEAR, CRYSTAL
            print(freq_settings["thermo_type"], file=f)
        
        # Temperature for calculation
        if "temperature" in freq_settings and "temprange" not in freq_settings:
            print("TEMP", file=f)
            print(format_crystal_float(freq_settings["temperature"]), file=f)
        
        # Pressure for calculation
        if "pressure" in freq_settings and "pressrange" not in freq_settings:
            print("PRES", file=f)
            print(format_crystal_float(freq_settings["pressure"]), file=f)
        
        print("END", file=f)
    
    # Anharmonic options
    if "anhapes" in freq_settings:
        anhapes = freq_settings["anhapes"]
        print("ANHAPES", file=f)
        print(len(anhapes["modes"]), file=f)
        print(" ".join(str(m) for m in anhapes["modes"]), file=f)
        print(anhapes.get("scheme", 0), file=f)
        if "step" in anhapes:
            print(format_crystal_float(anhapes["step"]), file=f)
    
    if "vscf" in freq_settings:
        vscf = freq_settings["vscf"]
        print("VSCF", file=f)
        if "tolerance" in vscf:
            print("TOLSCF", file=f)
            print(format_crystal_float(vscf["tolerance"]), file=f)
        if "mixing" in vscf:
            print("MIXING", file=f)
            print(format_crystal_float(vscf["mixing"]), file=f)
        print("END", file=f)
    
    if "vci" in freq_settings:
        vci = freq_settings["vci"]
        print("VCI", file=f)
        print(vci.get("quanta", 4), file=f)
        print(vci.get("modes", 10), file=f)
        print(vci.get("guess", 0), file=f)
    
    # High-symmetry only or special modes
    if freq_settings.get("nomodes", False):
        print("NOMODES", file=f)
    
    # End FREQCALC
    print("ENDFREQ", file=f)


def write_anharm_section(f, anharm_settings):
    """
    Write anharmonic X-H stretching section (outside FREQCALC)
    
    Args:
        f: File handle
        anharm_settings: Dictionary with anharmonic settings
            - atom_label: int - Label of H/D atom to displace
            - points: int - Number of points (7 or 26, default 7)
            - isotopes: dict - Modified isotope masses {atom_label: mass}
            - keepsymm: bool - Keep symmetry by moving all equivalent atoms
            - noguess: bool - Use atomic densities as SCF guess at each point
    """
    print("ANHARM", file=f)
    print(anharm_settings["atom_label"], file=f)
    
    # Modified isotopes
    if "isotopes" in anharm_settings:
        isotopes = anharm_settings["isotopes"]
        print("ISOTOPES", file=f)
        print(len(isotopes), file=f)
        for atom_label, mass in isotopes.items():
            print(f"{atom_label} {mass}", file=f)
    
    # Keep symmetry
    if anharm_settings.get("keepsymm", False):
        print("KEEPSYMM", file=f)
    
    # SCF guess
    if anharm_settings.get("noguess", False):
        print("NOGUESS", file=f)
    
    # Number of points
    if anharm_settings.get("points", 7) == 26:
        print("POINTS26", file=f)
    
    print("END", file=f)


def format_crystal_float(value):
    """
    Format a float value for CRYSTAL input according to its specific rules.
    CRYSTAL requires scientific notation for values outside certain ranges.
    
    For TOLDEG/TOLDEX in optimization, values like 0.00003 should be written
    as decimal notation, not scientific notation.
    """
    if abs(value) < 1e-10:
        return "0.0"
    elif 0.00001 <= abs(value) < 10000:  # Changed threshold from 0.0001 to 0.00001
        # For values in this range, use regular decimal notation
        # Ensure we don't accidentally use scientific notation for small values
        formatted = f"{value:.8f}".rstrip('0').rstrip('.')
        # Double-check it's not in scientific notation
        if 'e' in formatted.lower():
            # This shouldn't happen in this range, but just in case
            formatted = f"{value:.10f}".rstrip('0').rstrip('.')
        return formatted
    else:
        # For very small or large values, use scientific notation
        return f"{value:.6E}"


def get_high_symmetry_points(space_group, bravais):
    """
    Get default high-symmetry points based on space group and Bravais lattice.
    This is a placeholder function - you would need to implement the logic
    based on crystallographic tables.
    """
    # This is a simplified version - real implementation would use proper
    # crystallographic data
    if bravais in ["C", "F", "I"]:
        # Cubic systems
        return [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],  # Γ → X
            [0.5, 0.0, 0.0, 0.5, 0.5, 0.0],  # X → M
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],  # M → Γ
        ]
    else:
        # Default for other systems
        return [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],  # Γ → X
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # X → Γ
        ]


# For backwards compatibility, export the main function
get_frequency_settings = get_frequency_configuration