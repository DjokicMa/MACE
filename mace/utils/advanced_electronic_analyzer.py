#!/usr/bin/env python3
"""
Advanced Electronic Structure Analyzer for CRYSTAL Calculations
===============================================================
Analyzes BAND.DAT and DOSS.DAT files to calculate real effective masses,
accurate electronic classification, and transport properties.

Based on DOS analysis algorithms for metal/semimetal/semiconductor classification
and band structure analysis for effective mass calculation.

Usage:
    from advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    analyzer = AdvancedElectronicAnalyzer()
    results = analyzer.analyze_material(band_file, doss_file)
"""

import sys
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Single-sourced full-precision constants (fallback keeps this standalone-runnable).
try:
    from mace.constants import HARTREE_TO_EV, BOHR_TO_ANGSTROM
except ImportError:
    HARTREE_TO_EV = 27.211386245988
    BOHR_TO_ANGSTROM = 0.52917721067


class AdvancedElectronicAnalyzer:
    """Advanced electronic structure analysis using CRYSTAL BAND and DOSS data."""
    
    # Hartree above E_F up to which a band's maximum still counts as occupied
    OCCUPIED_TOL = 1e-3

    def __init__(self):
        self.ha_to_ev = HARTREE_TO_EV  # Hartree to eV conversion
        self.bohr_to_angstrom = BOHR_TO_ANGSTROM  # Bohr to Angstrom conversion
        
    def read_dos_data(self, doss_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read DOSS.DAT file and return energies and total DOS.
        
        Returns:
            E: Energy array in Hartree (relative to Fermi level = 0)
            g: Total DOS array in states/Ha/cell, the spin channels summed

        Read by DatFileProcessor's DOSS.DAT parser: CRYSTAL wraps a record
        with many projections over several lines, and repeats the energy
        grid per spin channel (beta DOS negative). Line by line, a 1002-point
        file of 26 projections and 2 spins read as 4008 points.
        """
        try:
            try:
                from mace.utils.dat_file_processor import DatFileProcessor
            except ImportError:
                from dat_file_processor import DatFileProcessor
            with open(doss_file, 'r') as fh:
                dat = DatFileProcessor()._parse_doss_dat_content(fh.read())
        except FileNotFoundError:
            print(f"Warning: DOSS file not found: {doss_file}")
            return np.array([]), np.array([])
        except Exception as e:
            print(f"Warning: Error reading DOSS file {doss_file}: {e}")
            return np.array([]), np.array([])

        energies = np.asarray(dat.get('energy_points') or [], float)
        total = np.asarray(dat.get('total_dos') or [], float)
        nspin = max(int(dat.get('num_spins') or 1), 1)
        if len(energies) == 0 or len(energies) != len(total) or len(energies) % nspin:
            return np.array([]), np.array([])
        n = len(energies) // nspin
        E = energies[:n]
        # total DOS = sum of the spin channels' (absolute) total columns
        g = np.abs(total.reshape(nspin, n)).sum(axis=0)
        return E, g
    
    def read_band_data(self, band_file: Path) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Read BAND.DAT file and return band edge information.
        
        Returns:
            Ev_max: Valence band maximum in Hartree
            Ec_min: Conduction band minimum in Hartree  
            k_points: k-point coordinates
            energies: Energy bands at each k-point

        The edges are by band, not by the sign of each eigenvalue: CRYSTAL
        puts E_F at the valence-band top, which can sit a few micro-Hartree
        above 0 (taking every E > 0 as a conduction state gave a molecule's
        7.8 eV gap as 0.16 eV). Per spin block, a band whose maximum is at
        most OCCUPIED_TOL above E_F is occupied: VBM is the highest occupied
        level, CBM the lowest level of the other bands (below VBM for a band
        crossing E_F, a metal).
        """
        Ev_max = -np.inf
        Ec_min = np.inf
        k_points = []
        all_energies = []
        nkpt = nspin = 0
        
        try:
            with open(band_file, 'r') as fh:
                nbnd = 0
                pending = []
                for line in fh:
                    if line.startswith('#'):
                        tok = line.lstrip('#').split()
                        if 'NBND' in tok:
                            try:
                                nbnd = int(tok[tok.index('NBND') + 1])
                            except (ValueError, IndexError):
                                pass
                        for key in ('NKPT', 'NSPIN'):
                            if key in tok:
                                try:
                                    val = int(tok[tok.index(key) + 1])
                                except (ValueError, IndexError):
                                    continue
                                if key == 'NKPT':
                                    nkpt = nkpt or val
                                else:
                                    nspin = nspin or val
                        continue
                    # Skip comments and empty lines
                    if line.startswith('@') or not line.strip():
                        continue
                    
                    try:
                        vals = pending + list(map(float, line.split()))
                        # CRYSTAL wraps a k-point row after 1000 fields
                        # (NBND >= 1000): keep reading until the row is whole
                        if nbnd and len(vals) < nbnd + 1:
                            pending = vals
                            continue
                        pending = []
                        if len(vals) >= 2:
                            k_coord = vals[0]  # k-point coordinate
                            energies = vals[1:]  # Energy bands in Hartree
                            
                            k_points.append(k_coord)
                            all_energies.append(energies)
                                    
                    except ValueError:
                        continue
                        
        except FileNotFoundError:
            print(f"Warning: BAND file not found: {band_file}")
            return -np.inf, np.inf, np.array([]), np.array([])
        except Exception as e:
            print(f"Warning: Error reading BAND file {band_file}: {e}")
            return -np.inf, np.inf, np.array([]), np.array([])

        # Band edges per spin block (the k path is written once per spin)
        widths = {len(row) for row in all_energies}
        if all_energies and len(widths) == 1:
            arr = np.asarray(all_energies, float)
            blocks = (np.split(arr, nspin) if nspin > 1 and nkpt and len(arr) == nkpt * nspin
                      else [arr])
            for block in blocks:
                occupied = block.max(axis=0) <= self.OCCUPIED_TOL
                if occupied.any():
                    Ev_max = max(Ev_max, float(block[:, occupied].max()))
                if (~occupied).any():
                    Ec_min = min(Ec_min, float(block[:, ~occupied].min()))
            
        return Ev_max, Ec_min, np.array(k_points), np.array(all_energies)
    
    def classify_electronic_behavior(self, E: np.ndarray, g: np.ndarray, 
                                   Ev_max: float, Ec_min: float, 
                                   gcrit_factor: float = 0.05) -> Dict[str, Any]:
        """
        Classify electronic behavior using DOS analysis.
        
        Based on the sophisticated algorithm from your provided code.
        """
        have_dos = len(E) > 0 and len(g) > 0
        have_band = Ev_max != -np.inf and Ec_min != np.inf
        if not have_dos and not have_band:
            return {
                'electronic_classification': 'unknown',
                'gap_hartree': 0.0,
                'gap_ev': 0.0,
                'dos_at_fermi': 0.0,
                'dos_threshold': 0.0,
                'classification_method': 'no_dos_data'
            }
        
        g_Ef = g_mean = g_crit = None
        if have_dos:
            # Find DOS at Fermi level (E = 0)
            Ef_index = np.argmin(np.abs(E))
            g_Ef = g[Ef_index]

            # Calculate mean DOS and threshold
            g_positive = g[g > 0]
            g_mean = g_positive.mean() if len(g_positive) > 0 else 0.0
            g_crit = gcrit_factor * g_mean
        
        # The gap comes from the band structure only. One read off the DOS
        # is not the material's: CRYSTAL puts E_F on a molecule's broadened
        # HOMO peak (DOS on both sides of it), and a DOSS band range leaves
        # states out (an empty window that is no gap). With the DOS alone
        # neither a gap nor a classification is reported.
        gap = None
        gap_source = None
        classification = None
        if have_band:
            gap = max(Ec_min - Ev_max, 0.0)  # bands overlapping E_F: none
            gap_source = "band_structure"
            if gap > 0.001:  # 0.001 Ha ≈ 0.027 eV threshold for clear gap
                if gap > 0.15:  # 0.15 Ha ≈ 4.0 eV
                    classification = "insulator"
                else:
                    classification = "semiconductor"
            elif have_dos:
                # metal vs semimetal needs the DOS at E_F
                classification = "metal" if g_Ef > g_crit else "semimetal"
        
        return {
            'electronic_classification': classification,
            'gap_hartree': gap,
            'gap_ev': gap * self.ha_to_ev if gap is not None else None,
            'dos_at_fermi': g_Ef,
            'dos_threshold': g_crit,
            'dos_mean': g_mean,
            'gap_source': gap_source,
            'classification_method': 'dos_band_analysis',
            'gcrit_factor': gcrit_factor
        }
    
    def calculate_effective_mass(self, k_points: np.ndarray, energies: np.ndarray,
                               Ev_max: float, Ec_min: float) -> Dict[str, Any]:
        """
        Calculate effective masses from band structure curvature.
        
        Uses second derivative d²E/dk² near band edges.
        """
        effective_mass_results = {}
        
        if len(k_points) == 0 or len(energies) == 0:
            return {
                'electron_effective_mass': None,
                'hole_effective_mass': None,
                'calculation_method': 'no_band_data'
            }
        
        try:
            # Find bands closest to Fermi level
            all_bands = energies.flatten()
            
            # Electron effective mass (conduction band minimum)
            if Ec_min != np.inf:
                electron_mass = self._calculate_band_curvature(k_points, energies, Ec_min, 'electron')
            else:
                electron_mass = None
            
            # Hole effective mass (valence band maximum)  
            if Ev_max != -np.inf:
                hole_mass = self._calculate_band_curvature(k_points, energies, Ev_max, 'hole')
            else:
                hole_mass = None
            
            # Calculate average effective mass
            if electron_mass is not None and hole_mass is not None:
                avg_mass = (electron_mass * hole_mass) ** 0.5
            elif electron_mass is not None:
                avg_mass = electron_mass
            elif hole_mass is not None:
                avg_mass = hole_mass
            else:
                avg_mass = None
            
            return {
                'electron_effective_mass': electron_mass,
                'hole_effective_mass': hole_mass,
                'average_effective_mass': avg_mass,
                'calculation_method': 'band_curvature',
                'band_points_analyzed': len(k_points)
            }
            
        except Exception as e:
            print(f"Warning: Error calculating effective mass: {e}")
            return {
                'electron_effective_mass': None,
                'hole_effective_mass': None,
                'calculation_method': 'calculation_failed'
            }
    
    def _calculate_band_curvature(self, k_points: np.ndarray, energies: np.ndarray, 
                                target_energy: float, carrier_type: str) -> Optional[float]:
        """
        Calculate effective mass from band curvature using finite differences.
        
        m* = ℏ² / (d²E/dk²)
        """
        try:
            # Find the band and k-points closest to target energy
            min_diff = np.inf
            best_band_idx = 0
            best_k_range = None
            
            for band_idx in range(energies.shape[1]):
                band_energies = energies[:, band_idx]
                
                # Find k-points near the target energy
                energy_diffs = np.abs(band_energies - target_energy)
                closest_k = np.argmin(energy_diffs)
                
                if energy_diffs[closest_k] < min_diff:
                    min_diff = energy_diffs[closest_k]
                    best_band_idx = band_idx
                    
                    # Find a range of k-points around the minimum/maximum
                    k_start = max(0, closest_k - 5)
                    k_end = min(len(k_points), closest_k + 6)
                    best_k_range = (k_start, k_end)
            
            if best_k_range is None or min_diff > 0.01:  # Energy difference > 0.27 eV
                return None
            
            # Extract the relevant k-points and energies
            k_start, k_end = best_k_range
            k_local = k_points[k_start:k_end]
            E_local = energies[k_start:k_end, best_band_idx]
            
            if len(k_local) < 5:  # Need at least 5 points for good curvature
                return None
            
            # Calculate second derivative using central differences
            # d²E/dk² ≈ (E[i+1] - 2*E[i] + E[i-1]) / (Δk)²
            dk = np.diff(k_local).mean()  # Average k-spacing
            
            if dk <= 0:
                return None
            
            # Calculate curvature at the center point
            center_idx = len(E_local) // 2
            if center_idx >= 1 and center_idx < len(E_local) - 1:
                d2E_dk2 = (E_local[center_idx + 1] - 2*E_local[center_idx] + E_local[center_idx - 1]) / (dk**2)
            else:
                return None
            
            # Effective mass: m* = ℏ² / |d²E/dk²|
            # In atomic units: ℏ = 1, so m* = 1 / |d²E/dk²|
            if abs(d2E_dk2) > 1e-10:  # Avoid division by zero
                effective_mass = 1.0 / abs(d2E_dk2)
                
                # Sanity check: effective mass should be reasonable (0.01 to 10 m_e)
                if 0.01 <= effective_mass <= 10.0:
                    return effective_mass
            
            return None
            
        except Exception as e:
            print(f"Warning: Error in band curvature calculation: {e}")
            return None
    
    def calculate_transport_properties(self, classification_results: Dict, 
                                     effective_mass_results: Dict) -> Dict[str, Any]:
        """
        Calculate transport properties based on classification and effective masses.
        """
        transport_props = {}
        
        # Get classification and masses
        classification = classification_results.get('electronic_classification', 'unknown')
        gap_ev = classification_results.get('gap_ev', 0.0)
        electron_mass = effective_mass_results.get('electron_effective_mass')
        hole_mass = effective_mass_results.get('hole_effective_mass')
        
        # Mobility calculation (Drude model approximation)
        # μ = qτ/m* where τ is scattering time
        # Assume scattering time ~1e-14 s for order of magnitude (set for the
        # hole mobility too: a hole mass without an electron one raised)
        scattering_time = 1e-14  # seconds
        elementary_charge = 1.602e-19  # Coulombs
        if electron_mass is not None and electron_mass > 0:
            electron_mass_kg = electron_mass * 9.109e-31  # kg
            
            electron_mobility = (elementary_charge * scattering_time / electron_mass_kg) * 1e4  # cm²/(V·s)
            transport_props['electron_mobility_estimate'] = electron_mobility
        
        if hole_mass is not None and hole_mass > 0:
            hole_mass_kg = hole_mass * 9.109e-31  # kg
            hole_mobility = (elementary_charge * scattering_time / hole_mass_kg) * 1e4  # cm²/(V·s)
            transport_props['hole_mobility_estimate'] = hole_mobility
        
        # Conductivity classification
        if classification == 'metal':
            transport_props['conductivity_type'] = 'high'
            transport_props['conductivity_estimate'] = 'metallic'
        elif classification == 'semimetal':
            transport_props['conductivity_type'] = 'moderate'
            transport_props['conductivity_estimate'] = 'semimetallic'
        elif classification == 'semiconductor':
            if gap_ev < 1.0:
                transport_props['conductivity_type'] = 'moderate'
            elif gap_ev < 3.0:
                transport_props['conductivity_type'] = 'low'
            else:
                transport_props['conductivity_type'] = 'very_low'
            transport_props['conductivity_estimate'] = 'semiconducting'
        elif classification == 'insulator':
            transport_props['conductivity_type'] = 'very_low'
            transport_props['conductivity_estimate'] = 'insulating'
        
        # Seebeck coefficient estimation
        if gap_ev is not None:
            transport_props['seebeck_coefficient_estimate'] = gap_ev * 80  # μV/K (rough estimate)
        
        return transport_props
    
    def analyze_material(self, band_file: Optional[Path] = None, 
                        doss_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Complete electronic structure analysis using BAND and DOSS data.
        
        Returns comprehensive dictionary with all electronic properties.
        """
        results = {
            'analysis_method': 'advanced_band_dos_analysis',
            'files_analyzed': {
                'band_file': str(band_file) if band_file else None,
                'doss_file': str(doss_file) if doss_file else None
            }
        }
        
        # Initialize default values
        E, g = np.array([]), np.array([])
        Ev_max, Ec_min = -np.inf, np.inf
        k_points, energies = np.array([]), np.array([])
        
        # Read DOS data if available
        if doss_file and doss_file.exists():
            E, g = self.read_dos_data(doss_file)
            results['dos_data_available'] = len(E) > 0
            if len(E) > 0:
                results['dos_energy_range'] = [float(E.min()), float(E.max())]
                results['dos_points'] = len(E)
        
        # Read BAND data if available
        if band_file and band_file.exists():
            Ev_max, Ec_min, k_points, energies = self.read_band_data(band_file)
            results['band_data_available'] = len(k_points) > 0
            if len(k_points) > 0:
                results['band_k_points'] = len(k_points)
                results['band_structure_range'] = [float(k_points.min()), float(k_points.max())]
        
        # Electronic classification
        classification_results = self.classify_electronic_behavior(E, g, Ev_max, Ec_min)
        results.update(classification_results)
        
        # Effective mass calculation
        effective_mass_results = self.calculate_effective_mass(k_points, energies, Ev_max, Ec_min)
        results.update(effective_mass_results)
        
        # Transport properties
        transport_results = self.calculate_transport_properties(classification_results, effective_mass_results)
        results.update(transport_results)
        
        # Additional derived properties
        results['is_semimetal'] = classification_results.get('electronic_classification') == 'semimetal'
        results['has_real_effective_mass'] = (effective_mass_results.get('electron_effective_mass') is not None or 
                                            effective_mass_results.get('hole_effective_mass') is not None)
        
        return results


def main():
    """Command line interface for testing."""
    if len(sys.argv) == 3:
        # Both BAND and DOSS files provided
        band_file = Path(sys.argv[1])
        doss_file = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        # Only one file provided - determine type
        input_file = Path(sys.argv[1])
        if 'BAND' in input_file.name.upper():
            band_file = input_file
            doss_file = None
        elif 'DOSS' in input_file.name.upper():
            band_file = None
            doss_file = input_file
        else:
            print("Error: Cannot determine file type from filename")
            sys.exit(1)
    else:
        print("Usage: python advanced_electronic_analyzer.py [BAND_file] [DOSS_file]")
        print("   or: python advanced_electronic_analyzer.py <BAND_or_DOSS_file>")
        sys.exit(1)
    
    # Perform analysis
    analyzer = AdvancedElectronicAnalyzer()
    results = analyzer.analyze_material(band_file, doss_file)
    
    # Print results
    print(f"\n{'='*60}")
    print("ADVANCED ELECTRONIC STRUCTURE ANALYSIS")
    print(f"{'='*60}")
    
    print(f"Files analyzed:")
    if band_file:
        print(f"  BAND: {band_file.name}")
    if doss_file:
        print(f"  DOSS: {doss_file.name}")
    
    print(f"\nElectronic Classification: {(results.get('electronic_classification') or 'unknown').upper()}")
    if results.get('gap_ev') is not None:
        print(f"Band Gap: {results['gap_ev']:.4f} eV ({results['gap_hartree']:.4f} Ha)")
    else:
        print("Band Gap: unknown (no band structure)")
    
    if results.get('dos_data_available'):
        print(f"DOS at Fermi Level: {results.get('dos_at_fermi', 0):.3e}")
        print(f"DOS Threshold: {results.get('dos_threshold', 0):.3e}")
    
    if results.get('has_real_effective_mass'):
        print(f"\nEffective Masses (m_e units):")
        if results.get('electron_effective_mass'):
            print(f"  Electron: {results['electron_effective_mass']:.3f}")
        if results.get('hole_effective_mass'):
            print(f"  Hole: {results['hole_effective_mass']:.3f}")
    
    if results.get('electron_mobility_estimate'):
        print(f"\nTransport Properties:")
        print(f"  Electron Mobility: {results['electron_mobility_estimate']:.1f} cm²/(V·s)")
        if results.get('hole_mobility_estimate'):
            print(f"  Hole Mobility: {results['hole_mobility_estimate']:.1f} cm²/(V·s)")
        print(f"  Conductivity Type: {results.get('conductivity_type', 'unknown')}")


if __name__ == "__main__":
    main()