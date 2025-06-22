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

class AdvancedElectronicAnalyzer:
    """Advanced electronic structure analysis using CRYSTAL BAND and DOSS data."""
    
    def __init__(self):
        self.ha_to_ev = 27.2114  # Hartree to eV conversion
        self.bohr_to_angstrom = 0.529177  # Bohr to Angstrom conversion
        
    def read_dos_data(self, doss_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read DOSS.DAT file and return energies and total DOS.
        
        Returns:
            E: Energy array in Hartree (relative to Fermi level = 0)
            g: Total DOS array in states/Ha/cell
        """
        E, g = [], []
        
        try:
            with open(doss_file, 'r') as fh:
                for line in fh:
                    # Skip comments and empty lines
                    if line.startswith(('#', '@')) or not line.strip():
                        continue
                    
                    try:
                        vals = list(map(float, line.split()))
                        if len(vals) >= 2:
                            E.append(vals[0])  # Energy in Hartree
                            g.append(sum(vals[1:]))  # Total DOS (sum all spin channels)
                    except ValueError:
                        continue
                        
        except FileNotFoundError:
            print(f"Warning: DOSS file not found: {doss_file}")
            return np.array([]), np.array([])
        except Exception as e:
            print(f"Warning: Error reading DOSS file {doss_file}: {e}")
            return np.array([]), np.array([])
            
        return np.asarray(E), np.asarray(g)
    
    def read_band_data(self, band_file: Path) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Read BAND.DAT file and return band edge information.
        
        Returns:
            Ev_max: Valence band maximum in Hartree
            Ec_min: Conduction band minimum in Hartree  
            k_points: k-point coordinates
            energies: Energy bands at each k-point
        """
        Ev_max = -np.inf
        Ec_min = np.inf
        k_points = []
        all_energies = []
        
        try:
            with open(band_file, 'r') as fh:
                for line in fh:
                    # Skip comments and empty lines
                    if line.startswith(('#', '@')) or not line.strip():
                        continue
                    
                    try:
                        vals = list(map(float, line.split()))
                        if len(vals) >= 2:
                            k_coord = vals[0]  # k-point coordinate
                            energies = vals[1:]  # Energy bands in Hartree
                            
                            k_points.append(k_coord)
                            all_energies.append(energies)
                            
                            # Find band edges relative to E_F = 0
                            for E in energies:
                                if E < 0 and E > Ev_max:  # Highest occupied
                                    Ev_max = E
                                if E > 0 and E < Ec_min:  # Lowest unoccupied
                                    Ec_min = E
                                    
                    except ValueError:
                        continue
                        
        except FileNotFoundError:
            print(f"Warning: BAND file not found: {band_file}")
            return -np.inf, np.inf, np.array([]), np.array([])
        except Exception as e:
            print(f"Warning: Error reading BAND file {band_file}: {e}")
            return -np.inf, np.inf, np.array([]), np.array([])
            
        return Ev_max, Ec_min, np.array(k_points), np.array(all_energies)
    
    def classify_electronic_behavior(self, E: np.ndarray, g: np.ndarray, 
                                   Ev_max: float, Ec_min: float, 
                                   gcrit_factor: float = 0.05) -> Dict[str, Any]:
        """
        Classify electronic behavior using DOS analysis.
        
        Based on the sophisticated algorithm from your provided code.
        """
        classification_results = {}
        
        if len(E) == 0 or len(g) == 0:
            return {
                'electronic_classification': 'unknown',
                'gap_hartree': 0.0,
                'gap_ev': 0.0,
                'dos_at_fermi': 0.0,
                'dos_threshold': 0.0,
                'classification_method': 'no_dos_data'
            }
        
        # Find DOS at Fermi level (E = 0)
        Ef_index = np.argmin(np.abs(E))
        g_Ef = g[Ef_index]
        
        # Calculate mean DOS and threshold
        g_positive = g[g > 0]
        g_mean = g_positive.mean() if len(g_positive) > 0 else 0.0
        g_crit = gcrit_factor * g_mean
        
        # Estimate gap from DOS data
        try:
            # First energy above E_F with finite DOS
            Ec_dos = E[(E > 0) & (g > 0)][0] if np.any((E > 0) & (g > 0)) else np.inf
            # Last energy below E_F with finite DOS  
            Ev_dos = E[(E < 0) & (g > 0)][-1] if np.any((E < 0) & (g > 0)) else -np.inf
            gap_dos = Ec_dos - Ev_dos if (Ec_dos != np.inf and Ev_dos != -np.inf) else 0.0
        except IndexError:
            gap_dos = 0.0
        
        # Use band structure gap if available, otherwise DOS gap
        if Ev_max != -np.inf and Ec_min != np.inf:
            gap = Ec_min - Ev_max
            gap_source = "band_structure"
        else:
            gap = gap_dos
            gap_source = "dos_analysis"
        
        # Classification logic
        if gap > 0.001:  # 0.001 Ha ≈ 0.027 eV threshold for clear gap
            if gap > 0.15:  # 0.15 Ha ≈ 4.0 eV
                classification = "insulator"
            else:
                classification = "semiconductor"
        elif g_Ef > g_crit:
            classification = "metal"
        else:
            classification = "semimetal"
        
        return {
            'electronic_classification': classification,
            'gap_hartree': gap,
            'gap_ev': gap * self.ha_to_ev,
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
        if electron_mass is not None and electron_mass > 0:
            # Assume scattering time ~1e-14 s for order of magnitude
            scattering_time = 1e-14  # seconds
            elementary_charge = 1.602e-19  # Coulombs
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
        else:  # insulator
            transport_props['conductivity_type'] = 'very_low'
            transport_props['conductivity_estimate'] = 'insulating'
        
        # Seebeck coefficient estimation
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
    
    print(f"\nElectronic Classification: {results.get('electronic_classification', 'unknown').upper()}")
    print(f"Band Gap: {results.get('gap_ev', 0):.4f} eV ({results.get('gap_hartree', 0):.4f} Ha)")
    
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