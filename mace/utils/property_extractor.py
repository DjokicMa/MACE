#!/usr/bin/env python3
"""
CRYSTAL Property Extractor
==========================
Comprehensive property extraction from CRYSTAL output files to populate the properties table.

Extracts:
- Structural properties: lattice parameters, atomic positions, cell volumes, densities
- Electronic properties: band gaps (direct/indirect, alpha/beta), total energies, D3 corrections
- Population analysis: Mulliken charges, overlap populations (alpha+beta, alpha-beta)
- Geometry optimization: initial vs final geometries, convergence information
- Crystallographic information: primitive vs crystallographic cells, space groups

Usage:
  python crystal_property_extractor.py --output-file file.out [--db-path materials.db]
  python crystal_property_extractor.py --scan-directory /path/to/outputs [--db-path materials.db]
"""

import os
import sys
import re
import math
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from mace.constants import HARTREE_TO_EV

# Import MACE components
try:
    from mace.database.materials import MaterialDatabase
except ImportError as e:
    print(f"Error importing MaterialDatabase: {e}")
    sys.exit(1)


class CrystalPropertyExtractor:
    """Extract comprehensive properties from CRYSTAL output files."""
    
    def __init__(self, db_path: str = "materials.db", enable_tracking: bool = True):
        # The database is created lazily (on first real use) so pure parsing —
        # e.g. extracting properties from a single .out with enable_tracking=False
        # — never touches or creates materials.db. Pass enable_tracking=False for
        # offline single-file extraction.
        self.db_path = db_path
        self.enable_tracking = enable_tracking
        self._db = None
        self.properties = []

    @property
    def db(self):
        """Lazily-opened MaterialDatabase, or None when tracking is disabled."""
        if self._db is None and self.enable_tracking and self.db_path:
            self._db = MaterialDatabase(self.db_path)
        return self._db
        
    def extract_all_properties(self, output_file: Path, material_id: str = None, calc_id: str = None) -> Dict[str, Any]:
        """Extract all properties from a CRYSTAL output file."""
        print(f"🔍 Extracting properties from: {output_file.name}")
        
        if not output_file.exists():
            print(f"❌ Output file not found: {output_file}")
            return {}
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return {}
        
        # Auto-detect material_id and calc_id if not provided
        if not material_id:
            material_id = self._extract_material_id_from_filename(output_file)
        if not calc_id:
            calc_id = self._find_calc_id_for_output(output_file)
        
        properties = {}
        
        # Extract different property categories
        properties.update(self._extract_structural_properties(content))
        properties.update(self._extract_electronic_properties(content))
        properties.update(self._extract_population_analysis(content))
        properties.update(self._extract_energy_properties(content))
        properties.update(self._extract_geometry_optimization(content))
        properties.update(self._extract_crystallographic_info(content))
        properties.update(self._extract_neighbor_information(content))
        properties.update(self._extract_computational_properties(content))
        properties.update(self._extract_band_structure_properties(content, output_file))
        properties.update(self._extract_dos_properties(content, output_file))
        properties.update(self._extract_frequency_properties(content))
        properties.update(self._extract_transport_properties(content, output_file))
        # CHARGE+POTENTIAL (CRYSTAL ECH3/POT3): grid metadata + references to the
        # generated .CUBE grid files. Additive and self-skipping -- returns {} when
        # the .out has no ECH3/POT3 markers, so non-charge calcs are untouched.
        properties.update(self._extract_charge_potential_properties(content, output_file))

        # Extract SCF settings from output
        properties.update(self._extract_scf_settings(content))
        
        # Add electronic classification based on band gap
        properties.update(self._classify_electronic_properties(properties))
        
        # Advanced electronic analysis using BAND/DOSS data if available
        properties.update(self._extract_advanced_band_dos_analysis(output_file, properties))
        
        # Add metadata
        properties['_metadata'] = {
            'material_id': material_id,
            'calc_id': calc_id,
            'output_file': str(output_file),
            'extracted_at': datetime.now().isoformat(),
            'extractor_version': '1.0'
        }
        
        # Process population analysis data if available. Package-qualified
        # import first: the bare name only resolves when mace/utils is on
        # sys.path directly, so in every production context the ImportError
        # was silently swallowed and population processing never ran.
        try:
            try:
                from mace.utils.population_analysis_processor import PopulationAnalysisProcessor
            except ImportError:
                from population_analysis_processor import PopulationAnalysisProcessor
            processor = PopulationAnalysisProcessor()
            
            # Check if we have population analysis data to process
            pop_data = {}
            for key, value in properties.items():
                if 'mulliken' in key or 'overlap' in key:
                    pop_data[key] = value
            
            if pop_data:
                processed_pop = processor.process_population_data(pop_data)
                
                # Add processed data as new properties
                for key, value in processed_pop.items():
                    if key != 'error':
                        properties[f'processed_{key}'] = value
                        
        except ImportError:
            pass  # Population processor not available
        except Exception as e:
            print(f"Warning: Error processing population analysis: {e}")
        
        return properties
    
    def _extract_structural_properties(self, content: str) -> Dict[str, Any]:
        """Extract structural properties."""
        props = {}
        
        # Extract both initial and final geometries
        initial_geometry = self._extract_initial_geometry(content)
        if initial_geometry:
            for key, value in initial_geometry.items():
                props[f'initial_{key}'] = value
        
        final_geometry = self._extract_final_geometry(content)
        if final_geometry:
            for key, value in final_geometry.items():
                props[f'final_{key}'] = value
        
        # Use final geometry as default if no prefix specified
        if final_geometry:
            props.update(final_geometry)
        elif initial_geometry:
            props.update(initial_geometry)
        
        # Number of atoms
        atoms_match = re.search(r'ATOMS IN THE UNIT CELL:\s*(\d+)', content)
        if atoms_match:
            props['atoms_in_unit_cell'] = int(atoms_match.group(1))
        
        return props
    
    def _extract_initial_geometry(self, content: str) -> Dict[str, Any]:
        """Extract initial geometry from beginning of output file."""
        props = {}
        
        # Look for initial lattice parameters (before optimization)
        # This appears early in the file in the geometry setup section
        initial_primitive_match = re.search(
            r'LATTICE PARAMETERS\s+\(ANGSTROMS AND DEGREES\)\s+-\s+PRIMITIVE CELL\s*\n'
            r'\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s+VOLUME\s*\n'
            r'\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)',
            content
        )
        
        if initial_primitive_match:
            props.update({
                'primitive_a': float(initial_primitive_match.group(1)),
                'primitive_b': float(initial_primitive_match.group(2)),
                'primitive_c': float(initial_primitive_match.group(3)),
                'primitive_alpha': float(initial_primitive_match.group(4)),
                'primitive_beta': float(initial_primitive_match.group(5)),
                'primitive_gamma': float(initial_primitive_match.group(6)),
                'primitive_cell_volume': float(initial_primitive_match.group(7))
            })
        else:
            # Alternative pattern for early geometry information
            alt_primitive_match = re.search(
                r'PRIMITIVE CELL.*?VOLUME=\s*([\d.]+).*?DENSITY\s*([\d.]+)\s*g/cm\^3.*?'
                r'A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*\n\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)',
                content[:5000], re.DOTALL  # Look only in first part of file
            )
            
            if alt_primitive_match:
                props.update({
                    'primitive_cell_volume': float(alt_primitive_match.group(1)),
                    'density': float(alt_primitive_match.group(2)),
                    'primitive_a': float(alt_primitive_match.group(3)),
                    'primitive_b': float(alt_primitive_match.group(4)),
                    'primitive_c': float(alt_primitive_match.group(5)),
                    'primitive_alpha': float(alt_primitive_match.group(6)),
                    'primitive_beta': float(alt_primitive_match.group(7)),
                    'primitive_gamma': float(alt_primitive_match.group(8))
                })
        
        # Initial crystallographic cell
        initial_crystal_match = re.search(
            r'LATTICE PARAMETERS\s+\(ANGSTROMS AND DEGREES\)\s+-\s+CONVENTIONAL CELL\s*\n'
            r'\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*\n'
            r'\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)',
            content[:5000]  # Look in first part of file
        )
        
        if initial_crystal_match:
            props.update({
                'crystallographic_a': float(initial_crystal_match.group(1)),
                'crystallographic_b': float(initial_crystal_match.group(2)),
                'crystallographic_c': float(initial_crystal_match.group(3)),
                'crystallographic_alpha': float(initial_crystal_match.group(4)),
                'crystallographic_beta': float(initial_crystal_match.group(5)),
                'crystallographic_gamma': float(initial_crystal_match.group(6))
            })
        
        # Extract initial atomic positions
        initial_positions = self._extract_atomic_positions(content, search_final=False)
        if initial_positions:
            props['initial_atomic_positions'] = initial_positions
            props['initial_atoms_count'] = len(initial_positions)
        
        return props
    
    def _extract_final_geometry(self, content: str) -> Dict[str, Any]:
        """Extract final geometry (for optimization calculations)."""
        props = {}
        
        # Look for final optimized geometry
        if 'FINAL OPTIMIZED GEOMETRY' in content or 'OPT END - CONVERGED' in content:
            # Final primitive cell
            final_primitive_match = re.search(
                r'FINAL OPTIMIZED GEOMETRY.*?PRIMITIVE CELL.*?VOLUME=\s*([\d.]+).*?DENSITY\s*([\d.]+)\s*g/cm\^3.*?'
                r'A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*\n\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)',
                content, re.DOTALL
            )
            
            if final_primitive_match:
                props.update({
                    'primitive_cell_volume': float(final_primitive_match.group(1)),
                    'density': float(final_primitive_match.group(2)),
                    'primitive_a': float(final_primitive_match.group(3)),
                    'primitive_b': float(final_primitive_match.group(4)),
                    'primitive_c': float(final_primitive_match.group(5)),
                    'primitive_alpha': float(final_primitive_match.group(6)),
                    'primitive_beta': float(final_primitive_match.group(7)),
                    'primitive_gamma': float(final_primitive_match.group(8))
                })
            
            # Final crystallographic cell
            final_crystal_match = re.search(
                r'FINAL OPTIMIZED GEOMETRY.*?CRYSTALLOGRAPHIC CELL \(VOLUME=\s*([\d.]+)\).*?'
                r'A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*\n\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)',
                content, re.DOTALL
            )
            
            if final_crystal_match:
                props.update({
                    'crystallographic_cell_volume': float(final_crystal_match.group(1)),
                    'crystallographic_a': float(final_crystal_match.group(2)),
                    'crystallographic_b': float(final_crystal_match.group(3)),
                    'crystallographic_c': float(final_crystal_match.group(4)),
                    'crystallographic_alpha': float(final_crystal_match.group(5)),
                    'crystallographic_beta': float(final_crystal_match.group(6)),
                    'crystallographic_gamma': float(final_crystal_match.group(7))
                })
            
            # Extract final atomic positions
            final_positions = self._extract_atomic_positions(content, search_final=True)
            if final_positions:
                props['final_atomic_positions'] = final_positions
                props['final_atoms_count'] = len(final_positions)
                
                # Set default atomic positions to final (for backward compatibility)
                props['atomic_positions'] = final_positions
        
        return props
    
    def _extract_electronic_properties(self, content: str) -> Dict[str, Any]:
        """Extract electronic properties."""
        props = {}
        
        # Band gaps - handle both spin-polarized and non-spin-polarized
        alpha_gap_match = re.search(r'ALPHA BAND GAP:\s*([\d.]+)\s*eV', content)
        beta_gap_match = re.search(r'BETA BAND GAP:\s*([\d.]+)\s*eV', content)
        
        if alpha_gap_match and beta_gap_match:
            # Spin-polarized calculation
            props.update({
                'alpha_band_gap': float(alpha_gap_match.group(1)),
                'beta_band_gap': float(beta_gap_match.group(1)),
                'spin_polarized': True
            })
        else:
            # Non-spin-polarized - look for general band gap
            gap_matches = re.findall(r'(?:DIRECT|INDIRECT)?\s*(?:ENERGY\s+)?BAND GAP:\s*([\d.]+)\s*eV', content)
            if gap_matches:
                props['band_gap'] = float(gap_matches[-1])  # Take the last (final) value
                props['spin_polarized'] = False
        
        # Direct vs indirect band gap. The negative lookbehind keeps "DIRECT"
        # from matching inside "INDIRECT", which stored a spurious
        # direct_band_gap for every indirect-gap material; take the LAST
        # occurrence (earlier ones are pre-optimization SCF cycles).
        direct_matches = re.findall(r'(?<!IN)DIRECT ENERGY BAND GAP:\s*([\d.]+)\s*eV', content)
        if direct_matches:
            props['direct_band_gap'] = float(direct_matches[-1])
            props['band_gap_type'] = 'direct'

        indirect_matches = re.findall(r'INDIRECT ENERGY BAND GAP:\s*([\d.]+)\s*eV', content)
        if indirect_matches:
            props['indirect_band_gap'] = float(indirect_matches[-1])
            props['band_gap_type'] = 'indirect'
        
        # Add advanced electronic properties
        props.update(self._extract_advanced_electronic_properties(content))
        
        return props
    
    def _extract_advanced_electronic_properties(self, content: str) -> Dict[str, Any]:
        """Extract advanced electronic properties like effective mass and transport properties."""
        props = {}
        
        # Effective mass estimation from band gaps
        # This is a simplified estimation - real effective mass requires band structure data
        if 'band_gap' in content.lower():
            gap_matches = re.findall(r'(?:DIRECT|INDIRECT)?\s*(?:ENERGY\s+)?BAND GAP:\s*([\d.]+)\s*eV', content)
            if gap_matches:
                band_gap = float(gap_matches[-1])
                
                # Simplified effective mass estimation (placeholder for now)
                # Real implementation would analyze band curvature from BAND calculations
                if band_gap > 0:
                    # Rough estimation: smaller gaps often correlate with lighter masses
                    # This is very approximate and should be replaced with real band structure analysis
                    estimated_electron_mass = 0.1 + (band_gap / 10.0)  # in m_e units
                    estimated_hole_mass = 0.2 + (band_gap / 8.0)       # in m_e units
                    
                    props.update({
                        'estimated_electron_effective_mass': estimated_electron_mass,
                        'estimated_hole_effective_mass': estimated_hole_mass,
                        'effective_mass_method': 'gap_based_estimation'
                    })
        
        # Carrier mobility estimation (very simplified)
        if 'estimated_electron_effective_mass' in props:
            # Simple mobility estimation using μ = qτ/m* with assumed scattering time
            # This is a placeholder - real mobility requires detailed scattering analysis
            assumed_scattering_time = 1e-14  # seconds (rough assumption)
            electron_charge = 1.602e-19      # Coulombs
            electron_mass_kg = 9.109e-31     # kg
            
            m_star_electron = props['estimated_electron_effective_mass'] * electron_mass_kg
            m_star_hole = props['estimated_hole_effective_mass'] * electron_mass_kg
            
            # Mobility in cm²/(V·s)
            electron_mobility = (electron_charge * assumed_scattering_time / m_star_electron) * 1e4
            hole_mobility = (electron_charge * assumed_scattering_time / m_star_hole) * 1e4
            
            props.update({
                'estimated_electron_mobility': electron_mobility,
                'estimated_hole_mobility': hole_mobility,
                'mobility_method': 'effective_mass_estimation'
            })
        
        # Conductivity type from band gap
        if 'band_gap' in content.lower():
            gap_matches = re.findall(r'(?:DIRECT|INDIRECT)?\s*(?:ENERGY\s+)?BAND GAP:\s*([\d.]+)\s*eV', content)
            if gap_matches:
                band_gap = float(gap_matches[-1])
                if band_gap < 0.1:
                    props['conductivity_classification'] = 'metallic'
                elif band_gap < 3.0:
                    props['conductivity_classification'] = 'semiconducting'
                else:
                    props['conductivity_classification'] = 'insulating'
        
        # Dielectric properties (placeholder - would need specific CRYSTAL output)
        # This would be extracted from frequency calculations or dielectric tensor output
        props.update({
            'has_dielectric_data': False,
            'dielectric_extraction_note': 'requires_frequency_calculation'
        })
        
        return props
    
    def _extract_energy_properties(self, content: str) -> Dict[str, Any]:
        """Extract energy-related properties."""
        props = {}
        
        # Total energy (final value from DFT calculation)
        energy_matches = re.findall(r'TOTAL ENERGY\(DFT\)\(AU\)\s*\(\s*\d+\)\s*([-\d.E+]+)', content)
        if energy_matches:
            props['total_energy_au'] = float(energy_matches[-1])
            props['total_energy_ev'] = float(energy_matches[-1]) * HARTREE_TO_EV  # Hartree to eV
        
        # Alternative energy extraction patterns
        if not energy_matches:
            # From SCF convergence
            scf_energy_match = re.search(r'== SCF ENDED - CONVERGENCE ON ENERGY\s+E\(AU\)\s*([-\d.E+]+)', content)
            if scf_energy_match:
                props['total_energy_au'] = float(scf_energy_match.group(1))
                props['total_energy_ev'] = float(scf_energy_match.group(1)) * HARTREE_TO_EV
            else:
                # From final optimization energy
                final_energy_match = re.search(r'E\(AU\):\s*([-\d.E+]+)', content)
                if final_energy_match:
                    props['total_energy_au'] = float(final_energy_match.group(1))
                    props['total_energy_ev'] = float(final_energy_match.group(1)) * HARTREE_TO_EV
        
        # D3 dispersion correction — take the LAST occurrence; the first is
        # printed for the initial geometry and is inconsistent with the
        # final total energy (which is taken last)
        d3_matches = re.findall(r'D3 DISPERSION ENERGY \(AU\)\s*([-\d.E+]+)', content)
        d3_energy = None
        if d3_matches:
            d3_energy = float(d3_matches[-1])
            props['d3_dispersion_energy_au'] = d3_energy
            props['d3_dispersion_energy_ev'] = d3_energy * HARTREE_TO_EV

            # Total energy + dispersion (D3 only). Kept under its historical
            # name for back-compat; for 3C methods this is NOT the fully
            # corrected total (those also add gCP — see total_energy_corrected_au).
            total_plus_disp_matches = re.findall(r'TOTAL ENERGY \+ DISP \(AU\)\s*([-\d.E+]+)', content)
            if total_plus_disp_matches:
                props['total_energy_plus_d3_au'] = float(total_plus_disp_matches[-1])
                props['total_energy_plus_d3_ev'] = float(total_plus_disp_matches[-1]) * HARTREE_TO_EV
            elif 'total_energy_au' in props:
                # Calculate if not explicitly given
                props['total_energy_plus_d3_au'] = props['total_energy_au'] + d3_energy
                props['total_energy_plus_d3_ev'] = props['total_energy_plus_d3_au'] * HARTREE_TO_EV

        # gCP (geometrical counterpoise) correction — printed by 3C methods
        # (e.g. HSESOL3C) alongside D3. Omitting it silently understates the
        # corrected total by the gCP term (~0.17 AU ≈ 108 kcal/mol).
        gcp_matches = re.findall(r'GCP ENERGY \(AU\)\s*([-\d.E+]+)', content)
        gcp_energy = None
        if gcp_matches:
            gcp_energy = float(gcp_matches[-1])
            props['gcp_energy_au'] = gcp_energy
            props['gcp_energy_ev'] = gcp_energy * HARTREE_TO_EV

        # Fully corrected total energy = total + D3 + gCP, built from the
        # FINAL-geometry components (total_energy_au, d3_energy, gcp_energy are
        # all taken as last occurrences = the converged geometry). We do NOT
        # prefer CRYSTAL's printed "TOTAL ENERGY + DISP + GCP" line: in an OPT
        # that combined line is emitted only for the INITIAL geometry and is
        # never re-printed after OPT END, so trusting it yields a stale total
        # (verified: 4LG OPT off by ~0.094 AU). For single-geometry SP runs the
        # computed sum matches the printed line to ~1e-10. Only fall back to the
        # printed line when the components themselves are unavailable.
        if 'total_energy_au' in props and (d3_energy is not None or gcp_energy is not None):
            corrected = props['total_energy_au'] + (d3_energy or 0.0) + (gcp_energy or 0.0)
            props['total_energy_corrected_au'] = corrected
            props['total_energy_corrected_ev'] = corrected * HARTREE_TO_EV
        else:
            corrected_line = (re.findall(r'TOTAL ENERGY \+ DISP \+ GCP \(AU\)\s*([-\d.E+]+)', content)
                              or re.findall(r'TOTAL ENERGY \+ DISP \(AU\)\s*([-\d.E+]+)', content))
            if corrected_line:
                props['total_energy_corrected_au'] = float(corrected_line[-1])
                props['total_energy_corrected_ev'] = float(corrected_line[-1]) * HARTREE_TO_EV

        # Extract energy components if available
        energy_components = self._extract_energy_components(content)
        props.update(energy_components)
        
        return props
    
    def _extract_energy_components(self, content: str) -> Dict[str, Any]:
        """Extract detailed energy components."""
        components = {}
        
        # Look for energy breakdown section
        if '+++ ENERGIES IN A.U. +++' in content:
            component_patterns = {
                'kinetic_energy_au': r'KINETIC ENERGY\s*([-\d.E+]+)',
                'electron_electron_energy_au': r'TOTAL E-E\s*([-\d.E+]+)',
                'electron_nuclear_energy_au': r'TOTAL E-N \+ N-E\s*([-\d.E+]+)',
                'nuclear_nuclear_energy_au': r'TOTAL N-N\s*([-\d.E+]+)',
                'exchange_energy_au': r'EXCHANGE ENERGY:\s*([-\d.E+]+)',
                'correlation_energy_au': r'CORRELATION ENERGY:\s*([-\d.E+]+)'
            }
            
            for prop_name, pattern in component_patterns.items():
                match = re.search(pattern, content)
                if match:
                    au_value = float(match.group(1))
                    components[prop_name] = au_value
                    # Also add eV version
                    ev_name = prop_name.replace('_au', '_ev')
                    components[ev_name] = au_value * HARTREE_TO_EV
        
        return components
    
    def _extract_population_analysis(self, content: str) -> Dict[str, Any]:
        """Extract Mulliken population analysis."""
        props = {}
        
        # Check for spin-resolved sections (both alpha+beta and alpha-beta)
        has_alpha_beta = "ALPHA+BETA ELECTRONS" in content
        has_alpha_minus_beta = "ALPHA-BETA ELECTRONS" in content
        is_spin_polarized = has_alpha_beta and has_alpha_minus_beta
        
        # Extract alpha+beta populations when available (even if not fully spin-polarized)
        if has_alpha_beta:
            alpha_beta_mulliken = self._extract_mulliken_section(content, "ALPHA+BETA ELECTRONS")
            alpha_beta_overlap = self._extract_overlap_populations(content, "ALPHA+BETA ELECTRONS")
            
            if alpha_beta_mulliken:
                props['mulliken_alpha_plus_beta'] = alpha_beta_mulliken
            if alpha_beta_overlap:
                props['overlap_population_alpha_plus_beta'] = alpha_beta_overlap
        
        # Extract alpha-beta populations when available
        if has_alpha_minus_beta:
            alpha_minus_beta_mulliken = self._extract_mulliken_section(content, "ALPHA-BETA ELECTRONS")
            alpha_minus_beta_overlap = self._extract_overlap_populations(content, "ALPHA-BETA ELECTRONS")
            
            if alpha_minus_beta_mulliken:
                props['mulliken_alpha_minus_beta'] = alpha_minus_beta_mulliken
            if alpha_minus_beta_overlap:
                props['overlap_population_alpha_minus_beta'] = alpha_minus_beta_overlap
        
        # Always try to extract general populations as fallback
        # This ensures we get overlap data even if spin-resolved extraction fails
        general_mulliken = self._extract_general_mulliken_section(content)
        if general_mulliken and not has_alpha_beta:  # Only use general if no alpha+beta
            props['mulliken_population'] = general_mulliken
            
        # Extract general overlap populations as fallback
        general_overlap = self._extract_general_overlap_populations(content)
        if general_overlap and not has_alpha_beta:  # Only use general if no alpha+beta
            props['overlap_population'] = general_overlap
        
        props['is_spin_polarized'] = is_spin_polarized
        
        return props
    
    def _extract_geometry_optimization(self, content: str) -> Dict[str, Any]:
        """Extract geometry optimization information."""
        props = {}
        
        # Check if this was an optimization
        if 'OPTGEOM' in content or 'OPT END - CONVERGED' in content:
            props['calculation_type'] = 'geometry_optimization'
            
            # Convergence information. Runs that converge at the first point
            # print "OPT END - CONVERGED" without the convergence-tests line,
            # so treat either as converged.
            converged_match = re.search(r'CONVERGENCE TESTS SATISFIED AFTER\s+(\d+)\s+ENERGY AND GRADIENT CALCULATIONS', content)
            opt_end_match = re.search(r'OPT END - CONVERGED.*?POINTS\s+(\d+)', content)
            if converged_match:
                props['optimization_cycles'] = int(converged_match.group(1))
                props['optimization_converged'] = True
            elif opt_end_match:
                props['optimization_cycles'] = int(opt_end_match.group(1))
                props['optimization_converged'] = True
            else:
                props['optimization_converged'] = False

            # Final gradient — last occurrence; the first GRADIENT NORM line
            # is the initial geometry's
            grad_norm_matches = re.findall(r'GRADIENT NORM\s+([\d.E+-]+)', content)
            if grad_norm_matches:
                props['final_gradient_norm'] = float(grad_norm_matches[-1])
        
        # Check for single point calculation
        elif 'SINGLE POINT CALCULATION' in content or 'SCFDIR' in content:
            props['calculation_type'] = 'single_point'
        
        return props
    
    def _extract_crystallographic_info(self, content: str) -> Dict[str, Any]:
        """Extract crystallographic information."""
        props = {}
        
        # Space group information
        space_group_match = re.search(r'SPACE GROUP NUMBER\s+(\d+)', content)
        if space_group_match:
            props['space_group_number'] = int(space_group_match.group(1))
        
        # Crystal system
        if 'CUBIC' in content:
            props['crystal_system'] = 'cubic'
        elif 'TETRAGONAL' in content:
            props['crystal_system'] = 'tetragonal'
        elif 'ORTHORHOMBIC' in content:
            props['crystal_system'] = 'orthorhombic'
        elif 'HEXAGONAL' in content:
            props['crystal_system'] = 'hexagonal'
        elif 'TRIGONAL' in content:
            props['crystal_system'] = 'trigonal'
        elif 'MONOCLINIC' in content:
            props['crystal_system'] = 'monoclinic'
        elif 'TRICLINIC' in content:
            props['crystal_system'] = 'triclinic'
        
        # Centering code
        centering_match = re.search(r'CENTRING CODE\s+([\d/]+)', content)
        if centering_match:
            props['centering_code'] = centering_match.group(1)
        
        return props
    
    def _extract_neighbor_information(self, content: str) -> Dict[str, Any]:
        """Extract neighbor information from CRYSTAL output.

        OPT outputs print the neighbor table twice: once for the initial
        geometry and again after the final optimized geometry. The first
        table keeps the original property names; when a later table exists,
        it is stored under final_* property names.
        """
        props = {}

        neighbor_section_matches = list(re.finditer(
            r'NEIGHBORS OF THE NON-EQUIVALENT ATOMS.*?N = NUMBER OF NEIGHBORS AT DISTANCE R\s*\n\s*ATOM\s+N\s+R/ANG\s+R/AU\s+NEIGHBORS.*?\n(.*?)(?=\n\s*SYMMETRY|\n\s*TTTT|\n\s*MMMM|\n\s*[A-Z]{3,}|$)',
            content, re.DOTALL
        ))

        if not neighbor_section_matches:
            return props

        props.update(self._parse_neighbor_section(neighbor_section_matches[0].group(1).strip()))

        if len(neighbor_section_matches) > 1:
            final_props = self._parse_neighbor_section(neighbor_section_matches[-1].group(1).strip())
            props.update({f'final_{name}': value for name, value in final_props.items()})

        return props

    def _parse_neighbor_section(self, neighbor_content: str) -> Dict[str, Any]:
        """Parse a single neighbor analysis table into property values."""
        props = {}
        neighbors_data = []
        coordination_numbers = []
        bond_distances = []
        
        # Parse each line for neighbor information
        # Pattern: ATOM_NUM ELEMENT N_NEIGHBORS DISTANCE_ANG DISTANCE_AU NEIGHBOR_LIST
        lines = neighbor_content.split('\n')
        current_atom = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts with atom number and element.
            # Shell header rows carry decimal R/ANG and R/AU values; wrapped
            # continuation rows hold only integer cell indices, so requiring
            # decimal points avoids misreading 5-token continuation lines
            # (e.g. "2 C    0 0 1") as new shells with bogus distances.
            parts = line.split()
            if (len(parts) >= 5 and parts[0].isdigit() and parts[1].isalpha()
                    and '.' in parts[3] and '.' in parts[4]):
                try:
                    atom_num = int(parts[0])
                    element = parts[1]
                    n_neighbors = int(parts[2])
                    dist_ang = float(parts[3])
                    dist_au = float(parts[4])
                    
                    # Find or create atom entry
                    current_atom = None
                    for atom in neighbors_data:
                        if atom['atom_number'] == atom_num:
                            current_atom = atom
                            break
                    
                    if current_atom is None:
                        current_atom = {
                            'atom_number': atom_num,
                            'element': element,
                            'neighbor_shells': []
                        }
                        neighbors_data.append(current_atom)
                    
                    # Add this neighbor shell
                    neighbor_info = ' '.join(parts[5:]) if len(parts) > 5 else ''
                    neighbors = self._parse_neighbor_list(neighbor_info)
                    
                    shell = {
                        'n_neighbors': n_neighbors,
                        'distance_ang': dist_ang,
                        'distance_au': dist_au,
                        'neighbors': neighbors
                    }
                    current_atom['neighbor_shells'].append(shell)
                    
                    # Track statistics
                    coordination_numbers.append(n_neighbors)
                    if dist_ang > 0:
                        bond_distances.append(dist_ang)
                        
                except (ValueError, IndexError):
                    continue
            
            elif current_atom is not None and line and not line.startswith(' ' * 50):
                # This might be a continuation line with more neighbors
                if current_atom['neighbor_shells']:
                    last_shell = current_atom['neighbor_shells'][-1]
                    neighbors = self._parse_neighbor_list(line)
                    last_shell['neighbors'].extend(neighbors)
        
        if neighbors_data:
            props['neighbor_analysis'] = neighbors_data
            
            # Extract summary statistics
            if coordination_numbers:
                props['max_coordination_number'] = max(coordination_numbers)
                props['total_coordination_shells'] = len(coordination_numbers)
            
            if bond_distances:
                props['min_bond_distance_ang'] = min(bond_distances)
                props['max_bond_distance_ang'] = max(bond_distances)
        
        return props
    
    def _parse_neighbor_list(self, neighbor_text: str) -> List[Dict]:
        """Parse neighbor list from text like '2 C 0 0 0 2 C 1 0 0'."""
        neighbors = []
        parts = neighbor_text.split()
        
        # Process in groups of 5: atom_num element i j k
        i = 0
        while i + 4 < len(parts):
            try:
                atom_num = int(parts[i])
                element = parts[i + 1]
                cell_i = int(parts[i + 2])
                cell_j = int(parts[i + 3])
                cell_k = int(parts[i + 4])
                
                neighbors.append({
                    'neighbor_atom_number': atom_num,
                    'neighbor_element': element,
                    'cell_indices': [cell_i, cell_j, cell_k]
                })
                
                i += 5
            except (ValueError, IndexError):
                i += 1
                continue
        
        return neighbors
    
    def _extract_atomic_positions(self, content: str, search_final: bool = True) -> List[Dict]:
        """Extract atomic positions from geometry."""
        positions = []
        
        if search_final:
            # Look for final atomic positions
            # Stop at TRANSFORMATION too: capturing through to "T = ATOM"
            # (printed only after the crystallographic table) spanned both
            # the primitive and crystallographic tables, doubling every atom
            final_geom_match = re.search(
                r'FINAL OPTIMIZED GEOMETRY.*?ATOM\s+X/A\s+Y/B\s+Z/C\s*\n\s*\*+\s*\n(.*?)\n\s*(?:T = ATOM|TRANSFORMATION)',
                content, re.DOTALL
            )
            
            if final_geom_match:
                atom_lines = final_geom_match.group(1).strip().split('\n')
                positions = self._parse_atomic_position_lines(atom_lines)
        
        if not positions:
            # Look for initial/general atomic positions
            geom_patterns = [
                r'GEOMETRY FOR WAVE FUNCTION.*?ATOM\s+X/A\s+Y/B\s+Z/C\s*\n\s*\*+\s*\n(.*?)\n\s*(?:T = ATOM|TRANSFORMATION)',
                r'ATOMS IN THE ASYMMETRIC UNIT.*?ATOM\s+X/A\s+Y/B\s+Z/C\s*\n\s*\*+\s*\n(.*?)\n\s*(?:T = ATOM|TRANSFORMATION)',
                r'ATOM\s+X/A\s+Y/B\s+Z/C\s*\n\s*\*+\s*\n(.*?)\n\s*(?:T = ATOM|TRANSFORMATION)'
            ]
            
            for pattern in geom_patterns:
                geom_match = re.search(pattern, content, re.DOTALL)
                if geom_match:
                    atom_lines = geom_match.group(1).strip().split('\n')
                    positions = self._parse_atomic_position_lines(atom_lines)
                    if positions:
                        break
        
        return positions
    
    def _parse_atomic_position_lines(self, atom_lines: List[str]) -> List[Dict]:
        """Parse atomic position lines into structured data."""
        positions = []
        
        for line in atom_lines:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            parts = line.split()
            if len(parts) >= 7:  # Need at least 7 parts for CRYSTAL format
                try:
                    # CRYSTAL format: ATOM_NUM ATOM_TYPE ATOMIC_NUM ELEMENT X Y Z
                    positions.append({
                        'atom_number': int(parts[0]),
                        'atom_type': parts[1],
                        'atomic_number': int(parts[2]),
                        'element': parts[3],
                        'x': float(parts[4]),
                        'y': float(parts[5]),
                        'z': float(parts[6])
                    })
                except (ValueError, IndexError):
                    continue
            elif len(parts) >= 6:  # Fallback for simpler format
                try:
                    positions.append({
                        'atom_number': int(parts[0]),
                        'atom_type': parts[1],
                        'element': parts[2],
                        'x': float(parts[3]),
                        'y': float(parts[4]),
                        'z': float(parts[5])
                    })
                except (ValueError, IndexError):
                    continue
        
        return positions
    
    def _extract_mulliken_section(self, content: str, section_type: str) -> Dict[str, Any]:
        """Extract Mulliken population analysis for a specific section."""
        # The section header line is printed IMMEDIATELY above the Mulliken
        # header; a loose `.*?` join matched the SPINLOCK echo line
        # ("ALPHA-BETA ELECTRONS LOCKED TO ...") and then skipped forward to
        # the alpha+beta table, storing full populations as spin densities.
        # The capture is bounded so it cannot run into the other spin section
        # or the overlap tables (which doubled atom counts), and the LAST
        # PPAN print is used (earlier ones belong to earlier SCF/geometries).
        escaped_section = re.escape(section_type)
        pattern = (f'{escaped_section}\\s*\\n\\s*MULLIKEN POPULATION ANALYSIS[^\\n]*?'
                   f'NO. OF ELECTRONS\\s+([\\d.-]+)(.*?)'
                   f'(?=ALPHA\\+BETA ELECTRONS|ALPHA-BETA ELECTRONS|'
                   f'OVERLAP POPULATION|MMMMM|TTTTTT|$)')
        matches = list(re.finditer(pattern, content, re.DOTALL))

        if not matches:
            return None

        match = matches[-1]
        total_electrons = float(match.group(1))
        section_content = match.group(2)
        
        # The section holds two tables with identical atom rows (A.O.
        # POPULATION followed by SHELL POPULATION) — parse only the first
        # so atoms aren't double-counted
        section_content = re.split(r'ATOM\s+Z\s+CHARGE\s+SHELL\s+POPULATION', section_content)[0]

        # Extract atomic charges and populations
        atoms = []
        atom_matches = re.finditer(r'(\d+)\s+(\w+)\s+(\d+)\s+([\d.-]+)(.*?)(?=\n\s*\d+|\n\s*ATOM|\n\s*OVERLAP|$)', section_content, re.DOTALL)
        
        for atom_match in atom_matches:
            atom_num = int(atom_match.group(1))
            element = atom_match.group(2)
            atomic_num = int(atom_match.group(3))
            charge = float(atom_match.group(4))
            
            # Extract orbital populations
            orbital_text = atom_match.group(5)
            orbitals = [float(x) for x in re.findall(r'[\d.-]+', orbital_text) if x.replace('.', '').replace('-', '').isdigit()]
            
            atoms.append({
                'atom_number': atom_num,
                'element': element,
                'atomic_number': atomic_num,
                'mulliken_charge': charge,
                'orbital_populations': orbitals
            })
        
        return {
            'total_electrons': total_electrons,
            'atoms': atoms
        }
    
    def _extract_overlap_populations(self, content: str, section_type: str) -> List[Dict]:
        """Extract overlap population data for spin-resolved sections."""
        # First, check if we're in a spin-resolved section
        if section_type in content:
            # Find the section and then look for overlap population after it
            section_start = content.find(section_type)
            remaining_content = content[section_start:]
            
            # Look for overlap population section in the remaining content
            overlap_match = re.search(r'OVERLAP POPULATION CONDENSED TO ATOMS(.*?)(?=EIGENVECTORS|MMMMM|TTTTTT|ALPHA|BETA|$)', 
                                    remaining_content, re.DOTALL)
        else:
            # Fallback to general search
            overlap_match = re.search(r'OVERLAP POPULATION CONDENSED TO ATOMS(.*?)(?=EIGENVECTORS|MMMMM|TTTTTT|$)', 
                                    content, re.DOTALL)
        
        if not overlap_match:
            return []
        
        overlap_content = overlap_match.group(1)
        overlaps = []
        
        # Parse overlap data with improved pattern
        # Look for ATOM A lines followed by neighbor data
        atom_pattern = r'ATOM A\s+(\d+)\s+(\w+)\s+ATOM B.*?\n(.*?)(?=ATOM A|\Z)'
        atom_matches = re.finditer(atom_pattern, overlap_content, re.DOTALL)
        
        for atom_match in atom_matches:
            atom_a_num = int(atom_match.group(1))
            atom_a_element = atom_match.group(2)
            neighbor_section = atom_match.group(3)
            
            # Parse neighbor interactions with more flexible parsing
            neighbors = []
            neighbor_lines = neighbor_section.strip().split('\n')
            
            for line in neighbor_lines:
                line = line.strip()
                if not line or 'ATOM' in line or 'CELL' in line:
                    continue
                
                # Look for neighbor line format: ATOM_NUM ELEMENT (CELL_I CELL_J CELL_K) DIST_AU DIST_ANG OVERLAP
                # or format: ATOM_NUM ELEMENT CELL_I CELL_J CELL_K DIST_AU DIST_ANG OVERLAP
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        # Handle format with parentheses: 2 C ( 0 0 0) 2.922 1.546 0.293
                        if '(' in line and ')' in line:
                            cell_match = re.search(r'\(\s*([-\d]+)\s+([-\d]+)\s+([-\d]+)\s*\)', line)
                            if cell_match:
                                atom_b_num = int(parts[0])
                                atom_b_element = parts[1]
                                cell_i = int(cell_match.group(1))
                                cell_j = int(cell_match.group(2))
                                cell_k = int(cell_match.group(3))
                                
                                # Find numeric values after the parentheses
                                after_paren = line.split(')')[1].strip().split()
                                if len(after_paren) >= 3:
                                    distance_au = float(after_paren[0])
                                    distance_ang = float(after_paren[1])
                                    overlap_pop = float(after_paren[2])
                                    
                                    neighbors.append({
                                        'atom_b_number': atom_b_num,
                                        'atom_b_element': atom_b_element,
                                        'cell_indices': [cell_i, cell_j, cell_k],
                                        'distance_au': distance_au,
                                        'distance_ang': distance_ang,
                                        'overlap_population': overlap_pop
                                    })
                        
                        # Handle format without parentheses: ATOM_NUM ELEMENT CELL_I CELL_J CELL_K DIST_AU DIST_ANG OVERLAP
                        elif len(parts) >= 8:
                            neighbors.append({
                                'atom_b_number': int(parts[0]),
                                'atom_b_element': parts[1],
                                'cell_indices': [int(parts[2]), int(parts[3]), int(parts[4])],
                                'distance_au': float(parts[5]),
                                'distance_ang': float(parts[6]),
                                'overlap_population': float(parts[7])
                            })
                    except (ValueError, IndexError):
                        continue
            
            if neighbors:  # Only add atoms that have neighbor data
                overlaps.append({
                    'atom_a_number': atom_a_num,
                    'atom_a_element': atom_a_element,
                    'neighbors': neighbors
                })
        
        return overlaps
    
    def _extract_general_mulliken_section(self, content: str) -> Dict[str, Any]:
        """Extract general Mulliken population analysis for non-spin-polarized calculations."""
        # Look for the general Mulliken section
        pattern = r'MULLIKEN POPULATION ANALYSIS - NO\. OF ELECTRONS\s+([\d.-]+)(.*?)(?=OVERLAP POPULATION|EIGENVECTORS|$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return None
        
        total_electrons = float(match.group(1))
        section_content = match.group(2)
        
        # Extract atomic charges and populations
        atoms = []
        
        # Look for ATOM Z CHARGE A.O. POPULATION section
        ao_pattern = r'ATOM\s+Z\s+CHARGE\s+A\.O\.\s+POPULATION(.*?)(?=ATOM\s+Z\s+CHARGE\s+SHELL|$)'
        ao_match = re.search(ao_pattern, section_content, re.DOTALL)
        
        if ao_match:
            ao_content = ao_match.group(1)
            # Parse individual atoms
            atom_lines = [line.strip() for line in ao_content.split('\n') if line.strip()]
            
            for line in atom_lines:
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    try:
                        atom_num = int(parts[0])
                        element = parts[1]
                        atomic_num = int(parts[2])
                        charge = float(parts[3])
                        
                        # Extract orbital populations (remaining numbers)
                        orbitals = [float(x) for x in parts[4:] if x.replace('.', '').replace('-', '').isdigit()]
                        
                        atoms.append({
                            'atom_number': atom_num,
                            'element': element,
                            'atomic_number': atomic_num,
                            'mulliken_charge': charge,
                            'orbital_populations': orbitals
                        })
                    except (ValueError, IndexError):
                        continue
        
        return {
            'total_electrons': total_electrons,
            'atoms': atoms
        }
    
    def _extract_general_overlap_populations(self, content: str) -> List[Dict]:
        """Extract general overlap populations for non-spin-polarized calculations."""
        overlaps = []
        
        # Look for the overlap population section
        pattern = r'OVERLAP POPULATION CONDENSED TO ATOMS.*?ATOM A\s+(\d+)\s+(\w+)\s+ATOM B.*?\n((?:.*?\([^)]*\).*?\n)+)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            atom_a_num = int(match.group(1))
            atom_a_element = match.group(2)
            overlap_data = match.group(3)
            
            # Parse neighbor interactions
            neighbors = []
            overlap_lines = overlap_data.strip().split('\n')
            
            for line in overlap_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for pattern: ATOM_NUM ELEMENT (CELL) DISTANCE_AU DISTANCE_ANG OVERLAP
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        # Extract cell indices from parentheses
                        cell_match = re.search(r'\(\s*([\d-]+)\s+([\d-]+)\s+([\d-]+)\s*\)', line)
                        if cell_match:
                            atom_b_num = int(parts[0])
                            atom_b_element = parts[1]
                            cell_i = int(cell_match.group(1))
                            cell_j = int(cell_match.group(2))
                            cell_k = int(cell_match.group(3))
                            
                            # Find distances and overlap in remaining parts
                            numeric_parts = [p for p in parts[2:] if p.replace('.', '').replace('-', '').isdigit()]
                            if len(numeric_parts) >= 3:
                                distance_au = float(numeric_parts[0])
                                distance_ang = float(numeric_parts[1])
                                overlap_pop = float(numeric_parts[2])
                                
                                neighbors.append({
                                    'atom_b_number': atom_b_num,
                                    'atom_b_element': atom_b_element,
                                    'cell_indices': [cell_i, cell_j, cell_k],
                                    'distance_au': distance_au,
                                    'distance_ang': distance_ang,
                                    'overlap_population': overlap_pop
                                })
                    except (ValueError, IndexError):
                        continue
            
            if neighbors:
                overlaps.append({
                    'atom_a_number': atom_a_num,
                    'atom_a_element': atom_a_element,
                    'neighbors': neighbors
                })
        
        return overlaps
    
    def _extract_material_id_from_filename(self, output_file: Path) -> str:
        """Extract material ID from filename."""
        filename = output_file.stem
        # Remove common suffixes
        for suffix in ['_opt', '_sp', '_band', '_doss', '_freq']:
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                break
        return filename
    
    def _find_calc_id_for_output(self, output_file: Path) -> Optional[str]:
        """Find the calculation ID associated with this output file."""
        if not self.enable_tracking or self.db is None:
            return None  # offline / untracked extraction: no DB to consult
        try:
            with self.db._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT calc_id FROM calculations WHERE output_file = ? OR work_dir = ?",
                    (str(output_file), str(output_file.parent))
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except:
            return None
    
    def save_properties_to_database(self, properties: Dict[str, Any]) -> int:
        """Save extracted properties to the database."""
        if not self.enable_tracking or self.db is None:
            return 0  # tracking disabled: nothing to persist
        if not properties or '_metadata' not in properties:
            return 0
        
        metadata = properties['_metadata']
        material_id = metadata['material_id']
        calc_id = metadata['calc_id']
        
        saved_count = 0
        
        # Save each property
        for prop_name, prop_value in properties.items():
            if prop_name.startswith('_'):
                continue  # Skip metadata
            
            # Determine property category
            category = self._categorize_property(prop_name)
            
            # Handle complex values (convert to JSON)
            if isinstance(prop_value, (dict, list)):
                value_text = json.dumps(prop_value)
                value_numeric = None
            elif isinstance(prop_value, (int, float)):
                value_numeric = float(prop_value)
                value_text = str(prop_value)
            else:
                value_numeric = None
                value_text = str(prop_value)
            
            # Determine units
            unit = self._get_property_unit(prop_name)
            
            try:
                # Check if property already exists. NULL-safe calc_id match:
                # untracked extractions have calc_id=None, and `calc_id = ?`
                # never matches NULL, so every re-extraction inserted a full
                # duplicate set of property rows.
                with self.db._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT property_id FROM properties WHERE material_id = ? AND property_name = ? "
                        "AND (calc_id = ? OR (calc_id IS NULL AND ? IS NULL))",
                        (material_id, prop_name, calc_id, calc_id)
                    )
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing property
                        conn.execute("""
                            UPDATE properties 
                            SET property_value = ?, property_value_text = ?, property_unit = ?,
                                extracted_at = ?, extractor_script = ?
                            WHERE property_id = ?
                        """, (value_numeric, value_text, unit, 
                              metadata['extracted_at'], 'crystal_property_extractor.py',
                              existing[0]))
                    else:
                        # Insert new property
                        conn.execute("""
                            INSERT INTO properties 
                            (material_id, calc_id, property_category, property_name, 
                             property_value, property_value_text, property_unit, 
                             extracted_at, extractor_script)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (material_id, calc_id, category, prop_name,
                              value_numeric, value_text, unit,
                              metadata['extracted_at'], 'crystal_property_extractor.py'))
                    
                    saved_count += 1
                    
            except Exception as e:
                print(f"⚠️  Error saving property {prop_name}: {e}")
        
        return saved_count
    
    def _extract_computational_properties(self, content: str) -> Dict[str, Any]:
        """Extract computational performance and timing properties."""
        props = {}
        
        # CPU time extraction
        cpu_time_match = re.search(r'TOTAL CPU TIME\s*[=:]\s*([\d.]+)', content, re.IGNORECASE)
        if cpu_time_match:
            props['total_cpu_time'] = float(cpu_time_match.group(1))
        
        # Alternative CPU time patterns
        cpu_patterns = [
            r'CPU TIME\s*[=:]\s*([\d.]+)',
            r'ELAPSED TIME\s*[=:]\s*([\d.]+)', 
            r'WALL TIME\s*[=:]\s*([\d.]+)'
        ]
        
        for pattern in cpu_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match and 'total_cpu_time' not in props:
                props['cpu_time'] = float(match.group(1))
                break
        
        # Fermi energy extraction - including SP calculation conducting state format
        fermi_patterns = [
            r'FERMI ENERGY\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)',
            r'FERMI LEVEL\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)',
            r'CHEMICAL POTENTIAL\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)',
            r'EFERMI\(AU\)\s+([-\d.E+\-]+)',  # SP calculation conducting state: "EFERMI(AU) -9.9423732E-02"
            r'POSSIBLY CONDUCTING STATE.*?EFERMI\(AU\)\s+([-\d.E+\-]+)'  # Full conducting state pattern
        ]
        
        for pattern in fermi_patterns:
            # Last occurrence: conducting-state EFERMI lines are printed per
            # SCF iteration and only the final one is converged
            fermi_matches = re.findall(pattern, content, re.IGNORECASE)
            if fermi_matches:
                props['fermi_energy'] = float(fermi_matches[-1])
                break
        
        # SCF cycles
        scf_match = re.search(r'SCF FIELD CONVERGENCE IN\s*(\d+)\s*CYCLES', content, re.IGNORECASE)
        if scf_match:
            props['scf_cycles'] = int(scf_match.group(1))
        
        # Memory usage (if available)
        memory_patterns = [
            r'MEMORY USAGE\s*[=:]\s*([\d.]+)\s*MB',
            r'MAX MEMORY\s*[=:]\s*([\d.]+)',
            r'TOTAL MEMORY\s*[=:]\s*([\d.]+)'
        ]
        
        for pattern in memory_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                props['memory_usage'] = float(match.group(1))
                break
        
        return props
    
    def _parse_band_dat_kpath_nodes(self, band_dat_file: Path):
        """Parse the high-symmetry node positions from a CRYSTAL BAND.DAT header.

        The ``# <kindex> <label>`` lines in the ``# NPANEL`` block give the
        cumulative (1-based) k-point index of each high-symmetry node along the
        path, plus a (usually placeholder, e.g. "A") label. ``DatFileProcessor``
        keeps only the labels, so re-scan the header here to also recover the
        node *indices* (needed to read band energies AT the high-symmetry
        points) and the explicit ``# EFERMI (HARTREE)`` value.

        Returns ``(node_indices, node_labels, efermi_hartree)`` using native
        Python ints/floats/strs. Scans only ``#``/``@`` header lines and stops
        at the first numeric data row, so it is cheap even for large files.
        """
        node_indices = []
        node_labels = []
        efermi_ha = None
        try:
            with open(band_dat_file, 'r') as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith('@'):
                        continue  # xmgrace directive
                    if line.startswith('#'):
                        upper = line.upper()
                        if 'EFERMI' in upper and efermi_ha is None:
                            m = re.search(r'(-?\d+\.\d+(?:[Ee][+-]?\d+)?)', line)
                            if m:
                                efermi_ha = float(m.group(1))
                            continue
                        tok = line.lstrip('#').split()
                        # Panel node line: "<int> <label>" (skip NKPT/NPANEL header
                        # lines, which have a keyword in tok[0]).
                        if (len(tok) >= 2 and tok[0].isdigit()
                                and not tok[1].isdigit()
                                and 'NKPT' not in tok and 'NPANEL' not in tok):
                            node_indices.append(int(tok[0]))
                            node_labels.append(tok[1])
                        continue
                    # Numeric data row. The node block sits in the header, but the
                    # "# EFERMI (HARTREE)" comment is written between the ALPHA and
                    # BETA data blocks (not in the header). Once we have the node
                    # indices, keep skipping data rows cheaply until we hit that
                    # comment (or EOF) so EFERMI is captured without re-reading the
                    # full BETA block.
                    if node_indices and efermi_ha is not None:
                        break
                    continue
        except Exception:
            return [], [], None
        return node_indices, node_labels, efermi_ha

    def _kpath_labels_from_d3(self, output_file: Path, num_nodes: int):
        """Resolve the REAL high-symmetry labels (G/X/L/...) from the band ``.d3``.

        CRYSTAL writes placeholder labels ("A") into BAND.DAT, but the input deck
        (``<stem>.d3``) carries the SeeKPath path string on its title line, e.g.
        ``... SeeKPath (w.I) - GAMMA-X-U-|-K-GAMMA-L-W-X``. A ``|`` marks a path
        discontinuity (the two endpoints it joins coincide as ONE node), so the
        node count after collapsing ``|`` matches the BAND.DAT panel-node count.
        Returns a list of ``num_nodes`` label strings, or ``[]`` if unavailable.
        """
        d3_file = output_file.parent / f"{output_file.stem}.d3"
        if not d3_file.exists():
            return []
        try:
            lines = d3_file.read_text(errors='ignore').splitlines()
        except Exception:
            return []
        title = ''
        for ln in lines[:5]:
            if 'SeeKPath' in ln or 'SEEKPATH' in ln.upper() or 'Band Structure' in ln:
                title = ln
                break
        if not title:
            return []
        m = re.search(r'SeeKPath[^-]*-\s*(.+)$', title)
        path_str = (m.group(1) if m else '').strip()
        if not path_str:
            return []
        raw = path_str.split('-')
        labels = []
        i = 0
        while i < len(raw):
            lab = raw[i].strip()
            if lab == '|':
                # Discontinuity: merge the next endpoint into the previous node.
                if labels and i + 1 < len(raw):
                    labels[-1] = labels[-1] + '|' + raw[i + 1].strip()
                    i += 2
                    continue
                i += 1
                continue
            if lab:
                labels.append(lab)
            i += 1
        if num_nodes and len(labels) == num_nodes:
            return labels
        return labels  # best-effort even on count mismatch (caller may zip-truncate)

    def _build_band_structure_summary(self, dat_info, content, output_file,
                                      band_dat_file, num_occ):
        """Build a compact, JSON-serializable band-structure summary.

        Per the storage analysis this stores the scientifically useful structure
        WITHOUT dumping the full ~1000k-point x ~100-band eigenvalue matrix into
        the properties table: the k-path with real high-symmetry labels, the band
        energies AT the high-symmetry points, direct vs indirect gap with their
        k-locations, band/k-point counts, and the Fermi energy. The raw BAND.DAT
        stays on disk and is re-parsed on demand. All values are native
        float/int/str/bool so they JSON-serialize cleanly.
        """
        eigenvalues = dat_info.get('eigenvalues') or []
        if not eigenvalues:
            return None, {}
        nk = int(dat_info.get('num_k_points') or 0)
        nb = int(dat_info.get('num_bands') or (len(eigenvalues[0]) if eigenvalues else 0))
        ns = int(dat_info.get('num_spins') or 1)
        k_abscissa = dat_info.get('k_points') or []

        node_indices, _placeholder_labels, efermi_ha = self._parse_band_dat_kpath_nodes(band_dat_file)
        if efermi_ha is None:
            # Fall back to the Fermi energy parsed from the .out (CRYSTAL writes
            # the same value, in Hartree, on the band .out's FERMI ENERGY line).
            m = re.search(r'FERMI ENERGY\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)', content)
            if m:
                try:
                    efermi_ha = float(m.group(1))
                except ValueError:
                    efermi_ha = None
        real_labels = self._kpath_labels_from_d3(output_file, len(node_indices))
        # Prefer real .d3 labels; fall back to BAND.DAT placeholders.
        if real_labels and len(real_labels) == len(node_indices):
            labels = real_labels
        elif dat_info.get('k_path_labels'):
            labels = list(dat_info['k_path_labels'])
        else:
            labels = list(_placeholder_labels)

        # Restrict to the ALPHA (first-spin) k-block for the per-k geometry; the
        # node indices are 1-based cumulative positions within one spin block.
        alpha = eigenvalues[:nk] if nk else eigenvalues

        summary = {
            'source_file': band_dat_file.name,
            'num_kpoints': nk,
            'num_bands': nb,
            'num_spins': ns,
            'num_panels': int(dat_info.get('num_panels') or 0),
            'fermi_energy_hartree': float(efermi_ha) if efermi_ha is not None else None,
            'energy_units': 'eV',
            'energy_reference': 'fermi',  # BAND.DAT eigenvalues are E - E_Fermi
            'kpath_node_indices': [int(i) for i in node_indices],
            'kpath_node_distances': [
                float(k_abscissa[i - 1]) for i in node_indices
                if 0 < i <= len(k_abscissa)
            ],
            'kpath_labels': [str(l) for l in labels],
        }

        H = HARTREE_TO_EV
        # Band energies (Fermi-referenced eV) at each high-symmetry node. Only the
        # node rows are stored (a few KB), not the full matrix.
        hs_bands = []
        for ni in node_indices:
            if 0 < ni <= len(alpha):
                hs_bands.append([float(e) * H for e in alpha[ni - 1]])
            else:
                hs_bands.append([])
        summary['high_symmetry_band_energies_ev'] = hs_bands

        # Direct vs indirect gap from the occupied-band index (referencing-
        # independent, same convention as DatFileProcessor._analyze_band_structure).
        #
        # Two correctness invariants, matching the legacy scalar analyzer:
        #   (1) Indices are over the UNFILTERED rows: ragged rows are skipped
        #       in place but never shift the surviving rows' indices, so the
        #       reported k-point index stays a valid index into the per-k
        #       geometry (k_abscissa / node_indices), which is defined over one
        #       spin block of `nk` k-points (kidx = global_row % nk).
        #   (2) The fundamental gap considers ALL spin channels (VBM = max
        #       occupied over every k AND every spin; CBM = min virtual over
        #       every k AND every spin), so it agrees with the legacy
        #       band_dat_band_gap_ev for spin-polarized systems too — not just
        #       the first spin block.
        extra_scalars = {}
        if num_occ and num_occ >= 1 and eigenvalues:
            n = num_occ
            kdiv = nk if nk else len(eigenvalues)

            def _label_for(kidx):
                # Map a 0-based per-spin-block k index to the high-symmetry
                # label if it sits exactly on a node, else None.
                for lbl, nidx in zip(labels, node_indices):
                    if nidx - 1 == kidx:
                        return str(lbl)
                return None

            vbm_e = None
            vbm_global = None   # global row index of the VBM (max occupied)
            cbm_e = None
            cbm_global = None   # global row index of the CBM (min virtual)
            direct = None       # smallest per-(k,spin) HOMO-LUMO gap (Hartree)
            for gi, row in enumerate(eigenvalues):
                # Skip ragged rows in place but KEEP the true global index.
                if len(row) < n + 1:
                    continue
                homo_e = row[n - 1]
                lumo_e = row[n]
                if vbm_e is None or homo_e > vbm_e:
                    vbm_e = homo_e
                    vbm_global = gi
                if cbm_e is None or lumo_e < cbm_e:
                    cbm_e = lumo_e
                    cbm_global = gi
                d = lumo_e - homo_e
                if direct is None or d < direct:
                    direct = d

            if vbm_e is not None and cbm_e is not None:
                # Per-spin-block k indices (valid into k_abscissa/node_indices).
                vbm_k = (vbm_global % kdiv) if kdiv else vbm_global
                cbm_k = (cbm_global % kdiv) if kdiv else cbm_global

                indirect = (cbm_e - vbm_e) * H
                if indirect < 0:
                    indirect = 0.0  # band overlap along the path -> metallic
                direct_ev = direct * H
                if direct_ev < 0:
                    direct_ev = 0.0
                is_direct = bool(vbm_k == cbm_k)

                summary['direct_gap_ev'] = float(direct_ev)
                summary['indirect_gap_ev'] = float(indirect)
                summary['fundamental_gap_ev'] = float(indirect)  # min over all k+spin
                summary['is_direct_gap'] = is_direct
                summary['vbm'] = {
                    'energy_ev': float(vbm_e * H),
                    'kpoint_index': int(vbm_k),
                    'kpath_distance': (float(k_abscissa[vbm_k]) if vbm_k < len(k_abscissa) else None),
                    'band_index': int(n),
                    'kpoint_label': _label_for(vbm_k),
                }
                summary['cbm'] = {
                    'energy_ev': float(cbm_e * H),
                    'kpoint_index': int(cbm_k),
                    'kpath_distance': (float(k_abscissa[cbm_k]) if cbm_k < len(k_abscissa) else None),
                    'band_index': int(n + 1),
                    'kpoint_label': _label_for(cbm_k),
                }
                summary['occupied_bands'] = int(n)

                # Queryable scalar rows (numeric/bool) — distinct from the JSON blob.
                extra_scalars['band_direct_gap_ev'] = float(direct_ev)
                extra_scalars['band_indirect_gap_ev'] = float(indirect)
                extra_scalars['band_fundamental_gap_ev'] = float(indirect)
                extra_scalars['band_gap_is_direct'] = is_direct
                extra_scalars['band_vbm_kpoint_index'] = int(vbm_k)
                extra_scalars['band_cbm_kpoint_index'] = int(cbm_k)

        if labels:
            extra_scalars['band_kpath_label_string'] = '-'.join(str(l) for l in labels)

        return summary, extra_scalars

    def _extract_band_structure_properties(self, content: str, output_file: Path) -> Dict[str, Any]:
        """Extract band structure specific properties from BAND calculations."""
        props = {}
        
        # Check if this is a BAND calculation
        if 'BAND STRUCTURE' not in content.upper() and 'FROM BAND' not in content:
            return props
            
        # Extract k-point information
        kpoint_match = re.search(r'TOTAL OF\s*(\d+)\s*K-POINTS ALONG THE PATH', content, re.IGNORECASE)
        if kpoint_match:
            props['total_kpoints'] = int(kpoint_match.group(1))
        
        # Extract band range information
        band_range_match = re.search(r'FROM BAND\s*(\d+)\s*TO BAND\s*(\d+)', content, re.IGNORECASE)
        if band_range_match:
            props['band_start'] = int(band_range_match.group(1))
            props['band_end'] = int(band_range_match.group(2))
            props['total_bands'] = int(band_range_match.group(2)) - int(band_range_match.group(1)) + 1
            
        # Extract Fermi energy from BAND calculation
        fermi_patterns = [
            r'FERMI ENERGY\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)',
            r'FERMI LEVEL\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)',
            r'EF\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)',
            r'FERMI ENERGY\s+([-\d.E+\-]+)',  # CRYSTAL format: "FERMI ENERGY -0.123E+00"
        ]
        
        for pattern in fermi_patterns:
            fermi_matches = re.findall(pattern, content, re.IGNORECASE)
            if fermi_matches:
                props['fermi_energy_band'] = float(fermi_matches[-1])
                break
                
        # Check for BAND.DAT file and extract information. CRYSTAL/the workflow
        # writes "<stem>.BAND.DAT" (e.g. "..._band.BAND.DAT"), not a bare
        # "BAND.DAT", so the fixed-name lookup never matched real files and the
        # band analysis stayed inert. Mirror the DOSS alt-name fallback.
        band_dat_default = output_file.parent / "BAND.DAT"
        band_dat_alt = output_file.parent / f"{output_file.stem}.BAND.DAT"
        band_dat_file = None
        if band_dat_default.exists():
            band_dat_file = band_dat_default
        elif band_dat_alt.exists():
            band_dat_file = band_dat_alt

        if band_dat_file:
            props['band_dat_exists'] = True
            props['band_dat_size'] = band_dat_file.stat().st_size
            
            # Try to extract information from DAT file using existing processor
            try:
                try:
                    from mace.utils.dat_file_processor import DatFileProcessor
                except ImportError:
                    from dat_file_processor import DatFileProcessor
                processor = DatFileProcessor()
                # CRYSTAL BAND.DAT eigenvalues are Fermi-referenced, so the gap is
                # computed by band index. Determine the occupied-band count (HOMO
                # index): prefer the explicit "TOP OF VALENCE BANDS - BAND N"
                # line, else fall back to (electrons per cell)//2 — exact for
                # closed-shell / non-magnetic systems (the band .out usually only
                # carries the electron count). Without it the BAND-derived gap is
                # left unset rather than wrong.
                num_occ = None
                nocc_match = re.findall(r'TOP OF VALENCE BANDS\s*-\s*BAND\s*(\d+)', content)
                if nocc_match:
                    num_occ = int(nocc_match[-1])
                else:
                    nelec_match = re.findall(r'N\.?\s*OF ELECTRONS PER CELL\s*(\d+)', content)
                    if nelec_match:
                        num_occ = int(nelec_match[-1]) // 2
                dat_info = processor.process_band_dat_file(
                    band_dat_file, fermi_energy=props.get('fermi_energy_band'),
                    num_occupied_bands=num_occ)
                if dat_info and 'error' not in dat_info:
                    # Merge only compact scalars — never the raw eigenvalue/k-point
                    # arrays, which would be JSON-dumped into the properties table.
                    props['band_dat_num_kpoints'] = dat_info.get('num_k_points')
                    props['band_dat_num_bands'] = dat_info.get('num_bands')
                    props['band_dat_num_spins'] = dat_info.get('num_spins')
                    props['band_dat_num_panels'] = dat_info.get('num_panels')
                    if dat_info.get('k_path_labels'):
                        props['band_dat_kpath_labels'] = dat_info['k_path_labels']
                    bep = dat_info.get('electronic_properties', {})
                    if bep.get('metallic') is not None:
                        props['band_dat_metallic'] = bep['metallic']
                    if bep.get('band_gap_ev') is not None:
                        props['band_dat_band_gap_ev'] = bep['band_gap_ev']
                    if bep.get('valence_band_maximum_ev') is not None:
                        props['band_dat_vbm_ev'] = bep['valence_band_maximum_ev']
                    if bep.get('conduction_band_minimum_ev') is not None:
                        props['band_dat_cbm_ev'] = bep['conduction_band_minimum_ev']

                    # LAYERED ADDITION: store the actual band STRUCTURE (k-path +
                    # high-symmetry-point band energies + direct/indirect gap with
                    # k-locations) as a compact JSON summary, plus queryable scalar
                    # rows. Per the storage analysis we do NOT dump the full
                    # per-k eigenvalue matrix (MBs); the raw BAND.DAT stays on disk
                    # and is re-parsed on demand. Failures here never affect the
                    # scalars above.
                    try:
                        summary, extra_scalars = self._build_band_structure_summary(
                            dat_info, content, output_file, band_dat_file, num_occ)
                        if summary:
                            props['band_structure'] = summary
                        for k, v in (extra_scalars or {}).items():
                            if v is not None:
                                props[k] = v
                    except Exception as e:
                        print(f"Warning: Could not build band-structure summary: {e}")
            except ImportError:
                pass  # DAT file processor not available
            except Exception as e:
                print(f"Warning: Could not process BAND.DAT file: {e}")
        
        # Extract band gap information specific to BAND calculations
        if 'BAND GAP' in content:
            # Look for VBM/CBM information
            vbm_match = re.search(r'TOP OF VALENCE BANDS\s*[^\d]*(-?[\d.]+)', content, re.IGNORECASE)
            if vbm_match:
                props['vbm_energy'] = float(vbm_match.group(1))
                
            cbm_match = re.search(r'BOTTOM OF CONDUCTION BANDS\s*[^\d]*(-?[\d.]+)', content, re.IGNORECASE)
            if cbm_match:
                props['cbm_energy'] = float(cbm_match.group(1))
        
        # Mark this as a band structure calculation
        if props:
            props['calculation_type'] = 'BAND'
            props['has_band_structure'] = True
            
        return props
    
    def _build_dos_summary(self, dat_info, content, output_file, doss_dat_file):
        """Build a compact, JSON-serializable density-of-states summary.

        Mirrors ``_build_band_structure_summary``: this stores the scientifically
        useful DOS structure WITHOUT dumping the full ~1000-pt x N-projection x
        2-spin matrix (which is 96 KB - 2 MB of JSON for real slabs) into the
        properties table. The raw ``DOSS.DAT`` stays on disk (referenced via
        ``source_file`` + the already-stored ``doss_dat_*`` scalars) and is
        re-parsed on demand. We persist:

          * the DOS @ Fermi PER SPIN channel (alpha + beta, not just the first),
          * the gap-from-DOS / band-edge energies (window straddling E_F),
          * the integrated number of states up to E_Fermi (occupied-state count),
          * a DOWNSAMPLED total-DOS curve per spin (<= 200 pts/spin, ~few KB) in
            eV so the shape can be replotted without re-reading the big file,
          * the projection count + per-projection integrated weight (a single
            scalar per projection — the atom/orbital decomposition magnitude —
            not the full per-energy projected curves).

        UNITS. CRYSTAL DOSS.DAT energies are Fermi-referenced Hartree (E -
        E_Fermi, Fermi at 0) and the DOS magnitudes are in states/HARTREE/cell.
        We convert the energy axis to eV (``* H``) AND the DOS magnitudes to
        per-eV (``/ H``) so the stored curve, the DOS@Fermi values and the
        projection weights are all self-consistent with the ``states/eV/cell``
        label. The legacy ``DatFileProcessor`` analyzer (``dos_at_fermi`` /
        ``total_states``) leaves DOS in states/HARTREE/cell, so the documented
        relationship between the eV values here and the legacy scalars is a
        single factor of ``H = HARTREE_TO_EV``:

            dos_at_fermi_<spin>_ev (states/eV) * H == legacy doss_dat_at_fermi
                                                      (states/Ha, single channel)

        For spin-polarized runs the energy grid is written once per spin block
        (concatenated in ``energy_points``); the beta block's DOS values are
        sign-negated for mirror plotting, so we take magnitudes for the physical
        DOS. Returns ``(summary, extra_scalars)`` and ``(None, {})`` when there
        is no curve to summarize. All values are native float/int/str/bool/list
        so they JSON-serialize cleanly.
        """
        energy_points = dat_info.get('energy_points') or []
        total_dos = dat_info.get('total_dos') or []
        if not energy_points or not total_dos:
            return None, {}
        # Guard ragged parser output (energy/DOS must stay aligned for slicing).
        if len(energy_points) != len(total_dos):
            return None, {}

        H = HARTREE_TO_EV
        nspin = int(dat_info.get('num_spins') or 1)
        if nspin < 1:
            nspin = 1
        nproj = int(dat_info.get('num_proj') or 0)
        # NEPTS = points PER spin block. Prefer the header value but never let it
        # run past the actual array; the grid is repeated once per spin block.
        nepts_hdr = int(dat_info.get('num_energy_points') or 0)
        total_len = len(energy_points)
        if nepts_hdr > 0 and nepts_hdr * nspin == total_len:
            nepts = nepts_hdr
        else:
            # Fall back to an even split; if it doesn't divide cleanly, treat the
            # whole thing as one block (never desync the per-block slices).
            if nspin > 1 and total_len % nspin == 0:
                nepts = total_len // nspin
            else:
                nepts = total_len
                nspin = 1

        da = dat_info.get('dos_analysis') or {}

        # Genuine projection count: the CRYSTAL header NPROJ counts the TOTAL DOS
        # column as a "projection", and the parser also exposes a projected_dos
        # column that is byte-identical to the total DOS. Neither is a real
        # atom/orbital projection, so we count only the projected_dos columns
        # that are NOT identical to the total DOS (computed below alongside the
        # per-projection weights). ``num_proj`` is then == len(weights).
        proj_all = dat_info.get('projected_dos') or {}
        total_dos_full = total_dos
        genuine_proj = {
            name: vals for name, vals in proj_all.items()
            if not (isinstance(vals, list) and vals == total_dos_full)
        }
        n_genuine_proj = len(genuine_proj)

        summary = {
            'source_file': Path(doss_dat_file).name,
            'num_energy_points': int(nepts),
            'num_proj': int(n_genuine_proj),
            'num_spins': int(nspin),
            'energy_units': 'eV',
            'energy_reference': 'fermi',  # DOSS.DAT energies are E - E_Fermi
            # DOS magnitudes are converted from states/HARTREE/cell to per-eV
            # (divided by HARTREE_TO_EV) so they match the eV energy axis below.
            'dos_units': 'states/eV/cell',
            'spin_polarized': bool(nspin >= 2),
        }

        erange = dat_info.get('energy_range')
        if erange and len(erange) == 2:
            summary['energy_min_ev'] = float(erange[0]) * H
            summary['energy_max_ev'] = float(erange[1]) * H

        # Absolute Fermi energy (Hartree): footer "# EFERMI (HARTREE) <val>",
        # printed once per spin block. Parse from the raw .DAT (it is NOT on the
        # energy axis, which is already Fermi-shifted). Best-effort only.
        efermi_ha = None
        try:
            dat_text = Path(doss_dat_file).read_text(errors='ignore')
            ef_matches = re.findall(
                r'EFERMI\s*\(HARTREE\)\s*(-?\d+\.?\d*(?:[Ee][+-]?\d+)?)', dat_text)
            if ef_matches:
                efermi_ha = float(ef_matches[-1])
        except Exception:
            efermi_ha = None
        summary['fermi_energy_hartree'] = (
            float(efermi_ha) if efermi_ha is not None else None)

        # --- Per-spin DOS @ Fermi (value at first grid point AT or ABOVE E=0) ---
        # Same convention as DatFileProcessor._analyze_density_of_states, but per
        # spin block so the alpha and beta Fermi-level DOS are both reported
        # (never silently collapsing to the first channel).
        def _dos_at_fermi(es, ds):
            at_or_above = [(e, d) for e, d in zip(es, ds) if e >= 0]
            if at_or_above:
                return abs(min(at_or_above, key=lambda ed: ed[0])[1])
            if not es:
                return None
            ci = min(range(len(es)), key=lambda i: abs(es[i]))
            return abs(ds[ci])

        spin_labels = ['alpha', 'beta'] if nspin >= 2 else ['total']
        dos_at_fermi_by_spin = {}      # states/eV/cell, per spin channel
        dos_at_fermi_per_channel = None  # states/eV/cell, single channel (legacy)
        downsampled = {}
        # Energy grid (eV) for the first block; identical across blocks.
        block0_es_ha = energy_points[0:nepts]
        for s in range(nspin):
            lo = s * nepts
            hi = lo + nepts
            es_ha = energy_points[lo:hi]
            # DOS magnitudes converted states/Ha -> states/eV (divide by H) so the
            # stored values match the 'states/eV/cell' label and the eV axis.
            ds_block = [abs(d) / H for d in total_dos[lo:hi]]
            if not es_ha:
                continue
            label = spin_labels[s] if s < len(spin_labels) else f'spin_{s}'
            daf = _dos_at_fermi(es_ha, ds_block)  # already per-eV
            daf_val = float(daf) if daf is not None else None
            dos_at_fermi_by_spin[label] = daf_val
            # The legacy analyzer reports a SINGLE channel (first block); keep an
            # explicit per-channel value so the relationship to doss_dat_at_fermi
            # is defined (per_channel_ev * H == legacy states/Ha value), with no
            # silent factor-of-2 vs the both-spins sum below.
            if s == 0:
                dos_at_fermi_per_channel = daf_val
            # Downsample the (E in eV, DOS in states/eV) curve to <= 200 points.
            # Stride = ceil(n/200) guarantees len(range(0,n,stride)) <= 200
            # (n//200 could yield 201 points: an off-by-one).
            n = len(es_ha)
            stride = max(1, -(-n // 200))  # ceil division
            curve = [[float(es_ha[i]) * H, float(ds_block[i])]
                     for i in range(0, n, stride)]
            downsampled[label] = curve
        summary['dos_at_fermi_by_spin'] = dos_at_fermi_by_spin
        summary['total_dos_curve_ev'] = downsampled

        # Two clearly-named Fermi-level DOS aggregates so there is NO silent 2x
        # disagreement with the single-channel legacy ``doss_dat_at_fermi``:
        #   * dos_at_fermi_per_channel_ev: single channel (matches legacy/H),
        #   * dos_at_fermi_both_spins_ev:  alpha + beta (the spin-summed total).
        finite_daf = [v for v in dos_at_fermi_by_spin.values() if v is not None]
        both_spins = float(sum(finite_daf)) if finite_daf else None
        summary['dos_at_fermi_per_channel_ev'] = dos_at_fermi_per_channel
        summary['dos_at_fermi_both_spins_ev'] = both_spins
        # Back-compat key: keep ``total_dos_at_fermi`` but make it the SINGLE
        # channel value (states/eV), matching the legacy convention (legacy/H),
        # NOT the doubled sum, so the headline no longer disagrees by 2x.
        total_dos_at_fermi = dos_at_fermi_per_channel
        summary['total_dos_at_fermi'] = total_dos_at_fermi

        # Gap-from-DOS / band edges: the zero-DOS window straddling E_F. Recompute
        # the edges here (the analyzer keeps only the width) using the first block
        # (insulator gap edges are spin-degenerate for the systems CRYSTAL writes
        # this way; abs() makes the beta block identical). Consistent-by-design
        # with doss_dat_band_gap_ev which the analyzer derived the same way.
        band_gap_ev = da.get('band_gap_ev')
        summary['band_gap_ev'] = (
            float(band_gap_ev) if band_gap_ev is not None else None)
        summary['metallic'] = (
            bool(da['metallic']) if da.get('metallic') is not None else None)

        vbm_ev = cbm_ev = None
        if block0_es_ha:
            es0 = [e for e in block0_es_ha]
            ds0 = [abs(d) for d in total_dos[0:nepts]]
            pairs = sorted(zip(es0, ds0), key=lambda p: p[0])
            es = [e for e, _ in pairs]
            ds = [d for _, d in pairs]
            span = (es[-1] - es[0]) if len(es) >= 2 else 0.0
            if span > 0:
                margin = 0.02 * span
                interior = [d for e, d in zip(es, ds)
                            if es[0] + margin <= e <= es[-1] - margin] or ds
                ordered = sorted(interior)
                scale = ordered[int(0.95 * (len(ordered) - 1))] if ordered else 0.0
                if scale > 0:
                    near_zero = 0.001 * scale
                    i, n = 0, len(ds)
                    best = None  # (lo_e, hi_e) of the gap window nearest E_F
                    while i < n:
                        if ds[i] < near_zero:
                            j = i
                            while j + 1 < n and ds[j + 1] < near_zero:
                                j += 1
                            lo_e, hi_e = es[i], es[j]
                            dist = (0.0 if lo_e <= 0 <= hi_e
                                    else min(abs(lo_e), abs(hi_e)))
                            if dist <= 0.01 and (best is None or
                                                 (hi_e - lo_e) > (best[1] - best[0])):
                                best = (lo_e, hi_e)
                            i = j + 1
                        else:
                            i += 1
                    if best is not None:
                        vbm_ev = float(best[0]) * H  # top of valence (gap onset)
                        cbm_ev = float(best[1]) * H  # bottom of conduction
        summary['vbm_ev'] = vbm_ev
        summary['cbm_ev'] = cbm_ev

        # numpy is NOT imported at module scope here, so use a self-contained
        # trapezoid (the integrand magnitudes are tiny per-point, no overflow).
        def _trapz(ys, xs):
            if len(ys) != len(xs) or len(xs) < 2:
                return None
            acc = 0.0
            for k in range(1, len(xs)):
                acc += (xs[k] - xs[k - 1]) * (ys[k] + ys[k - 1]) * 0.5
            return acc

        # Integrated states up to E_Fermi == approx number of OCCUPIED states
        # (electrons) per cell. The legacy ``total_states`` integrated the SIGNED
        # alpha(+)/beta(-) concatenated curve over the whole Hartree axis, giving
        # ~0 (physically meaningless), so we DROP that and instead integrate the
        # |DOS| magnitude (states/eV, summed over spins) on the eV axis from the
        # band bottom up to E_Fermi (E <= 0). Units: states/cell (eV cancels).
        integrated_states_to_fermi = None
        if block0_es_ha:
            es_ev_occ = []   # eV energies <= E_Fermi, ascending
            dos_occ = []     # summed-over-spin |DOS| (states/eV) at those energies
            for i in range(nepts):
                e_ha = energy_points[i]
                if e_ha > 0:
                    continue
                e_ev = float(e_ha) * H
                d_sum = 0.0
                for s in range(nspin):
                    idx = s * nepts + i
                    if idx < len(total_dos):
                        d_sum += abs(float(total_dos[idx])) / H
                es_ev_occ.append(e_ev)
                dos_occ.append(d_sum)
            if len(es_ev_occ) >= 2:
                order = sorted(range(len(es_ev_occ)), key=lambda k: es_ev_occ[k])
                xs = [es_ev_occ[k] for k in order]
                ys = [dos_occ[k] for k in order]
                w = _trapz(ys, xs)
                if w is not None:
                    integrated_states_to_fermi = float(w)
        summary['dos_integrated_states_to_fermi'] = integrated_states_to_fermi

        peaks = da.get('peak_positions') or []
        summary['peak_positions_ev'] = [float(p) * H for p in peaks]

        # Per-projection integrated weight (atom/orbital decomposition magnitude):
        # the trapezoid integral of |projection|/H over the first block, in
        # states/cell (states/eV * eV). A single scalar per projection — NOT the
        # per-energy curves. We integrate only the GENUINE projections (the column
        # byte-identical to the TOTAL DOS is excluded — it is not a projection),
        # so len(proj_weights) == num_proj. Keep the neutral projected_dos_N
        # indices (atom/orbital identity is not available from the .DAT).
        proj_weights = {}
        if block0_es_ha and genuine_proj:
            es0 = block0_es_ha
            es0_ev = [float(e) * H for e in es0]
            for name, vals in genuine_proj.items():
                if not isinstance(vals, list) or len(vals) < nepts:
                    continue
                block = [abs(v) / H for v in vals[0:nepts]]  # states/eV
                if len(block) == len(es0_ev) and len(es0_ev) >= 2:
                    w = _trapz(block, es0_ev)  # states/cell
                    if w is not None:
                        proj_weights[str(name)] = float(w)
        if proj_weights:
            summary['projection_integrated_weights'] = proj_weights

        # --- Queryable scalar rows (numeric/bool) — distinct from the JSON blob.
        extra_scalars = {}
        # Single-channel DOS@Fermi (states/eV); equals legacy doss_dat_at_fermi/H.
        if total_dos_at_fermi is not None:
            extra_scalars['dos_total_at_fermi'] = total_dos_at_fermi
        if both_spins is not None:
            extra_scalars['dos_at_fermi_both_spins_ev'] = both_spins
        for label, val in dos_at_fermi_by_spin.items():
            if val is not None and label in ('alpha', 'beta'):
                extra_scalars[f'dos_at_fermi_{label}'] = float(val)
        if band_gap_ev is not None:
            extra_scalars['dos_band_gap_from_dos_ev'] = float(band_gap_ev)
        if da.get('metallic') is not None:
            extra_scalars['dos_is_metallic'] = bool(da['metallic'])
        # Meaningful integrated occupied-state count (replaces the meaningless ~0
        # signed-trapezoid ``dos_total_states`` that was dropped from the blob).
        if integrated_states_to_fermi is not None:
            extra_scalars['dos_integrated_states_to_fermi'] = (
                float(integrated_states_to_fermi))
        # Genuine projection count == len(projection_integrated_weights); the
        # total-equals column and the header's TOTAL slot are not counted.
        extra_scalars['dos_num_projections'] = int(n_genuine_proj)

        return summary, extra_scalars

    def _extract_dos_properties(self, content: str, output_file: Path) -> Dict[str, Any]:
        """Extract density of states specific properties from DOSS calculations."""
        props = {}
        
        # Check if this is a DOSS calculation
        if 'DENSITY OF STATES' not in content.upper() and 'DOSS' not in content.upper():
            return props
            
        # Extract DOS energy range
        dos_range_match = re.search(r'ENERGY RANGE\s*FROM\s*([-\d.]+)\s*TO\s*([-\d.]+)', content, re.IGNORECASE)
        if dos_range_match:
            props['dos_energy_min'] = float(dos_range_match.group(1))
            props['dos_energy_max'] = float(dos_range_match.group(2))
            props['dos_energy_range'] = float(dos_range_match.group(2)) - float(dos_range_match.group(1))
        
        # Extract number of energy points
        points_match = re.search(r'NUMBER OF ENERGY POINTS\s*[=:]\s*(\d+)', content, re.IGNORECASE)
        if points_match:
            props['dos_energy_points'] = int(points_match.group(1))
            
        # Extract DOS broadening
        broadening_match = re.search(r'BROADENING\s*[=:]\s*([\d.]+)', content, re.IGNORECASE)
        if broadening_match:
            props['dos_broadening'] = float(broadening_match.group(1))
        
        # Extract Fermi energy from DOSS calculation
        fermi_patterns = [
            r'FERMI ENERGY\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)',
            r'FERMI LEVEL\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)',
            r'EF\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)',
            r'FERMI ENERGY\s+([-\d.E+\-]+)',  # CRYSTAL format: "FERMI ENERGY -0.123E+00"
        ]
        
        for pattern in fermi_patterns:
            fermi_matches = re.findall(pattern, content, re.IGNORECASE)
            if fermi_matches:
                props['fermi_energy_dos'] = float(fermi_matches[-1])
                break
                
        # Check for DOSS.DAT file and extract information
        doss_dat_file = output_file.parent / "DOSS.DAT"
        alt_doss_dat = output_file.parent / f"{output_file.stem}.DOSS.DAT"
        
        dat_file = None
        if doss_dat_file.exists():
            dat_file = doss_dat_file
        elif alt_doss_dat.exists():
            dat_file = alt_doss_dat
            
        if dat_file:
            props['doss_dat_exists'] = True
            props['doss_dat_size'] = dat_file.stat().st_size
            
            # Try to extract information from DAT file using existing processor
            try:
                try:
                    from mace.utils.dat_file_processor import DatFileProcessor
                except ImportError:
                    from dat_file_processor import DatFileProcessor
                processor = DatFileProcessor()
                dat_info = processor.process_doss_dat_file(dat_file)
                if dat_info and 'error' not in dat_info:
                    # Merge only compact scalars — never the raw DOS arrays.
                    props['doss_dat_num_energy_points'] = dat_info.get('num_energy_points')
                    props['doss_dat_num_spins'] = dat_info.get('num_spins')
                    erange = dat_info.get('energy_range')
                    if erange:
                        props['doss_dat_energy_min'] = erange[0]
                        props['doss_dat_energy_max'] = erange[1]
                    da = dat_info.get('dos_analysis', {})
                    if da.get('metallic') is not None:
                        props['doss_dat_metallic'] = da['metallic']
                    if da.get('band_gap_ev') is not None:
                        props['doss_dat_band_gap_ev'] = da['band_gap_ev']
                    if da.get('dos_at_fermi') is not None:
                        props['doss_dat_at_fermi'] = da['dos_at_fermi']

                    # LAYERED ADDITION: store the actual DENSITY OF STATES (the
                    # per-spin DOS @ Fermi, gap/band edges from the DOS, integrated
                    # states, projection decomposition weights, and a downsampled
                    # total-DOS curve) as a compact JSON summary, plus queryable
                    # scalar rows. Per the storage analysis we do NOT dump the full
                    # ~1000-pt x N-projection x 2-spin matrix; the raw DOSS.DAT
                    # stays on disk and is re-parsed on demand. Failures here never
                    # disturb the scalars merged above.
                    try:
                        summary, extra_scalars = self._build_dos_summary(
                            dat_info, content, output_file, dat_file)
                        if summary:
                            # Key prefixed 'dos_' so _categorize_property routes
                            # it to 'density_of_states' (a bare 'density_of_states'
                            # key would be caught by the earlier 'density' ->
                            # 'structural' branch).
                            props['dos_structure'] = summary
                        for k, v in (extra_scalars or {}).items():
                            if v is not None:
                                props[k] = v
                    except Exception as e:
                        print(f"Warning: Could not build density-of-states summary: {e}")
            except ImportError:
                pass  # DAT file processor not available
            except Exception as e:
                print(f"Warning: Could not process DOSS.DAT file: {e}")
        
        # Extract total DOS at Fermi level
        dos_fermi_match = re.search(r'DOS AT FERMI LEVEL\s*[=:]\s*([\d.]+)', content, re.IGNORECASE)
        if dos_fermi_match:
            props['dos_at_fermi'] = float(dos_fermi_match.group(1))
        
        # Mark this as a DOS calculation
        if props:
            props['calculation_type'] = 'DOSS'
            props['has_dos'] = True
            
        return props
    
    def _classify_electronic_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Classify electronic properties based on band gap values."""
        classification = {}
        
        # Find the primary band gap value
        band_gap = None
        gap_source = None
        
        # Priority order for band gap sources
        gap_sources = [
            ('band_gap', 'general'),
            ('indirect_band_gap', 'indirect'),
            ('direct_band_gap', 'direct'),
            ('band_gap_from_dos', 'dos_analysis'),
            ('alpha_band_gap', 'alpha_spin'),
            ('beta_band_gap', 'beta_spin')
        ]
        
        for gap_prop, source in gap_sources:
            if gap_prop in properties and properties[gap_prop] is not None:
                try:
                    gap_value = float(properties[gap_prop])
                    if gap_value >= 0:  # Valid band gap
                        band_gap = gap_value
                        gap_source = source
                        break
                except (ValueError, TypeError):
                    continue
        
        if band_gap is not None:
            # Electronic classification based on band gap
            if band_gap > 2.0:
                electronic_type = 'insulator'
            elif band_gap > 0.0:
                electronic_type = 'semiconductor'
            else:
                electronic_type = 'conductor'
            
            classification.update({
                'electronic_classification': electronic_type,
                'classification_band_gap': band_gap,
                'classification_gap_source': gap_source
            })
            
            # Additional detailed classification
            if electronic_type == 'insulator':
                if band_gap > 5.0:
                    classification['insulator_type'] = 'wide_band_gap'
                elif band_gap > 3.0:
                    classification['insulator_type'] = 'medium_band_gap'
                else:
                    classification['insulator_type'] = 'narrow_band_gap'
            
            elif electronic_type == 'semiconductor':
                if band_gap > 1.5:
                    classification['semiconductor_type'] = 'wide_band_gap'
                elif band_gap > 1.0:
                    classification['semiconductor_type'] = 'medium_band_gap'
                else:
                    classification['semiconductor_type'] = 'narrow_band_gap'
        
        # Magnetic classification for spin-polarized systems
        if properties.get('is_spin_polarized', False):
            classification['magnetic_classification'] = 'spin_polarized'
            
            # Check for magnetic moment if available
            alpha_gap = properties.get('alpha_band_gap')
            beta_gap = properties.get('beta_band_gap')
            
            if alpha_gap is not None and beta_gap is not None:
                try:
                    alpha_val = float(alpha_gap)
                    beta_val = float(beta_gap)
                    gap_difference = abs(alpha_val - beta_val)
                    
                    if gap_difference > 0.5:
                        classification['magnetic_type'] = 'strong_magnetic'
                    elif gap_difference > 0.1:
                        classification['magnetic_type'] = 'weak_magnetic'
                    else:
                        classification['magnetic_type'] = 'non_magnetic'
                        
                except (ValueError, TypeError):
                    pass
        else:
            classification['magnetic_classification'] = 'non_magnetic'
        
        # Metallic vs non-metallic classification
        if band_gap is not None:
            if band_gap <= 0.0:
                classification['conductivity_type'] = 'metallic'
            else:
                classification['conductivity_type'] = 'non_metallic'
        
        return classification
    
    def _extract_frequency_properties(self, content: str) -> Dict[str, Any]:
        """Extract frequency calculation properties."""
        props = {}
        
        # Check if this is a FREQ calculation
        if 'FORCE CONSTANT MATRIX' not in content and 'VIBRATIONAL' not in content:
            return props
            
        # Mark as frequency calculation
        props['calculation_type'] = 'FREQ'
        props['has_frequency_data'] = True
        
        # Extract vibrational frequencies
        # Note: real CRYSTAL output prints the units line as "(CM**-1)" (no
        # inner parens), so the anchor must tolerate both "(CM**-1)" and the
        # older "(CM**(-1))" form.
        freq_section = re.search(
            r'MODES\s+EIGV\s+FREQUENCIES\s+IRREP.*?\n\s*\(HARTREE\*\*2\)\s*\(CM\*\*-?\(?-?1\)?\)\s*\(THZ\).*?\n(.*?)(?=NORMAL MODES|VIBRATIONAL TEMPERATURES|<RAMAN>|\*{5,})',
            content, re.DOTALL
        )
        
        if freq_section:
            freq_data = []
            freq_lines = freq_section.group(1).strip().split('\n')
            
            for line in freq_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse frequency data lines. Real CRYSTAL output prints the
                # mode range space-separated, e.g. "1-   1" or "4-   6", so the
                # range and the three numeric columns (EIGV, CM**-1, THZ) must
                # be parsed from the line directly rather than from split()[0..3]
                # (a naive split puts "1-" in parts[0] and shifts every column).
                # Format: MODE_RANGE EIGV FREQ_CM FREQ_THZ (IRREP) IR_ACTIVE (INTENSITY) RAMAN_ACTIVE
                row_match = re.match(
                    r'(\d+)-\s*(\d+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)',
                    line
                )
                if row_match:
                    try:
                        mode_start = int(row_match.group(1))
                        mode_end = int(row_match.group(2))
                        eigv = float(row_match.group(3))
                        freq_cm = float(row_match.group(4))
                        freq_thz = float(row_match.group(5))

                        # Parse the symmetry/activity tail. Real layout after
                        # the three numeric columns is:
                        #   (IRREP)   IR_FLAG (INTENSITY)   RAMAN_FLAG
                        # e.g. "(A  )   A (     0.93)   A" where IR_FLAG /
                        # RAMAN_FLAG are single A(ctive)/I(nactive) letters and
                        # the bracketed value is the integrated IR intensity in
                        # KM/MOL. Default to None/0.0 when the tail is absent
                        # (files without IR/Raman columns must not crash).
                        irrep = None
                        ir_active = False
                        raman_active = False
                        intensity = 0.0
                        tail_match = re.search(
                            r'\(\s*([^)]*?)\s*\)\s*([AI])\s*\(\s*([-\d.E+]+)\s*\)\s*([AI])',
                            line[row_match.end():]
                        )
                        if tail_match:
                            irrep = tail_match.group(1).strip()
                            ir_active = tail_match.group(2) == 'A'
                            try:
                                intensity = float(tail_match.group(3))
                            except ValueError:
                                intensity = 0.0
                            raman_active = tail_match.group(4) == 'A'
                        else:
                            # Fallback: at least capture the IRREP label if the
                            # full activity tail does not match this row's shape.
                            irrep_match = re.search(r'\(([^)]+)\)', line[row_match.end():])
                            if irrep_match:
                                irrep = irrep_match.group(1).strip()

                        freq_data.append({
                            'mode_start': mode_start,
                            'mode_end': mode_end,
                            'eigenvalue': eigv,
                            'frequency_cm': freq_cm,
                            'frequency_thz': freq_thz,
                            'irrep': irrep,
                            'ir_active': ir_active,
                            'raman_active': raman_active,
                            'ir_intensity': intensity
                        })
                    except (ValueError, IndexError):
                        continue
            
            if freq_data:
                props['vibrational_frequencies'] = freq_data
                # Number of printed mode rows. Periodic outputs group degenerate
                # modes into one row (e.g. "4-   6"); num_vibrational_modes is
                # the number of parsed rows so it matches len(vibrational_frequencies).
                props['num_vibrational_modes'] = len(freq_data)
                # Degeneracy-expanded count (sums each row's mode range) — this
                # is the 3N-consistent number consumers should use for mode
                # bookkeeping; the row count above is kept for back-compat.
                props['num_vibrational_modes_total'] = sum(
                    (f['mode_end'] - f['mode_start'] + 1) for f in freq_data
                )

                # Imaginary modes are flagged by a NEGATIVE frequency in CRYSTAL
                # output (no trailing 'i'); use the cm**-1 sign as the primary
                # signal. This is always set when modes were parsed.
                props['num_imaginary_frequencies'] = sum(
                    1 for f in freq_data if f['frequency_cm'] < 0
                )

                # Highest / lowest among the genuine (non-translational) modes.
                non_zero_freqs = [f['frequency_cm'] for f in freq_data if f['frequency_cm'] > 0.1]
                if non_zero_freqs:
                    props['lowest_frequency_cm'] = min(non_zero_freqs)
                    props['highest_frequency_cm'] = max(non_zero_freqs)
        
        # Extract vibrational temperatures
        vib_temp_match = re.search(r'VIBRATIONAL TEMPERATURES \(K\).*?\n\s*TO MODES\s*\n(.*?)(?=\*{5,}|\n\n)', content, re.DOTALL)
        if vib_temp_match:
            temps = []
            temp_lines = vib_temp_match.group(1).strip().split('\n')
            for line in temp_lines:
                # Extract temperatures and mode info
                temp_values = re.findall(r'([\d.]+)\s*\[\s*(\d+);([^]]+)\]', line)
                for temp, mode, irrep in temp_values:
                    temps.append({
                        'temperature_k': float(temp),
                        'mode_number': int(mode),
                        'irrep': irrep.strip()
                    })
            if temps:
                props['vibrational_temperatures'] = temps
        
        # Extract thermodynamic properties (includes electronic_energy EL,
        # Gibbs, and enthalpy H = Gibbs + TS — see _extract_thermodynamic_properties)
        thermo_props = self._extract_thermodynamic_properties(content)
        props.update(thermo_props)

        # Extract zero-point energy
        zpe_match = re.search(r'E0\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', content)
        if zpe_match:
            props['zero_point_energy_au'] = float(zpe_match.group(1))
            props['zero_point_energy_ev'] = float(zpe_match.group(2))
            props['zero_point_energy_kj_mol'] = float(zpe_match.group(3))
        
        # Extract force constants information
        force_const_match = re.search(r'FORCE CONSTANT MATRIX.*?MAX ABS\(DGRAD\).*?\n(.*?)(?=GCALCO|\n\n)', content, re.DOTALL)
        if force_const_match:
            props['has_force_constants'] = True
            
            # Count number of displaced calculations
            displaced_atoms = len(re.findall(r'^\s*\d+\s+\w+\s+D[XYZ]', force_const_match.group(1), re.MULTILINE))
            if displaced_atoms > 0:
                props['num_displaced_calculations'] = displaced_atoms
        
        # Extract symmetry-allowed directions
        symm_dirs_match = re.search(r'SYMMETRY ALLOWED INTERNAL DEGREE\(S\) OF FREEDOM:\s*(\d+)', content)
        if symm_dirs_match:
            props['symmetry_allowed_dof'] = int(symm_dirs_match.group(1))
        else:
            no_symm_match = re.search(r'THERE ARE NO SYMMETRY ALLOWED DIRECTIONS', content)
            if no_symm_match:
                props['symmetry_allowed_dof'] = 0
        
        # Extract normal mode displacements if available
        normal_modes_match = re.search(r'NORMAL MODES NORMALIZED TO CLASSICAL AMPLITUDES.*?\n(.*?)(?=\*{5,}|VIBRATIONAL)', content, re.DOTALL)
        if normal_modes_match:
            props['has_normal_mode_displacements'] = True

        return props

    # ------------------------------------------------------------------
    # Thermoelectric / electronic transport (CRYSTAL BOLTZTRA / BoltzTraP)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_transport_dat(dat_file: Path):
        """Parse one CRYSTAL/BoltzTraP transport ``.dat`` file.

        These files store a tensor property as a function of chemical potential
        ``Mu(eV)`` and temperature ``T(K)``.  The per-temperature block markers
        (``# T(K) = 300.000 Npoints = ...``) are *commented*, so the temperature
        is taken from column 2 of every data row (which carries the same value
        and is therefore robust to the comment style).  Layout::

            # 0 [<property> in SI units, i.e. in <unit>]
            # Mu(eV) T(K) N(#carriers) <tensor components...>
            # T(K) = 300.000 Npoints = 711 V(cm^3) = 1.123E-21
              -3.400  300.000  1.351E+00  <xx> <xy> ... <zz>

        The off-diagonal components are ~0 for these cubic-ish cells; we keep the
        isotropic average of the diagonal (xx+yy+zz)/3 as the scalar property.

        Returns ``(unit_str_or_None, rows)`` where each row is the native tuple
        ``(T, mu, n_carriers, diag_avg)``.  Never raises: returns ``(None, [])``
        on a missing/unreadable/malformed file.
        """
        try:
            if dat_file is None or not dat_file.exists():
                return None, []
            text = dat_file.read_text(errors="ignore")
        except OSError:
            return None, []

        unit = None
        rows = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                # Units descriptor: "# 0 [Seebeck coefficient ... i.e. in V/K]".
                # Skip the column-header comment ("# Mu(eV) T(K) ...").
                if unit is None and 'units' in s.lower() and 'mu(ev)' not in s.lower():
                    m = re.search(r'\[(.*)\]', s)
                    if m:
                        desc = m.group(1).strip()
                        # The descriptor is verbose ("Seebeck coefficient in SI
                        # units, i.e. in V/K"); keep just the clean trailing unit
                        # ("... in V/K" -> "V/K"). Fall back to the full descriptor
                        # if no "in <unit>" tail is present.
                        um = re.search(r'\bin\s+(\S+)\s*$', desc)
                        unit = um.group(1) if um else desc
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if not all(math.isfinite(v) for v in vals):
                continue
            mu, T, n_carr = vals[0], vals[1], vals[2]
            tensor = vals[3:]
            n = len(tensor)
            # Diagonal of the property tensor depends on the column layout:
            #   9 cols -> full 3x3 (SEEBECK): xx,xy,xz,yx,yy,yz,zx,zy,zz
            #   6 cols -> symmetric (SIGMA/KAPPA): xx,xy,xz,yy,yz,zz
            if n == 9:
                diag = (tensor[0], tensor[4], tensor[8])
            elif n == 6:
                diag = (tensor[0], tensor[3], tensor[5])
            elif n >= 3:
                diag = (tensor[0], tensor[1], tensor[2])
            else:
                diag = (tensor[0],)
            rows.append((T, mu, n_carr, sum(diag) / len(diag)))
        return unit, rows

    def _extract_transport_properties(self, content: str, output_file: Path) -> Dict[str, Any]:
        """Extract thermoelectric / electronic transport properties.

        Detects a CRYSTAL "THERMOELECTRIC AND ELECTRONIC TRANSPORT PROPERTIES
        CALCULATION" (BOLTZTRA) run and, when its companion BoltzTraP ``.dat``
        files are present next to the ``.out`` (``*.SEEBECK.dat`` / ``*.SIGMA.dat``
        / ``*.KAPPA.dat``), stores:

          * a compact JSON ``transport`` summary (temperatures covered, mu range,
            peak |Seebeck| with its (T, mu) and carrier type, peak power factor,
            peak electronic ZT, units, and the source data files referenced), and
          * flat queryable scalar rows (``transport_seebeck_max_uv_per_k``,
            ``transport_power_factor_max``, ``transport_zt_max``,
            ``transport_n_temperatures``).

        Mirrors ``_build_band_structure_summary``: a compact payload + flat
        scalars, native python types, no schema change (the dict/list is
        ``json.dumps``-ed into ``property_value_text``).  Only peaks + a reference
        to the ``.dat`` are stored — never the full multi-thousand-row tables.
        Failures never propagate (no transport key emitted on absent/malformed
        data).

        Physical guards (these BoltzTraP outputs are noisy):
          * BoltzTraP sets S to a huge value where "THE DETERMINANT OF THE
            CONDUCTIVITY MATRIX IS VERY SMALL"; we only evaluate operating points
            whose electrical conductivity is within 1e-3 of the peak |sigma| so
            those determinant-singularity spikes are excluded.
          * Power factor PF = S^2 * sigma; electronic figure of merit
            ZT_e = PF * T / kappa_e (electronic kappa only -- no lattice term, so
            it is an upper bound the summary labels ``zt_electronic``).
        """
        props: Dict[str, Any] = {}

        # Only act on a real transport (BOLTZTRA) run.
        up = content.upper()
        if ('THERMOELECTRIC AND ELECTRONIC TRANSPORT' not in up
                and 'BOLTZTRA' not in up):
            return props

        try:
            props['calculation_type'] = 'TRANSPORT'
            props['has_transport'] = True

            # Temperature / chemical-potential grid from the .out header (if present).
            def _grab(label):
                m = re.search(label + r'\s*[:=]?\s*(-?\d+\.\d+)', content)
                return float(m.group(1)) if m else None

            t_min = _grab(r'MIN TEMPERATURE VALUE \(K\)')
            t_max = _grab(r'MAX TEMPERATURE VALUE \(K\)')
            mu_min_hdr = _grab(r'MIN CHEMICAL POTENTIAL VALUE \(eV\)')
            mu_max_hdr = _grab(r'MAX CHEMICAL POTENTIAL VALUE \(eV\)')

            # Locate companion BoltzTraP .dat files (case-insensitive stem match).
            out_dir = output_file.parent
            stem = output_file.stem

            def _find(suffix):
                # Prefer the exact-stem companion; fall back to a companion
                # whose stem starts at a separator boundary (same rule as the
                # CUBE sidecar match) so a sibling material's data in a shared
                # directory is never silently attributed to this one.
                exact = out_dir / f"{stem}.{suffix}"
                if exact.exists():
                    return exact
                cands = sorted(set(list(out_dir.glob(f"*.{suffix}"))
                                   + list(out_dir.glob(f"*.{suffix.lower()}"))))
                for c in cands:
                    c_stem = c.name[:-(len(suffix) + 1)]
                    if (c_stem == stem or c_stem.startswith(stem + '_')
                            or stem.startswith(c_stem + '_')):
                        return c
                # Unambiguous single companion: accept it (common layout —
                # one material per calc dir with differently-styled names).
                return cands[0] if len(cands) == 1 else None

            seebeck_file = _find('SEEBECK.dat')
            sigma_file = _find('SIGMA.dat')
            kappa_file = _find('KAPPA.dat')

            if seebeck_file is None:
                # No data to summarize beyond the calc-type marker.
                props['transport_data_available'] = False
                return props

            s_unit, s_rows = self._parse_transport_dat(seebeck_file)
            _, sig_rows = self._parse_transport_dat(sigma_file)
            _, kap_rows = self._parse_transport_dat(kappa_file)

            if not s_rows:
                props['transport_data_available'] = False
                return props

            # (T, mu) -> property lookups (round mu to align grids across files).
            sig_lut = {(round(T, 3), round(mu, 4)): v
                       for (T, mu, _nc, v) in sig_rows}
            kap_lut = {(round(T, 3), round(mu, 4)): v
                       for (T, mu, _nc, v) in kap_rows}

            sig_abs = [abs(v) for v in sig_lut.values() if v != 0.0]
            sig_max = max(sig_abs) if sig_abs else 0.0
            sig_floor = sig_max * 1e-3  # exclude determinant-singularity points
            # SEEBECK-only runs (no SIGMA.dat): the conductivity gate would
            # otherwise skip every row and store misleading zeros for a valid
            # Seebeck dataset. Compute Seebeck statistics unconditionally and
            # leave only the sigma-dependent PF/ZT unavailable.
            have_sigma = bool(sig_lut)

            temperatures = sorted({round(r[0], 3) for r in s_rows})
            mus = [r[1] for r in s_rows]
            mu_lo = min(mus) if mus else None
            mu_hi = max(mus) if mus else None

            peak_s = 0.0           # signed Seebeck (V/K) at peak |S|
            peak_s_loc = None      # (T, mu)
            peak_s_ncarr = None
            peak_pf = 0.0          # W/m/K^2
            peak_pf_loc = None
            peak_zt = 0.0          # dimensionless (electronic only)
            peak_zt_loc = None
            n_usable = 0

            for (T, mu, n_carr, s_val) in s_rows:
                key = (round(T, 3), round(mu, 4))
                g = sig_lut.get(key)
                if have_sigma:
                    # Require a physically meaningful conductivity at this point.
                    if sig_floor <= 0.0 or g is None or abs(g) < sig_floor:
                        continue
                n_usable += 1
                if abs(s_val) > abs(peak_s):
                    peak_s = s_val
                    peak_s_loc = (T, mu)
                    peak_s_ncarr = n_carr
                if g is not None:
                    pf = s_val * s_val * abs(g)
                    if pf > peak_pf:
                        peak_pf = pf
                        peak_pf_loc = (T, mu)
                    k = kap_lut.get(key)
                    if k and abs(k) > 0.0:
                        zt = pf * T / abs(k)
                        if zt > peak_zt:
                            peak_zt = zt
                            peak_zt_loc = (T, mu)

            has_usable = abs(peak_s) > 0.0
            # Carrier-type convention: N(#carriers) > 0 -> holes (p-type),
            # N(#carriers) < 0 -> electrons (n-type), matching the sign that
            # accompanies the dominant Seebeck operating point.
            carrier_type = None
            if peak_s_ncarr is not None and peak_s_ncarr != 0.0:
                carrier_type = 'p-type' if peak_s_ncarr > 0 else 'n-type'

            def _loc(loc):
                return {'temperature_k': float(loc[0]),
                        'mu_ev': float(loc[1])} if loc else None

            summary = {
                'source_files': [f.name for f in
                                 (seebeck_file, sigma_file, kappa_file) if f],
                'seebeck_source_file': seebeck_file.name,
                'temperatures_k': [float(t) for t in temperatures],
                'n_temperatures': int(len(temperatures)),
                'temperature_min_k': (float(t_min) if t_min is not None
                                      else (float(min(temperatures))
                                            if temperatures else None)),
                'temperature_max_k': (float(t_max) if t_max is not None
                                      else (float(max(temperatures))
                                            if temperatures else None)),
                'mu_min_ev': float(mu_min_hdr) if mu_min_hdr is not None
                             else (float(mu_lo) if mu_lo is not None else None),
                'mu_max_ev': float(mu_max_hdr) if mu_max_hdr is not None
                             else (float(mu_hi) if mu_hi is not None else None),
                'n_data_points': int(len(s_rows)),
                'n_usable_points': int(n_usable),
                'has_usable_data': bool(has_usable),
                # False => SEEBECK-only run: PF/ZT peaks below are not computed.
                'sigma_data_available': bool(have_sigma),
                'carrier_type': carrier_type,
                'tensor_reduction': ('diagonal isotropic average (xx+yy+zz)/3; '
                                     'off-diagonal tensor components discarded'),
                'units': {
                    'seebeck': s_unit or 'V/K',
                    'electrical_conductivity': '1/Ohm/m',
                    'electronic_thermal_conductivity': 'W/m/K',
                    'power_factor': 'W/m/K^2',
                    'zt': 'dimensionless (electronic, kappa_lattice excluded)',
                },
                'seebeck_max_signed_v_per_k': float(peak_s),
                'seebeck_max_abs_uv_per_k': float(abs(peak_s) * 1.0e6),
                'seebeck_max_at': _loc(peak_s_loc),
                'power_factor_max_w_per_m_k2': (float(peak_pf) if have_sigma
                                                else None),
                'power_factor_max_at': _loc(peak_pf_loc),
                'zt_electronic_max': (float(peak_zt) if have_sigma else None),
                'zt_electronic_max_at': _loc(peak_zt_loc),
            }
            props['transport'] = summary

            # Flat, numerically-queryable scalar rows.
            props['transport_n_temperatures'] = int(len(temperatures))
            props['transport_n_data_points'] = int(len(s_rows))
            props['transport_data_available'] = bool(has_usable)
            if has_usable:
                props['transport_seebeck_max_uv_per_k'] = float(abs(peak_s) * 1.0e6)
                if have_sigma:
                    # PF/ZT need conductivity; for SEEBECK-only runs storing
                    # 0.0 here would be the same misleading-zero bug.
                    props['transport_power_factor_max'] = float(peak_pf)
                    props['transport_zt_max'] = float(peak_zt)
                if carrier_type:
                    props['transport_carrier_type'] = carrier_type

        except Exception as e:  # never let transport extraction break the run
            print(f"Warning: Could not extract transport properties: {e}")

        return props

    def _extract_scf_settings(self, content: str) -> Dict[str, Any]:
        """Extract SCF and advanced electronic settings from output file."""
        props = {}
        
        # Extract SCF parameters reported in output
        # INFORMATION lines pattern - these show actual settings used
        info_patterns = {
            'scf_biposize': r'INFORMATION \*+\s*BIPOSIZE\s*\*+.*?COULOMB BIPOLAR BUFFER SET TO\s+(\d+)',
            'scf_exchsize': r'INFORMATION \*+\s*EXCHSIZE\s*\*+.*?EXCHANGE BIPOLAR BUFFER SIZE SET TO\s+(\d+)',
            'scf_maxcycle': r'INFORMATION \*+\s*MAXCYCLE\s*\*+.*?MAX NUMBER OF SCF CYCLES SET TO\s+(\d+)',
            'scf_ppan': r'INFORMATION \*+\s*PPAN\s*\*+.*?MULLIKEN POPULATION ANALYSIS',
            'scf_toldee': r'INFORMATION \*+\s*TOLDEE\s*\*+.*?SCF TOL ON TOTAL ENERGY SET TO\s+(\d+)',
            'scf_tolinteg': r'INFORMATION \*+\s*TOLINTEG\s*\*+.*?COULOMB AND EXCHANGE SERIES TOLERANCES',
            'scf_diis': r'INFORMATION \*+\s*DIIS\s*\*+.*?DIIS FOR SCF ACTIVE',
            'scf_scfdir': r'INFORMATION \*+\s*SCFDIR\s*\*+.*?DIRECT SCF',
            'scf_savewf': r'INFORMATION \*+\s*SAVEWF\s*\*+',
            'scf_savepred': r'INFORMATION \*+\s*SAVEPRED\s*\*+'
        }
        
        for param, pattern in info_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                if param in ['scf_ppan', 'scf_diis', 'scf_scfdir', 'scf_savewf', 'scf_savepred']:
                    props[param] = True
                elif param in ['scf_biposize', 'scf_exchsize', 'scf_maxcycle', 'scf_toldee']:
                    props[param] = int(match.group(1))
                else:
                    props[param] = True
        
        # Check for level shifter status
        if 'LEVEL SHIFTER DISABLED' in content:
            props['scf_levshift_enabled'] = False
        elif 'LEVEL SHIFTER ACTIVE' in content or 'LEVSHIFT' in content:
            props['scf_levshift_enabled'] = True
            
        # Extract shrink factors from output
        shrink_match = re.search(r'SHRINK\.\s*FACT\.\(MONKH\.\)\s+(\d+)\s+(\d+)\s+(\d+)', content)
        if shrink_match:
            props['scf_shrink_k1'] = int(shrink_match.group(1))
            props['scf_shrink_k2'] = int(shrink_match.group(2))
            props['scf_shrink_k3'] = int(shrink_match.group(3))
        
        # Extract Gilat shrink factor
        gilat_match = re.search(r'SHRINKING FACTOR\(GILAT NET\)\s+(\d+)', content)
        if gilat_match:
            props['scf_gilat_shrink'] = int(gilat_match.group(1))
        
        # FMIXING percentage
        fmixing_match = re.search(r'FMIXING\s*[:=]\s*(\d+)', content)
        if fmixing_match:
            props['scf_fmixing_percentage'] = int(fmixing_match.group(1))
        
        # Anderson mixing
        anderson_match = re.search(r'ANDERSON MIXING.*?FACTOR\s*[:=]\s*([\d.]+)', content)
        if anderson_match:
            props['scf_anderson_factor'] = float(anderson_match.group(1))
            props['scf_anderson_enabled'] = True
        
        # DIIS mixing parameters
        if 'DIIS FOR SCF ACTIVE' in content:
            props['scf_diis_enabled'] = True
            
            # Extract DIIS parameters
            diis_start_match = re.search(r'DIIS STARTING FROM CYCLE\s*(\d+)', content)
            if diis_start_match:
                props['scf_diis_start_cycle'] = int(diis_start_match.group(1))
                
            diis_size_match = re.search(r'DIIS SUBSPACE SIZE\s*[:=]\s*(\d+)', content)
            if diis_size_match:
                props['scf_diis_subspace_size'] = int(diis_size_match.group(1))
        
        # Broyden mixing
        broyden_match = re.search(r'BROYDEN MIXING.*?FACTOR\s*[:=]\s*([\d.]+)', content)
        if broyden_match:
            props['scf_broyden_factor'] = float(broyden_match.group(1))
            props['scf_broyden_enabled'] = True
        
        # Level shifter details
        levshift_match = re.search(r'LEVSHIFT\s*[:=]\s*([\d.]+)\s*CYCLES\s*[:=]\s*(\d+)', content)
        if levshift_match:
            props['scf_levshift_factor'] = float(levshift_match.group(1))
            props['scf_levshift_cycles'] = int(levshift_match.group(2))
        
        # Extract integration pack parameters
        intgpack_patterns = {
            'scf_ilasize': r'ILASIZE\s*[:=]\s*(\d+)',
            'scf_intgpack': r'INTGPACK\s*[:=]\s*(\d+)',
            'scf_madelimit': r'MADELIMIT\s*[:=]\s*(\d+)',
            'scf_poleordr': r'POLEORDR\s*[:=]\s*(\d+)'
        }
        
        for param, pattern in intgpack_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                props[param] = int(match.group(1))
        
        # Extract exchange parameters
        if 'EXCHPERM' in content:
            props['scf_exchange_permutation'] = True
            
        if 'BIPOLARIZ' in content:
            props['scf_bipolarization'] = True
        
        # Extract DFT grid information
        grid_match = re.search(r'DFT GRID.*?(\d+)\s+POINTS', content)
        if grid_match:
            props['scf_dft_grid_points'] = int(grid_match.group(1))
            
        # Extract pruning information
        if 'PRUNING' in content:
            props['scf_grid_pruning'] = True
        
        return props
    
    def _extract_thermodynamic_properties(self, content: str) -> Dict[str, Any]:
        """Extract thermodynamic properties from FREQ calculations."""
        props = {}
        
        # Temperature and pressure conditions
        conditions_match = re.search(r'AT \(T =\s*([\d.]+)\s*K,\s*P =\s*([\d.E+-]+)\s*MPA\)', content)
        if conditions_match:
            props['thermodynamic_temperature_k'] = float(conditions_match.group(1))
            props['thermodynamic_pressure_mpa'] = float(conditions_match.group(2))
        
        # Electronic energy (EL) — the baseline total energy printed at the top
        # of the harmonic thermodynamic block; H and G are built on top of it.
        el_match = re.search(
            r'^\s*EL\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)',
            content, re.MULTILINE
        )
        if el_match:
            props['electronic_energy_au'] = float(el_match.group(1))
            props['electronic_energy_ev'] = float(el_match.group(2))
            props['electronic_energy_kj_mol'] = float(el_match.group(3))

        # Extract thermodynamic functions. CRYSTAL labels the ET/PV/TS block
        # "...WITH VIBRATIONAL CONTRIBUTIONS" for periodic systems but
        # "...TAKING INTO ACCOUNT MOLECULAR ..." for molecules — match both,
        # else molecular FREQ runs lose Gibbs/ET/PV/TS entirely.
        thermo_section = re.search(
            r'THERMODYNAMIC FUNCTIONS '
            r'(?:WITH VIBRATIONAL CONTRIBUTIONS|TAKING INTO ACCOUNT MOLECULAR).*?'
            r'AU/CELL\s+EV/CELL\s+KJ/MOL\s*\n(.*?)(?=OTHER THERMODYNAMIC|\*{5,})',
            content, re.DOTALL
        )
        
        if thermo_section:
            thermo_content = thermo_section.group(1)
            
            # ET (thermal energy)
            et_match = re.search(r'ET\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', thermo_content)
            if et_match:
                props['thermal_energy_au'] = float(et_match.group(1))
                props['thermal_energy_ev'] = float(et_match.group(2))
                props['thermal_energy_kj_mol'] = float(et_match.group(3))
            
            # PV
            pv_match = re.search(r'PV\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', thermo_content)
            if pv_match:
                props['pv_term_au'] = float(pv_match.group(1))
                props['pv_term_ev'] = float(pv_match.group(2))
                props['pv_term_kj_mol'] = float(pv_match.group(3))
            
            # TS (entropy term)
            ts_match = re.search(r'TS\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', thermo_content)
            if ts_match:
                props['entropy_term_au'] = float(ts_match.group(1))
                props['entropy_term_ev'] = float(ts_match.group(2))
                props['entropy_term_kj_mol'] = float(ts_match.group(3))
            
            # ET+PV-TS (free energy correction)
            correction_match = re.search(r'ET\+PV-TS\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', thermo_content)
            if correction_match:
                props['free_energy_correction_au'] = float(correction_match.group(1))
                props['free_energy_correction_ev'] = float(correction_match.group(2))
                props['free_energy_correction_kj_mol'] = float(correction_match.group(3))
            
            # Total free energy (EL+E0+ET+PV-TS)
            total_match = re.search(r'EL\+E0\+ET\+PV-TS:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', thermo_content)
            if total_match:
                props['gibbs_free_energy_au'] = float(total_match.group(1))
                props['gibbs_free_energy_ev'] = float(total_match.group(2))
                props['gibbs_free_energy_kj_mol'] = float(total_match.group(3))

            # Enthalpy H = EL + E0 + ET + PV = Gibbs + TS. Built from CRYSTAL's
            # own printed columns (per unit) so it carries no conversion drift.
            if 'gibbs_free_energy_au' in props and 'entropy_term_au' in props:
                props['enthalpy_au'] = props['gibbs_free_energy_au'] + props['entropy_term_au']
                props['enthalpy_ev'] = props['gibbs_free_energy_ev'] + props['entropy_term_ev']
                props['enthalpy_kj_mol'] = props['gibbs_free_energy_kj_mol'] + props['entropy_term_kj_mol']
        
        # Extract entropy and heat capacity
        other_thermo = re.search(
            r'OTHER THERMODYNAMIC FUNCTIONS:.*?'
            r'mHARTREE/\(CELL\*K\)\s+mEV/\(CELL\*K\)\s+J/\(MOL\*K\)\s*\n(.*?)(?=\*{5,}|TTTT)',
            content, re.DOTALL
        )
        
        if other_thermo:
            other_content = other_thermo.group(1)
            
            # Entropy
            entropy_match = re.search(r'ENTROPY\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', other_content)
            if entropy_match:
                props['entropy_mhartree_cell_k'] = float(entropy_match.group(1))
                props['entropy_mev_cell_k'] = float(entropy_match.group(2))
                props['entropy_j_mol_k'] = float(entropy_match.group(3))
            
            # Heat capacity
            heat_cap_match = re.search(r'HEAT CAPACITY\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', other_content)
            if heat_cap_match:
                props['heat_capacity_mhartree_cell_k'] = float(heat_cap_match.group(1))
                props['heat_capacity_mev_cell_k'] = float(heat_cap_match.group(2))
                props['heat_capacity_j_mol_k'] = float(heat_cap_match.group(3))
        
        return props
    
    def _categorize_property(self, prop_name: str) -> str:
        """Categorize a property based on its name."""
        if any(x in prop_name.lower() for x in ['primitive_a', 'primitive_b', 'primitive_c', 'primitive_alpha', 'primitive_beta', 'primitive_gamma']):
            return 'lattice'
        elif any(x in prop_name.lower() for x in ['cell', 'volume', 'density', 'atomic', 'position']):
            return 'structural'
        elif any(x in prop_name.lower() for x in ['band_gap', 'energy', 'gap', 'electronic']):
            return 'electronic'
        elif any(x in prop_name.lower() for x in ['mulliken', 'overlap', 'charge', 'population']):
            return 'population_analysis'
        elif any(x in prop_name.lower() for x in ['optimization', 'gradient', 'converged', 'cycles']):
            return 'optimization'
        elif any(x in prop_name.lower() for x in ['space_group', 'crystal_system', 'centering']):
            return 'crystallographic'
        elif any(x in prop_name.lower() for x in ['cpu_time', 'scf_cycles', 'memory', 'fermi_energy']):
            return 'computational'
        elif any(x in prop_name.lower() for x in ['band_', 'total_kpoints', 'total_bands', 'vbm_', 'cbm_', 'has_band_structure']):
            return 'band_structure'
        elif any(x in prop_name.lower() for x in ['dos_', 'doss_', 'has_dos']):
            return 'density_of_states'
        elif any(x in prop_name.lower() for x in ['electronic_classification', 'classification_', 'insulator_type', 'semiconductor_type', 'magnetic_', 'conductivity_type']):
            return 'electronic_classification'
        elif any(x in prop_name.lower() for x in ['freq', 'vibrational', 'thermodynamic', 'entropy', 'heat_capacity', 'zero_point', 'gibbs', 'thermal_energy', 'force_constant']):
            return 'frequency'
        else:
            return 'other'
    
    def _get_property_unit(self, prop_name: str) -> str:
        """Get the appropriate unit for a property."""
        prop_lower = prop_name.lower()
        
        # Energy properties (check specific patterns first)
        if any(x in prop_lower for x in ['energy', 'gap']) and prop_name.endswith('_ev'):
            return 'eV'
        elif any(x in prop_lower for x in ['energy', 'gap']) and prop_name.endswith('_au'):
            return 'Hartree'
        elif 'band_gap' in prop_lower:
            return 'eV'
        elif 'fermi_energy' in prop_lower:
            # CRYSTAL prints FERMI ENERGY / EFERMI(AU) in Hartree and the
            # extractor stores the raw value
            return 'Hartree'
        elif prop_lower.endswith('_energy') and not 'lattice' in prop_lower:
            return 'eV'
        
        # Angles MUST come before length parameters to avoid conflicts
        elif any(x in prop_lower for x in ['alpha', 'beta', 'gamma']) and any(y in prop_lower for y in ['primitive', 'crystallographic', 'cell']):
            return 'degrees'
        
        # Volumes MUST come before length parameters
        elif 'volume' in prop_lower:
            return 'Å³'
        
        # Density
        elif 'density' in prop_lower:
            return 'g/cm³'
        
        # Length parameters (lattice constants, distances)
        elif any(x in prop_lower for x in ['primitive_a', 'primitive_b', 'primitive_c', 'crystallographic_a', 'crystallographic_b', 'crystallographic_c']):
            return 'Å'
        elif 'distance' in prop_lower:
            return 'Å'
        
        # Advanced electronic properties
        elif 'effective_mass' in prop_lower:
            return 'm_e'  # electron mass units
        elif 'mobility' in prop_lower:
            return 'cm²/(V·s)'
        elif 'conductivity' in prop_lower and 'classification' not in prop_lower:
            return 'S/m'  # Siemens per meter
        elif 'seebeck' in prop_lower:
            return 'μV/K'  # microvolts per Kelvin
        elif 'thermal_conductivity' in prop_lower:
            return 'W/(m·K)'
        elif 'dielectric' in prop_lower and 'constant' in prop_lower:
            return 'dimensionless'
        elif 'refractive_index' in prop_lower:
            return 'dimensionless'
        elif 'absorption' in prop_lower:
            return 'cm⁻¹'
        
        # Count properties (dimensionless)
        elif any(x in prop_name.lower() for x in ['atoms_count', 'coordination_number', 'atoms_in_unit_cell', 'shells']):
            return 'dimensionless'
        
        # Population analysis
        elif 'mulliken' in prop_name.lower():
            return 'electrons'
        elif 'overlap_population' in prop_name.lower():
            return 'dimensionless'
        
        # Boolean properties
        elif any(x in prop_name.lower() for x in ['converged', 'polarized']):
            return 'boolean'
        
        # Cycles and iterations
        elif 'cycles' in prop_name.lower():
            return 'cycles'
        
        # Gradients
        elif 'gradient' in prop_name.lower():
            return 'Hartree/Bohr'
        
        # Codes and identifiers
        elif any(x in prop_name.lower() for x in ['code', 'centering']):
            return 'code'
        elif 'space_group' in prop_name.lower():
            return 'number'
        
        # Positions and coordinates
        elif 'position' in prop_name.lower():
            return 'coordinates'
        
        # Complex data (JSON)
        elif any(x in prop_name.lower() for x in ['processed_', 'neighbor_analysis']):
            return 'JSON'
        
        # Computational properties
        elif 'cpu_time' in prop_name.lower() or 'time' in prop_name.lower():
            return 'seconds'
        elif 'fermi_energy' in prop_name.lower():
            return 'eV'
        elif 'scf_cycles' in prop_name.lower():
            return 'cycles'
        elif 'memory' in prop_name.lower():
            return 'MB'
        
        # Band structure properties
        elif any(x in prop_name.lower() for x in ['fermi_energy_band', 'fermi_energy_dos', 'vbm_energy', 'cbm_energy']):
            return 'eV'
        elif any(x in prop_name.lower() for x in ['total_kpoints', 'total_bands', 'band_start', 'band_end', 'dos_energy_points']):
            return 'dimensionless'
        elif any(x in prop_name.lower() for x in ['dos_energy_min', 'dos_energy_max', 'dos_energy_range', 'dos_broadening']):
            return 'eV'
        elif any(x in prop_name.lower() for x in ['dos_at_fermi']):
            return 'states/eV'
        elif any(x in prop_name.lower() for x in ['band_dat_size', 'doss_dat_size']):
            return 'bytes'
        elif any(x in prop_name.lower() for x in ['band_dat_exists', 'doss_dat_exists', 'has_band_structure', 'has_dos']):
            return 'boolean'
        
        # Electronic classification properties
        elif any(x in prop_name.lower() for x in ['electronic_classification', 'insulator_type', 'semiconductor_type', 'magnetic_classification', 'magnetic_type', 'conductivity_type', 'classification_gap_source']):
            return 'category'
        elif 'classification_band_gap' in prop_name.lower():
            return 'eV'
        
        # Frequency calculation properties
        elif 'frequency_cm' in prop_name.lower() or 'frequency_cm' in prop_name.lower():
            return 'cm⁻¹'
        elif 'frequency_thz' in prop_name.lower():
            return 'THz'
        elif 'vibrational_temperature' in prop_name.lower() or 'temperature_k' in prop_name.lower():
            return 'K'
        elif 'zero_point_energy_au' in prop_name.lower():
            return 'Hartree'
        elif 'zero_point_energy_ev' in prop_name.lower():
            return 'eV'
        elif 'zero_point_energy_kj_mol' in prop_name.lower():
            return 'kJ/mol'
        elif any(x in prop_name.lower() for x in ['thermal_energy_au', 'pv_term_au', 'entropy_term_au', 'free_energy_correction_au', 'gibbs_free_energy_au', 'enthalpy_au', 'enthalpy_total_au']):
            return 'Hartree'
        elif any(x in prop_name.lower() for x in ['thermal_energy_ev', 'pv_term_ev', 'entropy_term_ev', 'free_energy_correction_ev', 'gibbs_free_energy_ev', 'enthalpy_ev', 'enthalpy_total_ev']):
            return 'eV'
        elif any(x in prop_name.lower() for x in ['thermal_energy_kj_mol', 'pv_term_kj_mol', 'entropy_term_kj_mol', 'free_energy_correction_kj_mol', 'gibbs_free_energy_kj_mol', 'enthalpy_kj_mol', 'enthalpy_total_kj_mol']):
            return 'kJ/mol'
        elif 'entropy_mhartree_cell_k' in prop_name.lower() or 'heat_capacity_mhartree_cell_k' in prop_name.lower():
            return 'mHartree/(cell·K)'
        elif 'entropy_mev_cell_k' in prop_name.lower() or 'heat_capacity_mev_cell_k' in prop_name.lower():
            return 'meV/(cell·K)'
        elif 'entropy_j_mol_k' in prop_name.lower() or 'heat_capacity_j_mol_k' in prop_name.lower():
            return 'J/(mol·K)'
        elif 'pressure_mpa' in prop_name.lower():
            return 'MPa'
        elif 'eigenvalue' in prop_name.lower():
            return 'Hartree²'
        elif 'ir_intensity' in prop_name.lower():
            return 'km/mol'
        elif any(x in prop_name.lower() for x in ['num_vibrational_modes', 'num_imaginary_frequencies', 'num_displaced_calculations', 'symmetry_allowed_dof', 'mode_number']):
            return 'dimensionless'
        elif any(x in prop_name.lower() for x in ['has_frequency_data', 'has_force_constants', 'has_normal_mode_displacements', 'ir_active', 'raman_active']):
            return 'boolean'
        elif 'irrep' in prop_name.lower():
            return 'symmetry_label'
        
        # Default for unknown properties
        else:
            return 'dimensionless'


    def _extract_charge_potential_properties(self, content: str, output_file: Path) -> Dict[str, Any]:
        """Extract CHARGE+POTENTIAL (CRYSTAL ECH3/POT3) metadata.

        A CRYSTAL ECH3 (3D charge density) / POT3 (3D electrostatic potential)
        job stores its real numerical payload -- the 3D grids -- in BINARY/large
        sidecar files (``*_DENS.CUBE``, ``*_POT.CUBE``, ``*_SPIN.CUBE``), NOT in
        the .out text. A text extractor must NOT parse those grids. So, mirroring
        ``_build_band_structure_summary``, this stores ONLY what the .out text
        honestly exposes plus *references* to the grid files on disk:

          * confirmation that ECH3 / POT3 ran (the ``ECH3``/``POT3`` timing banners),
          * the grid divisions (``3D DENSITY, GRID IS nx*ny*nz``) and derived
            number of grid points,
          * the real-space bounding box (``X/Y/Z COORDINATE RANGE IS lo TO hi``),
          * multipolar-expansion order / spin-polarization flag when printed,
          * the SCF total energy / Fermi energy already on the page,
          * the names of the generated ``*.CUBE`` grid files in the same directory
            (so the mace plotting cube viewer, which globs ``*.CUBE``, can find them).

        Returns a compact JSON-serializable ``charge_potential`` summary plus a few
        flat queryable scalar/string rows. All values are native float/int/str/bool
        so they JSON-serialize cleanly. Returns ``{}`` (no keys) when the file has
        no ECH3/POT3 markers, so non-charge calculations are completely untouched.
        Any internal failure degrades gracefully -- it never raises.
        """
        try:
            has_ech3 = re.search(r'\bECH3\s+(?:START|END)\b', content) is not None
            has_pot3 = re.search(r'\bPOT3\s+(?:START|END)\b', content) is not None
            if not (has_ech3 or has_pot3):
                return {}  # not a charge/potential calc -> contribute nothing

            summary: Dict[str, Any] = {
                'source': 'ECH3/POT3 (.out text + .CUBE grid sidecars)',
                'has_charge_density': bool(has_ech3),
                'has_electrostatic_potential': bool(has_pot3),
            }

            # --- Grid divisions: "3D DENSITY, GRID IS nx*ny*nz" (space-padded
            # numbers like "100* 58* 86" appear, so tolerate inner whitespace).
            # ECH3 and POT3 print the same grid; take the first occurrence.
            grid_dims = None
            num_points = None
            gm = re.search(
                r'3D\s+\w+,\s*GRID\s+IS\s+(\d+)\s*\*\s*(\d+)\s*\*\s*(\d+)',
                content, re.IGNORECASE)
            if gm:
                grid_dims = [int(gm.group(1)), int(gm.group(2)), int(gm.group(3))]
                num_points = int(grid_dims[0] * grid_dims[1] * grid_dims[2])
                summary['grid_divisions'] = grid_dims
                summary['num_grid_points'] = num_points

            # --- Real-space bounding box of the grid (Bohr), if printed.
            box = {}
            for axis, lo, hi in re.findall(
                    r'([XYZ])\s+COORDINATE RANGE IS\s+(-?\d+\.\d+)\s+TO\s+(-?\d+\.\d+)',
                    content):
                box[axis.lower()] = [float(lo), float(hi)]
            if box:
                summary['coordinate_range'] = box

            # --- Multipolar expansion order (POT3 section), if printed.
            mm = re.search(r'MULTIPOLAR EXPANSION UP TO L=\s*(\d+)', content)
            if mm:
                summary['multipole_expansion_l'] = int(mm.group(1))

            # --- Spin-polarization flag.
            spin_polarized = 'SPIN POLARIZED SYSTEM' in content
            summary['spin_polarized'] = bool(spin_polarized)

            # --- Scalar metadata already on the page: SCF total energy / Fermi.
            em = re.search(r'TOTAL ENERGY\s+(-?\d+\.\d+E[+-]?\d+)', content)
            scf_energy = None
            if em:
                try:
                    scf_energy = float(em.group(1))
                    summary['scf_total_energy_au'] = scf_energy
                except ValueError:
                    pass
            fm = re.search(r'FERMI ENERGY\s+(-?\d+\.\d+E?[+-]?\d*)', content)
            fermi_energy = None
            if fm:
                try:
                    fermi_energy = float(fm.group(1))
                    summary['fermi_energy_au'] = fermi_energy
                except ValueError:
                    pass

            # --- Which 3D properties were computed (provenance list).
            computed = []
            if has_ech3:
                computed.append('charge_density')
                if spin_polarized:
                    computed.append('spin_density')
            if has_pot3:
                computed.append('electrostatic_potential')
            summary['computed_properties'] = computed

            # --- Reference the generated grid/cube files on disk so the mace
            # plotting cube viewer (which globs *.CUBE in the .out's directory)
            # can find them. We only RECORD names that actually exist -- never
            # parse the binary grids. The match is scoped to this job's stem.
            grid_files = []
            try:
                output_dir = output_file.parent
                stem = output_file.stem  # e.g. ..._charge+potential
                for cube in sorted(output_dir.glob('*.CUBE')):
                    # Require a separator boundary after the stem so a sibling
                    # material whose stem is a PREFIX of this one (e.g. "mat_1"
                    # vs "mat_12_DENS.CUBE") is not mis-attributed. CRYSTAL names
                    # the sidecars "<stem>_DENS/_POT/_SPIN.CUBE".
                    if cube.stem == stem or cube.stem.startswith(stem + '_'):
                        grid_files.append(cube.name)
            except OSError:
                grid_files = []
            if grid_files:
                summary['grid_files'] = grid_files

            props: Dict[str, Any] = {'charge_potential': summary}

            # --- Flat, queryable scalar/string rows (distinct from the JSON blob).
            props['chargepot_has_ech3'] = bool(has_ech3)
            props['chargepot_has_pot3'] = bool(has_pot3)
            props['chargepot_spin_polarized'] = bool(spin_polarized)
            if grid_dims is not None:
                props['chargepot_grid_dims'] = 'x'.join(str(d) for d in grid_dims)
            if num_points is not None:
                props['chargepot_grid_points'] = int(num_points)
            if computed:
                props['chargepot_computed_properties'] = ','.join(computed)
            if grid_files:
                props['chargepot_grid_files'] = ','.join(grid_files)
                props['chargepot_num_grid_files'] = int(len(grid_files))
            if scf_energy is not None:
                props['chargepot_scf_total_energy_au'] = float(scf_energy)

            return props
        except Exception as e:
            print(f"Warning: Could not extract charge/potential properties: {e}")
            return {}

    def _extract_advanced_band_dos_analysis(self, output_file: Path, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract advanced electronic properties using BAND/DOSS data analysis.
        
        Uses sophisticated DOS/BAND analysis for accurate effective mass and classification.
        """
        advanced_props = {}
        
        try:
            try:
                from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
            except ImportError:
                from advanced_electronic_analyzer import AdvancedElectronicAnalyzer
            
            # Look for BAND.DAT and DOSS.DAT files in the same directory
            output_dir = output_file.parent
            
            # Find BAND and DOSS data files
            band_files = list(output_dir.glob("*.BAND.DAT")) + list(output_dir.glob("*_band.BAND.DAT"))
            doss_files = list(output_dir.glob("*.DOSS.DAT")) + list(output_dir.glob("*_doss.DOSS.DAT"))
            
            band_file = band_files[0] if band_files else None
            doss_file = doss_files[0] if doss_files else None
            
            if not band_file and not doss_file:
                # No advanced data available
                advanced_props['advanced_analysis_available'] = False
                advanced_props['advanced_analysis_note'] = 'No BAND.DAT or DOSS.DAT files found'
                return advanced_props
            
            # Perform advanced analysis
            analyzer = AdvancedElectronicAnalyzer()
            analysis_results = analyzer.analyze_material(band_file, doss_file)
            
            # Map results to our property naming convention
            advanced_props['advanced_analysis_available'] = True
            advanced_props['advanced_analysis_method'] = analysis_results.get('analysis_method', 'unknown')
            
            # Electronic classification (override basic classification if we have better data)
            if 'electronic_classification' in analysis_results:
                advanced_props['electronic_classification_advanced'] = analysis_results['electronic_classification']
                
                # Update the main classification if we have better data
                if band_file and doss_file:
                    # We have both BAND and DOSS - this is the most accurate
                    advanced_props['electronic_classification'] = analysis_results['electronic_classification']
                    advanced_props['classification_gap_source'] = analysis_results.get('gap_source', 'unknown')
            
            # Real effective masses from band structure
            if analysis_results.get('has_real_effective_mass'):
                if analysis_results.get('electron_effective_mass') is not None:
                    advanced_props['real_electron_effective_mass'] = analysis_results['electron_effective_mass']
                if analysis_results.get('hole_effective_mass') is not None:
                    advanced_props['real_hole_effective_mass'] = analysis_results['hole_effective_mass']
                if analysis_results.get('average_effective_mass') is not None:
                    advanced_props['real_average_effective_mass'] = analysis_results['average_effective_mass']
                
                advanced_props['effective_mass_calculation_method'] = analysis_results.get('calculation_method', 'unknown')
            
            # Semimetal detection
            advanced_props['is_semimetal_advanced'] = analysis_results.get('is_semimetal', False)
            
            # DOS analysis results
            if analysis_results.get('dos_data_available'):
                advanced_props['dos_at_fermi_level'] = analysis_results.get('dos_at_fermi', 0.0)
                advanced_props['dos_threshold'] = analysis_results.get('dos_threshold', 0.0)
                advanced_props['dos_mean'] = analysis_results.get('dos_mean', 0.0)
                advanced_props['dos_analysis_gcrit_factor'] = analysis_results.get('gcrit_factor', 0.0)
            
            # Band structure analysis results
            if analysis_results.get('band_data_available'):
                advanced_props['band_structure_k_points'] = analysis_results.get('band_k_points', 0)
                if 'band_structure_range' in analysis_results:
                    k_range = analysis_results['band_structure_range']
                    advanced_props['band_structure_k_range_min'] = k_range[0]
                    advanced_props['band_structure_k_range_max'] = k_range[1]
            
            # Enhanced transport properties
            if 'electron_mobility_estimate' in analysis_results:
                advanced_props['real_electron_mobility'] = analysis_results['electron_mobility_estimate']
            if 'hole_mobility_estimate' in analysis_results:
                advanced_props['real_hole_mobility'] = analysis_results['hole_mobility_estimate']
            
            advanced_props['conductivity_type_advanced'] = analysis_results.get('conductivity_type', 'unknown')
            advanced_props['conductivity_estimate_advanced'] = analysis_results.get('conductivity_estimate', 'unknown')
            
            # Gap information with source tracking
            if 'gap_ev' in analysis_results:
                advanced_props['band_gap_advanced_ev'] = analysis_results['gap_ev']
            if 'gap_hartree' in analysis_results:
                advanced_props['band_gap_advanced_ha'] = analysis_results['gap_hartree']
            
            # Analysis metadata
            files_analyzed = analysis_results.get('files_analyzed', {})
            if files_analyzed.get('band_file'):
                advanced_props['band_file_analyzed'] = files_analyzed['band_file']
            if files_analyzed.get('doss_file'):
                advanced_props['doss_file_analyzed'] = files_analyzed['doss_file']
                
            print(f"      🚀 Advanced analysis: {analysis_results['electronic_classification']} classification")
            if analysis_results.get('has_real_effective_mass'):
                print(f"      📊 Real effective masses calculated from band structure")
            
        except ImportError:
            advanced_props['advanced_analysis_available'] = False
            advanced_props['advanced_analysis_note'] = 'AdvancedElectronicAnalyzer not available'
        except Exception as e:
            advanced_props['advanced_analysis_available'] = False
            advanced_props['advanced_analysis_error'] = str(e)
            print(f"      ⚠️  Advanced analysis failed: {e}")
        
        return advanced_props


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract properties from CRYSTAL output files")
    parser.add_argument("--output-file", help="Single output file to process")
    parser.add_argument("--scan-directory", help="Directory to scan for output files")
    parser.add_argument("--db-path", default="materials.db", help="Path to materials database")
    parser.add_argument("--material-id", help="Override material ID (or filter by material ID when scanning)")
    parser.add_argument("--calc-id", help="Override calculation ID")
    parser.add_argument("--calc-type", help="Filter by calculation type (OPT, SP, FREQ, BAND, DOSS, etc.)")
    parser.add_argument("--filter-material", help="Only extract properties for specific material ID")
    parser.add_argument("--filter-type", help="Only extract properties from specific calculation type")
    
    args = parser.parse_args()
    
    if not args.output_file and not args.scan_directory:
        print("❌ Please specify either --output-file or --scan-directory")
        sys.exit(1)
    
    extractor = CrystalPropertyExtractor(args.db_path)
    
    output_files = []
    if args.output_file:
        output_files = [Path(args.output_file)]
    elif args.scan_directory:
        scan_dir = Path(args.scan_directory)
        output_files = list(scan_dir.rglob("*.out"))
        
        # Apply filters if specified
        if args.filter_material or args.filter_type:
            filtered_files = []
            for f in output_files:
                # Check material ID filter
                if args.filter_material:
                    if args.filter_material not in str(f):
                        continue
                
                # Check calculation type filter
                if args.filter_type:
                    # Simple heuristic: check if the calc type is in the filename
                    file_str = str(f).lower()
                    filter_type = args.filter_type.lower()
                    
                    # Common patterns for different calc types
                    if filter_type == 'opt' and ('opt' not in file_str and 'optimization' not in file_str):
                        continue
                    elif filter_type == 'sp' and 'sp' not in file_str and 'opt' in file_str:
                        continue
                    elif filter_type == 'freq' and 'freq' not in file_str:
                        continue
                    elif filter_type == 'band' and 'band' not in file_str:
                        continue
                    elif filter_type == 'doss' and ('doss' not in file_str and 'dos' not in file_str):
                        continue
                
                filtered_files.append(f)
            
            print(f"🔍 Found {len(output_files)} output files, {len(filtered_files)} match filters")
            output_files = filtered_files
        else:
            print(f"🔍 Found {len(output_files)} output files to process")
    else:
        print(f"🔍 Found {len(output_files)} output files to process")
    
    total_properties = 0
    files_processed = 0
    materials_processed = set()
    
    for output_file in output_files:
        print(f"\n📊 Processing: {output_file}")
        
        # For better filtering, we can also check content if needed
        if args.calc_type or args.filter_type:
            # Read first few lines to determine calc type
            try:
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content_preview = f.read(5000)  # Read first 5KB
                    
                # Check if this matches the requested calc type
                calc_type_filter = args.calc_type or args.filter_type
                if calc_type_filter:
                    calc_type_filter = calc_type_filter.upper()
                    
                    # Define patterns for each calculation type
                    type_patterns = {
                        'OPT': ['OPTGEOM', 'OPTIMIZATION', 'GEOMETRY OPTIMIZATION'],
                        'SP': ['SINGLE POINT', 'SINGLE-POINT', 'SP CALCULATION'],
                        'FREQ': ['FREQUENCY', 'FREQUENCIES', 'VIBRATIONAL', 'FREQCALC'],
                        'BAND': ['BAND STRUCTURE', 'BANDSTRUCTURE', 'NEWK'],
                        'DOSS': ['DENSITY OF STATES', 'DOS CALCULATION', 'DOSS']
                    }
                    
                    # Check if content matches the requested type
                    if calc_type_filter in type_patterns:
                        patterns = type_patterns[calc_type_filter]
                        if not any(pattern in content_preview.upper() for pattern in patterns):
                            print(f"   ⏭️  Skipping - not a {calc_type_filter} calculation")
                            continue
            except Exception as e:
                print(f"   ⚠️  Could not read file for filtering: {e}")
        
        properties = extractor.extract_all_properties(
            output_file, 
            material_id=args.material_id,
            calc_id=args.calc_id
        )
        
        if properties:
            # Extract material ID from properties if available
            for prop in properties:
                if 'material_id' in prop:
                    materials_processed.add(prop['material_id'])
                    break
            
            saved_count = extractor.save_properties_to_database(properties)
            total_properties += saved_count
            files_processed += 1
            print(f"   ✅ Extracted and saved {saved_count} properties")
        else:
            print(f"   ⚠️  No properties extracted")
    
    print(f"\n🎉 Processing complete!")
    print(f"   Files processed: {files_processed}")
    print(f"   Total properties extracted: {total_properties}")
    print(f"   Materials processed: {len(materials_processed)}")
    
    if args.filter_material or args.filter_type or args.calc_type:
        print(f"\n📌 Filters applied:")
        if args.filter_material:
            print(f"   Material ID: {args.filter_material}")
        if args.filter_type or args.calc_type:
            print(f"   Calculation type: {args.filter_type or args.calc_type}")


if __name__ == "__main__":
    main()