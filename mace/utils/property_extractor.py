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
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Import MACE components
try:
    from mace.database.materials import MaterialDatabase
except ImportError as e:
    print(f"Error importing MaterialDatabase: {e}")
    sys.exit(1)


class CrystalPropertyExtractor:
    """Extract comprehensive properties from CRYSTAL output files."""
    
    def __init__(self, db_path: str = "materials.db"):
        self.db = MaterialDatabase(db_path)
        self.properties = []
        
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
        
        # Process population analysis data if available
        try:
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
        
        # Direct vs indirect band gap
        if 'DIRECT ENERGY BAND GAP' in content:
            direct_match = re.search(r'DIRECT ENERGY BAND GAP:\s*([\d.]+)\s*eV', content)
            if direct_match:
                props['direct_band_gap'] = float(direct_match.group(1))
                props['band_gap_type'] = 'direct'
        
        if 'INDIRECT ENERGY BAND GAP' in content:
            indirect_match = re.search(r'INDIRECT ENERGY BAND GAP:\s*([\d.]+)\s*eV', content)
            if indirect_match:
                props['indirect_band_gap'] = float(indirect_match.group(1))
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
            props['total_energy_ev'] = float(energy_matches[-1]) * 27.2114  # Hartree to eV
        
        # Alternative energy extraction patterns
        if not energy_matches:
            # From SCF convergence
            scf_energy_match = re.search(r'== SCF ENDED - CONVERGENCE ON ENERGY\s+E\(AU\)\s*([-\d.E+]+)', content)
            if scf_energy_match:
                props['total_energy_au'] = float(scf_energy_match.group(1))
                props['total_energy_ev'] = float(scf_energy_match.group(1)) * 27.2114
            else:
                # From final optimization energy
                final_energy_match = re.search(r'E\(AU\):\s*([-\d.E+]+)', content)
                if final_energy_match:
                    props['total_energy_au'] = float(final_energy_match.group(1))
                    props['total_energy_ev'] = float(final_energy_match.group(1)) * 27.2114
        
        # D3 dispersion correction
        d3_match = re.search(r'D3 DISPERSION ENERGY \(AU\)\s*([-\d.E+]+)', content)
        if d3_match:
            d3_energy = float(d3_match.group(1))
            props['d3_dispersion_energy_au'] = d3_energy
            props['d3_dispersion_energy_ev'] = d3_energy * 27.2114
            
            # Also look for the total energy + dispersion
            total_plus_disp_match = re.search(r'TOTAL ENERGY \+ DISP \(AU\)\s*([-\d.E+]+)', content)
            if total_plus_disp_match:
                props['total_energy_plus_d3_au'] = float(total_plus_disp_match.group(1))
                props['total_energy_plus_d3_ev'] = float(total_plus_disp_match.group(1)) * 27.2114
            elif 'total_energy_au' in props:
                # Calculate if not explicitly given
                props['total_energy_plus_d3_au'] = props['total_energy_au'] + d3_energy
                props['total_energy_plus_d3_ev'] = props['total_energy_plus_d3_au'] * 27.2114
        
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
                    components[ev_name] = au_value * 27.2114
        
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
            
            # Convergence information
            converged_match = re.search(r'CONVERGENCE TESTS SATISFIED AFTER\s+(\d+)\s+ENERGY AND GRADIENT CALCULATIONS', content)
            if converged_match:
                props['optimization_cycles'] = int(converged_match.group(1))
                props['optimization_converged'] = True
            else:
                props['optimization_converged'] = False
            
            # Final gradient
            grad_norm_match = re.search(r'GRADIENT NORM\s+([\d.E+-]+)', content)
            if grad_norm_match:
                props['final_gradient_norm'] = float(grad_norm_match.group(1))
        
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
        """Extract neighbor information from CRYSTAL output."""
        props = {}
        
        # Look for neighbor analysis section (take the first occurrence)
        neighbor_section_match = re.search(
            r'NEIGHBORS OF THE NON-EQUIVALENT ATOMS.*?N = NUMBER OF NEIGHBORS AT DISTANCE R\s*\n\s*ATOM\s+N\s+R/ANG\s+R/AU\s+NEIGHBORS.*?\n(.*?)(?=\n\s*SYMMETRY|\n\s*TTTT|\n\s*MMMM|\n\s*[A-Z]{3,}|$)',
            content, re.DOTALL
        )
        
        if not neighbor_section_match:
            return props
        
        neighbor_content = neighbor_section_match.group(1).strip()
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
            
            # Check if this line starts with atom number and element
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit() and parts[1].isalpha():
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
            final_geom_match = re.search(
                r'FINAL OPTIMIZED GEOMETRY.*?ATOM\s+X/A\s+Y/B\s+Z/C\s*\n\s*\*+\s*\n(.*?)\n\s*T = ATOM',
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
        # Escape special regex characters in section_type
        escaped_section = re.escape(section_type)
        pattern = f'{escaped_section}.*?MULLIKEN POPULATION ANALYSIS.*?NO. OF ELECTRONS\\s+([\\d.-]+)(.*?)(?=MMMMM|TTTTTT|$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return None
        
        total_electrons = float(match.group(1))
        section_content = match.group(2)
        
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
                # Check if property already exists
                with self.db._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT property_id FROM properties WHERE material_id = ? AND property_name = ? AND calc_id = ?",
                        (material_id, prop_name, calc_id)
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
            r'FERMI ENERGY\s*[=:]\s*([-\d.]+)',
            r'FERMI LEVEL\s*[=:]\s*([-\d.]+)',
            r'CHEMICAL POTENTIAL\s*[=:]\s*([-\d.]+)',
            r'EFERMI\(AU\)\s+([-\d.E+\-]+)',  # SP calculation conducting state: "EFERMI(AU) -9.9423732E-02"
            r'POSSIBLY CONDUCTING STATE.*?EFERMI\(AU\)\s+([-\d.E+\-]+)'  # Full conducting state pattern
        ]
        
        for pattern in fermi_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                props['fermi_energy'] = float(match.group(1))
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
            r'FERMI ENERGY\s*[=:]\s*([-\d.]+)',
            r'FERMI LEVEL\s*[=:]\s*([-\d.]+)',
            r'EF\s*[=:]\s*([-\d.]+)',
            r'FERMI ENERGY\s+([-\d.E+\-]+)',  # CRYSTAL format: "FERMI ENERGY -0.123E+00"
        ]
        
        for pattern in fermi_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                props['fermi_energy_band'] = float(match.group(1))
                break
                
        # Check for BAND.DAT file and extract information
        band_dat_file = output_file.parent / "BAND.DAT"
        if band_dat_file.exists():
            props['band_dat_exists'] = True
            props['band_dat_size'] = band_dat_file.stat().st_size
            
            # Try to extract information from DAT file using existing processor
            try:
                from dat_file_processor import DatFileProcessor
                processor = DatFileProcessor()
                dat_info = processor.process_band_dat_file(band_dat_file)
                if dat_info:
                    props.update(dat_info)
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
            r'FERMI ENERGY\s*[=:]\s*([-\d.]+)',
            r'FERMI LEVEL\s*[=:]\s*([-\d.]+)',
            r'EF\s*[=:]\s*([-\d.]+)',
            r'FERMI ENERGY\s+([-\d.E+\-]+)',  # CRYSTAL format: "FERMI ENERGY -0.123E+00"
        ]
        
        for pattern in fermi_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                props['fermi_energy_dos'] = float(match.group(1))
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
                from dat_file_processor import DatFileProcessor
                processor = DatFileProcessor()
                dat_info = processor.process_doss_dat_file(dat_file)
                if dat_info:
                    props.update(dat_info)
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
        freq_section = re.search(
            r'MODES\s+EIGV\s+FREQUENCIES\s+IRREP.*?\n\s*\(HARTREE\*\*2\)\s*\(CM\*\*\(-1\)\)\s*\(THZ\).*?\n(.*?)(?=NORMAL MODES|VIBRATIONAL TEMPERATURES|\*{5,})',
            content, re.DOTALL
        )
        
        if freq_section:
            freq_data = []
            freq_lines = freq_section.group(1).strip().split('\n')
            
            for line in freq_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse frequency data lines
                # Format: MODE_RANGE EIGV FREQ_CM FREQ_THZ (IRREP) IR_ACTIVE (INTENSITY) RAMAN_ACTIVE
                parts = line.split()
                if len(parts) >= 3 and parts[0].replace('-', '').isdigit():
                    try:
                        mode_match = re.match(r'(\d+)-\s*(\d+)', parts[0])
                        if mode_match:
                            mode_start = int(mode_match.group(1))
                            mode_end = int(mode_match.group(2))
                        else:
                            mode_start = mode_end = int(parts[0])
                        
                        eigv = float(parts[1])
                        freq_cm = float(parts[2])
                        freq_thz = float(parts[3]) if len(parts) > 3 else None
                        
                        # Extract irrep
                        irrep_match = re.search(r'\(([^)]+)\)', line)
                        irrep = irrep_match.group(1) if irrep_match else None
                        
                        # Extract IR/Raman activity
                        ir_active = 'A' in line.split(')')[-1] if ')' in line else False
                        raman_active = line.strip().endswith('A')
                        
                        # Extract intensity
                        intensity_match = re.search(r'\(\s*([\d.]+)\s*\)', line.split(')')[-1] if ')' in line else line)
                        intensity = float(intensity_match.group(1)) if intensity_match else 0.0
                        
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
                props['num_vibrational_modes'] = len(freq_data)
                
                # Extract summary statistics
                non_zero_freqs = [f['frequency_cm'] for f in freq_data if f['frequency_cm'] > 0.1]
                if non_zero_freqs:
                    props['lowest_frequency_cm'] = min(non_zero_freqs)
                    props['highest_frequency_cm'] = max(non_zero_freqs)
                    props['num_imaginary_frequencies'] = sum(1 for f in freq_data if f['eigenvalue'] < 0)
        
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
        
        # Extract thermodynamic properties
        thermo_props = self._extract_thermodynamic_properties(content)
        props.update(thermo_props)
        
        # Calculate enthalpy from other energies
        if 'zero_point_energy_au' in props and 'thermal_energy_au' in thermo_props and 'pv_term_au' in thermo_props:
            # Enthalpy = E0 + ET + PV
            enthalpy_au = props['zero_point_energy_au'] + thermo_props['thermal_energy_au'] + thermo_props['pv_term_au']
            props['enthalpy_au'] = enthalpy_au
            props['enthalpy_ev'] = enthalpy_au * 27.2114
            props['enthalpy_kj_mol'] = enthalpy_au * 2625.5  # Hartree to kJ/mol
            
            # Also calculate enthalpy including electronic energy if available
            if 'total_energy_au' in self._extract_energy_properties(content):
                el_energy = self._extract_energy_properties(content)['total_energy_au']
                props['enthalpy_total_au'] = el_energy + enthalpy_au
                props['enthalpy_total_ev'] = props['enthalpy_total_au'] * 27.2114
                props['enthalpy_total_kj_mol'] = props['enthalpy_total_au'] * 2625.5
        
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
        
        # Extract thermodynamic functions
        thermo_section = re.search(
            r'THERMODYNAMIC FUNCTIONS WITH VIBRATIONAL CONTRIBUTIONS.*?'
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
            return 'eV'
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


    def _extract_advanced_band_dos_analysis(self, output_file: Path, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract advanced electronic properties using BAND/DOSS data analysis.
        
        Uses sophisticated DOS/BAND analysis for accurate effective mass and classification.
        """
        advanced_props = {}
        
        try:
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