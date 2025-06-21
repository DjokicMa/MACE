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

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from material_database import MaterialDatabase
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
        
        # Add metadata
        properties['_metadata'] = {
            'material_id': material_id,
            'calc_id': calc_id,
            'output_file': str(output_file),
            'extracted_at': datetime.now().isoformat(),
            'extractor_version': '1.0'
        }
        
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
            props['atomic_positions'] = initial_positions
        
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
        
        # Find alpha+beta and alpha-beta sections
        alpha_beta_section = self._extract_mulliken_section(content, "ALPHA+BETA ELECTRONS")
        alpha_minus_beta_section = self._extract_mulliken_section(content, "ALPHA-BETA ELECTRONS")
        
        if alpha_beta_section:
            props['mulliken_alpha_plus_beta'] = alpha_beta_section
            
        if alpha_minus_beta_section:
            props['mulliken_alpha_minus_beta'] = alpha_minus_beta_section
        
        # Extract overlap populations
        overlap_alpha_beta = self._extract_overlap_populations(content, "ALPHA+BETA ELECTRONS")
        overlap_alpha_minus_beta = self._extract_overlap_populations(content, "ALPHA-BETA ELECTRONS")
        
        if overlap_alpha_beta:
            props['overlap_population_alpha_plus_beta'] = overlap_alpha_beta
            
        if overlap_alpha_minus_beta:
            props['overlap_population_alpha_minus_beta'] = overlap_alpha_minus_beta
        
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
            if len(parts) >= 6:
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
        pattern = f'{section_type}.*?MULLIKEN POPULATION ANALYSIS.*?NO. OF ELECTRONS\\s+([\\d.]+)(.*?)(?=MMMMM|TTTTTT|$)'
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
        """Extract overlap population data."""
        # Find the overlap population section
        pattern = f'{section_type}.*?OVERLAP POPULATION CONDENSED TO ATOMS(.*?)(?=MMMMM|TTTTTT|$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return []
        
        overlap_content = match.group(1)
        overlaps = []
        
        # Extract overlap data
        overlap_matches = re.finditer(
            r'ATOM A\s+(\d+)\s+(\w+)\s+ATOM B.*?\n((?:\s+\d+\s+\w+.*?\n)+)',
            overlap_content, re.DOTALL
        )
        
        for overlap_match in overlap_matches:
            atom_a_num = int(overlap_match.group(1))
            atom_a_element = overlap_match.group(2)
            neighbor_data = overlap_match.group(3)
            
            # Parse neighbor interactions
            neighbors = []
            neighbor_lines = neighbor_data.strip().split('\n')
            for line in neighbor_lines:
                parts = line.split()
                if len(parts) >= 8:
                    try:
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
    
    def _categorize_property(self, prop_name: str) -> str:
        """Categorize a property based on its name."""
        if any(x in prop_name.lower() for x in ['lattice', 'cell', 'volume', 'density', 'atomic', 'position']):
            return 'structural'
        elif any(x in prop_name.lower() for x in ['band_gap', 'energy', 'gap', 'electronic']):
            return 'electronic'
        elif any(x in prop_name.lower() for x in ['mulliken', 'overlap', 'charge', 'population']):
            return 'population_analysis'
        elif any(x in prop_name.lower() for x in ['optimization', 'gradient', 'converged', 'cycles']):
            return 'optimization'
        elif any(x in prop_name.lower() for x in ['space_group', 'crystal_system', 'centering']):
            return 'crystallographic'
        else:
            return 'other'
    
    def _get_property_unit(self, prop_name: str) -> str:
        """Get the unit for a property based on its name."""
        if 'au' in prop_name.lower():
            return 'Hartree'
        elif 'ev' in prop_name.lower():
            return 'eV'
        elif any(x in prop_name.lower() for x in ['gap', 'band_gap']):
            return 'eV'
        elif 'volume' in prop_name.lower():
            return 'Å³'
        elif 'density' in prop_name.lower():
            return 'g/cm³'
        elif any(x in prop_name.lower() for x in ['_a', '_b', '_c', 'distance']):
            return 'Å'
        elif any(x in prop_name.lower() for x in ['alpha', 'beta', 'gamma']):
            return 'degrees'
        elif 'charge' in prop_name.lower():
            return 'e'
        else:
            return ''


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract properties from CRYSTAL output files")
    parser.add_argument("--output-file", help="Single output file to process")
    parser.add_argument("--scan-directory", help="Directory to scan for output files")
    parser.add_argument("--db-path", default="materials.db", help="Path to materials database")
    parser.add_argument("--material-id", help="Override material ID")
    parser.add_argument("--calc-id", help="Override calculation ID")
    
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
    
    print(f"🔍 Found {len(output_files)} output files to process")
    
    total_properties = 0
    for output_file in output_files:
        print(f"\n📊 Processing: {output_file}")
        
        properties = extractor.extract_all_properties(
            output_file, 
            material_id=args.material_id,
            calc_id=args.calc_id
        )
        
        if properties:
            saved_count = extractor.save_properties_to_database(properties)
            total_properties += saved_count
            print(f"   ✅ Extracted and saved {saved_count} properties")
        else:
            print(f"   ⚠️  No properties extracted")
    
    print(f"\n🎉 Processing complete! Extracted {total_properties} total properties.")


if __name__ == "__main__":
    main()