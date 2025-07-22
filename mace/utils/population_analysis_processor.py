#!/usr/bin/env python3
"""
Population Analysis Processor
=============================
Process raw Mulliken and overlap population data to extract meaningful chemical information:
- Atomic charges and oxidation states
- Bonding behavior and coordination environments
- Magnetic moments and spin densities
- Bond orders and covalency analysis

Author: Generated for materials database project
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AtomicCharge:
    """Atomic charge information."""
    atom_number: int
    element: str
    mulliken_charge: float
    formal_charge: int
    oxidation_state: int
    spin_density: Optional[float] = None


@dataclass
class BondInformation:
    """Bond information between atoms."""
    atom_a: int
    atom_b: int
    element_a: str
    element_b: str
    bond_order: float
    bond_type: str  # ionic, covalent, metallic
    distance: float
    overlap_population: float


@dataclass
class CoordinationEnvironment:
    """Coordination environment information."""
    atom_number: int
    element: str
    coordination_number: int
    neighbors: List[Dict[str, Any]]
    geometry: str  # tetrahedral, octahedral, etc.


class PopulationAnalysisProcessor:
    """Process population analysis data to extract chemical insights."""
    
    def __init__(self):
        # Standard atomic numbers for reference
        self.atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20
        }
        
        # Typical oxidation states for common elements
        self.common_oxidation_states = {
            'H': [-1, 1], 'Li': [1], 'Be': [2], 'B': [3], 'C': [-4, -2, 2, 4],
            'N': [-3, -1, 1, 2, 3, 4, 5], 'O': [-2, -1, 1, 2], 'F': [-1],
            'Na': [1], 'Mg': [2], 'Al': [3], 'Si': [-4, 2, 4], 'P': [-3, 3, 5],
            'S': [-2, 2, 4, 6], 'Cl': [-1, 1, 3, 5, 7], 'K': [1], 'Ca': [2]
        }
    
    def process_population_data(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw population analysis data to extract chemical information.
        
        Args:
            property_data: Dictionary containing raw population analysis data
            
        Returns:
            Dictionary with processed chemical information
        """
        results = {
            'atomic_charges': [],
            'bonding_analysis': [],
            'coordination_environments': [],
            'magnetic_info': {},
            'chemical_summary': {}
        }
        
        # Process Mulliken population data
        if 'mulliken_population' in property_data:
            mulliken_data = json.loads(property_data['mulliken_population']) if isinstance(property_data['mulliken_population'], str) else property_data['mulliken_population']
            atomic_charges = self._process_atomic_charges(mulliken_data)
            results['atomic_charges'] = [self._dataclass_to_dict(charge) for charge in atomic_charges]
        
        # Process spin-polarized data
        if 'mulliken_alpha_plus_beta' in property_data and 'mulliken_alpha_minus_beta' in property_data:
            alpha_plus_beta = json.loads(property_data['mulliken_alpha_plus_beta']) if isinstance(property_data['mulliken_alpha_plus_beta'], str) else property_data['mulliken_alpha_plus_beta']
            alpha_minus_beta = json.loads(property_data['mulliken_alpha_minus_beta']) if isinstance(property_data['mulliken_alpha_minus_beta'], str) else property_data['mulliken_alpha_minus_beta']
            
            atomic_charges = self._process_spin_polarized_charges(alpha_plus_beta, alpha_minus_beta)
            results['atomic_charges'] = [self._dataclass_to_dict(charge) for charge in atomic_charges]
            results['magnetic_info'] = self._process_magnetic_information(alpha_plus_beta, alpha_minus_beta)
        
        # Process overlap population data
        if 'overlap_population' in property_data:
            overlap_data = json.loads(property_data['overlap_population']) if isinstance(property_data['overlap_population'], str) else property_data['overlap_population']
            bonding_analysis = self._process_bonding_analysis(overlap_data)
            coordination_envs = self._process_coordination_environments(overlap_data)
            
            results['bonding_analysis'] = [self._dataclass_to_dict(bond) for bond in bonding_analysis]
            results['coordination_environments'] = [self._dataclass_to_dict(env) for env in coordination_envs]
        
        # Generate chemical summary
        results['chemical_summary'] = self._generate_chemical_summary(results)
        
        return results
    
    def _process_atomic_charges(self, mulliken_data: Dict) -> List[AtomicCharge]:
        """Process atomic charges from Mulliken population analysis."""
        charges = []
        
        if not mulliken_data or 'atoms' not in mulliken_data:
            return charges
        
        for atom_data in mulliken_data['atoms']:
            # Calculate net charge (nuclear charge - electron population)
            nuclear_charge = atom_data['atomic_number']
            electron_population = atom_data['mulliken_charge']
            net_charge = nuclear_charge - electron_population
            
            # Estimate oxidation state
            oxidation_state = self._estimate_oxidation_state(atom_data['element'], net_charge)
            
            charge_info = AtomicCharge(
                atom_number=atom_data['atom_number'],
                element=atom_data['element'],
                mulliken_charge=net_charge,
                formal_charge=int(round(net_charge)),
                oxidation_state=oxidation_state
            )
            charges.append(charge_info)
        
        return charges
    
    def _process_spin_polarized_charges(self, alpha_plus_beta: Dict, alpha_minus_beta: Dict) -> List[AtomicCharge]:
        """Process charges from spin-polarized calculations."""
        charges = []
        
        if not alpha_plus_beta or 'atoms' not in alpha_plus_beta:
            return charges
        
        alpha_atoms = alpha_plus_beta['atoms']
        spin_atoms = alpha_minus_beta.get('atoms', []) if alpha_minus_beta else []
        
        for i, atom_data in enumerate(alpha_atoms):
            # Calculate net charge
            nuclear_charge = atom_data['atomic_number']
            total_electrons = atom_data['mulliken_charge']
            net_charge = nuclear_charge - total_electrons
            
            # Calculate spin density if available
            spin_density = None
            if i < len(spin_atoms):
                spin_density = spin_atoms[i]['mulliken_charge']
            
            # Estimate oxidation state
            oxidation_state = self._estimate_oxidation_state(atom_data['element'], net_charge)
            
            charge_info = AtomicCharge(
                atom_number=atom_data['atom_number'],
                element=atom_data['element'],
                mulliken_charge=net_charge,
                formal_charge=int(round(net_charge)),
                oxidation_state=oxidation_state,
                spin_density=spin_density
            )
            charges.append(charge_info)
        
        return charges
    
    def _process_bonding_analysis(self, overlap_data: List[Dict]) -> List[BondInformation]:
        """Analyze bonding from overlap population data."""
        bonds = []
        
        for atom_data in overlap_data:
            atom_a = atom_data['atom_a_number']
            element_a = atom_data['atom_a_element']
            
            for neighbor in atom_data.get('neighbors', []):
                atom_b = neighbor['atom_b_number']
                element_b = neighbor['atom_b_element']
                overlap_pop = neighbor['overlap_population']
                distance = neighbor.get('distance_ang', 0.0)
                
                # Classify bond type based on overlap population
                bond_type = self._classify_bond_type(element_a, element_b, overlap_pop)
                
                # Estimate bond order
                bond_order = self._estimate_bond_order(overlap_pop, element_a, element_b)
                
                bond_info = BondInformation(
                    atom_a=atom_a,
                    atom_b=atom_b,
                    element_a=element_a,
                    element_b=element_b,
                    bond_order=bond_order,
                    bond_type=bond_type,
                    distance=distance,
                    overlap_population=overlap_pop
                )
                bonds.append(bond_info)
        
        return bonds
    
    def _process_coordination_environments(self, overlap_data: List[Dict]) -> List[CoordinationEnvironment]:
        """Analyze coordination environments."""
        coord_envs = []
        
        for atom_data in overlap_data:
            atom_number = atom_data['atom_a_number']
            element = atom_data['atom_a_element']
            neighbors = atom_data.get('neighbors', [])
            
            # Filter for significant bonding interactions
            significant_neighbors = [
                n for n in neighbors 
                if abs(n['overlap_population']) > 0.1  # Threshold for significant bonding
            ]
            
            coordination_number = len(significant_neighbors)
            
            # Classify geometry based on coordination number
            geometry = self._classify_coordination_geometry(coordination_number)
            
            coord_env = CoordinationEnvironment(
                atom_number=atom_number,
                element=element,
                coordination_number=coordination_number,
                neighbors=significant_neighbors,
                geometry=geometry
            )
            coord_envs.append(coord_env)
        
        return coord_envs
    
    def _process_magnetic_information(self, alpha_plus_beta: Dict, alpha_minus_beta: Dict) -> Dict[str, Any]:
        """Process magnetic information from spin-polarized data."""
        magnetic_info = {
            'is_magnetic': False,
            'total_spin_moment': 0.0,
            'atomic_spin_moments': [],
            'magnetic_classification': 'diamagnetic'
        }
        
        if not alpha_minus_beta or 'atoms' not in alpha_minus_beta:
            return magnetic_info
        
        spin_atoms = alpha_minus_beta['atoms']
        total_spin = sum(atom['mulliken_charge'] for atom in spin_atoms)
        
        magnetic_info['total_spin_moment'] = total_spin
        magnetic_info['is_magnetic'] = abs(total_spin) > 0.01
        
        # Classify magnetic behavior
        if abs(total_spin) > 0.01:
            if total_spin > 0:
                magnetic_info['magnetic_classification'] = 'ferromagnetic'
            else:
                magnetic_info['magnetic_classification'] = 'antiferromagnetic'
        
        # Individual atomic moments
        for atom in spin_atoms:
            magnetic_info['atomic_spin_moments'].append({
                'atom_number': atom['atom_number'],
                'element': atom['element'],
                'spin_moment': atom['mulliken_charge']
            })
        
        return magnetic_info
    
    def _estimate_oxidation_state(self, element: str, charge: float) -> int:
        """Estimate oxidation state from charge and element."""
        if element not in self.common_oxidation_states:
            return int(round(charge))  # Best guess
        
        possible_states = self.common_oxidation_states[element]
        # Find closest common oxidation state
        closest_state = min(possible_states, key=lambda x: abs(x - charge))
        return closest_state
    
    def _classify_bond_type(self, element_a: str, element_b: str, overlap_pop: float) -> str:
        """Classify bond type based on elements and overlap population."""
        # Simple classification based on overlap population magnitude
        if abs(overlap_pop) > 0.5:
            return 'covalent'
        elif abs(overlap_pop) > 0.1:
            return 'ionic'
        else:
            return 'weak'
    
    def _estimate_bond_order(self, overlap_pop: float, element_a: str, element_b: str) -> float:
        """Estimate bond order from overlap population."""
        # Rough estimation based on overlap population
        return max(0, min(3, abs(overlap_pop)))
    
    def _classify_coordination_geometry(self, coordination_number: int) -> str:
        """Classify coordination geometry based on coordination number."""
        geometries = {
            1: 'linear',
            2: 'linear',
            3: 'trigonal_planar',
            4: 'tetrahedral',
            5: 'trigonal_bipyramidal',
            6: 'octahedral',
            7: 'pentagonal_bipyramidal',
            8: 'cubic'
        }
        return geometries.get(coordination_number, f'CN_{coordination_number}')
    
    def _generate_chemical_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall chemical summary."""
        summary = {
            'total_atoms': len(results.get('atomic_charges', [])),
            'charge_distribution': {},
            'coordination_statistics': {},
            'bonding_summary': {},
            'magnetic_summary': results.get('magnetic_info', {})
        }
        
        # Charge distribution
        charges = results.get('atomic_charges', [])
        if charges:
            elements = {}
            for charge in charges:
                # Handle both dataclass objects and dictionaries
                element = charge.get('element') if isinstance(charge, dict) else charge.element
                mulliken_charge = charge.get('mulliken_charge') if isinstance(charge, dict) else charge.mulliken_charge
                
                if element not in elements:
                    elements[element] = []
                elements[element].append(mulliken_charge)
            
            summary['charge_distribution'] = {
                element: {
                    'count': len(charges),
                    'average_charge': np.mean(charges),
                    'charge_range': [min(charges), max(charges)]
                }
                for element, charges in elements.items()
            }
        
        # Coordination statistics
        coord_envs = results.get('coordination_environments', [])
        if coord_envs:
            coord_stats = {}
            for env in coord_envs:
                # Handle both dataclass objects and dictionaries
                element = env.get('element') if isinstance(env, dict) else env.element
                cn = env.get('coordination_number') if isinstance(env, dict) else env.coordination_number
                
                if element not in coord_stats:
                    coord_stats[element] = {}
                if cn not in coord_stats[element]:
                    coord_stats[element][cn] = 0
                coord_stats[element][cn] += 1
            
            summary['coordination_statistics'] = coord_stats
        
        return summary

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass objects to dictionaries for JSON serialization."""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._dataclass_to_dict(value)
                elif isinstance(value, list):
                    result[key] = [self._dataclass_to_dict(item) if hasattr(item, '__dict__') else item for item in value]
                else:
                    result[key] = value
            return result
        else:
            return obj


def process_material_population_analysis(material_id: str, db_path: str = "materials.db") -> Dict[str, Any]:
    """
    Process population analysis for a specific material.
    
    Args:
        material_id: Material ID to process
        db_path: Path to materials database
        
    Returns:
        Processed population analysis results
    """
    from material_database import MaterialDatabase
    
    db = MaterialDatabase(db_path)
    processor = PopulationAnalysisProcessor()
    
    # Get all population analysis properties for this material
    with db._get_connection() as conn:
        cursor = conn.execute("""
            SELECT calc_id, property_name, property_value_text
            FROM properties 
            WHERE material_id = ? AND property_category = 'population_analysis'
        """, (material_id,))
        
        properties = {}
        for calc_id, prop_name, prop_text in cursor.fetchall():
            if prop_text:
                try:
                    properties[prop_name] = json.loads(prop_text)
                except (json.JSONDecodeError, TypeError):
                    properties[prop_name] = prop_text
    
    if not properties:
        return {"error": "No population analysis data found"}
    
    # Process the data
    results = processor.process_population_data(properties)
    return results


if __name__ == "__main__":
    # Test with a material from the database
    result = process_material_population_analysis("1_dia")
    print("🧪 Population Analysis Processing Test:")
    print("=" * 60)
    
    for key, value in result.items():
        if key == 'atomic_charges':
            print(f"\n{key}:")
            for charge in value:
                print(f"  Atom {charge.atom_number} ({charge.element}): charge={charge.mulliken_charge:.3f}, oxidation={charge.oxidation_state}")
        elif key == 'coordination_environments':
            print(f"\n{key}:")
            for env in value:
                print(f"  Atom {env.atom_number} ({env.element}): CN={env.coordination_number}, geometry={env.geometry}")
        elif key == 'coordination_statistics':
            print(f"\n{key}:")
            print(f"  {value}")
        else:
            print(f"\n{key}: {value}")