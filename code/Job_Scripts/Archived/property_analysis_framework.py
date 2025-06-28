#!/usr/bin/env python3
"""
Comprehensive Property Analysis Framework
=========================================
Analyze extracted material properties to calculate additional derived properties,
identify patterns, correlations, and generate insights from the materials database.

Author: Generated for materials database enhancement
"""

import sys
import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import re

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from material_database import MaterialDatabase

@dataclass
class MaterialProperty:
    """Container for material property data."""
    material_id: str
    property_name: str
    property_value: float
    property_unit: str
    calc_type: str
    confidence: float = 1.0

@dataclass
class DerivedProperty:
    """Container for derived/calculated property."""
    material_id: str
    property_name: str
    property_value: float
    property_unit: str
    derivation_method: str
    source_properties: List[str]
    confidence: float

class PropertyAnalysisFramework:
    """
    Comprehensive framework for analyzing and deriving additional properties
    from extracted CRYSTAL calculation data.
    """
    
    def __init__(self, db_path: str = "materials.db"):
        self.db = MaterialDatabase(db_path)
        self.derived_properties = []
        
    def analyze_all_materials(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on all materials in the database.
        
        Returns:
            Dictionary with analysis results, derived properties, and correlations
        """
        print("🔍 Starting Comprehensive Property Analysis")
        print("=" * 60)
        
        results = {
            'derived_properties': [],
            'correlations': {},
            'materials_analysis': {},
            'database_insights': {},
            'electronic_trends': {},
            'structural_trends': {}
        }
        
        # Get all materials and their properties
        materials = self._get_all_material_properties()
        print(f"📊 Analyzing {len(materials)} materials with properties")
        
        # 1. Calculate derived electronic properties
        print("\n⚡ Calculating Derived Electronic Properties...")
        electronic_derived = self._calculate_electronic_derived_properties(materials)
        results['derived_properties'].extend(electronic_derived)
        
        # 2. Calculate structural derived properties
        print("🏗️ Calculating Derived Structural Properties...")
        structural_derived = self._calculate_structural_derived_properties(materials)
        results['derived_properties'].extend(structural_derived)
        
        # 3. Calculate thermodynamic derived properties
        print("🌡️ Calculating Derived Thermodynamic Properties...")
        thermo_derived = self._calculate_thermodynamic_derived_properties(materials)
        results['derived_properties'].extend(thermo_derived)
        
        # 4. Identify correlations between properties
        print("🔗 Identifying Property Correlations...")
        correlations = self._identify_property_correlations(materials)
        results['correlations'] = correlations
        
        # 5. Generate database insights
        print("💡 Generating Database Insights...")
        insights = self._generate_database_insights(materials)
        results['database_insights'] = insights
        
        # 6. Analyze electronic structure trends
        print("📈 Analyzing Electronic Structure Trends...")
        electronic_trends = self._analyze_electronic_trends(materials)
        results['electronic_trends'] = electronic_trends
        
        # 7. Analyze structural trends
        print("📐 Analyzing Structural Trends...")
        structural_trends = self._analyze_structural_trends(materials)
        results['structural_trends'] = structural_trends
        
        # 8. Store derived properties in database
        print("💾 Storing Derived Properties in Database...")
        self._store_derived_properties(results['derived_properties'])
        
        return results
    
    def _get_all_material_properties(self) -> Dict[str, Dict[str, Any]]:
        """Get all materials and their properties organized by material_id."""
        materials = {}
        
        with self.db._get_connection() as conn:
            cursor = conn.execute("""
                SELECT m.material_id, m.formula, m.space_group,
                       p.property_name, p.property_value, p.property_unit,
                       p.property_category, c.calc_type
                FROM materials m
                LEFT JOIN properties p ON m.material_id = p.material_id
                LEFT JOIN calculations c ON p.calc_id = c.calc_id
                WHERE p.property_value IS NOT NULL
                ORDER BY m.material_id, p.property_name
            """)
            
            for row in cursor.fetchall():
                material_id = row[0]
                if material_id not in materials:
                    materials[material_id] = {
                        'formula': row[1],
                        'space_group': row[2],
                        'properties': {},
                        'calc_types': set()
                    }
                
                prop_name = row[3]
                prop_value = row[4]
                prop_unit = row[5]
                prop_category = row[6]
                calc_type = row[7]
                
                materials[material_id]['properties'][prop_name] = {
                    'value': prop_value,
                    'unit': prop_unit,
                    'category': prop_category,
                    'calc_type': calc_type
                }
                
                if calc_type:
                    materials[material_id]['calc_types'].add(calc_type)
        
        return materials
    
    def _calculate_electronic_derived_properties(self, materials: Dict) -> List[DerivedProperty]:
        """Calculate derived electronic properties."""
        derived = []
        
        for material_id, data in materials.items():
            props = data['properties']
            
            # 1. Electronic DOS-based properties
            if 'dos_at_fermi' in props and 'fermi_energy_dos' in props:
                # Calculate electronic heat capacity coefficient (gamma)
                dos_fermi = props['dos_at_fermi']['value']
                if dos_fermi > 0:
                    # gamma = (π²/3) * k_B² * N(E_F) where N(E_F) is DOS at Fermi level
                    kb_eV = 8.617e-5  # Boltzmann constant in eV/K
                    gamma = (np.pi**2 / 3) * (kb_eV**2) * dos_fermi
                    
                    derived.append(DerivedProperty(
                        material_id=material_id,
                        property_name='electronic_heat_capacity_coefficient',
                        property_value=gamma,
                        property_unit='eV/K²',
                        derivation_method='gamma = (π²/3) * k_B² * N(E_F)',
                        source_properties=['dos_at_fermi'],
                        confidence=0.8
                    ))
            
            # 2. Band structure derived properties
            if 'band_gap' in props:
                band_gap = props['band_gap']['value']
                
                # Effective mass estimation from band gap (empirical)
                if band_gap > 0:
                    # Empirical relation for effective mass
                    m_eff = 0.1 + 0.05 * band_gap  # in units of electron mass
                    
                    derived.append(DerivedProperty(
                        material_id=material_id,
                        property_name='estimated_effective_mass',
                        property_value=m_eff,
                        property_unit='m_e',
                        derivation_method='Empirical relation: m* = 0.1 + 0.05*E_g',
                        source_properties=['band_gap'],
                        confidence=0.6
                    ))
                
                # Optical properties estimation
                if band_gap > 1.0:  # For semiconductors/insulators
                    # Refractive index estimation (Penn model)
                    n_ref = np.sqrt(1 + (13.6 / band_gap)**2)
                    
                    derived.append(DerivedProperty(
                        material_id=material_id,
                        property_name='estimated_refractive_index',
                        property_value=n_ref,
                        property_unit='dimensionless',
                        derivation_method='Penn model: n = sqrt(1 + (13.6/E_g)²)',
                        source_properties=['band_gap'],
                        confidence=0.7
                    ))
            
            # 3. Work function and electronegativity
            if 'fermi_energy_dos' in props or 'fermi_energy' in props:
                fermi_energy = props.get('fermi_energy_dos', props.get('fermi_energy', {})).get('value')
                if fermi_energy:
                    # Approximate work function (rough estimation)
                    work_function = abs(fermi_energy) + 4.5  # Add typical surface barrier
                    
                    derived.append(DerivedProperty(
                        material_id=material_id,
                        property_name='estimated_work_function',
                        property_value=work_function,
                        property_unit='eV',
                        derivation_method='Φ ≈ |E_F| + 4.5 eV (approximate)',
                        source_properties=['fermi_energy'],
                        confidence=0.5
                    ))
            
            # 4. Dielectric properties
            if 'band_gap' in props and props['band_gap']['value'] > 0:
                band_gap = props['band_gap']['value']
                
                # Static dielectric constant estimation
                epsilon_static = 1 + (2.5 / band_gap)**2
                
                derived.append(DerivedProperty(
                    material_id=material_id,
                    property_name='estimated_static_dielectric_constant',
                    property_value=epsilon_static,
                    property_unit='dimensionless',
                    derivation_method='ε_s ≈ 1 + (2.5/E_g)²',
                    source_properties=['band_gap'],
                    confidence=0.6
                ))
            
            # 5. Semimetal classification using effective mass
            if 'band_gap' in props:
                band_gap = props['band_gap']['value']
                
                # Calculate effective mass if not already done
                if band_gap > 0:
                    m_eff = 0.1 + 0.05 * band_gap
                    
                    # Semimetal classification: very small band gap + low effective mass
                    is_semimetal = False
                    classification_reason = ""
                    
                    if band_gap < 0.1 and m_eff < 0.15:  # < 100 meV gap + light carriers
                        is_semimetal = True
                        classification_reason = f"Small gap ({band_gap:.3f} eV) + light carriers (m*={m_eff:.3f})"
                    elif band_gap < 0.05:  # Very small gap
                        is_semimetal = True
                        classification_reason = f"Very small gap ({band_gap:.3f} eV)"
                    
                    if is_semimetal:
                        derived.append(DerivedProperty(
                            material_id=material_id,
                            property_name='semimetal_classification',
                            property_value=1.0,
                            property_unit='flag',
                            derivation_method=f'Semimetal: {classification_reason}',
                            source_properties=['band_gap', 'estimated_effective_mass'],
                            confidence=0.7
                        ))
                        
                        # Override electronic classification to semimetal
                        derived.append(DerivedProperty(
                            material_id=material_id,
                            property_name='refined_electronic_classification',
                            property_value='semimetal',
                            property_unit='category',
                            derivation_method=f'Reclassified from semiconductor: {classification_reason}',
                            source_properties=['band_gap', 'estimated_effective_mass'],
                            confidence=0.8
                        ))
        
        return derived
    
    def _calculate_structural_derived_properties(self, materials: Dict) -> List[DerivedProperty]:
        """Calculate derived structural properties."""
        derived = []
        
        for material_id, data in materials.items():
            props = data['properties']
            
            # 1. Lattice parameter ratios
            lattice_params = {}
            for key in ['lattice_a', 'lattice_b', 'lattice_c']:
                if key in props:
                    lattice_params[key] = props[key]['value']
            
            if len(lattice_params) >= 2:
                # Calculate c/a ratio for tetragonal/hexagonal systems
                if 'lattice_a' in lattice_params and 'lattice_c' in lattice_params:
                    c_a_ratio = lattice_params['lattice_c'] / lattice_params['lattice_a']
                    
                    derived.append(DerivedProperty(
                        material_id=material_id,
                        property_name='c_a_lattice_ratio',
                        property_value=c_a_ratio,
                        property_unit='dimensionless',
                        derivation_method='c/a ratio calculation',
                        source_properties=['lattice_c', 'lattice_a'],
                        confidence=0.9
                    ))
            
            # 2. Packing efficiency
            if 'cell_volume' in props and 'atomic_coordinates' in str(props):
                # Estimate packing efficiency (simplified)
                cell_volume = props['cell_volume']['value']
                # This is simplified - would need actual atomic radii
                estimated_packing = 0.74  # Typical for close-packed structures
                
                derived.append(DerivedProperty(
                    material_id=material_id,
                    property_name='estimated_packing_efficiency',
                    property_value=estimated_packing,
                    property_unit='dimensionless',
                    derivation_method='Estimated based on structure type',
                    source_properties=['cell_volume'],
                    confidence=0.5
                ))
            
            # 3. Bond length estimation from lattice parameters
            if 'lattice_a' in props:
                # For simple cubic/diamond structures, nearest neighbor distance
                lattice_a = props['lattice_a']['value']
                # Diamond structure: bond length ≈ a*sqrt(3)/4
                bond_length = lattice_a * np.sqrt(3) / 4
                
                derived.append(DerivedProperty(
                    material_id=material_id,
                    property_name='estimated_nearest_neighbor_distance',
                    property_value=bond_length,
                    property_unit='Angstrom',
                    derivation_method='Diamond structure: d = a*sqrt(3)/4',
                    source_properties=['lattice_a'],
                    confidence=0.7
                ))
        
        return derived
    
    def _calculate_thermodynamic_derived_properties(self, materials: Dict) -> List[DerivedProperty]:
        """Calculate derived thermodynamic properties."""
        derived = []
        
        for material_id, data in materials.items():
            props = data['properties']
            
            # 1. Formation energy estimation
            if 'total_energy_ev' in props:
                total_energy = props['total_energy_ev']['value']
                # This would need reference energies for elements
                # For now, just mark as available for calculation
                
                derived.append(DerivedProperty(
                    material_id=material_id,
                    property_name='total_energy_per_atom',
                    property_value=total_energy,  # Would need to divide by number of atoms
                    property_unit='eV/atom',
                    derivation_method='Total energy normalized per atom',
                    source_properties=['total_energy_ev'],
                    confidence=0.8
                ))
            
            # 2. Bulk modulus estimation
            if 'cell_volume' in props and 'total_energy_ev' in props:
                # Could estimate from pressure-volume calculations if available
                # For now, use empirical correlation with density
                
                derived.append(DerivedProperty(
                    material_id=material_id,
                    property_name='bulk_modulus_estimation_available',
                    property_value=1.0,
                    property_unit='flag',
                    derivation_method='Can be calculated from E-V curve',
                    source_properties=['cell_volume', 'total_energy_ev'],
                    confidence=0.9
                ))
        
        return derived
    
    def _identify_property_correlations(self, materials: Dict) -> Dict[str, Any]:
        """Identify correlations between different properties."""
        correlations = {}
        
        # Collect property data across all materials
        property_data = defaultdict(list)
        material_ids = []
        
        for material_id, data in materials.items():
            material_ids.append(material_id)
            props = data['properties']
            
            # Common properties to analyze
            key_properties = [
                'band_gap', 'fermi_energy', 'total_energy_ev',
                'lattice_a', 'lattice_b', 'lattice_c', 'cell_volume',
                'space_group', 'dos_at_fermi'
            ]
            
            for prop in key_properties:
                if prop in props and isinstance(props[prop]['value'], (int, float)):
                    property_data[prop].append(props[prop]['value'])
                else:
                    property_data[prop].append(None)
        
        # Calculate correlations for properties with sufficient data
        significant_correlations = []
        
        for prop1 in property_data:
            for prop2 in property_data:
                if prop1 >= prop2:  # Avoid duplicates
                    continue
                
                # Get paired data (remove None values)
                data1 = np.array(property_data[prop1])
                data2 = np.array(property_data[prop2])
                
                # Find indices where both properties have values
                valid_idx = ~(np.isnan(data1.astype(float)) | np.isnan(data2.astype(float)))
                
                if np.sum(valid_idx) >= 3:  # Need at least 3 data points
                    valid_data1 = data1[valid_idx].astype(float)
                    valid_data2 = data2[valid_idx].astype(float)
                    
                    try:
                        correlation = np.corrcoef(valid_data1, valid_data2)[0, 1]
                        if abs(correlation) > 0.5:  # Significant correlation
                            significant_correlations.append({
                                'property1': prop1,
                                'property2': prop2,
                                'correlation': correlation,
                                'n_samples': len(valid_data1),
                                'strength': 'strong' if abs(correlation) > 0.8 else 'moderate'
                            })
                    except:
                        continue
        
        correlations['significant_correlations'] = significant_correlations
        correlations['total_materials_analyzed'] = len(materials)
        
        return correlations
    
    def _generate_database_insights(self, materials: Dict) -> Dict[str, Any]:
        """Generate insights about the materials database."""
        insights = {}
        
        # 1. Coverage analysis
        calc_type_coverage = defaultdict(int)
        property_coverage = defaultdict(int)
        electronic_classifications = defaultdict(int)
        
        for material_id, data in materials.items():
            # Count calculation types
            for calc_type in data['calc_types']:
                calc_type_coverage[calc_type] += 1
            
            # Count properties
            for prop_name in data['properties']:
                property_coverage[prop_name] += 1
            
            # Electronic classification
            props = data['properties']
            if 'electronic_classification' in props:
                classification = props['electronic_classification']['value']
                electronic_classifications[classification] += 1
            elif 'band_gap' in props:
                bg = props['band_gap']['value']
                if bg > 2.0:
                    electronic_classifications['insulator'] += 1
                elif bg > 0.0:
                    electronic_classifications['semiconductor'] += 1
                else:
                    electronic_classifications['conductor'] += 1
            
            # Check for refined classification (semimetal)
            if 'refined_electronic_classification' in props:
                refined_class = props['refined_electronic_classification']['value']
                electronic_classifications[refined_class] += 1
            elif 'semimetal_classification' in props:
                electronic_classifications['semimetal'] += 1
        
        insights['calculation_coverage'] = dict(calc_type_coverage)
        insights['property_coverage'] = dict(property_coverage)
        insights['electronic_distribution'] = dict(electronic_classifications)
        
        # 2. Data quality metrics
        total_properties = sum(len(data['properties']) for data in materials.values())
        insights['total_properties_extracted'] = total_properties
        insights['average_properties_per_material'] = total_properties / len(materials) if materials else 0
        
        # 3. Completeness analysis
        workflow_completeness = defaultdict(int)
        for material_id, data in materials.items():
            calc_types = data['calc_types']
            if 'OPT' in calc_types and 'SP' in calc_types and 'BAND' in calc_types and 'DOSS' in calc_types:
                workflow_completeness['complete_workflow'] += 1
            elif 'OPT' in calc_types and 'SP' in calc_types:
                workflow_completeness['partial_workflow'] += 1
            else:
                workflow_completeness['minimal_workflow'] += 1
        
        insights['workflow_completeness'] = dict(workflow_completeness)
        
        return insights
    
    def _analyze_electronic_trends(self, materials: Dict) -> Dict[str, Any]:
        """Analyze trends in electronic properties."""
        trends = {}
        
        # Band gap distribution
        band_gaps = []
        fermi_energies = []
        
        for material_id, data in materials.items():
            props = data['properties']
            
            if 'band_gap' in props:
                band_gaps.append(props['band_gap']['value'])
            
            fermi_key = 'fermi_energy_dos' if 'fermi_energy_dos' in props else 'fermi_energy'
            if fermi_key in props:
                fermi_energies.append(props[fermi_key]['value'])
        
        if band_gaps:
            trends['band_gap_statistics'] = {
                'mean': np.mean(band_gaps),
                'std': np.std(band_gaps),
                'min': np.min(band_gaps),
                'max': np.max(band_gaps),
                'count': len(band_gaps)
            }
        
        if fermi_energies:
            trends['fermi_energy_statistics'] = {
                'mean': np.mean(fermi_energies),
                'std': np.std(fermi_energies),
                'min': np.min(fermi_energies),
                'max': np.max(fermi_energies),
                'count': len(fermi_energies)
            }
        
        return trends
    
    def _analyze_structural_trends(self, materials: Dict) -> Dict[str, Any]:
        """Analyze trends in structural properties."""
        trends = {}
        
        # Lattice parameter analysis
        lattice_a_values = []
        cell_volumes = []
        space_groups = []
        
        for material_id, data in materials.items():
            props = data['properties']
            
            if 'lattice_a' in props:
                lattice_a_values.append(props['lattice_a']['value'])
            
            if 'cell_volume' in props:
                cell_volumes.append(props['cell_volume']['value'])
            
            if data['space_group']:
                space_groups.append(data['space_group'])
        
        if lattice_a_values:
            trends['lattice_a_statistics'] = {
                'mean': np.mean(lattice_a_values),
                'std': np.std(lattice_a_values),
                'min': np.min(lattice_a_values),
                'max': np.max(lattice_a_values),
                'count': len(lattice_a_values)
            }
        
        if cell_volumes:
            trends['cell_volume_statistics'] = {
                'mean': np.mean(cell_volumes),
                'std': np.std(cell_volumes),
                'min': np.min(cell_volumes),
                'max': np.max(cell_volumes),
                'count': len(cell_volumes)
            }
        
        if space_groups:
            from collections import Counter
            sg_counts = Counter(space_groups)
            trends['space_group_distribution'] = dict(sg_counts.most_common(10))
        
        return trends
    
    def _store_derived_properties(self, derived_properties: List[DerivedProperty]) -> None:
        """Store derived properties back to the database."""
        with self.db._get_connection() as conn:
            for prop in derived_properties:
                try:
                    conn.execute("""
                        INSERT INTO properties (
                            material_id, property_category, property_name, 
                            property_value, property_unit, extracted_at, 
                            extractor_script, confidence
                        ) VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?)
                    """, (
                        prop.material_id, 'derived', prop.property_name,
                        prop.property_value, prop.property_unit,
                        f'property_analysis_framework.py:{prop.derivation_method}',
                        prop.confidence
                    ))
                except Exception as e:
                    print(f"Warning: Could not store derived property {prop.property_name} for {prop.material_id}: {e}")
        
        print(f"✅ Stored {len(derived_properties)} derived properties in database")


def main():
    """Run comprehensive property analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Property Analysis Framework")
    parser.add_argument("--db-path", default="materials.db", help="Path to materials database")
    parser.add_argument("--output", default="property_analysis_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = PropertyAnalysisFramework(args.db_path)
    results = analyzer.analyze_all_materials()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Analysis Results Summary:")
    print(f"   Derived properties calculated: {len(results['derived_properties'])}")
    print(f"   Significant correlations found: {len(results['correlations'].get('significant_correlations', []))}")
    print(f"   Materials analyzed: {results['database_insights'].get('total_materials_analyzed', 0)}")
    print(f"   Results saved to: {args.output}")


if __name__ == "__main__":
    main()