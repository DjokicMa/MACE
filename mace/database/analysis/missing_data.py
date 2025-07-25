"""
Missing Data Analysis
=====================
Analyze missing properties and suggest calculations to run.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
import json


class MissingDataAnalyzer:
    """Analyzes missing properties across materials to guide calculations."""
    
    # Define expected properties for each calculation type
    CALC_TYPE_PROPERTIES = {
        'OPT': {
            'required': ['total_energy', 'final_a', 'final_b', 'final_c'],
            'optional': ['final_alpha', 'final_beta', 'final_gamma', 'optimization_cycles', 
                        'max_gradient', 'rms_gradient', 'density']
        },
        'SP': {
            'required': ['total_energy', 'band_gap', 'fermi_energy'],
            'optional': ['indirect_band_gap', 'direct_band_gap', 'homo_energy', 
                        'lumo_energy', 'vbm_energy', 'cbm_energy']
        },
        'FREQ': {
            'required': ['freq_n_modes', 'freq_min', 'freq_max'],
            'optional': ['freq_negative_count', 'freq_zero_point_energy', 
                        'freq_thermal_energy', 'freq_entropy']
        },
        'BAND': {
            'required': ['band_n_kpoints', 'band_n_bands'],
            'optional': ['band_gap_from_band', 'band_fermi_energy']
        },
        'DOSS': {
            'required': ['dos_n_points', 'dos_energy_range'],
            'optional': ['dos_gap', 'dos_fermi_energy', 'dos_n_electrons']
        },
        'TRANSPORT': {
            'required': ['transport_seebeck_300k', 'transport_conductivity_300k'],
            'optional': ['transport_power_factor_300k', 'transport_thermal_conductivity_300k']
        },
        'CHARGE+POTENTIAL': {
            'required': ['charge_min', 'charge_max', 'potential_min', 'potential_max'],
            'optional': ['charge_sum', 'potential_average']
        }
    }
    
    # Define property dependencies
    PROPERTY_DEPENDENCIES = {
        'band_gap': ['SP'],
        'fermi_energy': ['SP', 'BAND', 'DOSS'],
        'phonon_properties': ['FREQ'],
        'transport_properties': ['TRANSPORT'],
        'charge_density': ['CHARGE+POTENTIAL']
    }
    
    def __init__(self, db):
        """
        Initialize with database connection.
        
        Args:
            db: MaterialDatabase instance
        """
        self.db = db
        
    def analyze_missing_data(self, material_ids: List[str] = None,
                           target_properties: List[str] = None) -> Dict[str, Any]:
        """
        Analyze missing properties across materials.
        
        Args:
            material_ids: List of materials to analyze (None = all)
            target_properties: Specific properties to check (None = all known)
            
        Returns:
            Analysis results with missing data summary and recommendations
        """
        # Get materials to analyze
        if material_ids:
            materials = [self.db.get_material(mid) for mid in material_ids 
                        if self.db.get_material(mid)]
        else:
            materials = self.db.get_all_materials()
            
        if not materials:
            return {'error': 'No materials found to analyze'}
            
        results = {
            'summary': {
                'total_materials': len(materials),
                'materials_analyzed': []
            },
            'missing_by_material': {},
            'missing_by_property': defaultdict(list),
            'recommendations': {},
            'completeness_scores': {}
        }
        
        # Analyze each material
        for material in materials:
            mat_id = material['material_id']
            results['summary']['materials_analyzed'].append(mat_id)
            
            # Get existing properties
            properties = self.db.get_material_properties(mat_id)
            existing_props = {p['property_name'] for p in properties}
            
            # Get calculation history
            calculations = self.db.get_material_calculations(mat_id)
            completed_calcs = {c['calc_type'] for c in calculations 
                             if c.get('status') == 'completed'}
            
            # Analyze missing properties
            missing_analysis = self._analyze_material_missing(
                mat_id, existing_props, completed_calcs, target_properties
            )
            
            results['missing_by_material'][mat_id] = missing_analysis
            
            # Update missing by property
            for prop in missing_analysis['missing_properties']:
                results['missing_by_property'][prop].append(mat_id)
                
            # Calculate completeness score
            total_expected = len(missing_analysis['expected_properties'])
            if total_expected > 0:
                # Only count properties that are actually expected
                matching_props = existing_props.intersection(set(missing_analysis['expected_properties']))
                completeness = len(matching_props) / total_expected
            else:
                completeness = 1.0
            results['completeness_scores'][mat_id] = completeness
            
        # Generate recommendations
        self._generate_recommendations(results)
        
        # Add statistics
        self._add_statistics(results)
        
        return results
        
    def _analyze_material_missing(self, material_id: str, 
                                existing_props: Set[str],
                                completed_calcs: Set[str],
                                target_properties: List[str] = None) -> Dict:
        """Analyze missing properties for a single material."""
        analysis = {
            'existing_properties': sorted(existing_props),
            'completed_calculations': sorted(completed_calcs),
            'expected_properties': set(),
            'missing_properties': set(),
            'missing_by_calc_type': defaultdict(list),
            'suggested_calculations': []
        }
        
        # Determine expected properties based on completed calculations
        for calc_type in completed_calcs:
            if calc_type in self.CALC_TYPE_PROPERTIES:
                expected = self.CALC_TYPE_PROPERTIES[calc_type]
                analysis['expected_properties'].update(expected['required'])
                analysis['expected_properties'].update(expected['optional'])
                
        # If target properties specified, focus on those
        if target_properties:
            analysis['expected_properties'] = analysis['expected_properties'].intersection(
                set(target_properties)
            )
            
        # Find missing properties
        analysis['missing_properties'] = analysis['expected_properties'] - existing_props
        
        # Categorize missing by calculation type
        for prop in analysis['missing_properties']:
            for calc_type, props_dict in self.CALC_TYPE_PROPERTIES.items():
                all_props = props_dict['required'] + props_dict['optional']
                if prop in all_props:
                    analysis['missing_by_calc_type'][calc_type].append(prop)
                    
        # Suggest calculations to get missing properties
        suggested_calcs = set()
        for prop in analysis['missing_properties']:
            # Check if property has known dependencies
            for dep_name, dep_calcs in self.PROPERTY_DEPENDENCIES.items():
                if prop.startswith(dep_name.split('_')[0]):  # Match property category
                    suggested_calcs.update(dep_calcs)
                    
        # Also check direct calc type mapping
        for calc_type, missing_props in analysis['missing_by_calc_type'].items():
            if missing_props and calc_type not in completed_calcs:
                suggested_calcs.add(calc_type)
                
        analysis['suggested_calculations'] = sorted(suggested_calcs)
        
        # Convert sets to sorted lists for JSON serialization
        analysis['expected_properties'] = sorted(analysis['expected_properties'])
        analysis['missing_properties'] = sorted(analysis['missing_properties'])
        
        return analysis
        
    def _generate_recommendations(self, results: Dict):
        """Generate actionable recommendations based on missing data analysis."""
        recommendations = {
            'global': [],
            'by_material': {},
            'by_calculation_type': defaultdict(list)
        }
        
        # Global recommendations
        total_mats = results['summary']['total_materials']
        
        # Find most commonly missing properties
        missing_counts = [(prop, len(mats)) for prop, mats 
                         in results['missing_by_property'].items()]
        missing_counts.sort(key=lambda x: x[1], reverse=True)
        
        if missing_counts:
            top_missing = missing_counts[:5]
            recommendations['global'].append({
                'priority': 'high',
                'message': f"Most commonly missing properties across {total_mats} materials:",
                'details': [f"{prop}: missing in {count} materials ({count/total_mats*100:.1f}%)" 
                           for prop, count in top_missing]
            })
            
        # Find materials with lowest completeness
        sorted_completeness = sorted(results['completeness_scores'].items(), 
                                   key=lambda x: x[1])
        incomplete_mats = [(m, s) for m, s in sorted_completeness if s < 0.5]
        
        if incomplete_mats:
            recommendations['global'].append({
                'priority': 'medium',
                'message': f"Materials with less than 50% property completeness ({len(incomplete_mats)} found):",
                'details': [f"{mat}: {score*100:.1f}% complete" 
                           for mat, score in incomplete_mats[:10]]
            })
            
        # Per-material recommendations
        for mat_id, analysis in results['missing_by_material'].items():
            mat_recs = []
            
            # Suggest calculations
            if analysis['suggested_calculations']:
                mat_recs.append({
                    'priority': 'high',
                    'message': f"Run these calculations to obtain missing properties:",
                    'calculations': analysis['suggested_calculations']
                })
                
            # Check for incomplete calculations
            missing_by_calc = analysis['missing_by_calc_type']
            for calc_type in analysis['completed_calculations']:
                if calc_type in missing_by_calc and missing_by_calc[calc_type]:
                    mat_recs.append({
                        'priority': 'medium',
                        'message': f"{calc_type} calculation may have failed to extract properties:",
                        'missing_properties': missing_by_calc[calc_type]
                    })
                    
            if mat_recs:
                recommendations['by_material'][mat_id] = mat_recs
                
        # By calculation type recommendations
        calc_type_missing = defaultdict(set)
        for mat_id, analysis in results['missing_by_material'].items():
            for calc_type, props in analysis['missing_by_calc_type'].items():
                if props:
                    calc_type_missing[calc_type].add(mat_id)
                    
        for calc_type, materials in calc_type_missing.items():
            if len(materials) >= 0.1 * total_mats:  # If affects >10% of materials
                recommendations['by_calculation_type'][calc_type] = {
                    'priority': 'high',
                    'message': f"{calc_type} properties missing for {len(materials)} materials",
                    'action': f"Consider running {calc_type} calculations for these materials"
                }
                
        results['recommendations'] = recommendations
        
    def _add_statistics(self, results: Dict):
        """Add summary statistics to results."""
        stats = {
            'average_completeness': 0.0,
            'property_coverage': {},
            'calculation_coverage': {}
        }
        
        # Average completeness
        if results['completeness_scores']:
            stats['average_completeness'] = sum(results['completeness_scores'].values()) / len(results['completeness_scores'])
            
        # Property coverage
        all_props = set()
        for analysis in results['missing_by_material'].values():
            all_props.update(analysis['expected_properties'])
            
        for prop in all_props:
            missing_count = len(results['missing_by_property'].get(prop, []))
            present_count = results['summary']['total_materials'] - missing_count
            stats['property_coverage'][prop] = {
                'present': present_count,
                'missing': missing_count,
                'coverage': present_count / results['summary']['total_materials'] if results['summary']['total_materials'] > 0 else 0
            }
            
        # Calculation coverage
        calc_counts = defaultdict(int)
        for analysis in results['missing_by_material'].values():
            for calc_type in analysis['completed_calculations']:
                calc_counts[calc_type] += 1
                
        for calc_type in self.CALC_TYPE_PROPERTIES:
            count = calc_counts.get(calc_type, 0)
            stats['calculation_coverage'][calc_type] = {
                'completed': count,
                'coverage': count / results['summary']['total_materials'] if results['summary']['total_materials'] > 0 else 0
            }
            
        results['statistics'] = stats
        
    def format_missing_data_report(self, analysis_results: Dict, 
                                  detail_level: str = 'summary') -> str:
        """
        Format missing data analysis as readable report.
        
        Args:
            analysis_results: Results from analyze_missing_data()
            detail_level: 'summary', 'detailed', or 'full'
            
        Returns:
            Formatted report string
        """
        lines = []
        
        # Header
        lines.append("=== Missing Data Analysis Report ===")
        lines.append(f"Materials analyzed: {analysis_results['summary']['total_materials']}")
        lines.append(f"Average completeness: {analysis_results['statistics']['average_completeness']*100:.1f}%")
        lines.append("")
        
        # Global recommendations
        if analysis_results['recommendations']['global']:
            lines.append("=== Key Findings ===")
            for rec in analysis_results['recommendations']['global']:
                lines.append(f"\n[{rec['priority'].upper()}] {rec['message']}")
                if 'details' in rec:
                    for detail in rec['details']:
                        lines.append(f"  - {detail}")
            lines.append("")
            
        # Property coverage summary
        lines.append("=== Property Coverage ===")
        prop_coverage = analysis_results['statistics']['property_coverage']
        sorted_props = sorted(prop_coverage.items(), 
                            key=lambda x: x[1]['coverage'], reverse=True)
        
        if detail_level in ['detailed', 'full']:
            for prop, coverage in sorted_props[:20]:  # Top 20
                lines.append(f"{prop:30} {coverage['coverage']*100:5.1f}% "
                           f"({coverage['present']}/{analysis_results['summary']['total_materials']})")
        else:
            # Just show summary
            fully_covered = sum(1 for p, c in sorted_props if c['coverage'] == 1.0)
            partially_covered = sum(1 for p, c in sorted_props if 0 < c['coverage'] < 1.0)
            not_covered = sum(1 for p, c in sorted_props if c['coverage'] == 0)
            
            lines.append(f"Fully covered properties: {fully_covered}")
            lines.append(f"Partially covered properties: {partially_covered}")
            lines.append(f"Not covered properties: {not_covered}")
            
        lines.append("")
        
        # Calculation coverage
        lines.append("=== Calculation Coverage ===")
        calc_coverage = analysis_results['statistics']['calculation_coverage']
        for calc_type, coverage in sorted(calc_coverage.items()):
            lines.append(f"{calc_type:15} {coverage['coverage']*100:5.1f}% "
                       f"({coverage['completed']}/{analysis_results['summary']['total_materials']})")
        lines.append("")
        
        # Material-specific details (if requested)
        if detail_level == 'full':
            lines.append("=== Material Details ===")
            for mat_id, analysis in analysis_results['missing_by_material'].items():
                if analysis['missing_properties']:  # Only show materials with missing data
                    lines.append(f"\n{mat_id}:")
                    lines.append(f"  Completeness: {analysis_results['completeness_scores'][mat_id]*100:.1f}%")
                    lines.append(f"  Missing properties: {', '.join(analysis['missing_properties'][:10])}")
                    if len(analysis['missing_properties']) > 10:
                        lines.append(f"  ... and {len(analysis['missing_properties'])-10} more")
                    if analysis['suggested_calculations']:
                        lines.append(f"  Suggested calculations: {', '.join(analysis['suggested_calculations'])}")
                        
        # Recommendations summary
        if detail_level in ['detailed', 'full']:
            mat_recs = analysis_results['recommendations']['by_material']
            if mat_recs:
                lines.append("\n=== Top Material Recommendations ===")
                for mat_id, recs in list(mat_recs.items())[:5]:  # Top 5
                    lines.append(f"\n{mat_id}:")
                    for rec in recs:
                        lines.append(f"  [{rec['priority']}] {rec['message']}")
                        if 'calculations' in rec:
                            lines.append(f"    Calculations: {', '.join(rec['calculations'])}")
                            
        return "\n".join(lines)


def analyze_missing_data(db, material_ids: List[str] = None,
                       target_properties: List[str] = None,
                       output_format: str = 'report',
                       detail_level: str = 'summary') -> str:
    """
    Convenience function to analyze missing data.
    
    Args:
        db: MaterialDatabase instance
        material_ids: List of materials to analyze (None = all)
        target_properties: Specific properties to check (None = all)
        output_format: 'report', 'json', or 'dict'
        detail_level: For report format - 'summary', 'detailed', or 'full'
        
    Returns:
        Formatted analysis results
    """
    analyzer = MissingDataAnalyzer(db)
    results = analyzer.analyze_missing_data(material_ids, target_properties)
    
    if output_format == 'report':
        return analyzer.format_missing_data_report(results, detail_level)
    elif output_format == 'json':
        return json.dumps(results, indent=2, default=str)
    else:  # dict
        return results