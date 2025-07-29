"""
Property Aggregation by Groups
===============================
Aggregate and analyze properties by material groups.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import json
import statistics
from collections import defaultdict


class PropertyAggregator:
    """Aggregates properties by material groups."""
    
    # Common grouping criteria
    GROUP_BY_OPTIONS = {
        'space_group': 'Space group number',
        'crystal_system': 'Crystal system (cubic, hexagonal, etc.)',
        'formula_prefix': 'First element in formula',
        'conductivity_type': 'Conductivity type (metal, semiconductor, insulator)',
        'band_gap_range': 'Band gap ranges',
        'energy_range': 'Total energy ranges',
        'atoms_range': 'Number of atoms ranges',
        'calculation_type': 'Calculation type',
        'convergence_status': 'Convergence status'
    }
    
    # Aggregation functions
    AGGREGATION_FUNCTIONS = {
        'mean': statistics.mean,
        'median': statistics.median,
        'min': min,
        'max': max,
        'sum': sum,
        'count': len,
        'stdev': lambda x: statistics.stdev(x) if len(x) > 1 else 0,
        'range': lambda x: max(x) - min(x) if x else 0
    }
    
    def __init__(self, db):
        """
        Initialize with database connection.
        
        Args:
            db: MaterialDatabase instance
        """
        self.db = db
        
    def aggregate_by_group(self, group_by: str, properties: List[str],
                         aggregation: str = 'mean',
                         filters: List[str] = None) -> Dict[str, Any]:
        """
        Aggregate properties by specified grouping.
        
        Args:
            group_by: Grouping criterion
            properties: Properties to aggregate
            aggregation: Aggregation function (mean, median, min, max, etc.)
            filters: Optional property filters
            
        Returns:
            Aggregation results
        """
        # Get all materials
        materials = self.db.get_all_materials()
        
        # Apply filters if provided
        if filters:
            from ..query.filters import PropertyFilter
            filter_obj = PropertyFilter()
            for filter_str in filters:
                # Parse filter string
                parts = filter_str.split()
                if len(parts) >= 3:
                    prop = parts[0]
                    op = parts[1]
                    val = ' '.join(parts[2:])
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    filter_obj.add_filter(prop, op, val)
                    
            # Filter materials
            filtered_materials = []
            for mat in materials:
                if filter_obj.matches(mat['material_id'], self.db):
                    filtered_materials.append(mat)
            materials = filtered_materials
            
        # Group materials
        groups = self._group_materials(materials, group_by)
        
        # Aggregate properties for each group
        results = {
            'group_by': group_by,
            'aggregation': aggregation,
            'properties': properties,
            'total_materials': len(materials),
            'total_groups': len(groups),
            'groups': {}
        }
        
        for group_name, material_ids in groups.items():
            group_data = self._aggregate_group(
                group_name, material_ids, properties, aggregation
            )
            results['groups'][group_name] = group_data
            
        # Add summary statistics
        self._add_summary_stats(results)
        
        return results
        
    def _group_materials(self, materials: List[Dict], group_by: str) -> Dict[str, List[str]]:
        """Group materials by specified criterion."""
        groups = defaultdict(list)
        
        for material in materials:
            mat_id = material['material_id']
            
            if group_by == 'space_group':
                group = str(material.get('space_group', 'Unknown'))
                
            elif group_by == 'crystal_system':
                # Determine crystal system from space group
                sg = material.get('space_group')
                if sg:
                    group = self._get_crystal_system(sg)
                else:
                    group = 'Unknown'
                    
            elif group_by == 'formula_prefix':
                formula = material.get('formula', '')
                if formula:
                    # Extract first element
                    import re
                    match = re.match(r'^([A-Z][a-z]?)', formula)
                    group = match.group(1) if match else 'Unknown'
                else:
                    group = 'Unknown'
                    
            elif group_by == 'conductivity_type':
                # Get conductivity type from properties
                props = self.db.get_material_properties(mat_id)
                cond_type = None
                for prop in props:
                    if prop['property_name'] == 'conductivity_type':
                        cond_type = prop['property_value']
                        break
                group = cond_type or 'Unknown'
                
            elif group_by == 'band_gap_range':
                # Get band gap and categorize
                props = self.db.get_material_properties(mat_id)
                band_gap = None
                for prop in props:
                    if prop['property_name'] == 'band_gap':
                        try:
                            band_gap = float(prop['property_value'])
                        except ValueError:
                            pass
                        break
                        
                if band_gap is not None:
                    if band_gap < 0.1:
                        group = 'Metal (< 0.1 eV)'
                    elif band_gap < 1.5:
                        group = 'Small gap (0.1-1.5 eV)'
                    elif band_gap < 3.0:
                        group = 'Medium gap (1.5-3.0 eV)'
                    elif band_gap < 6.0:
                        group = 'Large gap (3.0-6.0 eV)'
                    else:
                        group = 'Very large gap (> 6.0 eV)'
                else:
                    group = 'No band gap data'
                    
            elif group_by == 'energy_range':
                # Get total energy and categorize
                props = self.db.get_material_properties(mat_id)
                energy = None
                for prop in props:
                    if prop['property_name'] == 'total_energy':
                        try:
                            energy = float(prop['property_value'])
                        except ValueError:
                            pass
                        break
                        
                if energy is not None:
                    if energy > 0:
                        group = 'Positive energy'
                    elif energy > -100:
                        group = 'High energy (-100 to 0)'
                    elif energy > -1000:
                        group = 'Medium energy (-1000 to -100)'
                    else:
                        group = 'Low energy (< -1000)'
                else:
                    group = 'No energy data'
                    
            elif group_by == 'atoms_range':
                # Get atom count and categorize
                props = self.db.get_material_properties(mat_id)
                atoms = None
                for prop in props:
                    if prop['property_name'] == 'atoms_in_unit_cell':
                        try:
                            atoms = int(prop['property_value'])
                        except ValueError:
                            pass
                        break
                        
                if atoms is not None:
                    if atoms <= 10:
                        group = 'Small (≤ 10 atoms)'
                    elif atoms <= 50:
                        group = 'Medium (11-50 atoms)'
                    elif atoms <= 200:
                        group = 'Large (51-200 atoms)'
                    else:
                        group = 'Very large (> 200 atoms)'
                else:
                    group = 'No atom count data'
                    
            else:
                group = 'Unknown'
                
            groups[group].append(mat_id)
            
        return dict(groups)
        
    def _get_crystal_system(self, space_group: int) -> str:
        """Determine crystal system from space group number."""
        if 1 <= space_group <= 2:
            return 'triclinic'
        elif 3 <= space_group <= 15:
            return 'monoclinic'
        elif 16 <= space_group <= 74:
            return 'orthorhombic'
        elif 75 <= space_group <= 142:
            return 'tetragonal'
        elif 143 <= space_group <= 167:
            return 'trigonal'
        elif 168 <= space_group <= 194:
            return 'hexagonal'
        elif 195 <= space_group <= 230:
            return 'cubic'
        else:
            return 'unknown'
            
    def _aggregate_group(self, group_name: str, material_ids: List[str],
                       properties: List[str], aggregation: str) -> Dict[str, Any]:
        """Aggregate properties for a group of materials."""
        group_data = {
            'name': group_name,
            'material_count': len(material_ids),
            'material_ids': material_ids[:10],  # Show first 10
            'properties': {}
        }
        
        if len(material_ids) > 10:
            group_data['material_ids_truncated'] = True
            
        # Get aggregation function
        agg_func = self.AGGREGATION_FUNCTIONS.get(aggregation, statistics.mean)
        
        # Aggregate each property
        for prop_name in properties:
            values = []
            
            # Collect values from all materials in group
            for mat_id in material_ids:
                props = self.db.get_material_properties(mat_id)
                for prop in props:
                    if prop['property_name'] == prop_name:
                        try:
                            value = float(prop['property_value'])
                            values.append(value)
                        except (ValueError, TypeError):
                            pass
                        break
                        
            # Calculate aggregation
            if values:
                if aggregation == 'count':
                    result = len(values)
                else:
                    result = agg_func(values)
                    
                group_data['properties'][prop_name] = {
                    'value': result,
                    'count': len(values),
                    'coverage': len(values) / len(material_ids)
                }
                
                # Add additional statistics for mean/median
                if aggregation in ['mean', 'median'] and len(values) > 1:
                    group_data['properties'][prop_name]['min'] = min(values)
                    group_data['properties'][prop_name]['max'] = max(values)
                    group_data['properties'][prop_name]['stdev'] = statistics.stdev(values)
                    
        return group_data
        
    def _add_summary_stats(self, results: Dict):
        """Add summary statistics to results."""
        summary = {
            'largest_group': None,
            'smallest_group': None,
            'property_coverage': {}
        }
        
        # Find largest and smallest groups
        groups = results['groups']
        if groups:
            largest = max(groups.items(), key=lambda x: x[1]['material_count'])
            smallest = min(groups.items(), key=lambda x: x[1]['material_count'])
            
            summary['largest_group'] = {
                'name': largest[0],
                'count': largest[1]['material_count']
            }
            summary['smallest_group'] = {
                'name': smallest[0],
                'count': smallest[1]['material_count']
            }
            
        # Calculate overall property coverage
        for prop_name in results['properties']:
            total_with_prop = 0
            for group_data in groups.values():
                if prop_name in group_data['properties']:
                    total_with_prop += group_data['properties'][prop_name]['count']
                    
            summary['property_coverage'][prop_name] = {
                'total_values': total_with_prop,
                'coverage': total_with_prop / results['total_materials'] if results['total_materials'] > 0 else 0
            }
            
        results['summary'] = summary
        
    def compare_groups(self, group_by: str, properties: List[str],
                      groups_to_compare: List[str] = None) -> Dict[str, Any]:
        """
        Compare property statistics across groups.
        
        Args:
            group_by: Grouping criterion
            properties: Properties to compare
            groups_to_compare: Specific groups to compare (None = all)
            
        Returns:
            Comparison results
        """
        # Get aggregated data
        agg_results = self.aggregate_by_group(group_by, properties, 'mean')
        
        # Filter groups if specified
        if groups_to_compare:
            filtered_groups = {
                g: data for g, data in agg_results['groups'].items()
                if g in groups_to_compare
            }
            agg_results['groups'] = filtered_groups
            
        # Create comparison matrix
        comparison = {
            'group_by': group_by,
            'properties': properties,
            'groups': list(agg_results['groups'].keys()),
            'comparison_matrix': {}
        }
        
        for prop_name in properties:
            prop_comparison = {}
            
            # Collect values across groups
            values = []
            for group, data in agg_results['groups'].items():
                if prop_name in data['properties']:
                    value = data['properties'][prop_name]['value']
                    values.append(value)
                    prop_comparison[group] = value
                    
            # Calculate statistics across groups
            if values:
                prop_stats = {
                    'min_value': min(values),
                    'max_value': max(values),
                    'range': max(values) - min(values),
                    'values_by_group': prop_comparison
                }
                
                # Find group with min and max
                for group, value in prop_comparison.items():
                    if value == prop_stats['min_value']:
                        prop_stats['min_group'] = group
                    if value == prop_stats['max_value']:
                        prop_stats['max_group'] = group
                        
                comparison['comparison_matrix'][prop_name] = prop_stats
                
        return comparison
        
    def format_aggregation_report(self, results: Dict, detailed: bool = False) -> str:
        """
        Format aggregation results as readable report.
        
        Args:
            results: Results from aggregate_by_group()
            detailed: Show detailed group information
            
        Returns:
            Formatted report string
        """
        lines = []
        
        # Header
        lines.append("=== Property Aggregation Report ===")
        lines.append(f"Grouped by: {results['group_by']}")
        lines.append(f"Aggregation: {results['aggregation']}")
        lines.append(f"Properties: {', '.join(results['properties'])}")
        lines.append(f"Total materials: {results['total_materials']}")
        lines.append(f"Total groups: {results['total_groups']}")
        lines.append("")
        
        # Summary
        if 'summary' in results:
            summary = results['summary']
            lines.append("=== Summary ===")
            
            if summary['largest_group']:
                lines.append(f"Largest group: {summary['largest_group']['name']} "
                           f"({summary['largest_group']['count']} materials)")
                           
            if summary['smallest_group']:
                lines.append(f"Smallest group: {summary['smallest_group']['name']} "
                           f"({summary['smallest_group']['count']} materials)")
                           
            lines.append("")
            
        # Group results
        lines.append("=== Group Results ===")
        
        # Sort groups by material count
        sorted_groups = sorted(
            results['groups'].items(),
            key=lambda x: x[1]['material_count'],
            reverse=True
        )
        
        for group_name, group_data in sorted_groups:
            lines.append(f"\n{group_name} ({group_data['material_count']} materials):")
            
            # Show property aggregations
            for prop_name in results['properties']:
                if prop_name in group_data['properties']:
                    prop_data = group_data['properties'][prop_name]
                    value = prop_data['value']
                    coverage = prop_data['coverage']
                    
                    # Format value
                    if isinstance(value, float):
                        if abs(value) < 0.001 or abs(value) > 10000:
                            value_str = f"{value:.3e}"
                        else:
                            value_str = f"{value:.3f}"
                    else:
                        value_str = str(value)
                        
                    line = f"  {prop_name}: {value_str}"
                    
                    # Add coverage if not 100%
                    if coverage < 1.0:
                        line += f" ({coverage:.1%} coverage)"
                        
                    # Add range for mean/median
                    if results['aggregation'] in ['mean', 'median'] and 'min' in prop_data:
                        line += f" [range: {prop_data['min']:.3f} - {prop_data['max']:.3f}]"
                        
                    lines.append(line)
                else:
                    lines.append(f"  {prop_name}: No data")
                    
            # Show example materials if detailed
            if detailed and group_data.get('material_ids'):
                example_mats = group_data['material_ids'][:5]
                lines.append(f"  Examples: {', '.join(example_mats)}")
                if group_data.get('material_ids_truncated'):
                    lines.append(f"  ... and {group_data['material_count'] - 10} more")
                    
        return "\n".join(lines)
        
    def format_comparison_report(self, comparison: Dict) -> str:
        """Format group comparison as readable report."""
        lines = []
        
        lines.append("=== Group Comparison Report ===")
        lines.append(f"Grouped by: {comparison['group_by']}")
        lines.append(f"Groups compared: {', '.join(comparison['groups'])}")
        lines.append("")
        
        for prop_name, prop_data in comparison['comparison_matrix'].items():
            lines.append(f"\n{prop_name}:")
            
            # Show range statistics
            lines.append(f"  Range: {prop_data['range']:.3f}")
            lines.append(f"  Min: {prop_data['min_value']:.3f} ({prop_data.get('min_group', 'N/A')})")
            lines.append(f"  Max: {prop_data['max_value']:.3f} ({prop_data.get('max_group', 'N/A')})")
            
            # Show values by group
            lines.append("  Values by group:")
            for group, value in sorted(prop_data['values_by_group'].items()):
                lines.append(f"    {group}: {value:.3f}")
                
        return "\n".join(lines)


def aggregate_by_groups(db, group_by: str, properties: List[str],
                       aggregation: str = 'mean',
                       filters: List[str] = None,
                       output_format: str = 'report',
                       detailed: bool = False) -> str:
    """
    Convenience function to aggregate properties by groups.
    
    Args:
        db: MaterialDatabase instance
        group_by: Grouping criterion
        properties: Properties to aggregate
        aggregation: Aggregation function
        filters: Optional property filters
        output_format: 'report', 'json', or 'dict'
        detailed: Show detailed information
        
    Returns:
        Formatted results
    """
    aggregator = PropertyAggregator(db)
    results = aggregator.aggregate_by_group(group_by, properties, aggregation, filters)
    
    if output_format == 'report':
        return aggregator.format_aggregation_report(results, detailed)
    elif output_format == 'json':
        return json.dumps(results, indent=2, default=str)
    else:  # dict
        return results