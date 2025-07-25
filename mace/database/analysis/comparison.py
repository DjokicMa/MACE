"""
Material Property Comparison
============================
Compare properties across multiple materials for structure-property relationships.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
from collections import defaultdict


class MaterialComparison:
    """Handles comparison of properties across multiple materials."""
    
    def __init__(self, db):
        """
        Initialize with database connection.
        
        Args:
            db: MaterialDatabase instance
        """
        self.db = db
        
    def compare_materials(self, material_ids: List[str], 
                         properties: List[str] = None,
                         include_metadata: bool = True) -> Dict[str, Any]:
        """
        Compare specified properties across materials.
        
        Args:
            material_ids: List of material IDs to compare
            properties: List of property names to compare (None = all common properties)
            include_metadata: Whether to include material metadata
            
        Returns:
            Comparison results dictionary
        """
        results = {
            'materials': {},
            'properties': {},
            'summary': {}
        }
        
        # Get material info and properties for each material
        for mat_id in material_ids:
            material = self.db.get_material(mat_id)
            if not material:
                results['materials'][mat_id] = {'error': 'Material not found'}
                continue
                
            mat_props = self.db.get_material_properties(mat_id)
            
            # Store material info
            mat_info = {
                'formula': material.get('formula', 'Unknown'),
                'space_group': material.get('space_group', 'Unknown'),
                'properties': {}
            }
            
            if include_metadata:
                mat_info['metadata'] = {
                    'created_at': material.get('created_at'),
                    'source_file': material.get('source_file'),
                    'dimensionality': material.get('dimensionality')
                }
                
            # Process properties
            for prop in mat_props:
                prop_name = prop['property_name']
                if properties is None or prop_name in properties:
                    mat_info['properties'][prop_name] = {
                        'value': prop['property_value'],
                        'unit': prop.get('property_unit', ''),
                        'category': prop.get('property_category', 'Other'),
                        'calc_id': prop.get('calc_id')
                    }
                    
            results['materials'][mat_id] = mat_info
            
        # Analyze properties across materials
        self._analyze_properties(results)
        
        # Generate summary
        self._generate_summary(results)
        
        return results
        
    def _analyze_properties(self, results: Dict):
        """Analyze properties across all materials."""
        all_properties = defaultdict(list)
        
        # Collect all property values
        for mat_id, mat_data in results['materials'].items():
            if 'error' in mat_data:
                continue
                
            for prop_name, prop_data in mat_data.get('properties', {}).items():
                all_properties[prop_name].append({
                    'material_id': mat_id,
                    'value': prop_data['value'],
                    'unit': prop_data.get('unit', '')
                })
                
        # Analyze each property
        for prop_name, values in all_properties.items():
            analysis = {
                'materials_with_property': len(values),
                'values': values
            }
            
            # Try to compute statistics for numeric properties
            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v['value']))
                except (ValueError, TypeError):
                    pass
                    
            if numeric_values and len(numeric_values) == len(values):
                analysis['is_numeric'] = True
                analysis['min'] = min(numeric_values)
                analysis['max'] = max(numeric_values)
                analysis['range'] = analysis['max'] - analysis['min']
                analysis['average'] = sum(numeric_values) / len(numeric_values)
                
                # Calculate relative differences
                if analysis['average'] != 0:
                    analysis['relative_range'] = analysis['range'] / abs(analysis['average'])
                else:
                    analysis['relative_range'] = float('inf') if analysis['range'] != 0 else 0
                    
                # Find materials with min/max values
                for i, v in enumerate(numeric_values):
                    if v == analysis['min']:
                        analysis['min_material'] = values[i]['material_id']
                    if v == analysis['max']:
                        analysis['max_material'] = values[i]['material_id']
            else:
                analysis['is_numeric'] = False
                
            results['properties'][prop_name] = analysis
            
    def _generate_summary(self, results: Dict):
        """Generate comparison summary."""
        summary = {
            'total_materials': len(results['materials']),
            'total_properties': len(results['properties']),
            'common_properties': [],
            'unique_properties': defaultdict(list),
            'largest_differences': []
        }
        
        # Find common properties (present in all materials)
        valid_materials = [m for m, d in results['materials'].items() if 'error' not in d]
        
        if valid_materials:
            first_mat_props = set(results['materials'][valid_materials[0]].get('properties', {}).keys())
            common_props = first_mat_props
            
            for mat_id in valid_materials[1:]:
                mat_props = set(results['materials'][mat_id].get('properties', {}).keys())
                common_props = common_props.intersection(mat_props)
                
            summary['common_properties'] = list(common_props)
            
            # Find unique properties per material
            for mat_id in valid_materials:
                mat_props = set(results['materials'][mat_id].get('properties', {}).keys())
                unique = mat_props - common_props
                if unique:
                    summary['unique_properties'][mat_id] = list(unique)
                    
        # Find properties with largest relative differences
        numeric_props = [(name, data) for name, data in results['properties'].items() 
                        if data.get('is_numeric', False) and data.get('relative_range', 0) > 0]
        
        numeric_props.sort(key=lambda x: x[1].get('relative_range', 0), reverse=True)
        
        for prop_name, prop_data in numeric_props[:10]:  # Top 10
            summary['largest_differences'].append({
                'property': prop_name,
                'relative_range': prop_data['relative_range'],
                'min_value': prop_data['min'],
                'max_value': prop_data['max'],
                'min_material': prop_data.get('min_material'),
                'max_material': prop_data.get('max_material')
            })
            
        results['summary'] = summary
        
    def format_comparison_table(self, comparison_results: Dict, 
                               properties_filter: List[str] = None) -> str:
        """
        Format comparison results as a table.
        
        Args:
            comparison_results: Results from compare_materials()
            properties_filter: List of properties to include in table
            
        Returns:
            Formatted table string
        """
        lines = []
        
        # Header
        materials = list(comparison_results['materials'].keys())
        valid_materials = [m for m in materials if 'error' not in comparison_results['materials'][m]]
        
        if not valid_materials:
            return "No valid materials to compare."
            
        # Determine properties to show
        if properties_filter:
            props_to_show = properties_filter
        else:
            # Use common properties
            props_to_show = comparison_results['summary'].get('common_properties', [])
            if not props_to_show:
                # Fall back to all properties
                props_to_show = list(comparison_results['properties'].keys())
                
        if not props_to_show:
            return "No properties to compare."
            
        # Build header
        header = ['Property'] + valid_materials
        # Set reasonable column widths
        col_widths = [25]  # Property column
        for mat_id in valid_materials:
            # Use material ID length but cap at 20 chars
            col_widths.append(min(20, max(12, len(mat_id))))
        
        # Update column widths based on content
        for prop in props_to_show:
            col_widths[0] = max(col_widths[0], len(prop))
            for i, mat_id in enumerate(valid_materials):
                mat_data = comparison_results['materials'][mat_id]
                if prop in mat_data.get('properties', {}):
                    value_str = str(mat_data['properties'][prop]['value'])
                    unit = mat_data['properties'][prop].get('unit', '')
                    if unit:
                        value_str += f" {unit}"
                    col_widths[i + 1] = max(col_widths[i + 1], len(value_str))
                    
        # Format header with truncated material IDs if needed
        header_display = ['Property']
        for i, mat_id in enumerate(valid_materials):
            if len(mat_id) > col_widths[i + 1]:
                header_display.append(mat_id[:col_widths[i + 1] - 3] + '...')
            else:
                header_display.append(mat_id)
                
        header_line = " | ".join(h.ljust(w) for h, w in zip(header_display, col_widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Add formula and space group rows
        formula_row = ["Formula".ljust(col_widths[0])]
        for i, m in enumerate(valid_materials):
            formula = comparison_results['materials'][m].get('formula', 'Unknown')
            if len(formula) > col_widths[i + 1]:
                formula = formula[:col_widths[i + 1] - 3] + '...'
            formula_row.append(formula.ljust(col_widths[i + 1]))
        lines.append(" | ".join(formula_row))
        
        sg_row = ["Space Group".ljust(col_widths[0])]
        for i, m in enumerate(valid_materials):
            sg = str(comparison_results['materials'][m].get('space_group', 'Unknown'))
            if len(sg) > col_widths[i + 1]:
                sg = sg[:col_widths[i + 1] - 3] + '...'
            sg_row.append(sg.ljust(col_widths[i + 1]))
        lines.append(" | ".join(sg_row))
        lines.append("-" * len(header_line))
        
        # Add property rows
        for prop in props_to_show:
            row = [prop.ljust(col_widths[0])]
            
            for i, mat_id in enumerate(valid_materials):
                mat_data = comparison_results['materials'][mat_id]
                if prop in mat_data.get('properties', {}):
                    value = mat_data['properties'][prop]['value']
                    unit = mat_data['properties'][prop].get('unit', '')
                    
                    # Format value
                    if isinstance(value, float):
                        if abs(value) < 0.001 or abs(value) > 10000:
                            value_str = f"{value:.4e}"
                        else:
                            value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                        
                    if unit:
                        value_str += f" {unit}"
                        
                    row.append(value_str.ljust(col_widths[i + 1]))
                else:
                    row.append("N/A".ljust(col_widths[i + 1]))
                    
            lines.append(" | ".join(row))
            
        # Add summary section
        lines.append("")
        lines.append("=== Summary ===")
        summary = comparison_results['summary']
        lines.append(f"Common properties: {len(summary['common_properties'])}")
        
        if summary['largest_differences']:
            lines.append("\nLargest relative differences:")
            for diff in summary['largest_differences'][:5]:
                lines.append(f"  {diff['property']}: {diff['relative_range']:.2%} difference")
                lines.append(f"    Min: {diff['min_value']} ({diff['min_material']})")
                lines.append(f"    Max: {diff['max_value']} ({diff['max_material']})")
                
        return "\n".join(lines)


def compare_materials(db, material_ids: List[str], properties: List[str] = None,
                     output_format: str = 'table') -> str:
    """
    Convenience function to compare materials.
    
    Args:
        db: MaterialDatabase instance
        material_ids: List of material IDs to compare
        properties: List of properties to compare (None = all)
        output_format: Output format ('table', 'json', 'dict')
        
    Returns:
        Formatted comparison results
    """
    comparator = MaterialComparison(db)
    results = comparator.compare_materials(material_ids, properties)
    
    if output_format == 'table':
        return comparator.format_comparison_table(results, properties)
    elif output_format == 'json':
        return json.dumps(results, indent=2, default=str)
    else:  # dict
        return results