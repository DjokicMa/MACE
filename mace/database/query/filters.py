"""
Property Range Filtering for MACE Database
==========================================
Provides flexible filtering of materials by property ranges.

Usage:
    filter = PropertyFilter()
    filter.add_filter("band_gap", ">", 3.0)
    filter.add_filter("total_energy", "<", -1000)
    
    filtered_materials = filter.apply(all_materials)
"""

import re
import operator
from typing import List, Dict, Any, Tuple, Optional, Union


class PropertyFilter:
    """Handles filtering of materials by property value ranges."""
    
    # Supported operators
    OPERATORS = {
        '>': operator.gt,
        '>=': operator.ge,
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        '=': operator.eq,  # Alias for ==
    }
    
    def __init__(self):
        """Initialize an empty filter."""
        self.filters = []
        self.logic = 'AND'  # Default to AND logic
        
    def add_filter(self, property_name: str, op: str, value: Union[float, str], 
                   property_type: str = 'numeric'):
        """
        Add a filter condition.
        
        Args:
            property_name: Name of the property to filter
            op: Operator (>, >=, <, <=, ==, !=)
            value: Value to compare against
            property_type: Type of property ('numeric' or 'string')
        """
        if op not in self.OPERATORS:
            raise ValueError(f"Unsupported operator: {op}")
            
        self.filters.append({
            'property': property_name,
            'operator': op,
            'value': value,
            'type': property_type,
            'op_func': self.OPERATORS[op]
        })
        
    def set_logic(self, logic: str):
        """Set the logic for combining filters (AND/OR)."""
        if logic.upper() not in ['AND', 'OR']:
            raise ValueError("Logic must be 'AND' or 'OR'")
        self.logic = logic.upper()
        
    def apply_to_materials(self, materials: List[Dict], properties: List[Dict]) -> List[str]:
        """
        Apply filters to materials based on their properties.
        
        Args:
            materials: List of material records
            properties: List of property records
            
        Returns:
            List of material IDs that match the filters
        """
        if not self.filters:
            return [m['material_id'] for m in materials]
            
        # Build property lookup by material and property name
        prop_lookup = {}
        for prop in properties:
            mat_id = prop['material_id']
            prop_name = prop['property_name']
            if mat_id not in prop_lookup:
                prop_lookup[mat_id] = {}
            prop_lookup[mat_id][prop_name] = prop['property_value']
            
        # Apply filters
        matching_materials = []
        
        for material in materials:
            mat_id = material['material_id']
            if mat_id not in prop_lookup:
                continue
                
            mat_props = prop_lookup[mat_id]
            
            # Check each filter
            filter_results = []
            for filt in self.filters:
                prop_name = filt['property']
                
                if prop_name not in mat_props:
                    filter_results.append(False)
                    continue
                    
                prop_value = mat_props[prop_name]
                
                # Convert to appropriate type
                try:
                    if filt['type'] == 'numeric':
                        prop_value = float(prop_value)
                        compare_value = float(filt['value'])
                    else:
                        prop_value = str(prop_value)
                        compare_value = str(filt['value'])
                        
                    result = filt['op_func'](prop_value, compare_value)
                    filter_results.append(result)
                except (ValueError, TypeError):
                    filter_results.append(False)
                    
            # Apply logic
            if self.logic == 'AND':
                if all(filter_results):
                    matching_materials.append(mat_id)
            else:  # OR
                if any(filter_results):
                    matching_materials.append(mat_id)
                    
        return matching_materials
        
    def apply_to_properties(self, properties: List[Dict]) -> List[Dict]:
        """
        Apply filters directly to properties.
        
        Args:
            properties: List of property records
            
        Returns:
            Filtered list of properties
        """
        if not self.filters:
            return properties
            
        filtered = []
        
        for prop in properties:
            # Check each filter
            filter_results = []
            
            for filt in self.filters:
                if filt['property'] != prop['property_name']:
                    continue
                    
                try:
                    if filt['type'] == 'numeric':
                        prop_value = float(prop['property_value'])
                        compare_value = float(filt['value'])
                    else:
                        prop_value = str(prop['property_value'])
                        compare_value = str(filt['value'])
                        
                    result = filt['op_func'](prop_value, compare_value)
                    filter_results.append(result)
                except (ValueError, TypeError):
                    pass
                    
            # For property filtering, we only care if the property matches ANY filter for its name
            if filter_results and any(filter_results):
                filtered.append(prop)
                
        return filtered
        
    def __str__(self):
        """String representation of the filter."""
        if not self.filters:
            return "No filters"
            
        parts = []
        for filt in self.filters:
            parts.append(f"{filt['property']} {filt['operator']} {filt['value']}")
            
        return f" {self.logic} ".join(parts)


def parse_filter_string(filter_str: str) -> Tuple[str, str, Union[float, str]]:
    """
    Parse a filter string like "band_gap > 3.0" into components.
    
    Args:
        filter_str: Filter string to parse
        
    Returns:
        Tuple of (property_name, operator, value)
    """
    # Pattern to match: property_name operator value
    # Operators: >, >=, <, <=, ==, !=, =
    pattern = r'^\s*(\w+)\s*(>=|<=|!=|==|>|<|=)\s*(.+)\s*$'
    
    match = re.match(pattern, filter_str)
    if not match:
        raise ValueError(f"Invalid filter format: {filter_str}")
        
    property_name = match.group(1)
    operator = match.group(2)
    value_str = match.group(3).strip()
    
    # Try to convert value to float
    try:
        value = float(value_str)
    except ValueError:
        # Keep as string, remove quotes if present
        value = value_str.strip('"\'')
        
    return property_name, operator, value


def create_filter_from_strings(filter_strings: List[str], logic: str = 'AND') -> PropertyFilter:
    """
    Create a PropertyFilter from a list of filter strings.
    
    Args:
        filter_strings: List of filter strings like ["band_gap > 3.0", "total_energy < -1000"]
        logic: Logic to combine filters ('AND' or 'OR')
        
    Returns:
        Configured PropertyFilter object
    """
    filter_obj = PropertyFilter()
    filter_obj.set_logic(logic)
    
    for filter_str in filter_strings:
        prop_name, op, value = parse_filter_string(filter_str)
        
        # Determine type
        prop_type = 'numeric' if isinstance(value, (int, float)) else 'string'
        
        filter_obj.add_filter(prop_name, op, value, prop_type)
        
    return filter_obj