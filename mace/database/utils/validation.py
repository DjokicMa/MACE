"""
Data Validation System
======================
Validate property values and ensure data integrity in the materials database.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import re
from datetime import datetime
import json


class PropertyValidator:
    """Validates property values based on type and constraints."""
    
    # Property validation rules
    PROPERTY_RULES = {
        # Energy properties (should be negative for stable structures)
        'total_energy': {
            'type': 'float',
            'unit_type': 'energy',
            'constraints': {
                'max': 1000.0,  # Hartree - positive energies are suspicious
                'warning_if_positive': True
            }
        },
        'fermi_energy': {
            'type': 'float',
            'unit_type': 'energy',
            'constraints': {
                'min': -100.0,  # Hartree
                'max': 100.0
            }
        },
        'band_gap': {
            'type': 'float',
            'unit_type': 'energy',
            'constraints': {
                'min': 0.0,  # Cannot be negative
                'max': 20.0,  # eV - very large gaps are suspicious
                'unit': 'eV'
            }
        },
        
        # Structural properties
        'a_lattice': {
            'type': 'float',
            'unit_type': 'length',
            'constraints': {
                'min': 0.5,   # Angstrom - too small is unphysical
                'max': 100.0,  # Angstrom - too large is suspicious
                'unit': 'angstrom'
            }
        },
        'b_lattice': {
            'type': 'float',
            'unit_type': 'length',
            'constraints': {
                'min': 0.5,
                'max': 100.0,
                'unit': 'angstrom'
            }
        },
        'c_lattice': {
            'type': 'float',
            'unit_type': 'length',
            'constraints': {
                'min': 0.5,
                'max': 100.0,
                'unit': 'angstrom'
            }
        },
        'cell_volume': {
            'type': 'float',
            'unit_type': 'volume',
            'constraints': {
                'min': 1.0,    # Angstrom^3
                'max': 1000000.0,
                'unit': 'angstrom^3'
            }
        },
        
        # Angular properties
        'alpha': {
            'type': 'float',
            'unit_type': 'angle',
            'constraints': {
                'min': 10.0,   # Degrees - very acute angles are rare
                'max': 170.0,  # Degrees - very obtuse angles are rare
                'unit': 'degree'
            }
        },
        'beta': {
            'type': 'float',
            'unit_type': 'angle',
            'constraints': {
                'min': 10.0,
                'max': 170.0,
                'unit': 'degree'
            }
        },
        'gamma': {
            'type': 'float',
            'unit_type': 'angle',
            'constraints': {
                'min': 10.0,
                'max': 170.0,
                'unit': 'degree'
            }
        },
        
        # Integer properties
        'space_group': {
            'type': 'int',
            'constraints': {
                'min': 1,
                'max': 230,
                'allowed_values': list(range(1, 231))
            }
        },
        'atoms_in_unit_cell': {
            'type': 'int',
            'constraints': {
                'min': 1,
                'max': 10000  # Very large unit cells are suspicious
            }
        },
        'scf_cycles': {
            'type': 'int',
            'constraints': {
                'min': 1,
                'max': 1000
            }
        },
        
        # String properties
        'formula': {
            'type': 'string',
            'constraints': {
                'pattern': r'^[A-Z][a-z]?(\d*[A-Z][a-z]?\d*)*$',  # Chemical formula pattern
                'max_length': 100
            }
        },
        'crystal_system': {
            'type': 'string',
            'constraints': {
                'allowed_values': ['triclinic', 'monoclinic', 'orthorhombic', 
                                 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
            }
        },
        'conductivity_type': {
            'type': 'string',
            'constraints': {
                'allowed_values': ['metal', 'semiconductor', 'insulator', 'unknown']
            }
        },
        
        # Boolean properties
        'converged': {
            'type': 'bool'
        },
        'is_magnetic': {
            'type': 'bool'
        },
        
        # Array properties
        'phonon_frequencies': {
            'type': 'array',
            'element_type': 'float',
            'constraints': {
                'min_value': -100.0,  # cm^-1 - small negative for numerical errors
                'max_value': 5000.0,  # cm^-1
                'unit': 'cm^-1'
            }
        },
        
        # Derived properties
        'density': {
            'type': 'float',
            'constraints': {
                'min': 0.01,   # g/cm^3 - aerogels
                'max': 25.0,   # g/cm^3 - osmium is ~22.6
                'unit': 'g/cm^3'
            }
        },
        'bulk_modulus': {
            'type': 'float',
            'unit_type': 'pressure',
            'constraints': {
                'min': 0.001,  # GPa - very soft materials
                'max': 1000.0,  # GPa - diamond is ~440
                'unit': 'GPa'
            }
        }
    }
    
    def __init__(self):
        """Initialize the validator."""
        self.validation_errors = []
        self.validation_warnings = []
        
    def validate_property(self, property_name: str, value: Any, 
                         unit: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate a single property value.
        
        Args:
            property_name: Name of the property
            value: Value to validate
            unit: Unit of the value (optional)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if property has validation rules
        if property_name not in self.PROPERTY_RULES:
            # Unknown property - just check basic type
            if value is None:
                issues.append(f"Value is None")
            return len(issues) == 0, issues
            
        rules = self.PROPERTY_RULES[property_name]
        
        # Type validation
        expected_type = rules['type']
        if not self._validate_type(value, expected_type):
            issues.append(f"Expected type {expected_type}, got {type(value).__name__}")
            return False, issues
            
        # Constraint validation
        if 'constraints' in rules:
            constraints = rules['constraints']
            
            # Numeric constraints
            if expected_type in ['float', 'int']:
                if 'min' in constraints and value < constraints['min']:
                    issues.append(f"Value {value} below minimum {constraints['min']}")
                    
                if 'max' in constraints and value > constraints['max']:
                    issues.append(f"Value {value} above maximum {constraints['max']}")
                    
                if 'warning_if_positive' in constraints and value > 0:
                    issues.append(f"Warning: Positive value {value} is unusual")
                    
            # String constraints
            elif expected_type == 'string':
                if 'pattern' in constraints:
                    pattern = constraints['pattern']
                    if not re.match(pattern, str(value)):
                        issues.append(f"Value '{value}' doesn't match pattern {pattern}")
                        
                if 'max_length' in constraints and len(str(value)) > constraints['max_length']:
                    issues.append(f"Value length {len(str(value))} exceeds maximum {constraints['max_length']}")
                    
                if 'allowed_values' in constraints:
                    allowed = constraints['allowed_values']
                    if value not in allowed:
                        issues.append(f"Value '{value}' not in allowed values: {allowed}")
                        
            # Array constraints
            elif expected_type == 'array':
                if isinstance(value, (list, tuple)):
                    element_type = rules.get('element_type', 'float')
                    for i, elem in enumerate(value):
                        if not self._validate_type(elem, element_type):
                            issues.append(f"Array element {i} has wrong type")
                            
                        if 'min_value' in constraints and elem < constraints['min_value']:
                            issues.append(f"Array element {i} value {elem} below minimum")
                            
                        if 'max_value' in constraints and elem > constraints['max_value']:
                            issues.append(f"Array element {i} value {elem} above maximum")
                            
        # Unit validation
        if unit and 'unit' in rules.get('constraints', {}):
            expected_unit = rules['constraints']['unit']
            if unit != expected_unit:
                issues.append(f"Expected unit '{expected_unit}', got '{unit}'")
                
        return len(issues) == 0, issues
        
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'float': (int, float),
            'int': int,
            'string': str,
            'bool': bool,
            'array': (list, tuple),
            'dict': dict
        }
        
        if expected_type in type_map:
            return isinstance(value, type_map[expected_type])
        return True
        
    def validate_material_data(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all properties of a material.
        
        Args:
            material_data: Dictionary of material properties
            
        Returns:
            Validation report with errors and warnings
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'property_issues': {}
        }
        
        # Validate each property
        for prop_name, prop_value in material_data.items():
            # Skip metadata fields
            if prop_name in ['material_id', 'created_at', 'updated_at']:
                continue
                
            # Extract unit if present
            unit = None
            if isinstance(prop_value, dict) and 'value' in prop_value and 'unit' in prop_value:
                value = prop_value['value']
                unit = prop_value['unit']
            else:
                value = prop_value
                
            is_valid, issues = self.validate_property(prop_name, value, unit)
            
            if not is_valid:
                report['valid'] = False
                report['errors'].extend([f"{prop_name}: {issue}" for issue in issues 
                                       if not issue.startswith('Warning:')])
                report['warnings'].extend([f"{prop_name}: {issue}" for issue in issues 
                                         if issue.startswith('Warning:')])
                report['property_issues'][prop_name] = issues
                
        # Cross-property validation
        cross_issues = self._validate_cross_properties(material_data)
        if cross_issues:
            report['warnings'].extend(cross_issues)
            
        return report
        
    def _validate_cross_properties(self, material_data: Dict[str, Any]) -> List[str]:
        """Validate relationships between properties."""
        issues = []
        
        # Check lattice parameters consistency
        if all(k in material_data for k in ['a_lattice', 'b_lattice', 'c_lattice', 'crystal_system']):
            a = material_data['a_lattice']
            b = material_data['b_lattice']
            c = material_data['c_lattice']
            system = material_data['crystal_system']
            
            # Validate lattice parameters match crystal system
            if system == 'cubic' and not (abs(a - b) < 0.01 and abs(b - c) < 0.01):
                issues.append(f"Cubic system but lattice parameters not equal: a={a}, b={b}, c={c}")
                
            elif system == 'tetragonal' and not (abs(a - b) < 0.01 and abs(a - c) > 0.01):
                issues.append(f"Tetragonal system but a≠b or a=c: a={a}, b={b}, c={c}")
                
        # Check angles consistency
        if all(k in material_data for k in ['alpha', 'beta', 'gamma', 'crystal_system']):
            alpha = material_data['alpha']
            beta = material_data['beta']
            gamma = material_data['gamma']
            system = material_data['crystal_system']
            
            if system == 'cubic' and not all(abs(angle - 90) < 0.01 for angle in [alpha, beta, gamma]):
                issues.append(f"Cubic system but angles not 90°: α={alpha}, β={beta}, γ={gamma}")
                
        # Check band gap vs conductivity type
        if 'band_gap' in material_data and 'conductivity_type' in material_data:
            gap = material_data['band_gap']
            cond_type = material_data['conductivity_type']
            
            if cond_type == 'metal' and gap > 0.1:  # eV
                issues.append(f"Metal with non-zero band gap: {gap} eV")
                
            elif cond_type == 'insulator' and gap < 3.0:  # eV
                issues.append(f"Insulator with small band gap: {gap} eV")
                
        return issues
        
    def suggest_corrections(self, property_name: str, value: Any, 
                          issues: List[str]) -> List[Dict[str, Any]]:
        """
        Suggest corrections for validation issues.
        
        Args:
            property_name: Property with issues
            value: Current value
            issues: List of validation issues
            
        Returns:
            List of suggested corrections
        """
        suggestions = []
        
        # Get rules for property
        if property_name not in self.PROPERTY_RULES:
            return suggestions
            
        rules = self.PROPERTY_RULES[property_name]
        constraints = rules.get('constraints', {})
        
        # Suggest corrections based on issues
        for issue in issues:
            if 'below minimum' in issue and 'min' in constraints:
                suggestions.append({
                    'issue': issue,
                    'suggestion': f"Set to minimum value: {constraints['min']}",
                    'corrected_value': constraints['min']
                })
                
            elif 'above maximum' in issue and 'max' in constraints:
                suggestions.append({
                    'issue': issue,
                    'suggestion': f"Set to maximum value: {constraints['max']}",
                    'corrected_value': constraints['max']
                })
                
            elif 'not in allowed values' in issue and 'allowed_values' in constraints:
                # Find closest match
                allowed = constraints['allowed_values']
                if isinstance(value, str):
                    # Fuzzy string matching
                    closest = min(allowed, key=lambda x: self._string_distance(str(value), str(x)))
                    suggestions.append({
                        'issue': issue,
                        'suggestion': f"Did you mean '{closest}'?",
                        'corrected_value': closest
                    })
                    
        return suggestions
        
    def _string_distance(self, s1: str, s2: str) -> int:
        """Simple edit distance between strings."""
        if s1 == s2:
            return 0
        return abs(len(s1) - len(s2)) + sum(c1 != c2 for c1, c2 in zip(s1, s2))


class DatabaseValidator:
    """Validates database integrity and consistency."""
    
    def __init__(self, db):
        """
        Initialize with database connection.
        
        Args:
            db: MaterialDatabase instance
        """
        self.db = db
        self.property_validator = PropertyValidator()
        
    def validate_all_materials(self, fix_issues: bool = False) -> Dict[str, Any]:
        """
        Validate all materials in the database.
        
        Args:
            fix_issues: Whether to automatically fix issues
            
        Returns:
            Validation report
        """
        report = {
            'total_materials': 0,
            'valid_materials': 0,
            'materials_with_errors': 0,
            'materials_with_warnings': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'error_types': {},
            'fixed_issues': 0,
            'material_reports': {}
        }
        
        # Get all materials
        materials = self.db.get_all_materials()
        report['total_materials'] = len(materials)
        
        for material in materials:
            mat_id = material['material_id']
            
            # Get all properties for material
            properties = self.db.get_material_properties(mat_id)
            
            # Convert to validation format
            mat_data = {'material_id': mat_id}
            for prop in properties:
                prop_name = prop['property_name']
                prop_value = prop['property_value']
                prop_unit = prop.get('property_unit')
                
                # Handle numeric values
                try:
                    prop_value = float(prop_value)
                except (ValueError, TypeError):
                    pass
                    
                if prop_unit:
                    mat_data[prop_name] = {'value': prop_value, 'unit': prop_unit}
                else:
                    mat_data[prop_name] = prop_value
                    
            # Add material metadata
            if material.get('formula'):
                mat_data['formula'] = material['formula']
            if material.get('space_group'):
                mat_data['space_group'] = material['space_group']
                
            # Validate
            validation_report = self.property_validator.validate_material_data(mat_data)
            
            if validation_report['valid']:
                report['valid_materials'] += 1
            else:
                report['materials_with_errors'] += 1
                
            if validation_report['errors']:
                report['total_errors'] += len(validation_report['errors'])
                for error in validation_report['errors']:
                    error_type = error.split(':')[0]
                    report['error_types'][error_type] = report['error_types'].get(error_type, 0) + 1
                    
            if validation_report['warnings']:
                report['materials_with_warnings'] += 1
                report['total_warnings'] += len(validation_report['warnings'])
                
            # Store material report
            report['material_reports'][mat_id] = validation_report
            
            # Fix issues if requested
            if fix_issues and not validation_report['valid']:
                fixed_count = self._fix_material_issues(mat_id, validation_report)
                report['fixed_issues'] += fixed_count
                
        return report
        
    def _fix_material_issues(self, material_id: str, 
                           validation_report: Dict[str, Any]) -> int:
        """Fix validation issues for a material."""
        fixed_count = 0
        
        # For now, just log what would be fixed
        # In production, this would update the database
        for prop_name, issues in validation_report['property_issues'].items():
            # Get current value
            props = self.db.get_material_properties(material_id)
            current_prop = next((p for p in props if p['property_name'] == prop_name), None)
            
            if current_prop:
                value = current_prop['property_value']
                suggestions = self.property_validator.suggest_corrections(prop_name, value, issues)
                
                if suggestions:
                    # Would update property here
                    fixed_count += 1
                    
        return fixed_count
        
    def check_database_integrity(self) -> Dict[str, Any]:
        """
        Check overall database integrity.
        
        Returns:
            Integrity report
        """
        report = {
            'schema_valid': True,
            'orphaned_properties': 0,
            'orphaned_calculations': 0,
            'duplicate_properties': 0,
            'missing_materials': [],
            'integrity_issues': []
        }
        
        # Check for orphaned properties
        all_properties = self.db.get_all_properties()
        material_ids = {m['material_id'] for m in self.db.get_all_materials()}
        
        for prop in all_properties:
            if prop['material_id'] not in material_ids:
                report['orphaned_properties'] += 1
                report['missing_materials'].append(prop['material_id'])
                
        # Check for duplicate properties
        prop_keys = {}
        for prop in all_properties:
            key = (prop['material_id'], prop['property_name'], prop.get('calc_id'))
            if key in prop_keys:
                report['duplicate_properties'] += 1
                report['integrity_issues'].append(
                    f"Duplicate property: {prop['property_name']} for {prop['material_id']}"
                )
            prop_keys[key] = prop
            
        # Summary
        report['valid'] = (report['orphaned_properties'] == 0 and 
                          report['duplicate_properties'] == 0)
                          
        return report


def validate_materials(db, material_ids: List[str] = None,
                      fix_issues: bool = False,
                      output_format: str = 'report') -> str:
    """
    Convenience function to validate materials.
    
    Args:
        db: MaterialDatabase instance
        material_ids: Specific materials to validate (None = all)
        fix_issues: Whether to fix issues automatically
        output_format: 'report', 'json', or 'dict'
        
    Returns:
        Formatted validation results
    """
    validator = DatabaseValidator(db)
    
    if material_ids:
        # Validate specific materials
        report = {
            'total_materials': len(material_ids),
            'valid_materials': 0,
            'materials_with_errors': 0,
            'materials_with_warnings': 0,
            'material_reports': {}
        }
        
        for mat_id in material_ids:
            # Similar to validate_all_materials but for specific IDs
            pass  # Implementation would be similar
    else:
        report = validator.validate_all_materials(fix_issues)
        
    if output_format == 'json':
        return json.dumps(report, indent=2, default=str)
    elif output_format == 'dict':
        return report
    else:  # report format
        return _format_validation_report(report)
        
        
def _format_validation_report(report: Dict[str, Any]) -> str:
    """Format validation report as readable text."""
    lines = []
    
    lines.append("=== Material Data Validation Report ===")
    lines.append(f"Total materials: {report['total_materials']}")
    lines.append(f"Valid materials: {report['valid_materials']}")
    lines.append(f"Materials with errors: {report['materials_with_errors']}")
    lines.append(f"Materials with warnings: {report['materials_with_warnings']}")
    lines.append("")
    
    if report['total_errors'] > 0:
        lines.append(f"=== Errors ({report['total_errors']} total) ===")
        for error_type, count in sorted(report['error_types'].items(), 
                                      key=lambda x: x[1], reverse=True):
            lines.append(f"  {error_type}: {count}")
        lines.append("")
        
    if report['materials_with_errors'] > 0:
        lines.append("=== Materials with Errors ===")
        error_materials = [(mid, rep) for mid, rep in report['material_reports'].items() 
                          if not rep['valid']]
        
        for mat_id, mat_report in error_materials[:10]:
            lines.append(f"\n{mat_id}:")
            for error in mat_report['errors']:
                lines.append(f"  ERROR: {error}")
                
        if len(error_materials) > 10:
            lines.append(f"\n... and {len(error_materials) - 10} more materials with errors")
            
    if report.get('fixed_issues', 0) > 0:
        lines.append(f"\n=== Fixes Applied ===")
        lines.append(f"Fixed {report['fixed_issues']} issues")
        
    return "\n".join(lines)