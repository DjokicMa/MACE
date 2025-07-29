"""
Unit Conversion System
======================
Convert property values between different units commonly used in materials science.
"""

from typing import Dict, List, Tuple, Optional, Union
import re


class UnitConverter:
    """Handles unit conversions for material properties."""
    
    # Define conversion factors to base units
    # Energy conversions (base: Hartree)
    ENERGY_CONVERSIONS = {
        'hartree': 1.0,
        'ha': 1.0,
        'au': 1.0,
        'ev': 27.211386245988,
        'rydberg': 2.0,
        'ry': 2.0,
        'kcal/mol': 627.509474,
        'kj/mol': 2625.499638,
        'cm^-1': 219474.631363,
        'cm-1': 219474.631363,
        'mev': 27211.386245988,
        'thz': 6579.683920711,
    }
    
    # Length conversions (base: Bohr)
    LENGTH_CONVERSIONS = {
        'bohr': 1.0,
        'au': 1.0,
        'angstrom': 0.52917721067,
        'a': 0.52917721067,
        'nm': 0.052917721067,
        'pm': 52.917721067,
        'm': 5.2917721067e-11,
        'cm': 5.2917721067e-9,
        'mm': 5.2917721067e-8,
    }
    
    # Pressure conversions (base: Hartree/Bohr^3)
    PRESSURE_CONVERSIONS = {
        'hartree/bohr^3': 1.0,
        'ha/bohr^3': 1.0,
        'gpa': 29421.0107637093,
        'mbar': 294210.107637093,
        'kbar': 294.210107637093,
        'pa': 2.9421e13,
        'mpa': 2.9421e7,
        'atm': 290246.6961,
        'psi': 4.269e9,
    }
    
    # Temperature conversions (base: Kelvin)
    TEMPERATURE_CONVERSIONS = {
        'k': 1.0,
        'kelvin': 1.0,
        'c': lambda t: t + 273.15,  # Celsius to Kelvin
        'celsius': lambda t: t + 273.15,
        'f': lambda t: (t - 32) * 5/9 + 273.15,  # Fahrenheit to Kelvin
        'fahrenheit': lambda t: (t - 32) * 5/9 + 273.15,
    }
    
    # Inverse temperature conversions (from Kelvin)
    TEMPERATURE_CONVERSIONS_INVERSE = {
        'k': 1.0,
        'kelvin': 1.0,
        'c': lambda t: t - 273.15,  # Kelvin to Celsius
        'celsius': lambda t: t - 273.15,
        'f': lambda t: (t - 273.15) * 9/5 + 32,  # Kelvin to Fahrenheit
        'fahrenheit': lambda t: (t - 273.15) * 9/5 + 32,
    }
    
    # Frequency conversions (base: THz)
    FREQUENCY_CONVERSIONS = {
        'thz': 1.0,
        'cm^-1': 33.35641,
        'cm-1': 33.35641,
        'mev': 4.13567,
        'ghz': 1000.0,
        'mhz': 1e6,
        'hz': 1e12,
        'ps^-1': 1.0,
        'ps-1': 1.0,
    }
    
    # Angle conversions (base: radians)
    ANGLE_CONVERSIONS = {
        'rad': 1.0,
        'radian': 1.0,
        'deg': 57.29577951308232,
        'degree': 57.29577951308232,
        'grad': 63.66197723675814,
        'gradian': 63.66197723675814,
    }
    
    # Property unit mapping
    PROPERTY_UNITS = {
        # Energy properties
        'total_energy': ENERGY_CONVERSIONS,
        'fermi_energy': ENERGY_CONVERSIONS,
        'band_gap': ENERGY_CONVERSIONS,
        'cohesive_energy': ENERGY_CONVERSIONS,
        'formation_energy': ENERGY_CONVERSIONS,
        'binding_energy': ENERGY_CONVERSIONS,
        'homo_energy': ENERGY_CONVERSIONS,
        'lumo_energy': ENERGY_CONVERSIONS,
        
        # Structural properties
        'a_lattice': LENGTH_CONVERSIONS,
        'b_lattice': LENGTH_CONVERSIONS,
        'c_lattice': LENGTH_CONVERSIONS,
        'bond_length': LENGTH_CONVERSIONS,
        'cell_volume': {k: v**3 for k, v in LENGTH_CONVERSIONS.items()},
        
        # Angular properties
        'alpha': ANGLE_CONVERSIONS,
        'beta': ANGLE_CONVERSIONS,
        'gamma': ANGLE_CONVERSIONS,
        
        # Pressure properties
        'pressure': PRESSURE_CONVERSIONS,
        'bulk_modulus': PRESSURE_CONVERSIONS,
        'young_modulus': PRESSURE_CONVERSIONS,
        'shear_modulus': PRESSURE_CONVERSIONS,
        
        # Frequency properties
        'phonon_frequencies': FREQUENCY_CONVERSIONS,
        'vibrational_frequencies': FREQUENCY_CONVERSIONS,
        
        # Temperature properties
        'temperature': TEMPERATURE_CONVERSIONS,
        'debye_temperature': TEMPERATURE_CONVERSIONS,
    }
    
    # Common unit aliases
    UNIT_ALIASES = {
        'angstroms': 'angstrom',
        'å': 'angstrom',
        'degrees': 'degree',
        '°': 'degree',
        'electronvolt': 'ev',
        'electron-volt': 'ev',
        'wavenumber': 'cm^-1',
        'wavenumbers': 'cm^-1',
    }
    
    def __init__(self):
        """Initialize the unit converter."""
        pass
        
    def convert(self, value: Union[float, List[float]], 
                from_unit: str, to_unit: str, 
                property_name: str = None) -> Union[float, List[float]]:
        """
        Convert a value from one unit to another.
        
        Args:
            value: Value(s) to convert
            from_unit: Source unit
            to_unit: Target unit
            property_name: Property name for context-aware conversion
            
        Returns:
            Converted value(s)
            
        Raises:
            ValueError: If units are incompatible or unknown
        """
        # Normalize units
        from_unit = self._normalize_unit(from_unit)
        to_unit = self._normalize_unit(to_unit)
        
        # Handle list input
        if isinstance(value, list):
            return [self._convert_single(v, from_unit, to_unit, property_name) 
                    for v in value]
        else:
            return self._convert_single(value, from_unit, to_unit, property_name)
            
    def _convert_single(self, value: float, from_unit: str, 
                       to_unit: str, property_name: str = None) -> float:
        """Convert a single value."""
        # If units are the same, return as-is
        if from_unit == to_unit:
            return value
            
        # Find appropriate conversion table
        conversion_table = self._get_conversion_table(from_unit, to_unit, property_name)
        
        if not conversion_table:
            raise ValueError(f"Cannot convert between {from_unit} and {to_unit}")
            
        # Convert to base unit first
        if from_unit in conversion_table:
            if callable(conversion_table[from_unit]):
                base_value = conversion_table[from_unit](value)
            else:
                base_value = value / conversion_table[from_unit]
        else:
            raise ValueError(f"Unknown unit: {from_unit}")
            
        # Convert from base unit to target
        if to_unit in conversion_table:
            # Special handling for temperature inverse conversions
            if conversion_table == self.TEMPERATURE_CONVERSIONS and to_unit in ['c', 'celsius', 'f', 'fahrenheit']:
                if callable(self.TEMPERATURE_CONVERSIONS_INVERSE[to_unit]):
                    return self.TEMPERATURE_CONVERSIONS_INVERSE[to_unit](base_value)
                else:
                    return base_value * self.TEMPERATURE_CONVERSIONS_INVERSE[to_unit]
            elif callable(conversion_table[to_unit]):
                # This shouldn't happen for our current conversions
                return base_value
            else:
                return base_value * conversion_table[to_unit]
        else:
            raise ValueError(f"Unknown unit: {to_unit}")
            
    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit string."""
        if not unit:
            return unit
            
        unit = unit.lower().strip()
        
        # Apply aliases
        if unit in self.UNIT_ALIASES:
            unit = self.UNIT_ALIASES[unit]
            
        return unit
        
    def _get_conversion_table(self, from_unit: str, to_unit: str, 
                            property_name: str = None) -> Optional[Dict]:
        """Get appropriate conversion table for units."""
        # First check if property-specific conversion exists
        if property_name and property_name in self.PROPERTY_UNITS:
            table = self.PROPERTY_UNITS[property_name]
            if from_unit in table and to_unit in table:
                return table
                
        # Check all conversion tables
        all_tables = [
            self.ENERGY_CONVERSIONS,
            self.LENGTH_CONVERSIONS,
            self.PRESSURE_CONVERSIONS,
            self.TEMPERATURE_CONVERSIONS,
            self.FREQUENCY_CONVERSIONS,
            self.ANGLE_CONVERSIONS,
        ]
        
        for table in all_tables:
            if from_unit in table and to_unit in table:
                return table
                
        # Check derived units (like volume)
        if property_name == 'cell_volume' and property_name in self.PROPERTY_UNITS:
            return self.PROPERTY_UNITS[property_name]
            
        return None
        
    def get_property_units(self, property_name: str) -> List[str]:
        """
        Get available units for a property.
        
        Args:
            property_name: Name of the property
            
        Returns:
            List of available units
        """
        if property_name in self.PROPERTY_UNITS:
            return list(self.PROPERTY_UNITS[property_name].keys())
        return []
        
    def get_default_unit(self, property_name: str) -> Optional[str]:
        """
        Get default unit for a property.
        
        Args:
            property_name: Name of the property
            
        Returns:
            Default unit or None
        """
        defaults = {
            'total_energy': 'hartree',
            'fermi_energy': 'ev',
            'band_gap': 'ev',
            'a_lattice': 'angstrom',
            'b_lattice': 'angstrom',
            'c_lattice': 'angstrom',
            'alpha': 'degree',
            'beta': 'degree',
            'gamma': 'degree',
            'pressure': 'gpa',
            'bulk_modulus': 'gpa',
            'phonon_frequencies': 'cm^-1',
            'temperature': 'k',
        }
        
        return defaults.get(property_name)
        
    def parse_value_with_unit(self, value_str: str) -> Tuple[float, Optional[str]]:
        """
        Parse a value string that may contain units.
        
        Args:
            value_str: String like "5.5 eV" or "300 K"
            
        Returns:
            Tuple of (value, unit) or (value, None)
        """
        value_str = str(value_str).strip()
        
        # Try to match number followed by optional unit
        match = re.match(r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(.*)$', value_str)
        
        if match:
            value = float(match.group(1))
            unit = match.group(2).strip() if match.group(2) else None
            
            if unit:
                unit = self._normalize_unit(unit)
                
            return value, unit
        else:
            raise ValueError(f"Cannot parse value and unit from: {value_str}")
            
    def format_value_with_unit(self, value: float, unit: str, 
                             precision: int = 4) -> str:
        """
        Format a value with its unit.
        
        Args:
            value: Numeric value
            unit: Unit string
            precision: Number of decimal places
            
        Returns:
            Formatted string
        """
        if abs(value) < 1e-10:
            return f"0.0 {unit}"
        elif abs(value) < 0.01 or abs(value) > 10000:
            return f"{value:.{precision}e} {unit}"
        else:
            return f"{value:.{precision}f} {unit}"


# Convenience functions
_converter = UnitConverter()

def convert_units(value: Union[float, List[float]], 
                 from_unit: str, to_unit: str,
                 property_name: str = None) -> Union[float, List[float]]:
    """Convert value(s) between units."""
    return _converter.convert(value, from_unit, to_unit, property_name)

def get_property_units(property_name: str) -> List[str]:
    """Get available units for a property."""
    return _converter.get_property_units(property_name)

def get_default_unit(property_name: str) -> Optional[str]:
    """Get default unit for a property."""
    return _converter.get_default_unit(property_name)

def parse_value_with_unit(value_str: str) -> Tuple[float, Optional[str]]:
    """Parse a value string that may contain units."""
    return _converter.parse_value_with_unit(value_str)

def format_value_with_unit(value: float, unit: str, precision: int = 4) -> str:
    """Format a value with its unit."""
    return _converter.format_value_with_unit(value, unit, precision)