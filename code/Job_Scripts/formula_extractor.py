#!/usr/bin/env python3
"""
Chemical Formula and Space Group Extractor
==========================================
Extract chemical formula and space group information from CRYSTAL files
to populate the materials table.

This module provides functions to:
1. Extract chemical formula from D12 input files
2. Extract space group from output files  
3. Extract formula from CIF files
4. Update materials table with correct information

Usage:
  from formula_extractor import extract_formula_from_d12, extract_space_group_from_output
"""

import re
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import Counter
from datetime import datetime


def extract_formula_from_d12(d12_file: Path) -> Optional[str]:
    """
    Extract chemical formula from a CRYSTAL D12 input file.
    
    Args:
        d12_file: Path to .d12 file
        
    Returns:
        Chemical formula string (e.g., 'C2', 'SiO2', 'CaCO3')
    """
    if not d12_file.exists():
        return None
    
    try:
        with open(d12_file, 'r') as f:
            content = f.read()
    except:
        return None
    
    # Look for atomic numbers and coordinates in geometry section
    atoms = []
    
    # Pattern for CRYSTAL geometry input
    # Look for lines with atomic number followed by coordinates
    atom_pattern = r'^\s*(\d+)\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+'
    
    lines = content.split('\n')
    in_geometry = False
    
    for line in lines:
        line = line.strip()
        
        # Detect start of geometry section
        if line.upper() in ['CRYSTAL', 'SLAB', 'POLYMER', 'MOLECULE']:
            in_geometry = True
            continue
        elif line.upper() in ['EXTERNAL', 'OPTGEOM', 'END']:
            in_geometry = False
            continue
        
        if in_geometry and line and not line.startswith('#'):
            # Try to extract atomic number
            match = re.match(atom_pattern, line)
            if match:
                atomic_num = int(match.group(1))
                element = atomic_number_to_symbol(atomic_num)
                if element:
                    atoms.append(element)
    
    # If no atoms found in main section, try alternative patterns
    if not atoms:
        # Look for patterns like "6 0.0 0.0 0.0" (Carbon at origin)
        all_matches = re.findall(r'^\s*(\d+)\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+', content, re.MULTILINE)
        for atomic_num_str in all_matches:
            atomic_num = int(atomic_num_str)
            element = atomic_number_to_symbol(atomic_num)
            if element:
                atoms.append(element)
    
    if not atoms:
        return None
    
    # Count atoms and create formula
    atom_counts = Counter(atoms)
    
    # Sort by electronegativity/convention (metals first, then nonmetals)
    element_order = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 
                     'Al', 'Ga', 'In', 'Tl', 'Si', 'Ge', 'Sn', 'Pb',
                     'N', 'P', 'As', 'Sb', 'Bi', 'C', 'S', 'Se', 'Te',
                     'F', 'Cl', 'Br', 'I', 'O', 'H']
    
    # Create formula string
    formula_parts = []
    
    # Add elements in order
    for element in element_order:
        if element in atom_counts:
            count = atom_counts[element]
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")
            del atom_counts[element]
    
    # Add any remaining elements
    for element in sorted(atom_counts.keys()):
        count = atom_counts[element]
        if count == 1:
            formula_parts.append(element)
        else:
            formula_parts.append(f"{element}{count}")
    
    return ''.join(formula_parts)


def extract_formula_from_cif(cif_file: Path) -> Optional[str]:
    """Extract chemical formula from a CIF file."""
    if not cif_file.exists():
        return None
    
    try:
        with open(cif_file, 'r') as f:
            content = f.read()
    except:
        return None
    
    # Look for chemical formula in CIF
    formula_patterns = [
        r'_chemical_formula_sum\s+[\'"]([^\'"]+)[\'"]',
        r'_chemical_formula_sum\s+(\S+)',
        r'_chemical_formula_analytical\s+[\'"]([^\'"]+)[\'"]',
        r'_chemical_formula_analytical\s+(\S+)'
    ]
    
    for pattern in formula_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            formula = match.group(1).strip()
            # Clean up formula (remove spaces, quotes)
            formula = re.sub(r'[\'"\s]', '', formula)
            return formula
    
    # If no formula found, try to extract from atom sites
    atom_sites = []
    site_pattern = r'_atom_site_type_symbol\s+(\w+)'
    matches = re.findall(site_pattern, content, re.IGNORECASE)
    
    if matches:
        atom_counts = Counter(matches)
        formula_parts = []
        for element in sorted(atom_counts.keys()):
            count = atom_counts[element]
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")
        return ''.join(formula_parts)
    
    return None


def extract_space_group_from_output(output_file: Path) -> Optional[int]:
    """
    Extract space group number from CRYSTAL output file.
    
    Args:
        output_file: Path to .out file
        
    Returns:
        Space group number (1-230) or None
    """
    if not output_file.exists():
        return None
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
    except:
        return None
    
    # Look for space group information
    patterns = [
        r'SPACE GROUP NUMBER\s+(\d+)',
        r'SPACE GROUP:\s+(\d+)',
        r'S\.G\.\s+(\d+)',
        r'SYMMOPS - SPACE GROUP\s+(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            sg_num = int(match.group(1))
            if 1 <= sg_num <= 230:  # Valid space group range
                return sg_num
    
    return None


def extract_material_info_from_files(d12_file: Path = None, 
                                    cif_file: Path = None, 
                                    output_file: Path = None) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract formula and space group from available files.
    
    Args:
        d12_file: D12 input file
        cif_file: CIF file  
        output_file: CRYSTAL output file
        
    Returns:
        Tuple of (formula, space_group_number)
    """
    formula = None
    space_group = None
    
    # Try to get formula from multiple sources
    if d12_file and d12_file.exists():
        formula = extract_formula_from_d12(d12_file)
    
    if not formula and cif_file and cif_file.exists():
        formula = extract_formula_from_cif(cif_file)
    
    # Get space group from output file
    if output_file and output_file.exists():
        space_group = extract_space_group_from_output(output_file)
    
    return formula, space_group


def atomic_number_to_symbol(atomic_num: int) -> Optional[str]:
    """Convert atomic number to element symbol."""
    elements = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
        19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
        29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd',
        47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe',
        55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
        65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
        72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
        81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn'
    }
    return elements.get(atomic_num)


def update_materials_table_info(db, material_id: str, d12_file: Path = None, 
                               cif_file: Path = None, output_file: Path = None):
    """
    Update materials table with extracted formula and space group.
    
    Args:
        db: MaterialDatabase instance
        material_id: Material ID to update
        d12_file: D12 input file
        cif_file: Original CIF file
        output_file: CRYSTAL output file
    """
    formula, space_group = extract_material_info_from_files(d12_file, cif_file, output_file)
    
    if formula or space_group:
        try:
            with db._get_connection() as conn:
                update_fields = []
                update_values = []
                
                if formula:
                    update_fields.append('formula = ?')
                    update_values.append(formula)
                
                if space_group:
                    update_fields.append('space_group = ?')
                    update_values.append(space_group)
                
                if update_fields:
                    update_fields.append('updated_at = ?')
                    update_values.append(datetime.now().isoformat())
                    update_values.append(material_id)
                    
                    query = f"UPDATE materials SET {', '.join(update_fields)} WHERE material_id = ?"
                    conn.execute(query, update_values)
                    
                    print(f"  📝 Updated material {material_id}: formula={formula}, space_group={space_group}")
                    
        except Exception as e:
            print(f"  ⚠️  Error updating material {material_id}: {e}")


if __name__ == "__main__":
    # Test the extractor
    import sys
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        if test_file.suffix == '.d12':
            formula = extract_formula_from_d12(test_file)
            print(f"Formula from {test_file}: {formula}")
        elif test_file.suffix == '.out':
            space_group = extract_space_group_from_output(test_file)
            print(f"Space group from {test_file}: {space_group}")
        elif test_file.suffix == '.cif':
            formula = extract_formula_from_cif(test_file)
            print(f"Formula from {test_file}: {formula}")
    else:
        print("Usage: python formula_extractor.py <file.d12|file.out|file.cif>")