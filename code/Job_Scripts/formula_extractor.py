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
    
    # First try to find explicit space group numbers
    number_patterns = [
        r'SPACE GROUP NUMBER\s+(\d+)',
        r'SPACE GROUP:\s+(\d+)',
        r'S\.G\.\s+(\d+)',
        r'SYMMOPS - SPACE GROUP\s+(\d+)'
    ]
    
    for pattern in number_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            sg_num = int(match.group(1))
            if 1 <= sg_num <= 230:  # Valid space group range
                return sg_num
    
    # If no number found, try to map space group symbols to numbers
    symbol_pattern = r'SPACE GROUP.*:\s*([A-Z0-9\s/-]{1,20})'
    match = re.search(symbol_pattern, content, re.IGNORECASE)
    if match:
        sg_symbol = match.group(1).strip()
        # Take only the first line if there are line breaks
        sg_symbol = sg_symbol.split('\n')[0].strip()
        # Remove extra spaces and normalize
        sg_symbol = ' '.join(sg_symbol.split()).upper()
        return _space_group_symbol_to_number(sg_symbol)
    
    return None


def _space_group_symbol_to_number(symbol: str) -> Optional[int]:
    """Map space group symbol to number."""
    # Common space group mappings (Hermann-Mauguin symbols)
    symbol_to_number = {
        'P1': 1, 'P-1': 2, 'P2': 3, 'P21': 4, 'C2': 5, 'PM': 6, 'PC': 7, 'CM': 8, 'CC': 9,
        'P2/M': 10, 'P21/M': 11, 'C2/M': 12, 'P2/C': 13, 'P21/C': 14, 'C2/C': 15,
        'P222': 16, 'P2221': 17, 'P21212': 18, 'P212121': 19, 'C2221': 20, 'C222': 21, 'F222': 22,
        'I222': 23, 'I212121': 24, 'PMM2': 25, 'PMC21': 26, 'PCC2': 27, 'PMA2': 28, 'PCA21': 29,
        'PNC2': 30, 'PMN21': 31, 'PBA2': 32, 'PNA21': 33, 'PNN2': 34, 'CMM2': 35, 'CMC21': 36,
        'CCC2': 37, 'AMM2': 38, 'AEM2': 39, 'AMA2': 40, 'AEA2': 41, 'FMM2': 42, 'FDD2': 43,
        'IMM2': 44, 'IBA2': 45, 'IMA2': 46, 'PMMM': 47, 'PNNN': 48, 'PCCM': 49, 'PBAN': 50,
        'PMMA': 51, 'PNNA': 52, 'PMNA': 53, 'PCCA': 54, 'PBAM': 55, 'PCCN': 56, 'PBCM': 57,
        'PNNM': 58, 'PMMN': 59, 'PBCN': 60, 'PBCA': 61, 'PNMA': 62, 'CMCM': 63, 'CMCE': 64,
        'CMMM': 65, 'CCCM': 66, 'CMME': 67, 'CCCE': 68, 'FMMM': 69, 'FDDD': 70, 'IMMM': 71,
        'IBAM': 72, 'IBCA': 73, 'IMMA': 74, 'P4': 75, 'P41': 76, 'P42': 77, 'P43': 78,
        'I4': 79, 'I41': 80, 'P-4': 81, 'I-4': 82, 'P4/M': 83, 'P42/M': 84, 'P4/N': 85,
        'P42/N': 86, 'I4/M': 87, 'I41/A': 88, 'P422': 89, 'P4212': 90, 'P4122': 91,
        'P41212': 92, 'P4222': 93, 'P42212': 94, 'P4322': 95, 'P43212': 96, 'I422': 97,
        'I4122': 98, 'P4MM': 99, 'P4BM': 100, 'P42CM': 101, 'P42NM': 102, 'P4CC': 103,
        'P4NC': 104, 'P42MC': 105, 'P42BC': 106, 'I4MM': 107, 'I4CM': 108, 'I41MD': 109,
        'I41CD': 110, 'P-42M': 111, 'P-42C': 112, 'P-421M': 113, 'P-421C': 114, 'P-4M2': 115,
        'P-4C2': 116, 'P-4B2': 117, 'P-4N2': 118, 'I-4M2': 119, 'I-4C2': 120, 'I-42M': 121,
        'I-42D': 122, 'P4/MMM': 123, 'P4/MCC': 124, 'P4/NBM': 125, 'P4/NNC': 126, 'P4/MBM': 127,
        'P4/MNC': 128, 'P4/NMM': 129, 'P4/NCC': 130, 'P42/MMC': 131, 'P42/MCM': 132, 'P42/NBC': 133,
        'P42/NNM': 134, 'P42/MBC': 135, 'P42/MNM': 136, 'P42/NMC': 137, 'P42/NCM': 138, 'I4/MMM': 139,
        'I4/MCM': 140, 'I41/AMD': 141, 'I41/ACD': 142, 'P3': 143, 'P31': 144, 'P32': 145,
        'R3': 146, 'P-3': 147, 'R-3': 148, 'P312': 149, 'P321': 150, 'P3112': 151,
        'P3121': 152, 'P3212': 153, 'P3221': 154, 'R32': 155, 'P3M1': 156, 'P31M': 157,
        'P3C1': 158, 'P31C': 159, 'R3M': 160, 'R3C': 161, 'P-31M': 162, 'P-31C': 163,
        'P-3M1': 164, 'P-3C1': 165, 'R-3M': 166, 'R-3C': 167, 'P6': 168, 'P61': 169,
        'P65': 170, 'P62': 171, 'P64': 172, 'P63': 173, 'P-6': 174, 'P6/M': 175,
        'P63/M': 176, 'P622': 177, 'P6122': 178, 'P6522': 179, 'P6222': 180, 'P6422': 181,
        'P6322': 182, 'P6MM': 183, 'P6CC': 184, 'P63CM': 185, 'P63MC': 186, 'P-6M2': 187,
        'P-6C2': 188, 'P-62M': 189, 'P-62C': 190, 'P6/MMM': 191, 'P6/MCC': 192, 'P63/MCM': 193,
        'P63/MMC': 194, 'P23': 195, 'F23': 196, 'I23': 197, 'P213': 198, 'I213': 199,
        'PM-3': 200, 'PN-3': 201, 'FM-3': 202, 'FD-3': 203, 'IM-3': 204, 'PA-3': 205,
        'IA-3': 206, 'P432': 207, 'P4232': 208, 'F432': 209, 'F4132': 210, 'I432': 211,
        'P4332': 212, 'P4132': 213, 'I4132': 214, 'P-43M': 215, 'F-43M': 216, 'I-43M': 217,
        'P-43N': 218, 'F-43C': 219, 'I-43D': 220, 'PM-3M': 221, 'PN-3N': 222, 'PM-3N': 223,
        'PN-3M': 224, 'FM-3M': 225, 'FM-3C': 226, 'FD-3M': 227, 'FD-3C': 228, 'IM-3M': 229, 'IA-3D': 230
    }
    
    # Try different variations of the symbol
    variations = [
        symbol,  # Original with spaces
        symbol.replace(' ', ''),  # No spaces: "FD3M"
        symbol.replace(' ', '-'),  # All spaces to dashes: "F-D-3-M"
        symbol.replace(' ', '/'),  # Spaces to slashes: "F/D/3/M"
        re.sub(r'(\d)', r'-\1', symbol.replace(' ', '')),  # Insert dash before numbers: "FD-3M"
        re.sub(r'(\d)', r'/\1', symbol.replace(' ', '')),  # Insert slash before numbers: "FD/3M"
    ]
    
    for var in variations:
        if var in symbol_to_number:
            return symbol_to_number[var]
    
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