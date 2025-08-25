#!/usr/bin/env python3
"""
Input Settings Extractor for Materials Database
===============================================
Extract and store D12/D3 input settings directly in the materials.db database.

This module extracts calculation settings from CRYSTAL input files and stores
them directly in the calculations.settings_json field of the materials database.

Usage:
  from input_settings_extractor import extract_and_store_input_settings
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import MACE components
try:
    from mace.database.materials import MaterialDatabase
except ImportError as e:
    print(f"Error importing MaterialDatabase: {e}")
    sys.exit(1)


def extract_input_settings(input_file: Path) -> Dict[str, Any]:
    """
    Extract comprehensive settings from a CRYSTAL D12/D3 input file.
    
    Args:
        input_file: Path to .d12 or .d3 file
        
    Returns:
        Dictionary with extracted settings
    """
    settings = {
        'file_type': input_file.suffix,
        'file_name': input_file.name,
        'extraction_timestamp': datetime.now().isoformat(),
        'crystal_keywords': [],
        'basis_set_info': {},
        'calculation_parameters': {},
        'geometry_info': {},
        'scf_parameters': {},
        'optimization_parameters': {},
        'property_parameters': {}
    }
    
    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except Exception as e:
        settings['extraction_error'] = str(e)
        return settings
    
    # Store original content (first 1000 chars for reference)
    settings['file_content_preview'] = content[:1000]
    
    # Extract CRYSTAL keywords
    crystal_keywords = [
        'CRYSTAL', 'SLAB', 'POLYMER', 'MOLECULE', 'EXTERNAL',
        'OPTGEOM', 'FREQCALC', 'SINGLEPOINT', 'RESTART',
        'DFT', 'HYBRID', 'EXCHANGE', 'CORRELAT', 'NONLOCAL',
        'SHRINK', 'TOLINTEG', 'TOLDEE', 'SCFDIR', 'FMIXING',
        'MAXCYCLE', 'ANDERSON', 'DIIS', 'BROYDEN', 'LEVSHIFT',
        'BIPOSIZE', 'EXCHSIZE', 'ILASIZE', 'INTGPACK',
        'MADELIMIT', 'BIPOLARIZ', 'EXCHPERM', 'POLEORDR'
    ]
    
    found_keywords = []
    for keyword in crystal_keywords:
        if keyword in content.upper():
            found_keywords.append(keyword)
    
    settings['crystal_keywords'] = found_keywords
    
    # Extract calculation type
    if '.d3' in input_file.suffix.lower():
        settings['calculation_type'] = 'properties'
        # Extract property-specific parameters
        settings['property_parameters'] = _extract_property_parameters(content)
    else:
        settings['calculation_type'] = 'scf'
        
    # Extract SCF and calculation parameters
    settings['calculation_parameters'] = _extract_calculation_parameters(content)
    settings['scf_parameters'] = _extract_scf_parameters(content)
    
    # Extract basis set information
    settings['basis_set_info'] = _extract_basis_set_info(content)
    
    # Extract geometry and optimization info
    settings['geometry_info'] = _extract_geometry_info(content)
    if 'OPTGEOM' in content.upper():
        settings['optimization_parameters'] = _extract_optimization_parameters(content)
    
    # Extract functional information
    settings['functional_info'] = _extract_functional_info(content)
    
    return settings


def _extract_calculation_parameters(content: str) -> Dict[str, Any]:
    """Extract general calculation parameters."""
    params = {}
    
    parameter_patterns = {
        'shrink_factor': r'SHRINK\s+(\d+)\s+(\d+)',
        'tolinteg_values': r'TOLINTEG\s+([\d\s]+)',
        'toldee_value': r'TOLDEE\s+(\d+)',
        'maxcycle_value': r'MAXCYCLE\s+(\d+)',
        'biposize': r'BIPOSIZE\s+(\d+)',
        'exchsize': r'EXCHSIZE\s+(\d+)',
        'ilasize': r'ILASIZE\s+(\d+)',
        'madelimit': r'MADELIMIT\s+(\d+)'
    }
    
    for param_name, pattern in parameter_patterns.items():
        match = re.search(pattern, content.upper())
        if match:
            if param_name == 'shrink_factor':
                params[param_name] = {'k_points': int(match.group(1)), 'density': int(match.group(2))}
            elif param_name == 'tolinteg_values':
                values = [int(x) for x in match.group(1).split()]
                params[param_name] = values
            else:
                params[param_name] = int(match.group(1))
    
    return params


def _extract_scf_parameters(content: str) -> Dict[str, Any]:
    """Extract SCF-specific parameters."""
    scf_params = {}
    
    scf_patterns = {
        'fmixing': r'FMIXING\s+(\d+)',
        'levshift': r'LEVSHIFT\s+([\d.]+)\s+(\d+)',
        'anderson_mixing': r'ANDERSON\s+([\d.]+)\s+(\d+)',
        'diis_mixing': r'DIIS\s+([\d.]+)\s+(\d+)',
        'broyden_mixing': r'BROYDEN\s+([\d.]+)\s+(\d+)'
    }
    
    for param, pattern in scf_patterns.items():
        match = re.search(pattern, content.upper())
        if match:
            if param == 'fmixing':
                scf_params[param] = int(match.group(1))
            elif param == 'levshift':
                scf_params[param] = {'factor': float(match.group(1)), 'cycles': int(match.group(2))}
            else:
                scf_params[param] = {'factor': float(match.group(1)), 'start_cycle': int(match.group(2))}
    
    # Check for SCFDIR
    if 'SCFDIR' in content.upper():
        scf_params['direct_scf'] = True
    
    return scf_params


def _extract_basis_set_info(content: str) -> Dict[str, Any]:
    """Extract basis set information."""
    basis_info = {}
    
    if 'EXTERNAL' in content.upper():
        basis_info['type'] = 'external'
        # Look for basis set file references
        basis_match = re.search(r'EXTERNAL\s*\n\s*(\S+)', content, re.IGNORECASE)
        if basis_match:
            basis_info['file'] = basis_match.group(1)
    else:
        basis_info['type'] = 'internal'
        
        # Count basis set definitions
        basis_definitions = re.findall(r'^\s*\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+', content, re.MULTILINE)
        basis_info['basis_functions_count'] = len(basis_definitions)
    
    return basis_info


def _extract_geometry_info(content: str) -> Dict[str, Any]:
    """Extract geometry information."""
    geom_info = {}
    
    # Determine dimensionality
    if 'CRYSTAL' in content.upper():
        geom_info['dimensionality'] = '3D'
    elif 'SLAB' in content.upper():
        geom_info['dimensionality'] = '2D'
    elif 'POLYMER' in content.upper():
        geom_info['dimensionality'] = '1D'
    elif 'MOLECULE' in content.upper():
        geom_info['dimensionality'] = '0D'
    
    # Count atoms in geometry section
    atom_lines = re.findall(r'^\s*\d+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+', content, re.MULTILINE)
    geom_info['atom_count'] = len(atom_lines)
    
    # Extract space group if specified
    space_group_match = re.search(r'^\s*(\d+)\s*$', content, re.MULTILINE)
    if space_group_match:
        geom_info['space_group'] = int(space_group_match.group(1))
    
    return geom_info


def _extract_optimization_parameters(content: str) -> Dict[str, Any]:
    """Extract geometry optimization parameters."""
    opt_params = {}
    
    # Find OPTGEOM section
    optgeom_match = re.search(r'OPTGEOM\s*\n(.*?)END', content, re.DOTALL | re.IGNORECASE)
    if optgeom_match:
        optgeom_content = optgeom_match.group(1)
        
        opt_patterns = {
            'fulloptg': r'FULLOPTG',
            'cellonly': r'CELLONLY',
            'atomsonly': r'ATOMSONLY',
            'maxcycle': r'MAXCYCLE\s+(\d+)',
            'toldeg': r'TOLDEG\s+([\d.E+-]+)',
            'toldex': r'TOLDEX\s+([\d.E+-]+)',
            'toldee': r'TOLDEE\s+([\d.E+-]+)',
            'finalrun': r'FINALRUN\s+(\d+)'
        }
        
        for param, pattern in opt_patterns.items():
            match = re.search(pattern, optgeom_content.upper())
            if match:
                if param in ['fulloptg', 'cellonly', 'atomsonly']:
                    opt_params[param] = True
                elif param in ['maxcycle', 'finalrun']:
                    opt_params[param] = int(match.group(1))
                else:
                    opt_params[param] = float(match.group(1))
    
    return opt_params


def _extract_functional_info(content: str) -> Dict[str, Any]:
    """Extract exchange-correlation functional information."""
    functional_info = {}
    
    # Check for DFT
    if 'DFT' in content.upper():
        functional_info['method'] = 'DFT'
        
        # Check for complete functional specifications first
        complete_functionals = ['PBESOL', 'B3LYP', 'HSE06', 'PBE0', 'BLYP', 'PBE', 'LDA', 'SVWN', 'PWGGA']
        for func in complete_functionals:
            if func in content.upper():
                functional_info['exchange'] = func
                break
        
        # If not found, try exchange patterns
        if 'exchange' not in functional_info:
            exchange_patterns = [
                r'EXCHANGE\s+(\w+)',
                r'(\w+)EXCHANGE',  # For combined functionals like B3LYP
            ]
            
            for pattern in exchange_patterns:
                match = re.search(pattern, content.upper())
                if match:
                    functional_info['exchange'] = match.group(1)
                    break
        
        # Extract correlation functional
        corr_patterns = [
            r'CORRELAT\s+(\w+)',
            r'(\w+)CORR',
        ]
        
        for pattern in corr_patterns:
            match = re.search(pattern, content.upper())
            if match:
                functional_info['correlation'] = match.group(1)
                break
        
        # Check for dispersion corrections
        if 'NONLOCAL' in content.upper():
            functional_info['dispersion'] = 'nonlocal'
        elif 'D3' in content.upper():
            functional_info['dispersion'] = 'D3'
        elif 'D2' in content.upper():
            functional_info['dispersion'] = 'D2'
            
    elif 'HYBRID' in content.upper():
        functional_info['method'] = 'Hybrid'
    elif 'HF' in content.upper() or 'HARTREE' in content.upper():
        # Only assign HF if it's clearly a Hartree-Fock calculation, not D3 property files
        if not any(prop_type in content.upper() for prop_type in ['BAND', 'DOSS', 'NEWK']):
            functional_info['method'] = 'HF'
    # For D3 files, don't assign method - settings inherited from previous calculation
    
    return functional_info


def _extract_property_parameters(content: str) -> Dict[str, Any]:
    """Extract parameters for properties calculations (D3 files)."""
    prop_params = {}
    
    # Common property calculation keywords
    property_keywords = [
        'BAND', 'DOSS', 'NEWK', 'COND', 'ECHG', 'POTM',
        'PPAN', 'RDFMWF', 'RHOLINE', 'MOLDRAW'
    ]
    
    found_properties = []
    for keyword in property_keywords:
        if keyword in content.upper():
            found_properties.append(keyword)
    
    prop_params['property_types'] = found_properties
    
    # Extract NEWK parameters for band structure
    newk_match = re.search(r'NEWK\s+([\d\s]+)', content.upper())
    if newk_match:
        k_values = [int(x) for x in newk_match.group(1).split()]
        prop_params['k_path_points'] = k_values
    
    # Extract k-path labels for BAND calculations
    if 'BAND' in content.upper():
        k_path_labels = []
        k_path_segments = []
        lines = content.split('\n')
        in_kpath_section = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.upper() == 'BAND':
                in_kpath_section = True
                continue
            elif line.upper() == 'END' and in_kpath_section:
                break
            elif in_kpath_section and line and not line.endswith('.out'):
                # Skip the first few lines (typically parameters) and find k-point labels
                if i > 2 and ' ' in line and not line.startswith('#'):
                    k_points = line.split()
                    if len(k_points) >= 2:
                        # Extract k-point labels (typically first two items)
                        start_point = k_points[0]
                        end_point = k_points[1]
                        k_path_labels.extend([start_point, end_point])
                        k_path_segments.append(f"{start_point} {end_point}")
        
        if k_path_labels:
            # Store both individual labels and path segments
            unique_labels = list(dict.fromkeys(k_path_labels))  # Preserve order, remove duplicates
            prop_params['k_path_labels'] = unique_labels
            prop_params['k_path_segments'] = k_path_segments
            
            # Create condensed k-path format with proper continuity handling
            # Example: 'X G', 'G L', 'L W', 'W G' -> 'X G L W G'
            # Example: 'X G', 'G L', 'G W', 'W G' -> 'X G L|G W G'
            condensed_segments = []
            current_path = []
            
            for segment in k_path_segments:
                points = segment.split()
                if len(points) == 2:
                    start_point, end_point = points
                    
                    # If this is the first segment or continues from previous
                    if not current_path:
                        current_path = [start_point, end_point]
                    elif current_path[-1] == start_point:
                        # Continuous path - just add the end point
                        current_path.append(end_point)
                    else:
                        # Discontinuous path - finish current and start new
                        condensed_segments.append(' '.join(current_path))
                        current_path = [start_point, end_point]
            
            # Add the final path segment
            if current_path:
                condensed_segments.append(' '.join(current_path))
            
            # Join with | for discontinuous segments
            prop_params['k_path_condensed'] = '|'.join(condensed_segments)
    
    # Extract DOSS parameters
    doss_match = re.search(r'DOSS\s+([\d\s.-]+)', content.upper())
    if doss_match:
        doss_values = doss_match.group(1).split()
        prop_params['dos_parameters'] = {
            'projections': len(doss_values),
            'values': doss_values
        }
    
    return prop_params


def extract_and_store_input_settings(calc_id: str, input_file: Path, 
                                    db_path: str = "materials.db") -> bool:
    """
    Extract settings from input file and store directly in materials database.
    
    Args:
        calc_id: Calculation ID
        input_file: Path to D12/D3 input file
        db_path: Path to materials database
        
    Returns:
        True if successful, False otherwise
    """
    if not input_file.exists():
        print(f"  ⚠️  Input file not found: {input_file}")
        return False
    
    try:
        # Extract settings
        settings = extract_input_settings(input_file)
        
        if not settings or 'extraction_error' in settings:
            print(f"  ⚠️  Failed to extract settings from {input_file.name}")
            return False
        
        # Store in database
        db = MaterialDatabase(db_path)
        
        with db._get_connection() as conn:
            # Update the calculation with extracted settings
            conn.execute("""
                UPDATE calculations 
                SET input_settings_json = ?
                WHERE calc_id = ?
            """, (json.dumps(settings), calc_id))
            
            # Also add file record
            conn.execute("""
                INSERT OR REPLACE INTO files 
                (calc_id, file_type, file_name, file_path, file_size, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                calc_id,
                'input',
                input_file.name,
                str(input_file),
                input_file.stat().st_size,
                datetime.now().isoformat()
            ))
        
        print(f"  ✅ Extracted and stored settings for {input_file.name}")
        print(f"    📋 Keywords: {', '.join(settings['crystal_keywords'][:5])}" + 
              (f" (+{len(settings['crystal_keywords'])-5} more)" if len(settings['crystal_keywords']) > 5 else ""))
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error extracting settings from {input_file}: {e}")
        return False


def query_calculation_settings(calc_id: str, db_path: str = "materials.db") -> Optional[Dict[str, Any]]:
    """Query stored settings for a calculation."""
    try:
        db = MaterialDatabase(db_path)
        
        with db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT settings_json FROM calculations WHERE calc_id = ?",
                (calc_id,)
            )
            result = cursor.fetchone()
            
            if result and result[0]:
                return json.loads(result[0])
            else:
                return None
                
    except Exception as e:
        print(f"Error querying settings for {calc_id}: {e}")
        return None


def main():
    """Test the input settings extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract CRYSTAL input settings")
    parser.add_argument("--input-file", required=True, help="D12/D3 input file")
    parser.add_argument("--calc-id", help="Calculation ID")
    parser.add_argument("--db-path", default="materials.db", help="Database path")
    parser.add_argument("--extract-only", action="store_true", help="Extract settings without storing")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    
    if args.extract_only:
        # Just extract and display settings
        settings = extract_input_settings(input_file)
        print(json.dumps(settings, indent=2))
    else:
        # Extract and store in database
        if not args.calc_id:
            print("❌ --calc-id required when storing to database")
            return
        
        success = extract_and_store_input_settings(args.calc_id, input_file, args.db_path)
        if success:
            print(f"✅ Settings stored successfully for {args.calc_id}")
        else:
            print(f"❌ Failed to store settings for {args.calc_id}")


if __name__ == "__main__":
    main()