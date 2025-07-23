#!/usr/bin/env python3
"""
SCF Settings Extractor from CRYSTAL Output Files
================================================
Extract SCF and advanced electronic settings from CRYSTAL output files.

This module extracts settings that are reported in output files including:
- BIPOSIZE: Coulomb bipolar buffer size
- EXCHSIZE: Exchange bipolar buffer size  
- LEVSHIFT: Level shifter parameters
- FMIXING: Fock/Kohn-Sham matrix mixing percentage
- PPAN: Population analysis settings
- Memory allocation parameters
- SCF convergence parameters

Usage:
  from scf_settings_extractor import extract_scf_settings_from_output
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


def extract_scf_settings_from_output(output_file: Path) -> Dict[str, Any]:
    """
    Extract SCF and advanced electronic settings from CRYSTAL output file.
    
    Args:
        output_file: Path to .out file
        
    Returns:
        Dictionary with extracted SCF settings
    """
    settings = {
        'scf_parameters': {},
        'memory_parameters': {},
        'convergence_parameters': {},
        'mixing_parameters': {},
        'population_analysis': {},
        'advanced_settings': {}
    }
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
    except Exception as e:
        settings['extraction_error'] = str(e)
        return settings
    
    # Extract SCF parameters reported in output
    settings['scf_parameters'] = _extract_scf_parameters_from_output(content)
    settings['memory_parameters'] = _extract_memory_parameters(content)
    settings['convergence_parameters'] = _extract_convergence_parameters(content)
    settings['mixing_parameters'] = _extract_mixing_parameters(content)
    settings['population_analysis'] = _extract_population_settings(content)
    settings['advanced_settings'] = _extract_advanced_settings(content)
    
    return settings


def _extract_scf_parameters_from_output(content: str) -> Dict[str, Any]:
    """Extract SCF parameters from output file."""
    params = {}
    
    # INFORMATION lines pattern - these show actual settings used
    info_patterns = {
        'biposize': r'INFORMATION \*+\s*BIPOSIZE\s*\*+.*?COULOMB BIPOLAR BUFFER SET TO\s+(\d+)',
        'exchsize': r'INFORMATION \*+\s*EXCHSIZE\s*\*+.*?EXCHANGE BIPOLAR BUFFER SIZE SET TO\s+(\d+)',
        'maxcycle': r'INFORMATION \*+\s*MAXCYCLE\s*\*+.*?MAX NUMBER OF SCF CYCLES SET TO\s+(\d+)',
        'ppan': r'INFORMATION \*+\s*PPAN\s*\*+.*?MULLIKEN POPULATION ANALYSIS',
        'toldee': r'INFORMATION \*+\s*TOLDEE\s*\*+.*?SCF TOL ON TOTAL ENERGY SET TO\s+(\d+)',
        'tolinteg': r'INFORMATION \*+\s*TOLINTEG\s*\*+.*?COULOMB AND EXCHANGE SERIES TOLERANCES',
        'diis': r'INFORMATION \*+\s*DIIS\s*\*+.*?DIIS FOR SCF ACTIVE',
        'scfdir': r'INFORMATION \*+\s*SCFDIR\s*\*+.*?DIRECT SCF',
        'savewf': r'INFORMATION \*+\s*SAVEWF\s*\*+',
        'savepred': r'INFORMATION \*+\s*SAVEPRED\s*\*+'
    }
    
    for param, pattern in info_patterns.items():
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            if param in ['ppan', 'diis', 'scfdir', 'savewf', 'savepred']:
                params[param] = True
            elif param in ['biposize', 'exchsize', 'maxcycle', 'toldee']:
                params[param] = int(match.group(1))
            else:
                params[param] = True
    
    # Check for level shifter status
    if 'LEVEL SHIFTER DISABLED' in content:
        params['levshift_enabled'] = False
    elif 'LEVEL SHIFTER ACTIVE' in content or 'LEVSHIFT' in content:
        params['levshift_enabled'] = True
        
    # Extract shrink factors from output
    shrink_match = re.search(r'SHRINK\.\s*FACT\.\(MONKH\.\)\s+(\d+)\s+(\d+)\s+(\d+)', content)
    if shrink_match:
        params['shrink_factors'] = {
            'k1': int(shrink_match.group(1)),
            'k2': int(shrink_match.group(2)),
            'k3': int(shrink_match.group(3))
        }
    
    # Extract Gilat shrink factor
    gilat_match = re.search(r'SHRINKING FACTOR\(GILAT NET\)\s+(\d+)', content)
    if gilat_match:
        params['gilat_shrink'] = int(gilat_match.group(1))
    
    return params


def _extract_memory_parameters(content: str) -> Dict[str, Any]:
    """Extract memory-related parameters."""
    memory = {}
    
    # Memory allocation patterns
    memory_patterns = {
        'intgpack_memory': r'INTGPACK MEMORY ALLOCATION\s*:\s*(\d+)',
        'total_memory': r'TOTAL MEMORY AVAILABLE\s*:\s*([\d.]+)\s*(\w+)',
        'memory_needed': r'MEMORY NEEDED\s*:\s*([\d.]+)\s*(\w+)',
        'biposize_mb': r'BIPOSIZE.*?(\d+)\s*MB',
        'exchsize_mb': r'EXCHSIZE.*?(\d+)\s*MB'
    }
    
    for param, pattern in memory_patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            if 'mb' in param or 'memory' in param:
                memory[param] = {
                    'value': float(match.group(1)),
                    'unit': match.group(2) if match.lastindex > 1 else 'MB'
                }
            else:
                memory[param] = int(match.group(1))
    
    # Extract bipolar buffer sizes from INFORMATION lines
    bipo_match = re.search(r'COULOMB BIPOLAR BUFFER SET TO\s+(\d+)', content)
    if bipo_match:
        memory['biposize_buffer'] = int(bipo_match.group(1))
        
    exch_match = re.search(r'EXCHANGE BIPOLAR BUFFER SIZE SET TO\s+(\d+)', content)
    if exch_match:
        memory['exchsize_buffer'] = int(exch_match.group(1))
    
    return memory


def _extract_convergence_parameters(content: str) -> Dict[str, Any]:
    """Extract convergence-related parameters."""
    convergence = {}
    
    # Convergence thresholds
    conv_patterns = {
        'toldee': r'TOLDEE\s*-?\s*TOTAL ENERGY CONVERGENCE TOLERANCE\s*:\s*10\*\*\(-(\d+)\)',
        'toldeg': r'TOLDEG\s*-?\s*GRADIENT CONVERGENCE TOLERANCE\s*:\s*([\d.E+-]+)',
        'toldex': r'TOLDEX\s*-?\s*GEOMETRY CONVERGENCE TOLERANCE\s*:\s*([\d.E+-]+)',
        'tolinteg': r'TOLINTEG\s*-?\s*COULOMB SERIES TOLERANCE\s*:\s*10\*\*\(-(\d+)\)'
    }
    
    for param, pattern in conv_patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            if param == 'toldee' or param == 'tolinteg':
                convergence[param] = int(match.group(1))
            else:
                convergence[param] = float(match.group(1))
    
    # Extract actual convergence achieved
    scf_conv_match = re.search(r'SCF FIELD CONVERGENCE\s+IN\s+(\d+)\s+CYCLES', content)
    if scf_conv_match:
        convergence['scf_cycles_actual'] = int(scf_conv_match.group(1))
    
    # Extract optimization convergence
    opt_conv_match = re.search(r'CONVERGENCE ON (\w+)\s*[:=]\s*([\d.E+-]+)', content)
    if opt_conv_match:
        convergence[f'convergence_{opt_conv_match.group(1).lower()}'] = float(opt_conv_match.group(2))
    
    return convergence


def _extract_mixing_parameters(content: str) -> Dict[str, Any]:
    """Extract SCF mixing parameters."""
    mixing = {}
    
    # FMIXING percentage
    fmixing_match = re.search(r'FMIXING\s*[:=]\s*(\d+)', content)
    if fmixing_match:
        mixing['fmixing_percentage'] = int(fmixing_match.group(1))
    
    # Anderson mixing
    anderson_match = re.search(r'ANDERSON MIXING.*?FACTOR\s*[:=]\s*([\d.]+)', content)
    if anderson_match:
        mixing['anderson_factor'] = float(anderson_match.group(1))
        mixing['anderson_enabled'] = True
    
    # DIIS mixing
    if 'DIIS FOR SCF ACTIVE' in content:
        mixing['diis_enabled'] = True
        
        # Extract DIIS parameters
        diis_start_match = re.search(r'DIIS STARTING FROM CYCLE\s*(\d+)', content)
        if diis_start_match:
            mixing['diis_start_cycle'] = int(diis_start_match.group(1))
            
        diis_size_match = re.search(r'DIIS SUBSPACE SIZE\s*[:=]\s*(\d+)', content)
        if diis_size_match:
            mixing['diis_subspace_size'] = int(diis_size_match.group(1))
    
    # Broyden mixing
    broyden_match = re.search(r'BROYDEN MIXING.*?FACTOR\s*[:=]\s*([\d.]+)', content)
    if broyden_match:
        mixing['broyden_factor'] = float(broyden_match.group(1))
        mixing['broyden_enabled'] = True
    
    # Level shifter
    if 'LEVEL SHIFTER DISABLED' in content:
        mixing['levshift_enabled'] = False
    else:
        levshift_match = re.search(r'LEVSHIFT\s*[:=]\s*([\d.]+)\s*CYCLES\s*[:=]\s*(\d+)', content)
        if levshift_match:
            mixing['levshift_enabled'] = True
            mixing['levshift_factor'] = float(levshift_match.group(1))
            mixing['levshift_cycles'] = int(levshift_match.group(2))
    
    return mixing


def _extract_population_settings(content: str) -> Dict[str, Any]:
    """Extract population analysis settings."""
    population = {}
    
    # Check for PPAN
    if 'MULLIKEN POPULATION ANALYSIS AT THE END OF SCF' in content:
        population['mulliken_analysis'] = True
        population['analysis_type'] = 'end_of_scf'
    elif 'MULLIKEN POPULATION ANALYSIS' in content:
        population['mulliken_analysis'] = True
        
    # Check for other population analysis types
    if 'HIRSHFELD POPULATION ANALYSIS' in content:
        population['hirshfeld_analysis'] = True
        
    if 'BADER ANALYSIS' in content:
        population['bader_analysis'] = True
        
    # Extract charge analysis settings
    if 'ATOMIC CHARGES' in content:
        population['atomic_charges'] = True
        
    if 'OVERLAP POPULATIONS' in content:
        population['overlap_populations'] = True
    
    return population


def _extract_advanced_settings(content: str) -> Dict[str, Any]:
    """Extract advanced electronic structure settings."""
    advanced = {}
    
    # Integration pack parameters
    intgpack_patterns = {
        'ilasize': r'ILASIZE\s*[:=]\s*(\d+)',
        'intgpack': r'INTGPACK\s*[:=]\s*(\d+)',
        'madelimit': r'MADELIMIT\s*[:=]\s*(\d+)',
        'poleordr': r'POLEORDR\s*[:=]\s*(\d+)'
    }
    
    for param, pattern in intgpack_patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            advanced[param] = int(match.group(1))
    
    # Extract exchange parameters
    if 'EXCHPERM' in content:
        advanced['exchange_permutation'] = True
        
    if 'BIPOLARIZ' in content:
        advanced['bipolarization'] = True
    
    # Extract DFT grid information
    grid_match = re.search(r'DFT GRID.*?(\d+)\s+POINTS', content)
    if grid_match:
        advanced['dft_grid_points'] = int(grid_match.group(1))
        
    # Extract pruning information
    if 'PRUNING' in content:
        advanced['grid_pruning'] = True
        
    # Extract symmetry handling
    if 'SYMMOPS' in content:
        symm_match = re.search(r'(\d+)\s+SYMMOPS', content)
        if symm_match:
            advanced['symmetry_operations'] = int(symm_match.group(1))
    
    return advanced


def extract_and_compare_settings(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Extract settings from both input and output files and compare.
    
    Args:
        input_file: Path to .d12 file
        output_file: Path to .out file
        
    Returns:
        Dictionary with input settings, output settings, and comparison
    """
    comparison = {
        'input_file': str(input_file),
        'output_file': str(output_file),
        'input_settings': {},
        'output_settings': {},
        'differences': {}
    }
    
    # Extract input settings
    try:
        from settings_extractor import extract_input_settings
        comparison['input_settings'] = extract_input_settings(input_file)
    except Exception as e:
        comparison['input_extraction_error'] = str(e)
    
    # Extract output settings
    try:
        comparison['output_settings'] = extract_scf_settings_from_output(output_file)
    except Exception as e:
        comparison['output_extraction_error'] = str(e)
    
    # Compare settings
    if comparison['input_settings'] and comparison['output_settings']:
        # Compare SCF parameters
        input_scf = comparison['input_settings'].get('scf_parameters', {})
        output_scf = comparison['output_settings'].get('scf_parameters', {})
        
        for param in ['biposize', 'exchsize', 'maxcycle', 'levshift']:
            if param in input_scf and param in output_scf:
                if input_scf[param] != output_scf[param]:
                    comparison['differences'][param] = {
                        'input': input_scf[param],
                        'output': output_scf[param]
                    }
    
    return comparison


def main():
    """Test the SCF settings extractor."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Extract SCF settings from CRYSTAL output files")
    parser.add_argument("--output-file", required=True, help="CRYSTAL output file (.out)")
    parser.add_argument("--input-file", help="Optional: corresponding input file (.d12) for comparison")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    output_file = Path(args.output_file)
    
    if args.input_file:
        # Compare input and output settings
        input_file = Path(args.input_file)
        results = extract_and_compare_settings(input_file, output_file)
    else:
        # Just extract output settings
        results = extract_scf_settings_from_output(output_file)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Pretty print results
        print(f"\n📄 SCF Settings from: {output_file.name}")
        print("=" * 60)
        
        if isinstance(results, dict) and 'scf_parameters' in results:
            # Output-only extraction
            for category, params in results.items():
                if params and not category.startswith('_'):
                    print(f"\n{category.replace('_', ' ').title()}:")
                    for key, value in params.items():
                        print(f"  {key}: {value}")
        else:
            # Comparison results
            print("\nInput Settings:")
            for key, value in results.get('input_settings', {}).get('scf_parameters', {}).items():
                print(f"  {key}: {value}")
                
            print("\nOutput Settings:")
            for key, value in results.get('output_settings', {}).get('scf_parameters', {}).items():
                print(f"  {key}: {value}")
                
            if results.get('differences'):
                print("\nDifferences Found:")
                for param, diff in results['differences'].items():
                    print(f"  {param}: {diff['input']} (input) → {diff['output']} (output)")


if __name__ == "__main__":
    main()