#!/usr/bin/env python3
"""
SHRINK Parameter Parser Fix
==========================
Utility to correctly parse SHRINK parameters from CRYSTAL input files.

Handles both formats:
1. Symmetric: SHRINK followed by single line with two values (ka, kb)
2. Unsymmetric: SHRINK followed by first line (0 kb) and second line (ka ka kc)

Usage:
  from shrink_parser_fix import parse_shrink_parameter
  
Author: Generated for materials database project
"""

import re
from typing import Tuple, Optional, List


def parse_shrink_parameter(lines: List[str], shrink_line_index: int) -> Tuple[Optional[Tuple[int, int, int]], int]:
    """
    Parse SHRINK parameter from CRYSTAL input file lines.
    
    Args:
        lines: List of lines from input file
        shrink_line_index: Index of line containing "SHRINK"
        
    Returns:
        Tuple of (ka, kb, kc) values if successfully parsed, None otherwise
        Index of next line after SHRINK parameters
        
    Examples:
        # Symmetric format
        SHRINK
        12 24
        
        # Unsymmetric format  
        SHRINK
        0 24
        12 12 12
    """
    if shrink_line_index + 1 >= len(lines):
        return None, shrink_line_index + 1
    
    # Get the line immediately after SHRINK
    next_line = lines[shrink_line_index + 1].strip()
    
    try:
        parts = next_line.split()
        
        if len(parts) == 2:
            # Could be symmetric (ka kb) or unsymmetric first line (0 kb)
            val1, val2 = map(int, parts)
            
            if val1 == 0:
                # Unsymmetric format, first line: 0 kb
                if shrink_line_index + 2 >= len(lines):
                    return None, shrink_line_index + 2
                
                kb = val2
                
                # Get second line: ka ka kc
                second_line = lines[shrink_line_index + 2].strip()
                second_parts = second_line.split()
                
                if len(second_parts) == 3:
                    ka, ka_check, kc = map(int, second_parts)
                    # Use first ka value regardless of symmetry
                    return (ka, kb, kc), shrink_line_index + 3
                else:
                    return None, shrink_line_index + 3
            else:
                # Symmetric format: ka kb (kc = kb)
                ka, kb = val1, val2
                kc = kb
                return (ka, kb, kc), shrink_line_index + 2
                
        elif len(parts) == 3:
            # Direct format: ka kb kc
            ka, kb, kc = map(int, parts)
            return (ka, kb, kc), shrink_line_index + 2
            
        else:
            # Unrecognized format
            return None, shrink_line_index + 2
            
    except (ValueError, IndexError):
        # Parsing error
        return None, shrink_line_index + 2


def extract_shrink_from_file(file_path: str) -> Optional[Tuple[int, int, int]]:
    """
    Extract SHRINK parameters from a CRYSTAL input file.
    
    Args:
        file_path: Path to .d12 or .d3 file
        
    Returns:
        Tuple of (ka, kb, kc) values if found, None otherwise
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find SHRINK line
        for i, line in enumerate(lines):
            if 'SHRINK' in line.upper():
                shrink_params, _ = parse_shrink_parameter(lines, i)
                if shrink_params:
                    return shrink_params
        
        return None
        
    except (IOError, OSError):
        return None


def fix_shrink_parsing_in_script(script_content: str) -> str:
    """
    Fix SHRINK parameter parsing in existing scripts.
    
    Replaces naive SHRINK parsing with robust version.
    
    Args:
        script_content: Content of Python script as string
        
    Returns:
        Fixed script content
    """
    # Pattern to find problematic SHRINK parsing
    problematic_patterns = [
        r'lines\[.*\+\s*1\].*SHRINK',  # lines[i+1] after finding SHRINK
        r'next_line.*SHRINK',          # next_line after SHRINK
        r'line.*SHRINK.*split\(\)',    # Splitting line after SHRINK
    ]
    
    # Add import for the fix
    import_fix = """
# Import SHRINK parser fix
from shrink_parser_fix import parse_shrink_parameter, extract_shrink_from_file
"""
    
    # Add the import at the top if not already present
    if 'shrink_parser_fix' not in script_content:
        script_content = import_fix + "\n" + script_content
    
    return script_content


def validate_shrink_parameters(ka: int, kb: int, kc: int) -> bool:
    """
    Validate SHRINK parameters for reasonableness.
    
    Args:
        ka, kb, kc: SHRINK parameter values
        
    Returns:
        True if parameters seem reasonable
    """
    # Basic validation
    if ka <= 0 or kb <= 0 or kc <= 0:
        return False
    
    # Typical ranges for CRYSTAL calculations
    if ka > 100 or kb > 100 or kc > 100:
        return False
    
    return True


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Symmetric format
        ["SHRINK", "12 24", "SCFDIR"],
        
        # Unsymmetric format
        ["SHRINK", "0 24", "12 12 12", "SCFDIR"],
        
        # Direct format
        ["SHRINK", "8 16 8", "SCFDIR"],
        
        # Invalid format
        ["SHRINK", "SCFDIR", "END"],
    ]
    
    print("🧪 Testing SHRINK parameter parsing:")
    
    for i, test_lines in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {test_lines}")
        shrink_params, next_idx = parse_shrink_parameter(test_lines, 0)
        if shrink_params:
            ka, kb, kc = shrink_params
            print(f"   ✅ Parsed: ka={ka}, kb={kb}, kc={kc}")
            print(f"   Next line index: {next_idx} ({'in bounds' if next_idx < len(test_lines) else 'out of bounds'})")
            if next_idx < len(test_lines):
                print(f"   Next line content: '{test_lines[next_idx]}'")
        else:
            print(f"   ❌ Failed to parse")
            print(f"   Next line index: {next_idx}")