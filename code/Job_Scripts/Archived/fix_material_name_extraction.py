#!/usr/bin/env python3
"""
Fix Material Name Extraction Issue
==================================
Fixes the issue where materials with numbers (test3, test4, etc.) are 
incorrectly named during SP generation.
"""

import re
from pathlib import Path


def extract_core_material_name_fixed(material_id: str) -> str:
    """
    Extract the core material name with proper handling of numbered materials.
    
    This fixed version correctly handles cases like:
    - test1_opt -> test1
    - test2_opt -> test2  
    - test3-RCSR-ums_opt -> test3-RCSR-ums
    - test4-CA_opt -> test4-CA
    """
    # Handle both full filenames and stems
    if material_id.endswith('.d12') or material_id.endswith('.d3'):
        name = Path(material_id).stem
    else:
        name = material_id
    
    # Strategy: Remove known calculation suffixes from the end
    calc_suffixes = ['_opt', '_sp', '_freq', '_band', '_doss', 
                     '_opt2', '_sp2', '_freq2', '_band2', '_doss2',
                     '_opt3', '_sp3', '_freq3', '_band3', '_doss3']
    
    # Remove calc suffix if present
    clean_name = name
    for suffix in calc_suffixes:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
            break
    
    return clean_name


def test_extraction():
    """Test the extraction with various material names"""
    test_cases = [
        ("test1_opt", "test1"),
        ("test2_opt", "test2"),
        ("test3-RCSR-ums_opt", "test3-RCSR-ums"),
        ("test4-CA_opt", "test4-CA"),
        ("test5_CA_opt", "test5_CA"),
        ("test6-CA_opt", "test6-CA"),
        ("test7_opt", "test7"),
        ("test8_opt", "test8"),
        ("test1_opt_BULK_OPTGEOM_symm_CRYSTAL_OPT_symm_PBE-D3_POB-TZVP-REV2", "test1"),
        ("test3-RCSR-ums_BULK_OPTGEOM_TZ_symm_CRYSTAL_OPT_symm_PBE-D3_POB-TZVP-REV2", "test3-RCSR-ums"),
    ]
    
    print("Testing material name extraction:")
    print("=" * 60)
    
    all_passed = True
    for input_name, expected in test_cases:
        result = extract_core_material_name_fixed(input_name)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"{status} {input_name}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        print()
    
    return all_passed


if __name__ == "__main__":
    if test_extraction():
        print("All tests passed!")
        
        # Show the fix that needs to be applied to workflow_engine.py
        print("\nTo fix workflow_engine.py, replace the extract_core_material_name method with:")
        print("=" * 60)
        print('''
    def extract_core_material_name(self, material_id: str) -> str:
        """Extract the core material name with proper handling of numbered materials"""
        # Handle both full filenames and stems
        from pathlib import Path
        if material_id.endswith('.d12') or material_id.endswith('.d3'):
            name = Path(material_id).stem
        else:
            name = material_id
        
        # Strategy: Remove known calculation suffixes from the end
        calc_suffixes = ['_opt', '_sp', '_freq', '_band', '_doss', 
                         '_opt2', '_sp2', '_freq2', '_band2', '_doss2',
                         '_opt3', '_sp3', '_freq3', '_band3', '_doss3']
        
        # Remove calc suffix if present
        clean_name = name
        for suffix in calc_suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        return clean_name
''')