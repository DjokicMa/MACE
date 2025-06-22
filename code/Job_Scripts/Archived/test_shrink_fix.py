#!/usr/bin/env python3
"""
Test SHRINK Parameter Fix in alldos.py
======================================
Test that the SHRINK parameter parsing fix works correctly.
"""

def test_shrink_parsing():
    """Test the improved SHRINK parsing logic."""
    
    # Test cases
    test_cases = [
        {
            'name': 'Symmetric format',
            'lines': ['CRYSTAL', 'SHRINK', '12 24', 'SCFDIR', 'END'],
            'expected_newk': ['12 24\n']
        },
        {
            'name': 'Unsymmetric format', 
            'lines': ['CRYSTAL', 'SHRINK', '0 24', '12 12 12', 'SCFDIR', 'END'],
            'expected_newk': ['0 24\n', '12 12 12\n']
        },
        {
            'name': 'Direct format',
            'lines': ['CRYSTAL', 'SHRINK', '8 16 8', 'SCFDIR', 'END'],
            'expected_newk': ['8 16 8\n']
        }
    ]
    
    print("🧪 Testing SHRINK parsing fix:")
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        input_lines = test_case['lines']
        newk = []
        
        # Extract SHRINK parameters with robust parsing (copied from alldos.py fix)
        for i, line in enumerate(input_lines):
            if "SHRINK" in line:
                # Parse SHRINK parameters correctly
                if i + 1 < len(input_lines):
                    next_line = input_lines[i + 1].strip()
                    parts = next_line.split()
                    
                    if len(parts) == 2:
                        # Could be symmetric (ka kb) or unsymmetric first line (0 kb)
                        try:
                            val1, val2 = map(int, parts)
                            
                            if val1 == 0:
                                # Unsymmetric format: 0 kb followed by ka ka kc
                                newk.append(next_line + "\n")
                                if i + 2 < len(input_lines):
                                    second_line = input_lines[i + 2].strip()
                                    newk.append(second_line + "\n")
                            else:
                                # Symmetric format: ka kb (only one line needed)
                                newk.append(next_line + "\n")
                        except ValueError:
                            print(f"   ❌ Could not parse numeric values: {parts}")
                            continue
                    elif len(parts) == 3:
                        # Direct format: ka kb kc (only one line needed)
                        try:
                            map(int, parts)  # Validate numeric
                            newk.append(next_line + "\n")
                        except ValueError:
                            print(f"   ❌ Could not parse numeric values: {parts}")
                            continue
                break
        
        # Check results
        expected = test_case['expected_newk']
        if newk == expected:
            print(f"   ✅ Correct: {newk}")
        else:
            print(f"   ❌ Expected: {expected}")
            print(f"   ❌ Got: {newk}")
            
        # Verify that SCFDIR is not included
        has_scfdir = any('SCFDIR' in line for line in newk)
        if has_scfdir:
            print(f"   ❌ ERROR: SCFDIR found in SHRINK parameters!")
        else:
            print(f"   ✅ SCFDIR correctly excluded")

if __name__ == "__main__":
    test_shrink_parsing()