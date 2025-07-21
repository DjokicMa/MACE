#!/usr/bin/env python3
"""Test script to verify Fermi energy extraction from CRYSTAL output files."""

from d3_interactive import get_band_info_from_output
import glob
import os

# Test on all .out files in test directory
test_dir = "test"
if os.path.exists(test_dir):
    out_files = glob.glob(os.path.join(test_dir, "*.out"))
    
    print("Testing Fermi energy extraction:\n")
    print(f"{'File':<25} {'Fermi (Ha)':<15} {'Fermi (eV)':<15} {'Source'}")
    print("-" * 70)
    
    for out_file in sorted(out_files):
        band_info = get_band_info_from_output(out_file)
        fermi_ha = band_info.get('fermi_energy')
        
        if fermi_ha is not None:
            fermi_ev = fermi_ha * 27.211386
            
            # Determine source more precisely
            with open(out_file, 'r') as f:
                content = f.read()
            
            # Check which pattern would match
            import re
            if re.search(r'FERMI ENERGY\s*\[.*?\]\s*:\s*([-\d.]+)', content):
                source = "FERMI ENERGY"
            else:
                scf_ended = re.search(r'== SCF ENDED[^\n]*\n', content)
                if scf_ended:
                    before_scf = content[:scf_ended.start()]
                    if re.findall(r'EFERMI\(AU\)\s*([-\d.E+]+)', before_scf):
                        source = "EFERMI(final)"
                    else:
                        after_scf = content[scf_ended.end():]
                        if re.findall(r'TOP OF VALENCE BANDS.*?EIG\s*([-\d.E+]+)\s*AU', after_scf):
                            source = "TOP OF VB(final)"
                        else:
                            source = "Unknown"
                else:
                    if 'EFERMI(AU)' in content:
                        source = "EFERMI(fallback)"
                    elif 'TOP OF VALENCE BANDS' in content:
                        source = "TOP OF VB(fallback)"
                    else:
                        source = "Unknown"
            
            print(f"{os.path.basename(out_file):<25} {fermi_ha:<15.6f} {fermi_ev:<15.3f} {source}")
        else:
            print(f"{os.path.basename(out_file):<25} {'Not found':<15} {'N/A':<15} N/A")
    
    # Test specific example with automatic mu range
    print("\n\nExample TRANSPORT configuration with automatic Fermi:")
    print("-" * 70)
    
    test_file = "2_dia2_sp.out"
    if test_file in [os.path.basename(f) for f in out_files]:
        test_path = os.path.join(test_dir, test_file)
        band_info = get_band_info_from_output(test_path)
        
        if band_info.get('fermi_energy'):
            fermi_ev = band_info['fermi_energy'] * 27.211386
            mu_min = fermi_ev - 2.0
            mu_max = fermi_ev + 2.0
            
            print(f"File: {test_file}")
            print(f"Fermi energy: {fermi_ev:.3f} eV")
            print(f"Chemical potential range: {mu_min:.3f} to {mu_max:.3f} eV")
            print(f"\nExpected MURANGE line in D3:")
            print(f"MURANGE")
            print(f"{mu_min:.3f} {mu_max:.3f} 0.01")
else:
    print("Test directory not found. Please run in Crystal_d3 directory.")