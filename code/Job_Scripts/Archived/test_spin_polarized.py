#!/usr/bin/env python3
"""Test spin-polarized calculation extraction accuracy."""

from crystal_property_extractor import CrystalPropertyExtractor
from pathlib import Path

def test_spin_polarized_extraction():
    """Test spin-polarized calculation extraction."""
    extractor = CrystalPropertyExtractor()
    sp_file = Path('workflow_outputs/workflow_20250621_170319/step_001_OPT/3.4^9T2_opt/3.4^9T2_opt.out')

    print('🧪 Testing TRUE Spin-Polarized Calculation:')
    print('=' * 45)

    with open(sp_file, 'r') as f:
        content = f.read()

    # Check for spin-polarized sections
    alpha_beta_found = 'ALPHA+BETA ELECTRONS' in content
    alpha_minus_beta_found = 'ALPHA-BETA ELECTRONS' in content

    print(f'\nSpin-polarized sections in {sp_file.name}:')
    print(f'   ALPHA+BETA ELECTRONS: {"✅" if alpha_beta_found else "❌"}')
    print(f'   ALPHA-BETA ELECTRONS: {"✅" if alpha_minus_beta_found else "❌"}')

    # Test population analysis extraction
    pop_props = extractor._extract_population_analysis(content)
    print(f'\n🔍 Population Analysis Results:')
    print(f'   is_spin_polarized: {pop_props.get("is_spin_polarized", "Not found")}')

    # Count the population analysis properties
    pop_count = sum(1 for key in pop_props.keys() if key != 'is_spin_polarized')
    print(f'   Population properties extracted: {pop_count}')

    # Show spin-polarized properties
    for key, value in pop_props.items():
        if 'alpha' in key.lower():
            value_preview = str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
            print(f'   {key}: {value_preview}')

    # Test band gap extraction for spin-polarized
    band_props = extractor._extract_electronic_properties(content)
    print(f'\n🔍 Electronic Properties (Band Gaps):')
    gap_count = 0
    for key, value in band_props.items():
        if 'gap' in key.lower() or 'band' in key.lower():
            print(f'   {key}: {value}')
            gap_count += 1

    print(f'\n📊 Summary:')
    print(f'   Total band gap properties: {gap_count}')

    # Check for alpha/beta separate band gaps
    if 'alpha_band_gap' in band_props or 'beta_band_gap' in band_props:
        print('   ✅ Separate alpha/beta band gaps detected')
    else:
        print('   ⚠️  No separate alpha/beta band gaps found')
    
    return pop_props, band_props

if __name__ == "__main__":
    test_spin_polarized_extraction()