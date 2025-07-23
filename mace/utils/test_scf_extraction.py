#!/usr/bin/env python3
"""
Test Script for SCF Settings Extraction
=======================================
Demonstrates extraction of SCF settings from both input (.d12) and output (.out) files.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

sys.path.append(str(Path(__file__).parent.parent))

from utils.settings_extractor import extract_input_settings
from utils.scf_settings_extractor import extract_scf_settings_from_output
from utils.property_extractor import CrystalPropertyExtractor


def test_scf_extraction():
    """Test SCF settings extraction from various files."""
    
    # Test files
    test_files = [
        ('cif/crystalouputs/3.4^9T2_BULK_OPTGEOM_TZ.d12', 'cif/crystalouputs/3.4^9T2_BULK_OPTGEOM_TZ.out'),
        ('cif/crystalouputs/1_dia_opt_BULK_OPTGEOM.d12', 'cif/crystalouputs/1_dia_opt_BULK_OPTGEOM.out'),
    ]
    
    for d12_path, out_path in test_files:
        d12_file = Path(d12_path)
        out_file = Path(out_path)
        
        if not d12_file.exists() or not out_file.exists():
            print(f"Skipping {d12_file.name} - files not found")
            continue
            
        print(f"\n{'='*80}")
        print(f"Testing: {d12_file.name} and {out_file.name}")
        print('='*80)
        
        # Extract from input file
        print("\n📄 Input File Settings (.d12):")
        print("-" * 40)
        input_settings = extract_input_settings(d12_file)
        
        # Show SCF-related settings from input
        scf_params = input_settings.get('scf_parameters', {})
        calc_params = input_settings.get('calculation_parameters', {})
        
        print("SCF Parameters:")
        for key, value in scf_params.items():
            print(f"  {key}: {value}")
            
        print("\nCalculation Parameters:")
        for key, value in calc_params.items():
            print(f"  {key}: {value}")
        
        # Extract from output file
        print("\n📄 Output File Settings (.out):")
        print("-" * 40)
        output_settings = extract_scf_settings_from_output(out_file)
        
        for category, params in output_settings.items():
            if params and not category.startswith('_'):
                print(f"\n{category.replace('_', ' ').title()}:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
        
        # Compare key parameters
        print("\n🔍 Comparison of Key Parameters:")
        print("-" * 40)
        
        # Compare BIPOSIZE
        input_bipo = calc_params.get('biposize')
        output_bipo = output_settings.get('scf_parameters', {}).get('biposize')
        if input_bipo and output_bipo:
            print(f"BIPOSIZE: {input_bipo} (input) → {output_bipo} (output)")
            
        # Compare EXCHSIZE
        input_exch = calc_params.get('exchsize')
        output_exch = output_settings.get('scf_parameters', {}).get('exchsize')
        if input_exch and output_exch:
            print(f"EXCHSIZE: {input_exch} (input) → {output_exch} (output)")
            
        # Compare FMIXING
        input_fmix = scf_params.get('fmixing')
        output_fmix = output_settings.get('mixing_parameters', {}).get('fmixing_percentage')
        if input_fmix and output_fmix:
            print(f"FMIXING: {input_fmix} (input) → {output_fmix} (output)")
            
        # Compare LEVSHIFT
        input_lev = scf_params.get('levshift')
        output_lev_enabled = output_settings.get('mixing_parameters', {}).get('levshift_enabled')
        if input_lev is not None:
            print(f"LEVSHIFT: {input_lev} (input) → enabled={output_lev_enabled} (output)")


def test_property_extractor_scf():
    """Test SCF extraction through the property extractor."""
    print(f"\n{'='*80}")
    print("Testing SCF Extraction via Property Extractor")
    print('='*80)
    
    extractor = CrystalPropertyExtractor()
    output_file = Path('cif/crystalouputs/3.4^9T2_BULK_OPTGEOM_TZ.out')
    
    if output_file.exists():
        properties = extractor.extract_all_properties(output_file)
        
        # Filter and show SCF-related properties
        scf_props = {k: v for k, v in properties.items() if 'scf_' in k}
        
        print("\nSCF Properties Extracted:")
        for key, value in sorted(scf_props.items()):
            print(f"  {key}: {value}")
    else:
        print(f"Output file not found: {output_file}")


def save_extraction_example():
    """Save an example of extracted settings to JSON."""
    output_file = Path('cif/crystalouputs/3.4^9T2_BULK_OPTGEOM_TZ.out')
    
    if output_file.exists():
        settings = extract_scf_settings_from_output(output_file)
        
        # Save to JSON
        json_file = Path('scf_settings_example.json')
        with open(json_file, 'w') as f:
            json.dump(settings, f, indent=2)
            
        print(f"\n✅ SCF settings saved to: {json_file}")
        print(f"   Total settings extracted: {sum(len(v) for v in settings.values() if isinstance(v, dict))}")


if __name__ == "__main__":
    print("SCF Settings Extraction Test")
    print("============================\n")
    
    # Run tests
    test_scf_extraction()
    test_property_extractor_scf()
    save_extraction_example()
    
    print("\n✅ All tests completed!")