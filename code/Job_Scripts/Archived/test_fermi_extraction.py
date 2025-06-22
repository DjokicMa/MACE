#!/usr/bin/env python3
"""Test Fermi energy extraction across all calculation types."""
import sys
sys.path.append('code/Job_Scripts')
from crystal_property_extractor import CrystalPropertyExtractor
from pathlib import Path

# Test Fermi energy extraction on different calculation types
test_files = [
    ('code/Job_Scripts/workflow_outputs/workflow_20250621_170319/step_001_OPT/1_dia_opt/1_dia_opt.out', 'OPT'),
    ('code/Job_Scripts/workflow_outputs/workflow_20250621_170319/step_002_SP/1_dia_opt_sp/1_dia_opt_sp.out', 'SP'),
    ('code/Job_Scripts/workflow_outputs/workflow_20250621_170319/step_003_BAND/1_dia_opt_band/1_dia_opt_band.out', 'BAND'),
    ('code/Job_Scripts/workflow_outputs/workflow_20250621_170319/step_004_DOSS/1_dia_opt_doss/1_dia_opt_doss.out', 'DOSS')
]

extractor = CrystalPropertyExtractor()

print('🔍 Testing Fermi Energy Extraction Across All Calculation Types')
print('=' * 70)

for file_path, calc_type in test_files:
    print(f'\n📁 {calc_type} Calculation: {Path(file_path).name}')
    print('-' * 50)
    
    if not Path(file_path).exists():
        print(f'❌ File not found: {file_path}')
        continue
        
    properties = extractor.extract_all_properties(Path(file_path), material_id='1_dia', calc_id=f'test_{calc_type.lower()}')
    
    # Look for all Fermi energy related properties
    fermi_props = {k: v for k, v in properties.items() if 'fermi' in k.lower()}
    
    if fermi_props:
        print('✅ Fermi Energy Properties Found:')
        for prop, value in sorted(fermi_props.items()):
            print(f'  {prop:30} = {value}')
    else:
        print('❌ No Fermi energy properties found')
        
    # Also check for energy-related properties that might contain Fermi info
    energy_props = {k: v for k, v in properties.items() if 'energy' in k.lower() and v is not None}
    if energy_props:
        print('📊 Other Energy Properties:')
        for prop, value in sorted(energy_props.items()):
            if prop not in fermi_props:  # Don't duplicate Fermi properties
                print(f'  {prop:30} = {value}')

print(f'\n📋 Fermi Energy Extraction Test Complete')