#!/usr/bin/env python3
"""Test electronic classification system."""
import sys
sys.path.append('code/Job_Scripts')
from crystal_property_extractor import CrystalPropertyExtractor
from pathlib import Path

# Test electronic classification on different calculation types
test_files = [
    ('code/Job_Scripts/workflow_outputs/workflow_20250621_170319/step_002_SP/1_dia_opt_sp/1_dia_opt_sp.out', 'SP'),
    ('code/Job_Scripts/workflow_outputs/workflow_20250621_170319/step_004_DOSS/1_dia_opt_doss/1_dia_opt_doss.out', 'DOSS'),
    ('code/Job_Scripts/workflow_outputs/workflow_20250621_170319/step_003_BAND/1_dia_opt_band/1_dia_opt_band.out', 'BAND')
]

extractor = CrystalPropertyExtractor()

for file_path, calc_type in test_files:
    print(f'\n🧪 Testing Electronic Classification for {calc_type}:')
    print(f'📁 File: {Path(file_path).name}')
    print('=' * 60)
    
    properties = extractor.extract_all_properties(Path(file_path), material_id='1_dia', calc_id=f'test_{calc_type.lower()}')
    
    # Show classification and gap properties
    classification_props = {k: v for k, v in properties.items() if any(x in k.lower() for x in ['classification', 'band_gap', 'gap', 'electronic', 'magnetic', 'conductivity', 'insulator', 'semiconductor'])}
    
    if classification_props:
        print('🔬 Electronic Classification Results:')
        for prop, value in sorted(classification_props.items()):
            print(f'  {prop:35} = {value}')
    else:
        print('❌ No classification properties found')
        
print(f'\n📊 Electronic Classification System Test Complete')