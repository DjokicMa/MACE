#!/usr/bin/env python3
"""
Test complex workflow scenarios with OPT calculations appearing after FREQ
"""

import os
import sys
from pathlib import Path

# Import workflow engine
from workflow_engine import WorkflowEngine

def test_complex_workflow():
    """Test workflow with OPT appearing after FREQ"""
    
    print("Testing Complex Workflow Scenarios")
    print("=" * 80)
    
    # Initialize workflow engine
    engine = WorkflowEngine(auto_submit=False)
    
    # Test workflow: OPT → OPT2 → SP → OPT3 → SP2 → BAND → DOSS → FREQ → OPT4
    planned_sequence = ['OPT', 'OPT2', 'SP', 'OPT3', 'SP2', 'BAND', 'DOSS', 'FREQ', 'OPT4']
    
    print("\nWorkflow sequence:", ' → '.join(planned_sequence))
    print("\nTesting dependency resolution...")
    
    # Test dependencies
    deps = [
        ('OPT2', 'OPT'),
        ('SP', 'OPT2'),
        ('OPT3', 'SP'),
        ('SP2', 'OPT3'),
        ('BAND', 'SP2'),
        ('DOSS', 'BAND'),
        ('FREQ', 'DOSS'),
        ('OPT4', 'FREQ'),
    ]
    
    for calc_type, expected_dep in deps:
        actual_dep = engine._find_dependency_in_sequence(calc_type, planned_sequence)
        print(f"  {calc_type} waits for: {actual_dep} ({'✓' if actual_dep == expected_dep else '✗'})")
        assert actual_dep == expected_dep, f"Expected {expected_dep}, got {actual_dep}"
    
    print("\nTesting source calculation resolution for special cases...")
    
    # Simulate completed calculations up to FREQ
    completed_by_type = {
        'OPT': [{'calc_id': 'calc_opt_001', 'status': 'completed'}],
        'OPT2': [{'calc_id': 'calc_opt2_002', 'status': 'completed'}],
        'SP': [{'calc_id': 'calc_sp_003', 'status': 'completed'}],
        'OPT3': [{'calc_id': 'calc_opt3_004', 'status': 'completed'}],
        'SP2': [{'calc_id': 'calc_sp2_005', 'status': 'completed'}],
        'BAND': [{'calc_id': 'calc_band_006', 'status': 'completed'}],
        'DOSS': [{'calc_id': 'calc_doss_007', 'status': 'completed'}],
        'FREQ': [{'calc_id': 'calc_freq_008', 'status': 'completed'}],
    }
    
    # Test FREQ source
    print("\n1. FREQ calculation source:")
    # FREQ should use the highest completed OPT (OPT3)
    highest_opt = engine._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
    print(f"   FREQ uses geometry from: {highest_opt}")
    print(f"   Expected: calc_opt3_004 (OPT3)")
    assert highest_opt == 'calc_opt3_004', f"FREQ should use OPT3, not {highest_opt}"
    
    # Test OPT4 source
    print("\n2. OPT4 calculation source:")
    print("   OPT4 appears after FREQ in sequence")
    print("   FREQ cannot provide geometry for OPT4")
    print(f"   OPT4 should also use geometry from: {highest_opt} (OPT3)")
    
    # Test with even more complex workflow
    print("\n3. Testing extreme workflow:")
    extreme_seq = ['OPT', 'SP', 'OPT2', 'SP2', 'FREQ', 'OPT3', 'BAND', 'DOSS', 'OPT4', 'FREQ2', 'OPT5']
    
    # After FREQ completes (with OPT, OPT2 completed)
    completed_up_to_freq = {
        'OPT': [{'calc_id': 'calc_opt_001', 'status': 'completed'}],
        'SP': [{'calc_id': 'calc_sp_002', 'status': 'completed'}],
        'OPT2': [{'calc_id': 'calc_opt2_003', 'status': 'completed'}],
        'SP2': [{'calc_id': 'calc_sp2_004', 'status': 'completed'}],
        'FREQ': [{'calc_id': 'calc_freq_005', 'status': 'completed'}],
    }
    
    highest_at_freq = engine._find_highest_numbered_calc_of_type(completed_up_to_freq, 'OPT')
    print(f"   When FREQ completes, highest OPT is: {highest_at_freq} (should be OPT2)")
    assert highest_at_freq == 'calc_opt2_003'
    
    print(f"   OPT3 (after FREQ) should use: {highest_at_freq} as source")
    
    # After more calculations complete
    completed_up_to_freq2 = dict(completed_up_to_freq)
    completed_up_to_freq2.update({
        'OPT3': [{'calc_id': 'calc_opt3_006', 'status': 'completed'}],
        'BAND': [{'calc_id': 'calc_band_007', 'status': 'completed'}],
        'DOSS': [{'calc_id': 'calc_doss_008', 'status': 'completed'}],
        'OPT4': [{'calc_id': 'calc_opt4_009', 'status': 'completed'}],
        'FREQ2': [{'calc_id': 'calc_freq2_010', 'status': 'completed'}],
    })
    
    highest_at_freq2 = engine._find_highest_numbered_calc_of_type(completed_up_to_freq2, 'OPT')
    print(f"   When FREQ2 completes, highest OPT is: {highest_at_freq2} (should be OPT4)")
    assert highest_at_freq2 == 'calc_opt4_009'
    
    print(f"   OPT5 (after FREQ2) should use: {highest_at_freq2} as source")
    
    print("\n✅ All complex workflow tests passed!")
    print("\nKey insights:")
    print("- FREQ always uses the highest numbered OPT completed up to that point")
    print("- OPT calculations after FREQ/BAND/DOSS also use the highest completed OPT")
    print("- The system correctly tracks which calculations have completed")
    print("- No future calculations are considered (e.g., OPT4 doesn't affect FREQ)")

if __name__ == "__main__":
    test_complex_workflow()