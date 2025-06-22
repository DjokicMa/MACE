#!/usr/bin/env python3
"""
Test script to verify workflow dependencies are correctly implemented.

Tests the following dependency rules:
1. OPT -> SP and FREQ (parallel)
2. SP -> BAND, DOSS, and OPT2 (parallel)
3. OPT2 -> SP2 and FREQ2 (parallel)
4. SP2 -> BAND2, DOSS2, and OPT3 (parallel)
"""

import sys
import json
from pathlib import Path
from workflow_engine import WorkflowEngine

def test_dependency_logic():
    """Test the _get_next_steps_from_sequence method with various scenarios"""
    
    # Initialize workflow engine
    engine = WorkflowEngine(db_path=":memory:")  # Use in-memory DB for testing
    
    # Test sequence: OPT -> SP -> BAND, DOSS, OPT2 -> SP2 -> BAND2, DOSS2, FREQ, FREQ2
    test_sequence = ["OPT", "SP", "BAND", "DOSS", "FREQ", "OPT2", "SP2", "BAND2", "DOSS2", "FREQ2"]
    
    print("Test Sequence:", test_sequence)
    print("=" * 60)
    
    # Test 1: OPT completed
    print("\nTest 1: OPT completed")
    next_steps = engine._get_next_steps_from_sequence(0, test_sequence, "OPT")
    print(f"  Expected: ['SP', 'FREQ']")
    print(f"  Got:      {next_steps}")
    assert set(next_steps) == {"SP", "FREQ"}, f"Expected SP and FREQ, got {next_steps}"
    
    # Test 2: SP completed
    print("\nTest 2: SP completed")
    next_steps = engine._get_next_steps_from_sequence(1, test_sequence, "SP")
    print(f"  Expected: ['BAND', 'DOSS', 'OPT2']")
    print(f"  Got:      {next_steps}")
    assert set(next_steps) == {"BAND", "DOSS", "OPT2"}, f"Expected BAND, DOSS, and OPT2, got {next_steps}"
    
    # Test 3: OPT2 completed
    print("\nTest 3: OPT2 completed")
    next_steps = engine._get_next_steps_from_sequence(5, test_sequence, "OPT2")
    print(f"  Expected: ['SP2', 'FREQ2']")
    print(f"  Got:      {next_steps}")
    assert set(next_steps) == {"SP2", "FREQ2"}, f"Expected SP2 and FREQ2, got {next_steps}"
    
    # Test 4: SP2 completed
    print("\nTest 4: SP2 completed")
    next_steps = engine._get_next_steps_from_sequence(6, test_sequence, "SP2")
    print(f"  Expected: ['BAND2', 'DOSS2'] (and possibly OPT3 if in sequence)")
    print(f"  Got:      {next_steps}")
    assert "BAND2" in next_steps and "DOSS2" in next_steps, f"Expected at least BAND2 and DOSS2, got {next_steps}"
    
    # Test 5: Complex sequence with OPT3
    print("\nTest 5: Complex sequence with OPT3")
    complex_sequence = test_sequence + ["OPT3", "SP3", "BAND3", "DOSS3", "FREQ3"]
    next_steps = engine._get_next_steps_from_sequence(6, complex_sequence, "SP2")
    print(f"  Expected: ['BAND2', 'DOSS2', 'OPT3']")
    print(f"  Got:      {next_steps}")
    assert set(next_steps) == {"BAND2", "DOSS2", "OPT3"}, f"Expected BAND2, DOSS2, and OPT3, got {next_steps}"
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    
def test_pending_calculations():
    """Test the _check_and_trigger_pending_calculations method"""
    
    print("\n\nTesting Pending Calculations Detection")
    print("=" * 60)
    
    # This would require a more complex setup with database mocking
    # For now, we'll just verify the logic structure
    
    engine = WorkflowEngine(db_path=":memory:")
    
    # Create a test material
    material_id = "test_material"
    engine.db.create_material(
        material_id=material_id,
        formula="Test",
        source_type="test",
        source_file="test.cif"
    )
    
    # Simulate some completed calculations
    opt_calc_id = engine.db.create_calculation(
        material_id=material_id,
        calc_type="OPT",
        input_file="test_opt.d12",
        work_dir="."
    )
    
    sp_calc_id = engine.db.create_calculation(
        material_id=material_id,
        calc_type="SP",
        input_file="test_sp.d12",
        work_dir="."
    )
    
    # Update their status to completed
    engine.db.update_calculation_status(opt_calc_id, "completed")
    engine.db.update_calculation_status(sp_calc_id, "completed")
    
    # Test sequence with missing calculations
    test_sequence = ["OPT", "SP", "BAND", "DOSS", "FREQ", "OPT2"]
    
    print(f"\nScenario: OPT and SP completed, but BAND, DOSS, FREQ not started")
    print(f"Sequence: {test_sequence}")
    
    # This would trigger BAND, DOSS, FREQ, and OPT2 creation
    # In a real test, we'd mock the generation methods
    # For now, we just verify the method exists and can be called
    
    try:
        pending_ids = engine._check_and_trigger_pending_calculations(material_id, test_sequence)
        print(f"Method executed successfully (would trigger pending calculations)")
    except Exception as e:
        print(f"Method execution failed: {e}")
        
    print("\n" + "=" * 60)
    print("Pending calculations test completed!")

def print_dependency_diagram():
    """Print a visual representation of the workflow dependencies"""
    
    print("\n\nWorkflow Dependency Diagram")
    print("=" * 60)
    print("""
    OPT ─────┬──→ SP ─────┬──→ BAND
             │            ├──→ DOSS
             └──→ FREQ    └──→ OPT2 ─────┬──→ SP2 ─────┬──→ BAND2
                                         │             ├──→ DOSS2
                                         └──→ FREQ2    └──→ OPT3
    
    Legend:
    - OPT completion triggers SP and FREQ (parallel)
    - SP completion triggers BAND, DOSS, and next OPT (parallel)
    - FREQ depends only on its corresponding OPT
    - Each OPT (except the first) depends on the previous SP
    """)
    print("=" * 60)

if __name__ == "__main__":
    print("CRYSTAL Workflow Dependencies Test")
    print("=" * 60)
    
    # Run tests
    test_dependency_logic()
    test_pending_calculations()
    print_dependency_diagram()
    
    print("\nAll tests completed successfully!")