#!/usr/bin/env python3
"""
Test script for numbered workflow calculations
"""

from workflow_engine import WorkflowEngine

def test_parse_calc_type():
    """Test calculation type parsing"""
    we = WorkflowEngine()
    
    test_cases = [
        ("OPT", ("OPT", 1)),
        ("OPT2", ("OPT", 2)),
        ("OPT3", ("OPT", 3)),
        ("SP", ("SP", 1)),
        ("SP2", ("SP", 2)),
        ("BAND", ("BAND", 1)),
        ("BAND2", ("BAND", 2)),
        ("DOSS", ("DOSS", 1)),
        ("DOSS3", ("DOSS", 3)),
        ("FREQ", ("FREQ", 1)),
    ]
    
    print("Testing calculation type parsing:")
    for input_type, expected in test_cases:
        result = we._parse_calc_type(input_type)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {input_type} -> {result} (expected {expected})")

def test_workflow_sequences():
    """Test various workflow sequences"""
    we = WorkflowEngine()
    
    # Test finding next steps
    sequences = [
        ["OPT", "SP", "BAND", "DOSS"],
        ["OPT", "OPT2", "SP", "BAND", "DOSS"],
        ["OPT", "SP", "BAND", "DOSS", "OPT2", "OPT3", "SP2", "BAND2", "DOSS2", "FREQ"],
        ["SP", "BAND", "DOSS"],  # Starting with SP
        ["OPT", "SP", "OPT2", "SP2", "BAND", "DOSS", "BAND2", "DOSS2"],
    ]
    
    print("\nTesting workflow sequences:")
    for seq in sequences:
        print(f"\n  Sequence: {' → '.join(seq)}")
        
        # Test getting next steps at various positions
        for i in range(len(seq)):
            next_steps = we._get_next_steps_from_sequence(i, seq)
            if next_steps:
                print(f"    After {seq[i]}: Next steps = {next_steps}")

def test_get_next_calc_suffix():
    """Test suffix generation for numbered calculations"""
    from pathlib import Path
    import tempfile
    
    we = WorkflowEngine()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        workflow_base = Path(tmpdir)
        
        # Create some test directories
        (workflow_base / "step_001_OPT" / "mat_1_dia_opt").mkdir(parents=True)
        (workflow_base / "step_002_SP" / "mat_1_dia_sp").mkdir(parents=True)
        (workflow_base / "step_003_OPT" / "mat_1_dia_opt2").mkdir(parents=True)
        
        print("\nTesting suffix generation:")
        
        # Test getting next suffix
        test_cases = [
            ("mat_1_dia", "OPT", "_opt3"),  # Should be _opt3 since _opt and _opt2 exist
            ("mat_1_dia", "SP", "_sp2"),     # Should be _sp2 since _sp exists
            ("mat_1_dia", "BAND", "_band"),  # Should be _band since none exist
            ("mat_2_dia", "OPT", "_opt"),    # Should be _opt for new material
        ]
        
        for core_name, calc_type, expected in test_cases:
            result = we.get_next_calc_suffix(core_name, calc_type, workflow_base)
            status = "✓" if result == expected else "✗"
            print(f"  {status} {core_name} + {calc_type} -> {result} (expected {expected})")

if __name__ == "__main__":
    test_parse_calc_type()
    test_workflow_sequences()
    test_get_next_calc_suffix()