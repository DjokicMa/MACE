#!/usr/bin/env python3
"""
Fix Workflow Sequence Execution
===============================
Fixes the issue where workflow engine ignores the planned sequence and 
uses hard-coded dependency rules instead.
"""

def _get_next_steps_from_sequence_fixed(current_index: int, planned_sequence: list, 
                                       completed_calc_type: str) -> list:
    """
    Get the next calculation steps from the planned sequence.
    
    This fixed version:
    1. Follows the exact sequence order when specified
    2. Only applies dependency rules for parallel execution
    
    Args:
        current_index: Current position in the sequence
        planned_sequence: The full planned calculation sequence
        completed_calc_type: The type of calculation that just completed
        
    Returns:
        List of calculation types that should be started next
    """
    if not planned_sequence or current_index >= len(planned_sequence) - 1:
        return []
    
    # The next step is simply the next item in the sequence
    next_index = current_index + 1
    if next_index < len(planned_sequence):
        next_calc = planned_sequence[next_index]
        
        # Check if there are parallel calculations at the same "level"
        # For example, BAND and DOSS often run in parallel after SP
        parallel_calcs = [next_calc]
        
        # Look ahead for calculations that can run in parallel
        # Only specific combinations run in parallel:
        # - BAND and DOSS after SP
        # - FREQ can run in parallel with SP after OPT
        if next_calc == "SP" and next_index + 1 < len(planned_sequence):
            if planned_sequence[next_index + 1] == "FREQ":
                # FREQ can run in parallel with SP after OPT
                parallel_calcs.append("FREQ")
        elif next_calc == "BAND" and next_index + 1 < len(planned_sequence):
            if planned_sequence[next_index + 1] == "DOSS":
                # BAND and DOSS run in parallel
                parallel_calcs.append("DOSS")
        elif next_calc == "DOSS" and next_index - 1 >= 0:
            if planned_sequence[next_index - 1] == "BAND":
                # If BAND was supposed to run but didn't, include it
                # This handles the case where we're at DOSS but BAND hasn't started yet
                return ["BAND", "DOSS"]
                
        return parallel_calcs
    
    return []


def test_sequence_logic():
    """Test the fixed sequence logic"""
    
    test_sequence = ["OPT", "OPT2", "SP", "OPT3", "SP2", "BAND", "DOSS", "FREQ"]
    
    test_cases = [
        (0, "OPT", ["OPT2"]),  # After OPT, should get OPT2
        (1, "OPT2", ["SP"]),   # After OPT2, should get SP  
        (2, "SP", ["OPT3"]),   # After SP, should get OPT3
        (3, "OPT3", ["SP2"]),  # After OPT3, should get SP2
        (4, "SP2", ["BAND", "DOSS"]),  # After SP2, should get BAND and DOSS in parallel
        (5, "BAND", []),       # After BAND, nothing (DOSS runs in parallel)
        (6, "DOSS", ["FREQ"]), # After DOSS, should get FREQ
        (7, "FREQ", []),       # After FREQ, nothing
    ]
    
    print("Testing workflow sequence logic:")
    print(f"Sequence: {test_sequence}")
    print("=" * 60)
    
    all_passed = True
    for current_index, completed_type, expected in test_cases:
        result = _get_next_steps_from_sequence_fixed(current_index, test_sequence, completed_type)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"{status} After {completed_type} (position {current_index})")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        print()
    
    # Test alternate sequence with FREQ in parallel with SP
    alt_sequence = ["OPT", "SP", "FREQ", "BAND", "DOSS", "OPT2"]
    print("\nTesting alternate sequence with parallel FREQ:")
    print(f"Sequence: {alt_sequence}")
    print("-" * 60)
    
    result = _get_next_steps_from_sequence_fixed(0, alt_sequence, "OPT")
    print(f"After OPT: {result}")
    print(f"Expected: ['SP', 'FREQ'] (parallel execution)")
    
    return all_passed


if __name__ == "__main__":
    if test_sequence_logic():
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        
    print("\nTo fix workflow_engine.py, the _get_next_steps_from_sequence method needs to be updated to:")
    print("1. Follow the exact sequence order")
    print("2. Only apply parallel execution for specific known combinations")
    print("3. Not use hardcoded dependency rules that override the planned sequence")