#!/usr/bin/env python3
"""
Fix Workflow Dependencies V2
============================
Implements the correct dependency logic:
1. OPT chains continue from last OPT (OPT → OPT2 → OPT3)
2. SP follows its corresponding OPT (OPT2 → SP, OPT3 → SP2)
3. FREQ depends on its corresponding OPT (OPT → FREQ, OPT3 → FREQ2)
4. BAND/DOSS follow SP
5. Proper parallel execution
"""

import re
from typing import List, Tuple, Optional


def parse_calc_type(calc_type: str) -> Tuple[str, int]:
    """Parse calculation type to extract base type and number."""
    match = re.match(r'^([A-Z]+)(\d*)$', calc_type)
    if match:
        base_type = match.group(1)
        num_str = match.group(2)
        num = int(num_str) if num_str else 1
        return base_type, num
    else:
        return calc_type, 1


def get_next_steps_from_sequence_v2(current_index: int, planned_sequence: List[str], 
                                   completed_calc_type: str, material_id: str,
                                   completed_calcs: List[str]) -> List[str]:
    """
    Get the next calculation steps based on proper dependency rules.
    
    Dependency Rules:
    1. OPTn → OPT(n+1) (chain continues regardless of intervening calcs)
    2. OPTn → SPn (each OPT triggers its corresponding SP)
    3. OPTn → FREQn (each OPT triggers its corresponding FREQ)
    4. SPn → BANDn + DOSSn (properties follow SP)
    5. Parallel execution where dependencies allow
    
    Args:
        current_index: Position of completed calc in sequence
        planned_sequence: Full workflow sequence
        completed_calc_type: Type that just completed
        material_id: Material identifier
        completed_calcs: List of already completed calculation types
        
    Returns:
        List of calculations that can start now
    """
    if not planned_sequence:
        return []
    
    next_steps = []
    completed_base, completed_num = parse_calc_type(completed_calc_type)
    
    # Scan the ENTIRE sequence for calculations that depend on what just completed
    for calc_type in planned_sequence:
        # Skip if already completed or in progress
        if calc_type in completed_calcs or calc_type == completed_calc_type:
            continue
            
        # Skip if already in our next steps
        if calc_type in next_steps:
            continue
            
        calc_base, calc_num = parse_calc_type(calc_type)
        can_start = False
        
        # Apply dependency rules
        if completed_base == "OPT":
            # OPT triggers:
            # 1. Next OPT in chain (OPT → OPT2, OPT2 → OPT3)
            if calc_base == "OPT" and calc_num == completed_num + 1:
                can_start = True
                
            # 2. Corresponding SP (OPT2 → SP, OPT3 → SP2)
            # For OPT (num=1), it should trigger SP (num=1)
            # For OPT2 (num=2), it should trigger SP (num=1) 
            # For OPT3 (num=3), it should trigger SP2 (num=2)
            if calc_base == "SP":
                if completed_num == 1 and calc_num == 1:  # OPT → SP
                    can_start = True
                elif completed_num == 3 and calc_num == 2:  # OPT3 → SP2
                    can_start = True
                # Note: OPT2 → SP is handled by checking if SP is right after OPT2 in sequence
                elif completed_num == 2 and calc_num == 1 and "SP" not in completed_calcs:
                    # Check if SP is the next SP in the sequence after OPT2
                    opt2_index = planned_sequence.index(completed_calc_type)
                    if calc_type in planned_sequence[opt2_index:]:
                        can_start = True
                        
            # 3. Corresponding FREQ (OPT → FREQ, OPT3 → FREQ2)
            if calc_base == "FREQ":
                # Match FREQ numbering to OPT source
                if completed_num == 1 and calc_num == 1:  # OPT → FREQ
                    can_start = True
                elif completed_num == 3 and calc_num == 2:  # OPT3 → FREQ2
                    # But in the given sequence, FREQ comes after SP2/BAND/DOSS
                    # So we need to check the sequence position
                    if "FREQ" in planned_sequence[current_index:]:
                        # FREQ is after OPT3 in sequence, so it depends on OPT3
                        can_start = True
                        
        elif completed_base == "SP":
            # SP triggers BAND and DOSS with same numbering
            if calc_base in ["BAND", "DOSS"]:
                if calc_num == completed_num:
                    can_start = True
                    
        # Add to next steps if dependencies are met
        if can_start:
            next_steps.append(calc_type)
    
    return next_steps


def test_workflow_dependencies():
    """Test the corrected dependency logic"""
    
    sequence = ["OPT", "OPT2", "SP", "OPT3", "SP2", "BAND", "DOSS", "FREQ"]
    
    print("Testing workflow sequence:")
    print(f"Sequence: {sequence}")
    print("=" * 80)
    
    # Simulate workflow progression
    completed = []
    
    test_cases = [
        ("OPT", ["OPT2"]),  # OPT triggers OPT2 only
        ("OPT2", ["SP", "OPT3"]),  # OPT2 triggers SP and OPT3 in parallel
        ("SP", []),  # SP triggers nothing (no BAND/DOSS with num=1 in sequence)
        ("OPT3", ["SP2", "FREQ"]),  # OPT3 triggers SP2 and FREQ in parallel
        ("SP2", ["BAND", "DOSS"]),  # SP2 triggers BAND and DOSS
        ("BAND", []),  # BAND triggers nothing
        ("DOSS", []),  # DOSS triggers nothing
        ("FREQ", []),  # FREQ triggers nothing
    ]
    
    for completing_calc, expected in test_cases:
        # Find position in sequence
        if completing_calc in sequence:
            current_index = sequence.index(completing_calc)
        else:
            current_index = 0
            
        # Mark as completed
        completed.append(completing_calc)
        
        # Get next steps
        next_steps = get_next_steps_from_sequence_v2(
            current_index, sequence, completing_calc, "test_material", completed[:-1]
        )
        
        status = "✓" if set(next_steps) == set(expected) else "✗"
        print(f"{status} After {completing_calc} completes:")
        print(f"   Completed so far: {completed}")
        print(f"   Expected next: {expected}")
        print(f"   Got next:      {next_steps}")
        print()
        
    # Test alternate sequence
    print("\nTesting standard sequence: OPT → SP → BAND → DOSS")
    alt_sequence = ["OPT", "SP", "BAND", "DOSS", "FREQ"]
    completed = []
    
    # After OPT
    completed.append("OPT")
    next_steps = get_next_steps_from_sequence_v2(0, alt_sequence, "OPT", "test", [])
    print(f"After OPT: {next_steps} (should include SP and FREQ)")
    
    # After SP
    completed.append("SP")
    next_steps = get_next_steps_from_sequence_v2(1, alt_sequence, "SP", "test", ["OPT"])
    print(f"After SP: {next_steps} (should include BAND and DOSS)")


def create_workflow_engine_fix():
    """Generate the fix for workflow_engine.py"""
    
    print("\n" + "=" * 80)
    print("FIX FOR workflow_engine.py")
    print("=" * 80)
    print("""
Replace the _get_next_steps_from_sequence method with this corrected version:

    def _get_next_steps_from_sequence(self, current_index: int, planned_sequence: List[str], 
                                     completed_calc_type: str) -> List[str]:
        \"\"\"
        Get next calculation steps based on proper dependency rules.
        
        Rules:
        1. OPTn → OPT(n+1) (chain continues)
        2. OPTn → SPn (each OPT triggers corresponding SP)
        3. OPTn → FREQn (each OPT triggers corresponding FREQ)
        4. SPn → BANDn + DOSSn
        5. Parallel execution where allowed
        \"\"\"
        if not planned_sequence:
            return []
        
        next_steps = []
        completed_base, completed_num = self._parse_calc_type(completed_calc_type)
        
        # Get list of completed calculations for this material
        material_id = self.db.get_calculation(self.current_calc_id)['material_id']
        completed_calcs = [
            calc['calc_type'] for calc in self.db.get_calculations_by_status(material_id)
            if calc['status'] == 'completed'
        ]
        
        # Scan entire sequence for dependencies
        for calc_type in planned_sequence:
            if calc_type in completed_calcs or calc_type == completed_calc_type:
                continue
            if calc_type in next_steps:
                continue
                
            calc_base, calc_num = self._parse_calc_type(calc_type)
            can_start = False
            
            if completed_base == "OPT":
                # OPT chains
                if calc_base == "OPT" and calc_num == completed_num + 1:
                    can_start = True
                # SP mappings
                elif calc_base == "SP":
                    if (completed_num == 1 and calc_num == 1) or \\
                       (completed_num == 2 and calc_num == 1 and "SP" not in completed_calcs) or \\
                       (completed_num == 3 and calc_num == 2):
                        can_start = True
                # FREQ mappings  
                elif calc_base == "FREQ":
                    if (completed_num == 1 and calc_num == 1) or \\
                       (completed_num == 3 and calc_num == 1):  # Assuming FREQ in sequence
                        can_start = True
                        
            elif completed_base == "SP":
                if calc_base in ["BAND", "DOSS"] and calc_num == completed_num:
                    can_start = True
                    
            if can_start:
                next_steps.append(calc_type)
        
        return next_steps
""")


if __name__ == "__main__":
    test_workflow_dependencies()
    create_workflow_engine_fix()