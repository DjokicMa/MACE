#!/usr/bin/env python3
"""
Workflow Robustness Analysis
Analyzes how the workflow engine tracks and progresses through steps
"""

import json
from datetime import datetime
from pathlib import Path

# Define test scenarios
SCENARIOS = [
    {
        "name": "Simple Electronic Workflow",
        "sequence": ["OPT", "SP", "BAND", "DOSS"],
        "description": "Basic electronic structure workflow"
    },
    {
        "name": "Extended Workflow with FREQ",
        "sequence": ["OPT", "SP", "BAND", "DOSS", "FREQ", "OPT2", "SP2"],
        "description": "Extended workflow with frequency calculation and second optimization"
    },
    {
        "name": "Multi-OPT Workflow",
        "sequence": ["OPT", "OPT2", "OPT3", "SP", "BAND", "DOSS", "FREQ", "OPT4"],
        "description": "Complex workflow with multiple optimization steps"
    }
]

def analyze_workflow_behavior():
    """Analyze expected workflow behavior based on code inspection"""
    
    print("WORKFLOW ENGINE ROBUSTNESS ANALYSIS")
    print("=" * 70)
    print()
    
    print("KEY COMPONENTS:")
    print("-" * 40)
    print("1. Workflow Plan Storage:")
    print("   - Stored as JSON in workflow_configs/workflow_plan_*.json")
    print("   - Contains: workflow_id, workflow_sequence, step_configurations")
    print()
    
    print("2. Step Tracking Mechanism:")
    print("   - Each calculation stores workflow_id in settings")
    print("   - Step number determined from position in workflow_sequence")
    print("   - Uses get_workflow_step_number() to find correct step")
    print()
    
    print("3. Progression Logic (execute_workflow_step):")
    print("   - Triggered when calculation completes")
    print("   - Extracts workflow_id from completed calculation")
    print("   - Looks up planned sequence")
    print("   - Determines next steps based on current position")
    print()
    
    print("ROBUSTNESS FEATURES:")
    print("-" * 40)
    print("✓ Workflow ID propagation through calculation chain")
    print("✓ Step number determination from plan (not hardcoded)")
    print("✓ Fallback to default behavior if workflow_id lost")
    print("✓ Handles numbered calculation types (OPT2, SP2, etc.)")
    print("✓ Parallel step support (BAND & DOSS after SP)")
    print()
    
    print("POTENTIAL WEAKNESSES:")
    print("-" * 40)
    print("✗ Workflow ID can be lost if calculation fails before settings saved")
    print("✗ No persistent workflow state outside of calculation records")
    print("✗ Manual intervention calculations may not have workflow_id")
    print("✗ Dependency on file system for workflow plans")
    print()
    
    # Analyze each scenario
    for scenario in SCENARIOS:
        analyze_scenario(scenario)
        
def analyze_scenario(scenario):
    """Analyze expected behavior for a specific workflow scenario"""
    
    print(f"\nSCENARIO: {scenario['name']}")
    print("=" * 70)
    print(f"Sequence: {' → '.join(scenario['sequence'])}")
    print(f"Description: {scenario['description']}")
    print()
    
    # Expected behavior when all succeed
    print("EXPECTED BEHAVIOR (All Success):")
    print("-" * 40)
    
    sequence = scenario['sequence']
    for i, step in enumerate(sequence):
        if i == 0:
            print(f"1. {step} submitted manually with workflow_id")
        else:
            prev_step = sequence[i-1]
            print(f"{i+1}. {step} auto-generated after {prev_step} completes")
            
            # Special logic for certain transitions
            if prev_step == "OPT" and step == "SP":
                print(f"   - Uses CRYSTALOptToD12.py to extract geometry")
                print(f"   - Inherits settings from OPT")
            elif prev_step == "SP" and step in ["BAND", "DOSS"]:
                print(f"   - Uses wavefunction from SP")
                print(f"   - May generate both BAND and DOSS in parallel")
            elif step.startswith("OPT") and step != "OPT":
                print(f"   - Uses CRYSTALOptToD12.py with previous OPT")
                print(f"   - May have modified settings (basis, functional)")
            elif step == "FREQ":
                print(f"   - Uses highest numbered OPT for geometry")
                print(f"   - Requires tighter tolerances")
                
    print()
    
    # Expected behavior with failures
    print("EXPECTED BEHAVIOR (With Failures):")
    print("-" * 40)
    
    # Simulate failure at midpoint
    fail_index = len(sequence) // 2
    fail_step = sequence[fail_index]
    
    print(f"If {fail_step} fails:")
    print(f"- Workflow stops at step {fail_index + 1}")
    print(f"- No subsequent steps are generated")
    print(f"- Completed: {' → '.join(sequence[:fail_index])}")
    print(f"- Not started: {' → '.join(sequence[fail_index+1:])}")
    print()
    
    # Recovery options
    print("Recovery options:")
    print("1. Fix and resubmit failed calculation")
    print("2. Manually create next step with workflow_id")
    print("3. Use enhanced_queue_manager recovery features")
    print()
    
    # Edge cases
    print("EDGE CASES:")
    print("-" * 40)
    
    if "FREQ" in sequence:
        print("- FREQ requires completed OPT with .out file")
        print("- If multiple OPTs, uses highest numbered one")
        
    if any(s.endswith("2") for s in sequence):
        print("- Numbered calculations preserve step ordering")
        print("- Each gets unique step directory")
        
    if "BAND" in sequence and "DOSS" in sequence:
        print("- BAND and DOSS may run in parallel")
        print("- Both depend on same SP wavefunction")

def show_workflow_progression_logic():
    """Show the actual logic flow for workflow progression"""
    
    print("\n\nWORKFLOW PROGRESSION LOGIC FLOW:")
    print("=" * 70)
    
    print("""
    Calculation Completes
            ↓
    execute_workflow_step() called
            ↓
    Extract workflow_id from settings
            ↓
    Load workflow_sequence from plan
            ↓
    Find current position in sequence
            ↓
    Determine next steps
            ↓
    For each next step:
        - OPT/SP → use generate_numbered_calculation()
        - BAND/DOSS → use generate_property_calculation()
        - FREQ → use generate_freq_from_opt()
            ↓
    New calculations created with:
        - workflow_id propagated
        - correct step number
        - proper dependencies
    """)
    
    print("\nKEY FUNCTIONS:")
    print("-" * 40)
    print("- get_workflow_sequence(): Loads plan from JSON")
    print("- get_workflow_step_number(): Determines step number from sequence")
    print("- _find_calc_position_in_sequence(): Finds current position")
    print("- _get_next_steps_from_sequence(): Determines what comes next")
    print("- execute_workflow_step(): Main progression orchestrator")

def main():
    """Run the analysis"""
    analyze_workflow_behavior()
    show_workflow_progression_logic()
    
    print("\n\nCONCLUSION:")
    print("=" * 70)
    print("The workflow system is reasonably robust with:")
    print("- Clear progression logic based on workflow plans")
    print("- Proper step numbering from sequence position")
    print("- Workflow ID propagation through calculation chain")
    print("- Fallback behavior for missing workflow context")
    print()
    print("Main vulnerability is loss of workflow_id which breaks automation.")
    print("Recommend implementing workflow state persistence independent of calculations.")

if __name__ == "__main__":
    main()