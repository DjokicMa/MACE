#!/usr/bin/env python3
"""
Improved workflow continuation that respects critical dependencies
"""

# Define which calculations are optional (can fail without blocking workflow)
OPTIONAL_CALC_TYPES = {'BAND', 'DOSS', 'FREQ'}

# Define what each calculation type depends on
CALCULATION_DEPENDENCIES = {
    # Property calculations
    'BAND': {'SP'},     # BAND needs wavefunction
    'DOSS': {'SP'},     # DOSS needs wavefunction
    
    # Sequential optimizations
    'OPT2': {'OPT'},    # OPT2 needs OPT geometry
    'OPT3': {'OPT2'},   # OPT3 needs OPT2 geometry
    'OPT4': {'OPT3'},   # And so on...
    'OPT5': {'OPT4'},
    
    # SP calculations need geometry from their corresponding OPT
    'SP': {'OPT'},
    'SP2': {'OPT2'},
    'SP3': {'OPT3'},
    
    # FREQ calculations need optimized geometry
    'FREQ': {'OPT'},
    'FREQ2': {'OPT2'},
    'FREQ3': {'OPT3'},
}

def is_calculation_optional(calc_type: str) -> bool:
    """Check if a calculation type is optional"""
    base_type, _ = parse_calc_type(calc_type)
    return base_type in OPTIONAL_CALC_TYPES

def get_dependencies(calc_type: str) -> set:
    """Get dependencies for a calculation type"""
    base_type, num = parse_calc_type(calc_type)
    
    # Handle numbered variants
    if num > 1:
        # For numbered variants, adjust dependencies
        if base_type in ['BAND', 'DOSS']:
            return {f'SP{num}' if num > 1 else 'SP'}
        elif base_type == 'SP':
            return {f'OPT{num}' if num > 1 else 'OPT'}
        elif base_type == 'FREQ':
            return {f'OPT{num}' if num > 1 else 'OPT'}
        elif base_type == 'OPT' and num > 2:
            return {f'OPT{num-1}'}
    
    # Default dependencies
    return CALCULATION_DEPENDENCIES.get(base_type, set())

def check_dependencies_met(calc_type: str, completed_calcs: set) -> tuple[bool, str]:
    """
    Check if dependencies for a calculation are met
    Returns (success, missing_dependency)
    """
    dependencies = get_dependencies(calc_type)
    
    for dep in dependencies:
        # Check both base type and numbered variants
        base_dep, dep_num = parse_calc_type(dep)
        
        # Check if exact dependency is completed
        if dep in completed_calcs:
            continue
            
        # For some dependencies, a higher numbered variant is acceptable
        # e.g., if we need OPT but have OPT2, that's OK
        found = False
        if base_dep == 'OPT':
            for completed in completed_calcs:
                if completed.startswith('OPT'):
                    found = True
                    break
        
        if not found:
            return False, dep
    
    return True, None

def improved_execute_workflow_step(self, material_id: str, completed_calc_id: str) -> list[str]:
    """
    Improved version that respects dependencies
    """
    # ... initialization code ...
    
    # Track what calculations have been completed successfully
    completed_calcs = set()
    all_calcs = self.db.get_calculations_by_status(material_id=material_id)
    for calc in all_calcs:
        if calc['status'] == 'completed':
            completed_calcs.add(calc['calc_type'])
    
    # Track failed generations in this run
    failed_generations = set()
    
    # Generate calculations for all next steps
    for next_calc_type in next_steps:
        next_base_type, next_num = self._parse_calc_type(next_calc_type)
        
        # Check dependencies first
        deps_met, missing_dep = check_dependencies_met(next_calc_type, completed_calcs)
        
        if not deps_met:
            print(f"Skipping {next_calc_type} - missing dependency: {missing_dep}")
            continue
        
        # Check if a dependency failed in this run
        skip = False
        for dep in get_dependencies(next_calc_type):
            if dep in failed_generations:
                print(f"Skipping {next_calc_type} - dependency {dep} failed to generate")
                skip = True
                break
        
        if skip:
            continue
        
        try:
            # Attempt generation based on type
            calc_id = None
            
            if next_base_type == "DOSS":
                print(f"Generating {next_calc_type} from planned sequence...")
                calc_id = self.generate_property_calculation(completed_calc_id, next_calc_type)
                
            elif next_base_type == "BAND":
                print(f"Generating {next_calc_type} from planned sequence...")
                calc_id = self.generate_property_calculation(completed_calc_id, next_calc_type)
                
            # ... other calculation types ...
            
            if calc_id:
                new_calc_ids.append(calc_id)
                print(f"✓ Successfully generated {next_calc_type}")
            else:
                # Generation returned None (failed)
                handle_generation_failure(next_calc_type, next_base_type, failed_generations)
                
        except Exception as e:
            print(f"Exception generating {next_calc_type}: {e}")
            handle_generation_failure(next_calc_type, next_base_type, failed_generations)
    
    return new_calc_ids

def handle_generation_failure(calc_type: str, base_type: str, failed_generations: set):
    """Handle a failed calculation generation"""
    failed_generations.add(calc_type)
    
    if is_calculation_optional(calc_type):
        print(f"Optional {calc_type} generation failed, continuing with other calculations...")
    else:
        print(f"CRITICAL: {calc_type} generation failed!")
        print(f"  This may block dependent calculations.")
        # Could implement notification system here

def parse_calc_type(calc_type: str) -> tuple[str, int]:
    """Parse calculation type into base and number"""
    import re
    match = re.match(r'^([A-Z]+)(\d*)$', calc_type)
    if match:
        base = match.group(1)
        num = int(match.group(2)) if match.group(2) else 1
        return base, num
    return calc_type, 1

# Example workflow execution with the improved logic:
print("\nEXAMPLE WORKFLOW SCENARIOS")
print("=" * 50)

print("\nScenario 1: SP fails (critical for BAND/DOSS)")
print("Completed: [OPT]")
print("Next steps: [SP, BAND, DOSS]")
print("Result:")
print("  - Generate SP → FAILS")
print("  - Skip BAND (depends on SP)")
print("  - Skip DOSS (depends on SP)")
print("  - Workflow continues to next independent calc")

print("\nScenario 2: BAND fails (optional)")
print("Completed: [OPT, SP]")
print("Next steps: [BAND, DOSS, FREQ]")
print("Result:")
print("  - Generate BAND → FAILS (but it's optional)")
print("  - Generate DOSS → SUCCESS")
print("  - Generate FREQ → SUCCESS")

print("\nScenario 3: OPT2 fails (critical for OPT3)")
print("Completed: [OPT, SP, BAND, DOSS]")
print("Next steps: [OPT2, OPT3, SP2]")
print("Result:")
print("  - Generate OPT2 → FAILS")
print("  - Skip OPT3 (depends on OPT2)")
print("  - Skip SP2 (depends on OPT2)")
print("  - But FREQ could still run (depends only on OPT)")

print("\n" + "=" * 50)