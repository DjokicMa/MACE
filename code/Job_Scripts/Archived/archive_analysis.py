#!/usr/bin/env python3
"""
Script Analysis for Archiving
=============================
Identify scripts that can be archived vs those needed for active workflows.
"""

import os

# Core active scripts (definitely needed)
core_scripts = {
    'material_database.py',           # Core database system
    'enhanced_queue_manager.py',      # Main queue manager
    'crystal_property_extractor.py',  # Property extraction
    'workflow_executor.py',           # Workflow execution  
    'workflow_planner.py',            # Workflow planning
    'run_workflow.py',               # Main workflow interface
    'error_recovery.py',             # Error handling
    'workflow_engine.py',            # Workflow automation
    'formula_extractor.py',          # Formula extraction
    'dat_file_processor.py',         # DAT file processing
    'population_analysis_processor.py',  # Population analysis
    'input_settings_extractor.py',   # Input settings
    'submitcrystal23.py',            # SLURM submission
    'submit_prop.py',                # Properties submission
}

# Test/development scripts (archivable)
test_dev_scripts = {
    'integration_test.py',
    'test_shrink_fix.py', 
    'test_spin_polarized.py',
    'check_property_units.py',
    'fix_property_units.py',
    'fix_atomic_position_naming.py',
    'shrink_parser_fix.py',
    'additional_properties_analyzer.py'
}

# Legacy/superseded scripts (archivable)
legacy_scripts = {
    'crystal_queue_manager.py',      # Legacy queue manager (superseded by enhanced)
    'submitcrystal17.py',           # CRYSTAL17 submission (superseded by 23)
    'submit_prop_17.py',            # CRYSTAL17 properties (superseded)
    'monitor_workflow.py',          # Basic monitoring (superseded by material_monitor)
}

# Utility scripts (may be archivable)
utility_scripts = {
    'create_fresh_database.py',     # Database creation utility
    'database_status_report.py',   # Status reporting
    'show_properties.py',          # Property display
    'query_input_settings.py',     # Settings query
    'populate_completed_jobs.py',  # Population utility
    'crystal_file_manager.py',     # File management
    'material_monitor.py',         # Monitoring
    'error_detector.py',           # Error detection
    'copy_dependencies.py'         # Dependency copying
}

def main():
    all_scripts = set(os.listdir('.'))
    python_scripts = {f for f in all_scripts if f.endswith('.py')}

    print('📊 Script Analysis for Archiving:')
    print('=' * 50)

    print(f'\n🔧 Core Scripts (Keep Active): {len(core_scripts)}')
    for script in sorted(core_scripts):
        if script in python_scripts:
            print(f'  ✅ {script}')

    print(f'\n📦 Legacy/Superseded Scripts (Archive): {len(legacy_scripts)}')
    for script in sorted(legacy_scripts):
        if script in python_scripts:
            print(f'  📦 {script}')

    print(f'\n🧪 Test/Development Scripts (Archive): {len(test_dev_scripts)}') 
    for script in sorted(test_dev_scripts):
        if script in python_scripts:
            print(f'  📦 {script}')

    # Calculate archival candidates
    archival_candidates = set()
    archival_candidates.update(test_dev_scripts)
    archival_candidates.update(legacy_scripts)
    archival_candidates = archival_candidates & python_scripts

    print(f'\n📋 Total Archival Candidates: {len(archival_candidates)}')
    print('   These scripts can be moved to an Archived/ directory:')
    for script in sorted(archival_candidates):
        print(f'     📦 {script}')
    
    print(f'\n📈 Summary:')
    print(f'  Total Python scripts: {len(python_scripts)}')
    print(f'  Core scripts (keep active): {len(core_scripts & python_scripts)}')  
    print(f'  Archival candidates: {len(archival_candidates)}')
    print(f'  Utilities (review): {len(utility_scripts & python_scripts)}')

    return archival_candidates

if __name__ == "__main__":
    main()