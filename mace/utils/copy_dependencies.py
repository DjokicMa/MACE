#!/usr/bin/env python3
"""
Copy Workflow Dependencies
==========================
Standalone script to copy all required dependency scripts to a working directory.
This ensures all workflow components can find their dependencies locally.

Usage:
  python copy_dependencies.py [target_directory]
  
If no target directory is specified, uses current working directory.
"""

import os
import sys
import shutil
from pathlib import Path

def copy_dependencies(target_dir: str = "."):
    """Copy all workflow dependency scripts to target directory"""
    target_path = Path(target_dir).resolve()
    print(f"Copying workflow dependencies to: {target_path}")
    
    # Get source directories
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    # Define file groups and their source directories. Each name is the file's
    # current name (several were renamed when the scripts moved into the mace
    # package). recovery_config.yaml is not copied: the recovery engine reads
    # the shipped mace/config copy itself, and a copy in the job directory
    # would take precedence over it.
    repo_dir = base_dir.parent
    file_groups = {
        "Job Scripts": {
            "source_dir": base_dir,
            "files": [
                "enhanced_queue_manager.py",
                "run_workflow.py",
                "run_mace.py",  # run_workflow.py imports it by bare name
                "material_monitor.py",
            ]
        },
        "Database": {
            "source_dir": base_dir / "database",
            "files": [
                "materials.py",  # was material_database.py
                "populate_completed_jobs.py",
            ]
        },
        "Database utilities": {
            "source_dir": base_dir / "database" / "utils",
            "files": [
                "create_fresh_database.py",
                "database_status_report.py",
            ]
        },
        "Database queries": {
            "source_dir": base_dir / "database" / "query",
            "files": [
                "queries.py",  # was query_input_settings.py
            ]
        },
        "Queue": {
            "source_dir": base_dir / "queue",
            "files": [
                "queue_lock_manager.py",
            ]
        },
        "Workflow": {
            "source_dir": base_dir / "workflow",
            "files": [
                "engine.py",  # was workflow_engine.py
                "planner.py",  # was workflow_planner.py
                "executor.py",  # was workflow_executor.py
            ]
        },
        "Recovery": {
            "source_dir": base_dir / "recovery",
            "files": [
                "recovery.py",  # was error_recovery.py
                "detector.py",  # was error_detector.py
            ]
        },
        "Utilities": {
            "source_dir": script_dir,
            "files": [
                "file_manager.py",  # was crystal_file_manager.py
                "settings_extractor.py",  # was input_settings_extractor.py
                "formula_extractor.py",
                "property_extractor.py",  # was crystal_property_extractor.py
                "advanced_electronic_analyzer.py",
                "dat_file_processor.py",
                "population_analysis_processor.py",
                "show_properties.py",
                "copy_dependencies.py"
            ]
        },
        "Crystal_d12": {
            "source_dir": repo_dir / "Crystal_d12",
            "files": [
                "NewCifToD12.py",
                "CRYSTALOptToD12.py",
                "d12_calc_basic.py",
                "d12_calc_freq.py",
                "d12_config.py",
                "d12_constants.py",
                "d12_interactive.py",
                "d12_parsers.py",
                "d12_writer.py",
                "menu_nav.py",
                "spglib_compat.py",
                "soc_ecp.py"
            ]
        },
        "Crystal_d3": {
            "source_dir": repo_dir / "Crystal_d3",
            "files": [
                # Legacy scripts removed - use CRYSTALOptToD3.py instead
                # "alldos.py",  # Deprecated
                # "create_band_d3.py",  # Deprecated
                "CRYSTALOptToD3.py",
                "d3_interactive.py",
                "d3_config.py",
                "d3_kpoints.py",
                "seekpath_interface.py"
            ]
        }
    }

    copied_count = 0
    missing_count = 0
    
    for group_name, group_info in file_groups.items():
        print(f"\n{group_name}:")
        source_dir = group_info["source_dir"]
        
        for filename in group_info["files"]:
            source_file = source_dir / filename
            dest_file = target_path / filename
            
            if source_file.exists():
                try:
                    shutil.copy2(source_file, dest_file)
                    print(f"  ✓ Copied: {filename}")
                    copied_count += 1
                    
                    # Make Python scripts executable
                    if filename.endswith('.py'):
                        dest_file.chmod(0o755)
                        
                except Exception as e:
                    print(f"  ✗ Error copying {filename}: {e}")
            else:
                print(f"  ✗ Missing: {filename}")
                missing_count += 1
    
    print(f"\nSummary:")
    print(f"  Copied: {copied_count} files")
    print(f"  Missing: {missing_count} files")
    
    if missing_count == 0:
        print(f"  ✓ All dependencies successfully copied to {target_path}")
        print(f"\nYou can now run workflow operations from {target_path}")
        print(f"  Example: python enhanced_queue_manager.py --callback-mode completion")
    else:
        print(f"  ⚠ Some dependencies are missing from the source repository")
    
    return copied_count, missing_count

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    copy_dependencies(target_dir)
