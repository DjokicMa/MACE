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
    
    # Define file groups and their source directories
    file_groups = {
        "Job Scripts": {
            "source_dir": script_dir,
            "files": [
                "enhanced_queue_manager.py",
                "material_database.py", 
                "workflow_engine.py",
                "error_recovery.py",
                "error_detector.py",
                "material_monitor.py",
                "crystal_file_manager.py",
                "populate_completed_jobs.py",
                "crystal_queue_manager.py",
                "input_settings_extractor.py",
                "query_input_settings.py", 
                "formula_extractor.py",
                "crystal_property_extractor.py",
                "workflows.yaml",
                "recovery_config.yaml"
            ]
        },
        "Crystal_To_CIF": {
            "source_dir": base_dir / "Crystal_To_CIF",
            "files": [
                "NewCifToD12.py",
                "CRYSTALOptToD12.py", 
                "d12creation.py"
            ]
        },
        "Creation_Scripts": {
            "source_dir": base_dir / "Creation_Scripts",
            "files": [
                "alldos.py",
                "create_band_d3.py"
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