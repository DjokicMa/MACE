#!/usr/bin/env python3
"""
Fix Existing SLURM Scripts in Current Workflow
==============================================
This script updates the queue manager paths in existing SLURM scripts
to point to the correct location (../../../enhanced_queue_manager.py)
"""

import os
import re
from pathlib import Path

def fix_slurm_scripts():
    """Fix existing SLURM scripts to use correct queue manager path"""
    
    # Look for workflow directory in current working directory
    current_dir = Path.cwd()
    workflow_outputs = current_dir / "workflow_outputs"
    
    if not workflow_outputs.exists():
        print("❌ No workflow_outputs directory found in current directory")
        print(f"Current directory: {current_dir}")
        print("Please run this script from your ~/test directory")
        return
    
    # Find workflow directories
    workflow_dirs = list(workflow_outputs.glob("workflow_*"))
    if not workflow_dirs:
        print("❌ No workflow directories found")
        return
    
    latest_workflow = max(workflow_dirs, key=lambda x: x.stat().st_mtime)
    print(f"🔧 Fixing SLURM scripts in: {latest_workflow.name}")
    
    # Find all SLURM scripts
    slurm_scripts = list(latest_workflow.rglob("*.sh"))
    
    if not slurm_scripts:
        print("❌ No SLURM scripts found")
        return
    
    print(f"📝 Found {len(slurm_scripts)} SLURM scripts to fix")
    
    # Define the old and new queue manager sections
    old_pattern = re.compile(
        r'# ADDED: Auto-submit new jobs when this one completes\s*\n'
        r'if \[ -f \$DIR/enhanced_queue_manager\.py \]; then\s*\n'
        r'    cd \$DIR\s*\n'
        r'    python enhanced_queue_manager\.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion\s*\n'
        r'elif \[ -f \$DIR/crystal_queue_manager\.py \]; then\s*\n'
        r'    cd \$DIR\s*\n'
        r'    \./crystal_queue_manager\.py  --max-jobs 250 --reserve 30 --max-submit 5\s*\n'
        r'fi',
        re.MULTILINE
    )
    
    new_section = '''# ADDED: Auto-submit new jobs when this one completes
# Queue manager is in the base working directory (4 levels up from material directory)
# Current location: ~/test/workflow_outputs/workflow_ID/step_XXX_TYPE/material_name/
# Queue manager location: ~/test/enhanced_queue_manager.py (../../../../enhanced_queue_manager.py)

if [ -f ../../../../enhanced_queue_manager.py ]; then
    echo "Found enhanced_queue_manager.py in base directory (../../../../)"
    cd ../../../../
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
elif [ -f ../../../../crystal_queue_manager.py ]; then
    echo "Found crystal_queue_manager.py in base directory (../../../../)"
    cd ../../../../
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
else
    echo "Warning: No queue manager found in base directory (../../../../)"
    echo "Expected location: ../../../../enhanced_queue_manager.py"
    echo "Current working directory: $(pwd)"
    echo "Listing base directory:"
    ls -la ../../../../ | grep -E "(enhanced_queue_manager|crystal_queue_manager)"
fi'''
    
    fixed_count = 0
    
    for script_file in slurm_scripts:
        try:
            # Read the script
            with open(script_file, 'r') as f:
                content = f.read()
            
            # Check if it has the old pattern
            if old_pattern.search(content):
                # Replace with new pattern
                new_content = old_pattern.sub(new_section, content)
                
                # Write back the fixed script
                with open(script_file, 'w') as f:
                    f.write(new_content)
                
                print(f"  ✅ Fixed: {script_file.relative_to(workflow_outputs)}")
                fixed_count += 1
            else:
                print(f"  ➖ No fix needed: {script_file.relative_to(workflow_outputs)}")
                
        except Exception as e:
            print(f"  ❌ Error fixing {script_file}: {e}")
    
    print(f"\n✅ Fixed {fixed_count} SLURM scripts")
    
    if fixed_count > 0:
        print(f"\n🎯 Next Steps:")
        print(f"1. Your existing SLURM scripts now point to ../../../enhanced_queue_manager.py")
        print(f"2. When jobs complete, they'll find the queue manager in ~/test/")
        print(f"3. SP step should automatically trigger when OPT jobs finish")
        print(f"\nTo test immediately:")
        print(f"cd {latest_workflow}/step_001_OPT/1_dia_opt/")
        print(f"bash 1_dia_opt.sh  # (if OPT is completed)")

if __name__ == "__main__":
    fix_slurm_scripts()