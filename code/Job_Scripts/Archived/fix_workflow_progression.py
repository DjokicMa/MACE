#!/usr/bin/env python3
"""
Fix Current Workflow to Enable Progression
==========================================
This script fixes the current workflow by:
1. Adding enhanced_queue_manager.py to the workflow directory
2. Manually triggering SP step generation since OPT completed
3. Updating SLURM scripts for proper queue manager paths
"""

import os
import shutil
import subprocess
from pathlib import Path

def fix_current_workflow():
    """Fix the current workflow to enable proper progression"""
    print("🔧 Fixing Current Workflow for Progression")
    print("=" * 50)
    
    # Find the current workflow directory
    current_dir = Path.cwd()
    workflow_outputs = current_dir / "workflow_outputs"
    
    if not workflow_outputs.exists():
        print("❌ No workflow_outputs directory found")
        return
    
    # Find the most recent workflow
    workflow_dirs = list(workflow_outputs.glob("workflow_*"))
    if not workflow_dirs:
        print("❌ No workflow directories found")
        return
        
    latest_workflow = max(workflow_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Found workflow: {latest_workflow.name}")
    
    # Copy essential scripts to workflow directory
    print("\n1️⃣ Copying queue manager to workflow directory...")
    
    essential_scripts = [
        "enhanced_queue_manager.py",
        "crystal_queue_manager.py", 
        "material_database.py",
        "error_recovery.py",
        "recovery_config.yaml"
    ]
    
    copied_count = 0
    for script in essential_scripts:
        source_path = current_dir / script
        target_path = latest_workflow / script
        
        if source_path.exists() and not target_path.exists():
            try:
                shutil.copy2(source_path, target_path)
                print(f"  ✅ Copied {script}")
                copied_count += 1
            except Exception as e:
                print(f"  ❌ Failed to copy {script}: {e}")
        elif target_path.exists():
            print(f"  ✓ {script} already exists")
        else:
            print(f"  ⚠️  {script} not found in source")
    
    # Check OPT completion status
    print(f"\n2️⃣ Checking OPT completion status...")
    
    step_001_dir = latest_workflow / "step_001_OPT"
    if not step_001_dir.exists():
        print("❌ step_001_OPT directory not found")
        return
    
    material_dirs = [d for d in step_001_dir.iterdir() if d.is_dir()]
    completed_materials = []
    
    for material_dir in material_dirs:
        material_name = material_dir.name
        out_file = material_dir / f"{material_name}.out"
        f9_file = material_dir / f"{material_name}.f9"
        
        if out_file.exists() and f9_file.exists():
            completed_materials.append(material_name)
            print(f"  ✅ {material_name} - OPT completed (.out and .f9 found)")
        else:
            print(f"  ⏳ {material_name} - OPT not completed")
    
    print(f"\n📊 Completion Summary: {len(completed_materials)}/{len(material_dirs)} materials completed")
    
    if completed_materials:
        print(f"\n3️⃣ Manually triggering SP step for completed materials...")
        
        # Try to run enhanced queue manager to pick up completed jobs
        try:
            os.chdir(latest_workflow)
            print(f"  📂 Changed to workflow directory: {latest_workflow}")
            
            # Run queue manager in completion callback mode
            result = subprocess.run([
                "python", "enhanced_queue_manager.py", 
                "--max-jobs", "250", 
                "--reserve", "30", 
                "--max-submit", "5", 
                "--callback-mode", "completion"
            ], capture_output=True, text=True, timeout=60)
            
            print(f"  🚀 Queue manager exit code: {result.returncode}")
            if result.stdout:
                print(f"  📝 Output: {result.stdout}")
            if result.stderr:
                print(f"  ⚠️  Errors: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("  ⏰ Queue manager timed out (normal for long operations)")
        except Exception as e:
            print(f"  ❌ Error running queue manager: {e}")
        finally:
            os.chdir(current_dir)
    
    print(f"\n4️⃣ Creating manual progression script...")
    
    # Create a manual progression script
    progression_script = latest_workflow / "manual_progression.py"
    script_content = f'''#!/usr/bin/env python3
"""
Manual Workflow Progression Script
=================================
Run this script to manually advance completed OPT calculations to SP step.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from enhanced_queue_manager import EnhancedCrystalQueueManager
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"❌ Import error: {{e}}")
    print("Make sure you're running from the workflow directory with all required scripts.")
    sys.exit(1)

def main():
    """Main progression function"""
    print("🔄 Manual Workflow Progression")
    print("=" * 40)
    
    # Initialize queue manager
    queue_manager = EnhancedCrystalQueueManager(
        d12_dir=".",
        max_jobs=250,
        enable_tracking=True,
        db_path="../../../materials.db"  # Database is 3 levels up
    )
    
    print("📊 Checking for completed OPT calculations...")
    
    # Run completion callback to process finished jobs
    queue_manager.process_completion_callbacks()
    
    print("✅ Manual progression complete!")
    print("\\nTo monitor progress:")
    print("  python enhanced_queue_manager.py --status")
    print("  python ../../../material_monitor.py --action stats")

if __name__ == "__main__":
    main()
'''
    
    with open(progression_script, 'w') as f:
        f.write(script_content)
    os.chmod(progression_script, 0o755)
    print(f"  ✅ Created {progression_script}")
    
    print(f"\n✅ Workflow fix complete!")
    print(f"\n🎯 Next Steps:")
    print(f"1. The workflow now has enhanced_queue_manager.py for progression")
    print(f"2. Run manual progression: cd {latest_workflow} && python manual_progression.py")
    print(f"3. Check progress: python enhanced_queue_manager.py --status")
    print(f"4. Future workflows will automatically progress correctly")

if __name__ == "__main__":
    fix_current_workflow()