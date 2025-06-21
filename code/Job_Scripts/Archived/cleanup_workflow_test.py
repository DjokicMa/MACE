#!/usr/bin/env python3
"""
Clean up test workflow and demonstrate the fixes
"""
import os
import shutil
from pathlib import Path

def cleanup_test_workflow():
    """Clean up the test workflow to demonstrate the fixes"""
    current_dir = Path.cwd()
    
    # Remove duplicate D12 files from step directory (keep only in material folders)
    step_dir = current_dir / "workflow_outputs" / "workflow_20250620_235719" / "step_001_OPT"
    
    if step_dir.exists():
        print(f"Cleaning up duplicate files in: {step_dir}")
        
        # Get list of D12 files in step directory (these are duplicates)
        step_d12_files = list(step_dir.glob("*.d12"))
        print(f"Found {len(step_d12_files)} duplicate D12 files to remove")
        
        for d12_file in step_d12_files:
            material_name = d12_file.stem
            material_dir = step_dir / material_name
            
            # Check if the material directory has the file
            if material_dir.exists() and (material_dir / d12_file.name).exists():
                print(f"  Removing duplicate: {d12_file.name} (kept in {material_name}/)")
                d12_file.unlink()
            else:
                print(f"  Keeping: {d12_file.name} (no material folder found)")
    
    # Add monitoring scripts to workflow directory
    workflow_dir = current_dir / "workflow_outputs" / "workflow_20250620_235719"
    if workflow_dir.exists():
        print(f"\nAdding monitoring scripts to: {workflow_dir}")
        
        # List of monitoring scripts to copy
        monitoring_scripts = [
            "material_database.py",
            "crystal_file_manager.py", 
            "error_detector.py",
            "material_monitor.py",
            "enhanced_queue_manager.py",
            "workflow_engine.py",
            "error_recovery.py"
        ]
        
        copied_count = 0
        for script in monitoring_scripts:
            source_path = current_dir / script
            target_path = workflow_dir / script
            
            if source_path.exists() and not target_path.exists():
                try:
                    shutil.copy2(source_path, target_path)
                    print(f"  ✓ Copied {script}")
                    copied_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to copy {script}: {e}")
            elif target_path.exists():
                print(f"  - {script} already exists")
            else:
                print(f"  ✗ Source {script} not found")
        
        # Create monitoring helper
        helper_script = workflow_dir / "monitor_workflow.py"
        if not helper_script.exists():
            helper_content = '''#!/usr/bin/env python3
"""Workflow monitoring helper"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from material_monitor import MaterialMonitor
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"Error: Required monitoring modules not found: {e}")
    sys.exit(1)

def quick_status():
    """Show quick status overview."""
    try:
        monitor = MaterialMonitor()
        stats = monitor.get_quick_stats()
        
        print("=== Quick Workflow Status ===")
        print(f"Materials in database: {stats['materials']}")
        print(f"Total calculations: {stats['calculations']}")
        print(f"Database size: {stats['db_size_mb']} MB")
        print(f"Active queue jobs: {stats['queue_jobs']}")
        
        # Show recent calculations
        db = MaterialDatabase()
        recent = db.get_recent_calculations(5)
        if recent:
            print("\\nRecent calculations:")
            for calc in recent:
                print(f"  {calc['material_id']} ({calc['calc_type']}) - {calc['status']}")
        else:
            print("\\nNo recent calculations found")
            
    except Exception as e:
        print(f"Error getting status: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["status", "materials", "calculations", "stats"], 
                       default="status")
    args = parser.parse_args()
    
    if args.action == "status":
        quick_status()
    else:
        print(f"Action '{args.action}' - use database queries directly for now")
'''
            
            with open(helper_script, 'w') as f:
                f.write(helper_content)
            os.chmod(helper_script, 0o755)
            print(f"  ✓ Created monitor_workflow.py")
            copied_count += 1
        
        # Create README
        readme_path = workflow_dir / "MONITORING_README.md"
        if not readme_path.exists():
            readme_content = f"""# Workflow Monitoring

This directory contains monitoring scripts for your CRYSTAL workflow.

## Quick Commands

```bash
# Check status
python material_monitor.py --action stats

# Quick helper
python monitor_workflow.py --action status

# Live dashboard (press Ctrl+C to stop)
python material_monitor.py --action dashboard
```

## Database Queries

```python
from material_database import MaterialDatabase
db = MaterialDatabase()

# Get all materials
for mat in db.get_all_materials():
    print(f"{{mat['material_id']}}: {{mat['formula']}}")

# Get recent calculations
for calc in db.get_recent_calculations(10):
    print(f"{{calc['material_id']}} - {{calc['calc_type']}} - {{calc['status']}}")
```

Monitoring scripts copied: {copied_count}
"""
            
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            print(f"  ✓ Created MONITORING_README.md")
        
        print(f"\n✅ Monitoring setup complete! {copied_count} scripts copied")
    
    print("\n📁 Final directory structure:")
    if step_dir.exists():
        print("workflow_outputs/workflow_20250620_235719/step_001_OPT/")
        for item in sorted(step_dir.iterdir()):
            if item.is_dir():
                print(f"  {item.name}/")
                for sub_item in sorted(item.iterdir()):
                    print(f"    {sub_item.name}")
            else:
                print(f"  {item.name}")
    
    print(f"\n🎯 Next workflow run will have:")
    print(f"  ✅ Clean directory structure (files only in material folders)")
    print(f"  ✅ Monitoring scripts auto-copied")
    print(f"  ✅ No duplicate files")
    print(f"  ✅ Consistent naming")

if __name__ == "__main__":
    cleanup_test_workflow()