#!/usr/bin/env python3
"""
Setup Workflow Monitoring Scripts
=================================
Copies all necessary monitoring and database scripts to the current workflow directory
to enable comprehensive monitoring capabilities for CRYSTAL workflow tracking.

This ensures that monitoring commands work properly in workflow contexts where
the user may not have direct access to the full repository.
"""

import os
import shutil
import sys
from pathlib import Path

def setup_monitoring_scripts(target_dir="."):
    """Copy all necessary monitoring scripts to target directory."""
    target_dir = Path(target_dir).resolve()
    
    # Get source directory (where this script is located)
    source_dir = Path(__file__).parent
    
    # List of required monitoring scripts
    required_scripts = [
        "material_database.py",
        "crystal_file_manager.py", 
        "error_detector.py",
        "material_monitor.py",
        "enhanced_queue_manager.py",
        "workflow_engine.py",
        "error_recovery.py"
    ]
    
    print(f"Setting up monitoring scripts in: {target_dir}")
    
    copied_count = 0
    for script in required_scripts:
        source_path = source_dir / script
        target_path = target_dir / script
        
        if source_path.exists():
            if target_path.exists():
                print(f"  ✓ {script} already exists")
            else:
                try:
                    shutil.copy2(source_path, target_path)
                    print(f"  ✓ Copied {script}")
                    copied_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to copy {script}: {e}")
        else:
            print(f"  ✗ Source {script} not found at {source_path}")
    
    # Create monitoring helper script
    helper_script = target_dir / "monitor_workflow.py"
    if not helper_script.exists():
        create_monitoring_helper(helper_script)
        print(f"  ✓ Created monitoring helper script")
        copied_count += 1
    
    print(f"\nSetup complete! Copied {copied_count} files.")
    print("\nNow you can use these monitoring commands:")
    print("  python material_monitor.py --action dashboard")
    print("  python material_monitor.py --action status") 
    print("  python material_monitor.py --action stats")
    print("  python monitor_workflow.py")
    
    return copied_count

def create_monitoring_helper(helper_path):
    """Create a simple monitoring helper script."""
    content = '''#!/usr/bin/env python3
"""
Workflow Monitoring Helper
=========================
Quick access to common monitoring functions for CRYSTAL workflows.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from material_monitor import MaterialMonitor
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"Error: Required monitoring modules not found: {e}")
    print("Please run: python setup_workflow_monitoring.py")
    sys.exit(1)

def quick_status():
    """Show quick status overview."""
    monitor = MaterialMonitor()
    stats = monitor.get_quick_stats()
    
    print("=== Quick Workflow Status ===")
    print(f"Materials in database: {stats['materials']}")
    print(f"Total calculations: {stats['calculations']}")
    print(f"Database size: {stats['db_size_mb']} MB")
    print(f"Active queue jobs: {stats['queue_jobs']}")
    
    # Show recent calculations
    try:
        db = MaterialDatabase()
        recent = db.get_recent_calculations(limit=5)
        if recent:
            print("\\nRecent calculations:")
            for calc in recent:
                print(f"  {calc['material_id']} ({calc['calc_type']}) - {calc['status']}")
        else:
            print("\\nNo recent calculations found")
    except Exception as e:
        print(f"\\nError getting recent calculations: {e}")

def database_query(query_type="materials"):
    """Run common database queries."""
    try:
        db = MaterialDatabase()
        
        if query_type == "materials":
            materials = db.get_all_materials()
            print(f"=== All Materials ({len(materials)}) ===")
            for mat in materials:
                print(f"  {mat['material_id']} - {mat['formula']} ({mat['status']})")
                
        elif query_type == "calculations":
            calcs = db.get_recent_calculations(limit=10)
            print(f"=== Recent Calculations ({len(calcs)}) ===")
            for calc in calcs:
                print(f"  {calc['calc_id']} - {calc['status']} ({calc['calc_type']})")
                
        elif query_type == "stats":
            stats = db.get_database_stats()
            print("=== Database Statistics ===")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error running database query: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Workflow monitoring helper")
    parser.add_argument("--action", choices=["status", "materials", "calculations", "stats"], 
                       default="status", help="Action to perform")
    
    args = parser.parse_args()
    
    if args.action == "status":
        quick_status()
    elif args.action in ["materials", "calculations", "stats"]:
        database_query(args.action)
'''
    
    with open(helper_path, 'w') as f:
        f.write(content)
    
    # Make it executable
    os.chmod(helper_path, 0o755)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup workflow monitoring scripts")
    parser.add_argument("--target-dir", default=".", help="Target directory for setup")
    
    args = parser.parse_args()
    
    setup_monitoring_scripts(args.target_dir)