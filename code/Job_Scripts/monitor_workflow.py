#!/usr/bin/env python3
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
            print("\nRecent calculations:")
            for calc in recent:
                print(f"  {calc['material_id']} ({calc['calc_type']}) - {calc['status']}")
        else:
            print("\nNo recent calculations found")
    except Exception as e:
        print(f"\nError getting recent calculations: {e}")

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
