#!/usr/bin/env python3
"""
Manual workflow check and progression
=====================================
Use this to manually trigger workflow progression if automatic callbacks aren't working.

Usage:
  python check_workflows.py           # Check and process all workflows
  python check_workflows.py --dry-run # Just check, don't process
"""

import sys
import json
import argparse
from pathlib import Path

# Import MACE components
try:
    from database.materials import MaterialDatabase
    from workflow.engine import WorkflowEngine
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Check and process workflow progression")
    parser.add_argument("--dry-run", action="store_true", help="Just check, don't process")
    parser.add_argument("--db-path", default="materials.db", help="Database path")
    args = parser.parse_args()
    
    print("Checking for completed calculations that need workflow progression...")
    
    db = MaterialDatabase(args.db_path)
    engine = WorkflowEngine(args.db_path, ".")
    
    # Get all completed calculations
    completed_calcs = db.get_calculations_by_status('completed')
    
    # Filter for those that haven't been processed
    unprocessed = []
    for calc in completed_calcs:
        settings = json.loads(calc.get('settings_json', '{}'))
        if not settings.get('workflow_processed') and settings.get('workflow_id'):
            unprocessed.append(calc)
    
    if not unprocessed:
        print("✅ All completed calculations have been processed")
        return
    
    print(f"Found {len(unprocessed)} unprocessed completed calculations:")
    
    # Group by workflow
    by_workflow = {}
    for calc in unprocessed:
        settings = json.loads(calc.get('settings_json', '{}'))
        workflow_id = settings.get('workflow_id', 'Unknown')
        if workflow_id not in by_workflow:
            by_workflow[workflow_id] = []
        by_workflow[workflow_id].append(calc)
    
    # Display summary
    for workflow_id, calcs in by_workflow.items():
        print(f"\nWorkflow: {workflow_id}")
        sequence = engine.get_workflow_sequence(workflow_id)
        if sequence:
            print(f"  Sequence: {' → '.join(sequence)}")
        
        for calc in calcs:
            print(f"  - {calc['material_id']} {calc['calc_type']} (ID: {calc['calc_id']})")
    
    if args.dry_run:
        print("\n--dry-run specified, not processing")
        return
    
    print("\nProcessing calculations...")
    
    # Process each one
    total_new = 0
    for calc in unprocessed:
        print(f"\n📋 Processing: {calc['material_id']} {calc['calc_type']}")
        new_calc_ids = engine.execute_workflow_step(calc['material_id'], calc['calc_id'])
        
        if new_calc_ids:
            total_new += len(new_calc_ids)
            print(f"  ✅ Generated {len(new_calc_ids)} new calculations:")
            for new_id in new_calc_ids:
                new_calc = db.get_calculation(new_id)
                if new_calc:
                    print(f"    - {new_calc['calc_type']}: {new_calc.get('status', 'Unknown')}")
        else:
            print("  ℹ️  No new calculations generated (may be end of workflow)")
    
    print(f"\n✅ Done! Generated {total_new} new calculations total")
    
    if total_new > 0:
        print("\nNext steps:")
        print("  1. Check status: python workflow_status.py --all")
        print("  2. Monitor queue: python enhanced_queue_manager.py --status")

if __name__ == "__main__":
    main()