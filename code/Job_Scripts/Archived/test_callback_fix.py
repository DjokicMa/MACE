#!/usr/bin/env python3
"""
Test Callback Fix
-----------------
Script to test the fixed callback mechanism for workflow progression.
"""

import os
import sys
from pathlib import Path

def test_callback_mechanism():
    """Test the callback mechanism by simulating a job completion."""
    
    print("=== Testing Callback Mechanism Fixes ===\n")
    
    # Test 1: Database population
    print("1. Testing database population...")
    try:
        from populate_completed_jobs import scan_for_completed_calculations, populate_database
        from material_database import MaterialDatabase
        
        db = MaterialDatabase("materials.db")
        completed_calcs = scan_for_completed_calculations(Path.cwd())
        
        if completed_calcs:
            print(f"   ✓ Found {len(completed_calcs)} completed calculations")
            added_count = populate_database(completed_calcs, db)
            print(f"   ✓ Added {added_count} calculations to database")
        else:
            print("   ! No completed calculations found")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Enhanced queue manager workflow context detection
    print("\n2. Testing enhanced queue manager workflow detection...")
    try:
        from enhanced_queue_manager import EnhancedQueueManager
        
        manager = EnhancedQueueManager(".", enable_tracking=True)
        print(f"   ✓ Workflow context detected: {manager.is_workflow_context}")
        print(f"   ✓ Script paths configured: {len(manager.script_paths)} scripts")
        
        for script_type, path in manager.script_paths.items():
            exists = Path(path).exists()
            status = "✓" if exists else "✗"
            print(f"     {status} {script_type}: {path}")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Workflow engine integration
    print("\n3. Testing workflow engine integration...")
    try:
        from workflow_engine import WorkflowEngine
        
        engine = WorkflowEngine("materials.db", ".")
        new_steps = engine.process_completed_calculations()
        print(f"   ✓ Workflow engine processed: {new_steps} new steps initiated")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Manual callback simulation
    print("\n4. Testing manual callback simulation...")
    try:
        manager = EnhancedQueueManager(".", enable_tracking=True)
        print("   ✓ Running callback check...")
        manager.run_callback_check('completion')
        print("   ✓ Callback completed successfully")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n=== Test Complete ===")
    print("\nTo manually test the callback mechanism:")
    print("  python enhanced_queue_manager.py --callback-mode completion")
    print("\nTo check workflow status:")
    print("  python workflow_engine.py --action status --material-id <material_name>")
    print("  python run_workflow.py --status")


if __name__ == "__main__":
    test_callback_mechanism()