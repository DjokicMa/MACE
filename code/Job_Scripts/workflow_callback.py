#!/usr/bin/env python3
"""
Workflow Callback Handler
=========================
Processes completed calculations and triggers next workflow steps.
This should be called automatically when jobs complete.

Usage:
  python workflow_callback.py
  python workflow_callback.py --calc-id <calc_id>
"""

import os
import sys
import argparse
from pathlib import Path

# Add script directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from material_database import MaterialDatabase
    from workflow_engine import WorkflowEngine
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def process_workflow_callbacks(calc_id: str = None):
    """Process workflow callbacks for completed calculations"""
    print("Processing workflow callbacks...")
    
    # Initialize database and engine
    db = MaterialDatabase("materials.db")
    engine = WorkflowEngine("materials.db", ".")
    
    if calc_id:
        # Process specific calculation
        calc = db.get_calculation(calc_id)
        if calc and calc['status'] == 'completed':
            print(f"Processing completed calculation: {calc_id}")
            new_calc_ids = engine.execute_workflow_step(calc['material_id'], calc_id)
            if new_calc_ids:
                print(f"Generated {len(new_calc_ids)} new calculations")
                for new_id in new_calc_ids:
                    new_calc = db.get_calculation(new_id)
                    if new_calc:
                        print(f"  - {new_calc['calc_type']}: {new_calc['calc_id']}")
            else:
                print("No new calculations generated")
        else:
            print(f"Calculation {calc_id} not found or not completed")
    else:
        # Process all unprocessed completed calculations
        new_steps = engine.process_completed_calculations()
        print(f"Generated {new_steps} new workflow steps")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Workflow callback handler")
    parser.add_argument("--calc-id", help="Specific calculation ID to process")
    parser.add_argument("--db-path", default="materials.db", help="Database path")
    
    args = parser.parse_args()
    
    # Change to script directory to ensure proper paths
    original_cwd = os.getcwd()
    try:
        os.chdir(script_dir)
        process_workflow_callbacks(args.calc_id)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()