#!/usr/bin/env python3
"""
Debug SP Detection
Debug script to understand why SP calculations aren't being detected
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from populate_completed_jobs import scan_for_completed_calculations, extract_calc_type_from_output, is_calculation_completed

def debug_sp_detection():
    """Debug SP calculation detection."""
    print("=== SP Detection Debug ===")
    
    base_dir = Path.cwd()
    print(f"Scanning from: {base_dir}")
    
    # Find all .out files in workflow_outputs
    workflow_outputs = base_dir / "workflow_outputs"
    if not workflow_outputs.exists():
        print(f"❌ No workflow_outputs directory found at {workflow_outputs}")
        return
    
    print(f"✅ Found workflow_outputs at {workflow_outputs}")
    
    # Look for SP output files
    sp_out_files = list(workflow_outputs.glob("**/step_002_SP/**/*.out"))
    print(f"\nFound {len(sp_out_files)} .out files in step_002_SP:")
    
    for out_file in sp_out_files:
        print(f"\n--- Checking: {out_file.name} ---")
        print(f"Full path: {out_file}")
        print(f"Exists: {out_file.exists()}")
        print(f"Size: {out_file.stat().st_size if out_file.exists() else 'N/A'} bytes")
        
        if out_file.exists():
            # Check if it's completed
            is_completed = is_calculation_completed(out_file)
            print(f"Is completed: {is_completed}")
            
            if is_completed:
                # Check calc type detection
                calc_type = extract_calc_type_from_output(out_file)
                print(f"Detected calc type: {calc_type}")
                
                # Check for corresponding files
                base_name = out_file.stem
                d12_file = out_file.parent / f"{base_name}.d12"
                f9_file = out_file.parent / f"{base_name}.f9"
                
                print(f"D12 file exists: {d12_file.exists()}")
                print(f"F9 file exists: {f9_file.exists()}")
            else:
                # Show why it's not completed
                print("❌ Not detected as completed")
                # Sample first 500 chars of output to see what's there
                try:
                    with open(out_file, 'r') as f:
                        content = f.read(500)
                    print(f"File content start: {content[:200]}...")
                except Exception as e:
                    print(f"Error reading file: {e}")
    
    # Now run the full scan and see what it finds
    print(f"\n=== Full Scan Results ===")
    completed_calcs = scan_for_completed_calculations(base_dir)
    print(f"Total completed calculations found: {len(completed_calcs)}")
    
    sp_calcs = [calc for calc in completed_calcs if calc['calc_type'] == 'SP']
    print(f"SP calculations found: {len(sp_calcs)}")
    
    for calc in sp_calcs:
        print(f"  SP: {calc['material_id']} - {calc['output_file']}")

if __name__ == "__main__":
    debug_sp_detection()