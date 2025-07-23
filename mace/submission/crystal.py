#!/usr/bin/env python3
"""
This script is in place to submit all .d12 input files in the current directory to CRYSTAL23
"""
import os, sys, math
import re
import linecache
import shutil
import itertools
from pathlib import Path

def main():
    """Main function to submit D12 files"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    submitcrystal_script = script_dir / "submitcrystal23.sh"
    
    # Check if submitcrystal23.sh exists
    if not submitcrystal_script.exists():
        print(f"Error: submitcrystal23.sh not found at {submitcrystal_script}")
        sys.exit(1)
    
    # Get target from command line or use current directory
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isfile(target) and target.endswith('.d12'):
            # Single file submission
            data_folder = os.path.dirname(target) or os.getcwd()
            data_files = [os.path.basename(target)]
        elif os.path.isdir(target):
            # Directory submission
            data_folder = target
            data_files = os.listdir(data_folder)
        else:
            print(f"Error: {target} is not a valid D12 file or directory")
            sys.exit(1)
    else:
        # No argument provided, use current directory
        data_folder = os.getcwd()
        data_files = os.listdir(data_folder)
    
    # Count D12 files
    d12_files = [f for f in data_files if f.endswith(".d12")]
    if not d12_files:
        print(f"No D12 files found in {data_folder}")
        return
    
    print(f"Found {len(d12_files)} D12 file(s) to submit")
    
    # Submit each D12 file
    for file_name in d12_files:
        submit_name = file_name.split(".d12")[0]
        print(f"Submitting: {file_name}")
        
        # Change to the directory containing the D12 file for submission
        original_dir = os.getcwd()
        os.chdir(data_folder)
        
        # Submit using the absolute path to submitcrystal23.sh
        cmd = f"{submitcrystal_script} {submit_name} 100"
        result = os.system(cmd)
        
        # Return to original directory
        os.chdir(original_dir)
        
        if result != 0:
            print(f"Warning: Failed to submit {file_name}")
    
    print(f"\nSubmission complete. Use 'mace monitor' to track job status.")

if __name__ == "__main__":
    main()
