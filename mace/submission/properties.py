#!/usr/bin/env python3
"""
This script finds all .d3 files in the current directory and submits them for processing 
by running submit_prop.sh with a parameter of 100 for each file.
"""
import os, sys, math
import re
import linecache
import shutil
import itertools
from pathlib import Path

def main():
    """Main function to submit D3 property files"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    submit_prop_script = script_dir / "submit_prop.sh"
    
    # Check if submit_prop.sh exists
    if not submit_prop_script.exists():
        print(f"Error: submit_prop.sh not found at {submit_prop_script}")
        sys.exit(1)
    
    # Get target from command line or use current directory
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isfile(target) and target.endswith('.d3'):
            # Single file submission
            data_folder = os.path.dirname(target) or os.getcwd()
            data_files = [os.path.basename(target)]
        elif os.path.isdir(target):
            # Directory submission
            data_folder = target
            data_files = os.listdir(data_folder)
        else:
            print(f"Error: {target} is not a valid D3 file or directory")
            sys.exit(1)
    else:
        # No argument provided, use current directory
        data_folder = os.getcwd()
        data_files = os.listdir(data_folder)
    
    # Count D3 files
    d3_files = [f for f in data_files if f.endswith(".d3")]
    if not d3_files:
        print(f"No D3 files found in {data_folder}")
        return
    
    print(f"Found {len(d3_files)} D3 property file(s) to submit")
    
    # Submit each D3 file
    for file_name in d3_files:
        submit_name = file_name.split(".d3")[0]
        print(f"Submitting: {file_name}")
        
        # Change to the directory containing the D3 file for submission
        original_dir = os.getcwd()
        os.chdir(data_folder)
        
        # Submit using the absolute path to submit_prop.sh
        cmd = f"{submit_prop_script} {submit_name} 100"
        result = os.system(cmd)
        
        # Return to original directory
        os.chdir(original_dir)
        
        if result != 0:
            print(f"Warning: Failed to submit {file_name}")
    
    print(f"\nSubmission complete. Use 'mace monitor' to track job status.")

if __name__ == "__main__":
    main()
