#!/usr/bin/env python3
"""
This script moves completed files to a 'completed/' subdirectory,
based on entries from complete_list.csv and completesp_list.csv.
Creates subdirectories for each completion type.

Requires:
- updatelists.py to be run beforehand
- 'complete_list.csv' and/or 'completesp_list.csv' to exist

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
import os
import shutil
import pandas as pd

# === Configuration === #
base_dir = os.path.abspath(os.path.dirname(__file__))

# Completion categories to process
completion_categories = ["complete", "completesp"]

# File extensions to move
extensions = [".sh", ".out", ".d12", ".f9"]

# === Process each completion category === #
for category in completion_categories:
    csv_path = os.path.join(base_dir, f"{category}_list.csv")
    
    # Skip if CSV doesn't exist
    if not os.path.exists(csv_path):
        print(f"Skipping {category}: {csv_path} not found.")
        continue
    
    # Create subdirectory for this category
    category_dir = os.path.join(base_dir, "completed", category)
    os.makedirs(category_dir, exist_ok=True)
    
    # Load file list
    data_files = pd.read_csv(csv_path)
    print(f"\nProcessing {category}: Found {len(data_files)} structures")
    
    # Process and move files
    for row in data_files.itertuples(index=False):
        base_name = row.data_files
        moved_files = []
        
        for ext in extensions:
            src = os.path.join(base_dir, base_name + ext)
            dest = os.path.join(category_dir, base_name + ext)
            
            if os.path.exists(src):
                shutil.move(src, dest)
                moved_files.append(ext)
        
        if moved_files:
            print(f"  Moved {base_name}: {', '.join(moved_files)}")
        else:
            print(f"  Warning: No files found for {base_name} (may have been moved already)")

print("\nDone moving completed job files.")
