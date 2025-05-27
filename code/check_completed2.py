#!/usr/bin/env python3

"""
Moves files associated with completed runs into the 'done/' directory.

Requires:
- updatelists.py to be run beforehand
- 'completesp_list.csv' or 'complete_list.csv' to exist
- 'done/' directory to be created beforehand
"""

import os
import shutil
import pandas as pd

# === Configuration === #
base_dir = os.path.abspath(os.path.dirname(__file__))
completed_dir = os.path.abspath(os.path.join(base_dir, "../1"))

# Create 'done' directory if it doesn't exist
os.makedirs(completed_dir, exist_ok=True)

# === Load file list === #
csv_path = os.path.join(base_dir, "too_many_scf_list.csv")
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Run updatelists.py first.")
    exit(1)

data_files = pd.read_csv(csv_path)
print(f"Found {len(data_files)} completed structures in {csv_path}\n")

# === File extensions to move === #
extensions = [".sh", ".out", ".d12", ".f9"]

# === Process and move files === #
for row in data_files.itertuples(index=False):
    base_name = row.data_files
    print(f"Moving files for: {base_name}")

    for ext in extensions:
        src = os.path.join(base_dir, base_name + ext)
        dest = os.path.join(completed_dir, base_name + ext)

        if os.path.exists(src):
            shutil.move(src, dest)
            print(f"  Moved: {base_name + ext}")
        else:
            print(f"  Skipped (not found): {base_name + ext}")

print("\nDone moving completed job files.")
