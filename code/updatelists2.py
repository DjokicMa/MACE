#!/usr/bin/env python3

"""
Categorizes quantum chemistry .out files based on completion and error messages.
Exports categorized file names into separate CSVs by category.
"""

import os
import pandas as pd

# === Define known error and completion message patterns === #
ERROR_PATTERNS = {
    'too_many_scf': ["TOO MANY CYCLES"],
    'memory': ["out-of-memory handler"],
    'quota': ["error during write"],
    'time': ["DUE TO TIME LIMIT"],
    'geometry_small_dist': ["**** NEIGHB ****"],
    'shrink_error': ["ANISOTROPIC SHRINKING FACTOR"],
    'linear_basis': ["BASIS SET LINEARLY DEPENDENT"],
    'potential': [
        "segmentation fault",
        "=   bad termination of",
        "abort(1) on node",
        "srun: error:",
	"slurmstepd: error: ***",
	"forrtl: error (78):",
	"Stack trace terminated abnormally."
    ],
}

# === Initialize buckets === #
categories = list(ERROR_PATTERNS.keys()) + ["complete", "completesp", "unknown", "ongoing"]
result_buckets = {cat: pd.DataFrame(columns=["data_files"]) for cat in categories}

# === Function to categorize a single output file === #
def categorize_output_file(file_name):
    submit_name = file_name.split(".out")[0]
    temp_df = pd.DataFrame([[submit_name]], columns=["data_files"])

    try:
        with open(file_name, 'r', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Could not read file {file_name}: {e}")
        return

    # === Error pattern matching FIRST === #
    for line in lines:
        for category, keywords in ERROR_PATTERNS.items():
            if any(keyword.lower() in line.lower() for keyword in keywords):
                result_buckets[category] = pd.concat([result_buckets[category], temp_df], ignore_index=True)
                return

    # === Completion checks only if no error found === #
    has_opt_end = any("OPT END" in line for line in lines)
    has_cpu_time = any("    TOTAL CPU TIME =" in line for line in lines)

    if has_opt_end:
        result_buckets['complete'] = pd.concat([result_buckets['complete'], temp_df], ignore_index=True)
        return
    elif has_cpu_time:
        result_buckets['completesp'] = pd.concat([result_buckets['completesp'], temp_df], ignore_index=True)
        return

    # === Fallback: Check for generic 'error' === #
    if any("error" in line.lower() for line in lines):
        result_buckets['unknown'] = pd.concat([result_buckets['unknown'], temp_df], ignore_index=True)
    else:
        result_buckets['ongoing'] = pd.concat([result_buckets['ongoing'], temp_df], ignore_index=True)


# === Main loop: Process all .out files in directory === #
for file in os.listdir():
    if file.endswith(".out"):
        categorize_output_file(file)

# === Export results and print summary to terminal === #
for category, df in result_buckets.items():
    if not df.empty:
        print(f"\n {category.upper()} ({len(df)} structures):")
        print(df["data_files"].to_string(index=False))
        df.to_csv(f"{category}_list.csv", index=False)





