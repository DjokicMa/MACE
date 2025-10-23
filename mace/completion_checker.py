#!/usr/bin/env python3
"""
MACE Completion Checker
=======================

Categorizes CRYSTAL calculation output files based on completion and error status.
Can optionally organize files into categorized folders.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

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

# Error descriptions for better reporting
ERROR_DESCRIPTIONS = {
    'too_many_scf': "SCF convergence failure",
    'memory': "Out of memory error",
    'quota': "Disk quota/write error",
    'time': "Time limit exceeded",
    'geometry_small_dist': "Geometry error (atoms too close)",
    'shrink_error': "SHRINK parameter error",
    'linear_basis': "Linear dependency in basis set",
    'potential': "Segmentation fault/runtime error",
}

# === Initialize result buckets === #
def initialize_buckets():
    """Initialize categorization buckets"""
    categories = list(ERROR_PATTERNS.keys()) + ["complete", "completesp", "unknown", "ongoing"]
    return {cat: [] for cat in categories}

# === Function to categorize a single output file === #
def categorize_output_file(file_path):
    """
    Categorize a CRYSTAL output file based on error patterns and completion status.

    Args:
        file_path: Path to the .out file

    Returns:
        tuple: (category, base_name) where category is the status and base_name is the file stem
    """
    base_name = Path(file_path).stem

    try:
        with open(file_path, 'r', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Warning: Could not read {file_path}: {e}")
        return 'unknown', base_name

    # === Error pattern matching FIRST === #
    for line in lines:
        for category, keywords in ERROR_PATTERNS.items():
            if any(keyword.lower() in line.lower() for keyword in keywords):
                return category, base_name

    # === Completion checks only if no error found === #
    has_opt_end = any("OPT END" in line for line in lines)
    has_cpu_time = any("    TOTAL CPU TIME =" in line for line in lines)

    if has_opt_end:
        return 'complete', base_name
    elif has_cpu_time:
        return 'completesp', base_name

    # === Fallback: Check for generic 'error' === #
    if any("error" in line.lower() for line in lines):
        return 'unknown', base_name
    else:
        return 'ongoing', base_name

# === Process all .out files in directory === #
def scan_directory(directory='.'):
    """
    Scan directory for .out files and categorize them.

    Args:
        directory: Directory to scan (default: current directory)

    Returns:
        dict: Dictionary mapping categories to lists of base names
    """
    result_buckets = initialize_buckets()

    out_files = list(Path(directory).glob('*.out'))

    if not out_files:
        return result_buckets

    for file_path in out_files:
        category, base_name = categorize_output_file(file_path)
        result_buckets[category].append(base_name)

    return result_buckets

# === Print summary to terminal === #
def print_summary(result_buckets, detailed=False):
    """
    Print a formatted summary of categorized files.

    Args:
        result_buckets: Dictionary of categorized files
        detailed: Whether to print detailed file listings
    """
    total_files = sum(len(files) for files in result_buckets.values())

    if total_files == 0:
        print("\nNo .out files found in current directory.")
        return

    print(f"\n{'='*70}")
    print(f"CALCULATION STATUS SUMMARY")
    print(f"{'='*70}")
    print(f"Total files scanned: {total_files}\n")

    # Completion status
    complete_count = len(result_buckets['complete'])
    completesp_count = len(result_buckets['completesp'])
    total_complete = complete_count + completesp_count

    if total_complete > 0:
        print(f"✓ COMPLETED: {total_complete} calculation(s)")
        if complete_count > 0:
            print(f"  └─ Optimization complete (OPT END): {complete_count}")
        if completesp_count > 0:
            print(f"  └─ Single point complete: {completesp_count}")

    # Error status
    total_errors = sum(len(result_buckets[cat]) for cat in ERROR_PATTERNS.keys())
    if total_errors > 0:
        print(f"\n✗ ERRORS: {total_errors} calculation(s)")
        for category in ERROR_PATTERNS.keys():
            count = len(result_buckets[category])
            if count > 0:
                desc = ERROR_DESCRIPTIONS.get(category, category)
                print(f"  └─ {desc}: {count}")

    # Other status
    if len(result_buckets['ongoing']) > 0:
        print(f"\n⧗ ONGOING/INCOMPLETE: {len(result_buckets['ongoing'])} calculation(s)")

    if len(result_buckets['unknown']) > 0:
        print(f"\n? UNKNOWN ERRORS: {len(result_buckets['unknown'])} calculation(s)")

    # Detailed listings
    if detailed:
        print(f"\n{'='*70}")
        print("DETAILED FILE LISTINGS")
        print(f"{'='*70}")

        for category, files in result_buckets.items():
            if files:
                title = category.upper().replace('_', ' ')
                desc = ERROR_DESCRIPTIONS.get(category, '')
                if desc:
                    title = f"{title} ({desc})"

                print(f"\n{title} ({len(files)} files):")
                print("-" * 70)
                for fname in sorted(files):
                    print(f"  • {fname}")

# === Move files to organized folders === #
def organize_files(result_buckets, target_dir='sorted', extensions=None):
    """
    Move files into categorized subdirectories.

    Args:
        result_buckets: Dictionary of categorized files
        target_dir: Base directory for organized files
        extensions: List of file extensions to move (default: ['.sh', '.out', '.d12', '.d3', '.f9'])
    """
    if extensions is None:
        extensions = ['.sh', '.out', '.d12', '.d3', '.f9']

    base_dir = Path.cwd()
    moved_count = 0

    print(f"\n{'='*70}")
    print(f"ORGANIZING FILES")
    print(f"{'='*70}")

    for category, file_list in result_buckets.items():
        if not file_list:
            continue

        # Create category directory
        category_dir = base_dir / target_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{category.upper()} ({len(file_list)} calculations):")
        print("-" * 70)

        for base_name in file_list:
            moved_files = []

            for ext in extensions:
                src = base_dir / f"{base_name}{ext}"
                dest = category_dir / f"{base_name}{ext}"

                if src.exists():
                    try:
                        shutil.move(str(src), str(dest))
                        moved_files.append(ext)
                        moved_count += 1
                    except Exception as e:
                        print(f"  ✗ Error moving {src.name}: {e}")

            if moved_files:
                print(f"  ✓ {base_name}: {', '.join(moved_files)}")
            else:
                print(f"  ⚠ {base_name}: No associated files found")

    print(f"\n{'='*70}")
    print(f"Moved {moved_count} file(s) to {target_dir}/")
    print(f"{'='*70}")

# === Move only completed files === #
def organize_completed(result_buckets, target_dir='completed', extensions=None):
    """
    Move only successfully completed files to a completed directory.

    Args:
        result_buckets: Dictionary of categorized files
        target_dir: Directory for completed files
        extensions: List of file extensions to move
    """
    if extensions is None:
        extensions = ['.sh', '.out', '.d12', '.d3', '.f9']

    # Filter to only completed calculations
    completed_only = {
        'complete': result_buckets['complete'],
        'completesp': result_buckets['completesp']
    }

    total_completed = len(result_buckets['complete']) + len(result_buckets['completesp'])

    if total_completed == 0:
        print("\nNo completed calculations to organize.")
        return

    organize_files(completed_only, target_dir, extensions)

# === Main function === #
def main():
    """Main entry point for the completion checker"""
    parser = argparse.ArgumentParser(
        description='Check CRYSTAL calculation completion status and optionally organize files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status in current directory
  mace completion

  # Show detailed file listings
  mace completion --detailed

  # Move all completed files to 'completed' folder
  mace completion --move-completed

  # Organize all files by status
  mace completion --organize

  # Custom output directory
  mace completion --organize --output-dir results

  # Specify directory to check
  mace completion --directory /path/to/calculations
        """
    )

    parser.add_argument('--directory', '-d', default='.',
                       help='Directory to scan for .out files (default: current directory)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed file listings for each category')
    parser.add_argument('--move-completed', action='store_true',
                       help='Move completed files to a "completed" folder')
    parser.add_argument('--organize', action='store_true',
                       help='Organize all files into categorized folders')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for organized files (default: "completed" or "sorted")')
    parser.add_argument('--extensions', nargs='+',
                       default=['.sh', '.out', '.d12', '.d3', '.f9'],
                       help='File extensions to move (default: .sh .out .d12 .d3 .f9)')

    args = parser.parse_args()

    # Scan directory
    directory = Path(args.directory).resolve()

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    # Change to directory if needed
    original_dir = Path.cwd()
    os.chdir(directory)

    try:
        result_buckets = scan_directory('.')

        # Print summary
        print_summary(result_buckets, detailed=args.detailed)

        # Organize files if requested
        if args.move_completed:
            target_dir = args.output_dir or 'completed'
            organize_completed(result_buckets, target_dir, args.extensions)
        elif args.organize:
            target_dir = args.output_dir or 'sorted'
            organize_files(result_buckets, target_dir, args.extensions)

    finally:
        # Return to original directory
        os.chdir(original_dir)

if __name__ == '__main__':
    main()
