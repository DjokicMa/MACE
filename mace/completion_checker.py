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
import re
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple

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
def scan_directory(directory='.', recursive=False):
    """
    Scan directory for .out files and categorize them.

    Args:
        directory: Directory to scan (default: current directory)
        recursive: If True, scan subdirectories recursively

    Returns:
        tuple: (result_buckets dict, file_paths dict mapping base_name to Path)
    """
    result_buckets = initialize_buckets()
    file_paths = {}  # Map base_name to full path for zombie detection

    if recursive:
        out_files = list(Path(directory).glob('**/*.out'))
    else:
        out_files = list(Path(directory).glob('*.out'))

    if not out_files:
        return result_buckets, file_paths

    for file_path in out_files:
        category, base_name = categorize_output_file(file_path)
        result_buckets[category].append(base_name)
        file_paths[base_name] = file_path

    return result_buckets, file_paths

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

# === Zombie job detection functions === #
def find_slurm_job_id(out_file_path: Path) -> Optional[str]:
    """
    Find SLURM job ID from associated .o file.

    Looks for patterns like:
    - material_name-12345678.o
    - material_name.o12345678

    Args:
        out_file_path: Path to the .out file

    Returns:
        Job ID string if found, None otherwise
    """
    base = out_file_path.stem
    parent = out_file_path.parent

    # Pattern 1: name-jobid.o (e.g., 4^2T37-CA_SCF_opt-62676309.o)
    for f in parent.glob(f"{base}-*.o"):
        match = re.search(r'-(\d+)\.o$', f.name)
        if match:
            return match.group(1)

    # Pattern 2: name.ojobid (e.g., material.o12345678)
    for f in parent.glob(f"{base}.o*"):
        match = re.search(r'\.o(\d+)$', f.name)
        if match:
            return match.group(1)

    # Pattern 3: name_jobid.o or name-jobid.o with underscores
    for f in parent.glob(f"*{base}*.o"):
        # Look for any file containing the base name with a job ID
        match = re.search(r'[-_](\d{6,})\.o$', f.name)
        if match:
            return match.group(1)

    return None

def get_running_jobs() -> Set[str]:
    """
    Get set of currently running job IDs from squeue.

    Returns:
        Set of job ID strings that are currently running
    """
    try:
        user = os.environ.get('USER', '')
        result = subprocess.run(
            ['squeue', '-u', user, '-h', '-o', '%i'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return set(result.stdout.strip().split('\n'))
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Warning: Could not query squeue: {e}")

    return set()

def detect_zombie_jobs(result_buckets: Dict, file_paths: Dict[str, Path]) -> List[Dict]:
    """
    Detect jobs that have failed but are still running in SLURM.

    Args:
        result_buckets: Dictionary of categorized files
        file_paths: Dictionary mapping base_name to file Path

    Returns:
        List of zombie job info dicts with keys: name, job_id, category, description, path
    """
    running_jobs = get_running_jobs()

    if not running_jobs:
        return []

    zombies = []

    # Check errored jobs and ongoing jobs (which might have errors not yet detected)
    error_categories = list(ERROR_PATTERNS.keys()) + ['unknown', 'ongoing']

    for category in error_categories:
        for base_name in result_buckets.get(category, []):
            out_path = file_paths.get(base_name)
            if not out_path:
                continue

            job_id = find_slurm_job_id(out_path)
            if job_id and job_id in running_jobs:
                # For ongoing jobs, note they might still be legitimately running
                if category == 'ongoing':
                    description = "Still running (check if stuck)"
                else:
                    description = ERROR_DESCRIPTIONS.get(category, 'Unknown error')

                zombies.append({
                    'name': base_name,
                    'job_id': job_id,
                    'category': category,
                    'description': description,
                    'path': out_path
                })

    return zombies

def remove_zombie_jobs(result_buckets: Dict, file_paths: Dict[str, Path]):
    """
    Detect zombie jobs and offer to cancel them interactively.

    Args:
        result_buckets: Dictionary of categorized files
        file_paths: Dictionary mapping base_name to file Path
    """
    zombies = detect_zombie_jobs(result_buckets, file_paths)

    if not zombies:
        print("\n✓ No zombie jobs detected.")
        print("  (All failed jobs have already stopped in SLURM)")
        return

    # Separate actual errors from ongoing jobs
    error_zombies = [z for z in zombies if z['category'] != 'ongoing']
    ongoing_zombies = [z for z in zombies if z['category'] == 'ongoing']

    # Print header and explanation
    print(f"\n{'='*70}")
    print("ZOMBIE JOB DETECTION")
    print(f"{'='*70}")
    print("\nZombie jobs are calculations that have FAILED (error in output)")
    print("but are STILL RUNNING in SLURM, wasting CPU hours.\n")

    # Show error zombies first (definite zombies)
    if error_zombies:
        print(f"Found {len(error_zombies)} zombie job(s) with errors:\n")

        for z in error_zombies:
            print(f"  • {z['name']}")
            print(f"    Job ID: {z['job_id']}")
            print(f"    Error: {z['description']}")
            print(f"    Path: {z['path'].parent}")
            print()

    # Show ongoing zombies (might be legitimate or stuck)
    if ongoing_zombies:
        print(f"Found {len(ongoing_zombies)} job(s) still running (may be stuck):\n")

        for z in ongoing_zombies:
            print(f"  • {z['name']}")
            print(f"    Job ID: {z['job_id']}")
            print(f"    Status: {z['description']}")
            print(f"    Path: {z['path'].parent}")
            print()

    # Confirm cancellation
    print(f"{'='*70}")

    if error_zombies:
        # Ask about error zombies first
        error_ids = [z['job_id'] for z in error_zombies]
        response = input(f"Cancel {len(error_zombies)} failed zombie job(s)? [y/N]: ").strip().lower()

        if response == 'y':
            result = subprocess.run(
                ['scancel'] + error_ids,
                capture_output=True, text=True
            )

            if result.returncode == 0:
                print(f"\n✓ Cancelled {len(error_ids)} zombie job(s):")
                for jid in error_ids:
                    print(f"  scancel {jid}")
            else:
                print(f"\n✗ Error cancelling jobs: {result.stderr}")
        else:
            print("\nNo failed jobs cancelled.")
            if error_ids:
                print("To cancel manually:")
                for jid in error_ids:
                    print(f"  scancel {jid}")

    # Optionally ask about ongoing jobs
    if ongoing_zombies:
        ongoing_ids = [z['job_id'] for z in ongoing_zombies]
        print()
        response = input(f"Also cancel {len(ongoing_zombies)} ongoing job(s) that may be stuck? [y/N]: ").strip().lower()

        if response == 'y':
            result = subprocess.run(
                ['scancel'] + ongoing_ids,
                capture_output=True, text=True
            )

            if result.returncode == 0:
                print(f"\n✓ Cancelled {len(ongoing_ids)} ongoing job(s):")
                for jid in ongoing_ids:
                    print(f"  scancel {jid}")
            else:
                print(f"\n✗ Error cancelling jobs: {result.stderr}")
        else:
            print("\nNo ongoing jobs cancelled.")

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

  # Scan workflow subdirectories recursively
  mace completion --recursive

  # Show detailed file listings
  mace completion --detailed

  # Detect and cancel zombie jobs (failed but still running)
  mace completion --remove-zombie-jobs

  # Scan recursively and remove zombie jobs
  mace completion --recursive --remove-zombie-jobs

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
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Scan subdirectories recursively (for workflow folder structures)')
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
    parser.add_argument('--remove-zombie-jobs', action='store_true',
                       help='Detect and cancel jobs that failed but are still running in SLURM')

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
        result_buckets, file_paths = scan_directory('.', recursive=args.recursive)

        # Print summary
        print_summary(result_buckets, detailed=args.detailed)

        # Zombie job detection and removal
        if args.remove_zombie_jobs:
            remove_zombie_jobs(result_buckets, file_paths)

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
