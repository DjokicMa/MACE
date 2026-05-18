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

# === Completed-calc subtype handling === #
# Maps detected calc subtype to bucket name used for organizing files
CALC_TYPE_TO_BUCKET = {
    'OPT': 'complete',
    'SP': 'completesp',
    'FREQ': 'completefreq',
    'BAND': 'completeband',
    'DOSS': 'completedoss',
    'TRANSPORT': 'completetransport',
    'CHARGE+POTENTIAL': 'completecharge_potential',
}

COMPLETED_BUCKETS = list(CALC_TYPE_TO_BUCKET.values())

COMPLETED_BUCKET_DESCRIPTIONS = {
    'complete': "Geometry optimization (OPT END)",
    'completesp': "Single point energy (SP)",
    'completefreq': "Frequency calculation (FREQ)",
    'completeband': "Band structure (D3 BAND)",
    'completedoss': "Density of states (D3 DOSS)",
    'completetransport': "Transport properties (D3 TRANSPORT)",
    'completecharge_potential': "Charge density + potential (D3)",
}

# Filename suffix → calc type. Numbered variants like _opt2, _band3 also match.
# Patterns are evaluated in order; the first match wins so list more specific
# tokens (charge_potential) before generic ones (charge, potential).
_FILENAME_CALC_TYPE_PATTERNS = [
    (re.compile(r'_band\d*(?:_|$)'), 'BAND'),
    (re.compile(r'_doss\d*(?:_|$)'), 'DOSS'),
    (re.compile(r'_dos\d*(?:_|$)'), 'DOSS'),
    (re.compile(r'_transport\d*(?:_|$)'), 'TRANSPORT'),
    (re.compile(r'_transp\d*(?:_|$)'), 'TRANSPORT'),
    (re.compile(r'_charge[_+]potential\d*(?:_|$)'), 'CHARGE+POTENTIAL'),
    (re.compile(r'_chargepot\d*(?:_|$)'), 'CHARGE+POTENTIAL'),
    (re.compile(r'_cp\d*(?:_|$)'), 'CHARGE+POTENTIAL'),
    (re.compile(r'_charge\d*(?:_|$)'), 'CHARGE+POTENTIAL'),
    (re.compile(r'_potential\d*(?:_|$)'), 'CHARGE+POTENTIAL'),
    (re.compile(r'_freq\d*(?:_|$)'), 'FREQ'),
    (re.compile(r'_sp\d*(?:_|$)'), 'SP'),
    (re.compile(r'_opt\d*(?:_|$)'), 'OPT'),
]

# === Default extensions for organizing files === #
# Includes all files produced/copied back by submit_prop.sh for d3 calcs
# (BAND.DAT, DOSS.DAT, fort.25 → .f25, transport .DAT files, cube files)
# plus standard inputs/outputs.
DEFAULT_EXTENSIONS = [
    # Inputs and submission artifacts
    '.sh', '.out', '.d12', '.d3',
    # CRYSTAL wavefunction / phonon binaries
    '.f9', '.f25',
    # D3 BAND/DOSS/POT 1D outputs
    '.BAND.DAT', '.DOSS.DAT', '.POTC.DAT',
    # Transport outputs
    '.SIGMA.DAT', '.SEEBECK.DAT', '.SIGMAS.DAT', '.KAPPA.DAT', '.TDF.DAT',
    # 3D cube outputs (CHARGE+POTENTIAL)
    '_DENS.CUBE', '_POT.CUBE', '_SPIN.CUBE',
    # FREQ-related outputs (when the user copies them back from scratch)
    '.FREQINFO.DAT', '.BORN.DAT', '.IRSPEC.DAT', '.RAMSPEC.DAT',
    '.HESSOPT.DAT', '.IRREFR.DAT',
]


def _detect_calc_type_from_d3(d3_file: Path) -> Optional[str]:
    """Inspect a .d3 input file to decide which property calc it drives."""
    try:
        with open(d3_file, 'r', errors='ignore') as f:
            content = f.read().upper()
    except Exception:
        return None

    if 'BOLTZTRA' in content:
        return 'TRANSPORT'
    if 'ECHG' in content or 'POTC' in content:
        return 'CHARGE+POTENTIAL'
    if 'DOSS' in content:
        return 'DOSS'
    if 'BAND' in content:
        return 'BAND'
    return None


def determine_completed_subtype(file_path: Path, lines, has_opt_end: bool = False) -> str:
    """
    Decide which calc type a successfully-completed .out file came from.

    Resolution order (first match wins):
      1. Filename suffix (_opt, _sp, _freq, _band, _doss, _transport, _charge*, etc.)
      2. Sibling .d3 file's keywords (BAND/DOSS/BOLTZTRA/ECHG/POTC)
      3. Content tells (TRANSPORT, FREQ markers)
      4. has_opt_end fallback → OPT, otherwise SP
    """
    base_lower = file_path.stem.lower()
    for pattern, calc_type in _FILENAME_CALC_TYPE_PATTERNS:
        if pattern.search(base_lower):
            return calc_type

    parent = file_path.parent
    base_name = file_path.stem
    for ext in ('.d3', '.D3'):
        d3_candidate = parent / f"{base_name}{ext}"
        if d3_candidate.exists():
            d3_type = _detect_calc_type_from_d3(d3_candidate)
            if d3_type:
                return d3_type
            # .d3 exists but unrecognized — still definitely a properties calc
            return 'BAND'

    content = ''.join(lines)
    if re.search(r'SEEBECK COEFFICIENT|BOLTZTRA', content, re.IGNORECASE):
        return 'TRANSPORT'
    if re.search(r'VIBRATIONAL FREQUENCIES|FREQUENCY CALCULATION|MODES\s+EIGV', content, re.IGNORECASE):
        return 'FREQ'

    return 'OPT' if has_opt_end else 'SP'


# === Initialize result buckets === #
def initialize_buckets():
    """Initialize categorization buckets"""
    categories = list(ERROR_PATTERNS.keys()) + COMPLETED_BUCKETS + ["unknown", "ongoing"]
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
    # Detection logic unchanged: OPT END and TOTAL CPU TIME = are the same
    # signals that have always decided "did this calc finish?". Subtype
    # routing happens only after one of these fires.
    has_opt_end = any("OPT END" in line for line in lines)
    has_cpu_time = any("    TOTAL CPU TIME =" in line for line in lines)

    if has_opt_end:
        subtype = determine_completed_subtype(Path(file_path), lines, has_opt_end=True)
        return CALC_TYPE_TO_BUCKET.get(subtype, 'complete'), base_name
    elif has_cpu_time:
        subtype = determine_completed_subtype(Path(file_path), lines, has_opt_end=False)
        return CALC_TYPE_TO_BUCKET.get(subtype, 'completesp'), base_name

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
    bucket_counts = {b: len(result_buckets.get(b, [])) for b in COMPLETED_BUCKETS}
    total_complete = sum(bucket_counts.values())

    if total_complete > 0:
        print(f"✓ COMPLETED: {total_complete} calculation(s)")
        for bucket in COMPLETED_BUCKETS:
            count = bucket_counts[bucket]
            if count > 0:
                desc = COMPLETED_BUCKET_DESCRIPTIONS.get(bucket, bucket)
                print(f"  └─ {desc}: {count}")

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
        extensions: List of file suffixes to move. Defaults to DEFAULT_EXTENSIONS,
            which covers .d12/.d3 inputs, .out, .sh, .f9/.f25, and every
            properties-style output that submit_prop.sh copies back
            (.BAND.DAT, .DOSS.DAT, .POTC.DAT, transport .DAT files, cube files).
    """
    if extensions is None:
        extensions = list(DEFAULT_EXTENSIONS)

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

    Buckets each calc into a per-type subfolder (complete, completesp,
    completefreq, completeband, completedoss, completetransport,
    completecharge_potential) so d3 outputs land separately from SP/OPT.

    Args:
        result_buckets: Dictionary of categorized files
        target_dir: Directory for completed files
        extensions: List of file suffixes to move (default: DEFAULT_EXTENSIONS)
    """
    if extensions is None:
        extensions = list(DEFAULT_EXTENSIONS)

    completed_only = {bucket: result_buckets.get(bucket, []) for bucket in COMPLETED_BUCKETS}
    total_completed = sum(len(v) for v in completed_only.values())

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
    Detect jobs that have finished (completed or failed) but are still running in SLURM.

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

    # Check ALL categories: completed jobs, errored jobs, and ongoing jobs
    # Completed jobs should NOT still be running in SLURM
    all_categories = (
        COMPLETED_BUCKETS +  # Successfully completed (all complete* variants)
        list(ERROR_PATTERNS.keys()) +  # Failed with errors
        ['unknown', 'ongoing']  # Unknown errors or still running
    )

    for category in all_categories:
        for base_name in result_buckets.get(category, []):
            out_path = file_paths.get(base_name)
            if not out_path:
                continue

            job_id = find_slurm_job_id(out_path)
            if job_id and job_id in running_jobs:
                # Set appropriate description based on category
                if category in COMPLETED_BUCKET_DESCRIPTIONS:
                    description = f"{COMPLETED_BUCKET_DESCRIPTIONS[category]} completed but SLURM job still running"
                elif category == 'ongoing':
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
        print("  (All finished jobs have already stopped in SLURM)")
        return

    # Separate into three categories: completed, errors, and ongoing
    completed_buckets_set = set(COMPLETED_BUCKETS)
    completed_zombies = [z for z in zombies if z['category'] in completed_buckets_set]
    error_zombies = [z for z in zombies if z['category'] not in completed_buckets_set and z['category'] != 'ongoing']
    ongoing_zombies = [z for z in zombies if z['category'] == 'ongoing']

    # Print header and explanation
    print(f"\n{'='*70}")
    print("ZOMBIE JOB DETECTION")
    print(f"{'='*70}")
    print("\nZombie jobs are calculations that have FINISHED (completed or failed)")
    print("but are STILL RUNNING in SLURM, wasting CPU hours.\n")

    # Show completed zombies first (these should definitely be cancelled)
    if completed_zombies:
        print(f"Found {len(completed_zombies)} COMPLETED job(s) still running in SLURM:\n")

        for z in completed_zombies:
            print(f"  • {z['name']}")
            print(f"    Job ID: {z['job_id']}")
            print(f"    Status: {z['description']}")
            print(f"    Path: {z['path'].parent}")
            print()

    # Show error zombies (also definite zombies)
    if error_zombies:
        print(f"Found {len(error_zombies)} FAILED job(s) still running in SLURM:\n")

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

    # Combine completed and error zombies - both should be cancelled
    definite_zombies = completed_zombies + error_zombies

    if definite_zombies:
        # Ask about definite zombies (completed + errors)
        definite_ids = [z['job_id'] for z in definite_zombies]
        num_completed = len(completed_zombies)
        num_errors = len(error_zombies)

        prompt_parts = []
        if num_completed > 0:
            prompt_parts.append(f"{num_completed} completed")
        if num_errors > 0:
            prompt_parts.append(f"{num_errors} failed")

        prompt = f"Cancel {' + '.join(prompt_parts)} zombie job(s)? [y/N]: "
        response = input(prompt).strip().lower()

        if response == 'y':
            result = subprocess.run(
                ['scancel'] + definite_ids,
                capture_output=True, text=True
            )

            if result.returncode == 0:
                print(f"\n✓ Cancelled {len(definite_ids)} zombie job(s):")
                for jid in definite_ids:
                    print(f"  scancel {jid}")
            else:
                print(f"\n✗ Error cancelling jobs: {result.stderr}")
        else:
            print("\nNo zombie jobs cancelled.")
            if definite_ids:
                print("To cancel manually:")
                for jid in definite_ids:
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
                       default=list(DEFAULT_EXTENSIONS),
                       help=('File suffixes to move. Default covers .sh/.out/.d12/.d3 inputs, '
                             '.f9/.f25, all d3 .DAT outputs (BAND/DOSS/POTC/SIGMA/SEEBECK/'
                             'SIGMAS/KAPPA/TDF), cube files (_DENS/_POT/_SPIN.CUBE), and '
                             'FREQ-related .DAT files.'))
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
