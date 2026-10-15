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

try:
    from mace.utils.calc_detection import (
        PROPERTY_CALC_TYPES, calc_type_from_filename, d3_calc_type, deck_records,
        is_band_output, is_charge_potential_output, is_doss_output,
        is_frequency_output, is_optimization_output, is_transport_output)
except ImportError:  # pragma: no cover - run as a script from mace/
    from utils.calc_detection import (
        PROPERTY_CALC_TYPES, calc_type_from_filename, d3_calc_type, deck_records,
        is_band_output, is_charge_potential_output, is_doss_output,
        is_frequency_output, is_optimization_output, is_transport_output)

# === Define known error and completion message patterns === #
# The error lines and their priority are mace.utils.crystal_errors' table,
# shared with the recovery detector and the queue manager; this is that table
# per bucket (the bucket folders keep their names), for the GUI and livefile.
try:
    from mace.utils import crystal_errors
except ImportError:  # pragma: no cover - run as a script from mace/
    from utils import crystal_errors

# `mace run` stopped it (Ctrl-C, hang-up); its timeout is 'time'
_STOPPED = crystal_errors.ErrorClass(
    "stopped", ("mace run: STOPPED (interrupted",), "stopped", "stopped",
    "Stopped by mace run before it finished", "stopped on purpose")
_CLASSES = (_STOPPED,) + crystal_errors.ERROR_CLASSES

ERROR_PATTERNS = crystal_errors.bucket_patterns(
    ['too_many_scf', 'memory', 'quota', 'time', 'geometry_small_dist', 'shrink_error',
     'linear_basis', 'scratch', 'opt_unconverged', 'stopped', 'potential'])
ERROR_PATTERNS['stopped'] = list(_STOPPED.patterns)

# Error descriptions for better reporting
ERROR_DESCRIPTIONS = {
    'too_many_scf': "SCF convergence failure",
    'memory': "Out of memory error",
    'quota': "Disk quota/write error",
    'time': "Time limit exceeded",
    'geometry_small_dist': "Geometry error (atoms too close)",
    'shrink_error': "SHRINK parameter error",
    'linear_basis': "Linear dependency in basis set",
    'opt_unconverged': "Geometry optimization not converged (OPT END - FAILED)",
    'potential': "Segmentation fault/runtime error",
    'scratch': "No writable scratch directory (CRYSTAL not run)",
    'stopped': "Stopped by mace run before it finished",
}

# The line `mace run` appends to the .out of a deck it stopped. It says why the
# run ended, so it goes before the first error line - mpirun's "BAD
# TERMINATION" banner that the stop itself caused.
MACE_RUN_STOP_MARK = "mace run: STOPPED"

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
    # A properties run whose .d3 drives none of the types above (a matrix
    # dump - BASISSET with print codes 60/64 - for one)
    'PROPERTIES': 'completeproperties',
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
    'completeproperties': "Other properties (D3)",
}

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
            records = deck_records(f.read(), is_d3=True)
    except Exception:
        return None

    # Whole-line records, and never the free-text title that follows BAND:
    # MACE writes the material name there, which can hold any of these words.
    # CHARGE+POTENTIAL is ECH3/POT3 in MACE's own decks (ECHG/POTC for maps).
    return d3_calc_type(records)


def determine_completed_subtype(file_path: Path, lines, has_opt_end: bool = False) -> str:
    """
    Decide which calc type a successfully-completed .out file came from.

    Resolution order (first match wins):
      1. Sibling .d3 file's records (BAND/DOSS/BOLTZTRA/ECH3/POT3/ECHG/POTC;
         PROPERTIES for a .d3 with none of them)
      2. Lines CRYSTAL prints: TRANSPORT, FREQ, OPT (has_opt_end or the
         optimization lines), then the CHARGE+POTENTIAL, DOSS and BAND
         properties lines. Content goes before the file name because MACE
         chains the type into every follow-up name: an SP of an OPT is
         "X_opt_..._optimized_sp_..._optimized", and user names such as
         "X_SLAB_OPT_FSI" or "X_opt_tier7.freq" name the parent, not the run.
      3. The last properties type token of the file name (_band, _doss, ...),
         for a properties output with no .d3 beside it and no tell (a run
         that stopped early)
      4. Otherwise SP
    """
    parent = file_path.parent
    base_name = file_path.stem
    for ext in ('.d3', '.D3'):
        d3_candidate = parent / f"{base_name}{ext}"
        if d3_candidate.exists():
            d3_type = _detect_calc_type_from_d3(d3_candidate)
            if d3_type:
                return d3_type
            # .d3 exists but drives no type we know: still a properties run,
            # not a BAND unless its name says so
            name_type = calc_type_from_filename(file_path.name)
            return name_type if name_type in PROPERTY_CALC_TYPES else 'PROPERTIES'

    # From lines CRYSTAL prints, so an echoed title cannot supply the keyword
    content = ''.join(lines)
    if is_transport_output(content):
        return 'TRANSPORT'
    if is_frequency_output(content):
        return 'FREQ'
    if has_opt_end or is_optimization_output(content):
        return 'OPT'
    if is_charge_potential_output(content):
        return 'CHARGE+POTENTIAL'
    if is_doss_output(content):
        return 'DOSS'
    if is_band_output(content):
        return 'BAND'

    # A finished SCF with none of the tells above is an SP, whatever its name
    # says ("MoS2_band.out", "Fe_cp.out" are material names). The name only
    # decides for a properties run that stopped before printing its tell.
    if '== SCF ENDED' in content:
        return 'SP'
    name_type = calc_type_from_filename(file_path.name)
    if name_type in PROPERTY_CALC_TYPES:
        return name_type
    return 'SP'


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

    for line in lines:
        if line.startswith(MACE_RUN_STOP_MARK):
            return ('time' if "DUE TO TIME LIMIT" in line else 'stopped'), base_name

    # === Error pattern matching FIRST === #
    # In the shared priority order: an OOM kill before the BAD TERMINATION
    # banner printed above it, CRYSTAL's error before its MPI_Abort.
    match = crystal_errors.classify(lines, _CLASSES)
    if match:
        return match.bucket, base_name

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

    Only the calculation's own logs count, the names MACE's job scripts
    give them (``#SBATCH -o <name>-%J.o``; a job array task's link
    ``<name>-<A>_<T>.o``) and the other two common forms:

    - material_name-12345678.o
    - material_name_12345678.o
    - material_name.o12345678
    - material_name-12345678_3.o  (job array task 12345678_3)

    A log of another calculation whose name starts with this one
    (``si_sp-1.o`` for ``si``) is never read. When there are several, the
    newest wins: the newest log when an array task log is among them, else
    the highest job id.

    Args:
        out_file_path: Path to the .out file

    Returns:
        Job ID string if found, None otherwise
    """
    logs = _own_logs(out_file_path.parent, out_file_path.stem)
    if not logs:
        return None
    if any(is_task for _, _, is_task in logs):
        # an array task log is among them: the newest log wins (an older
        # plain log must not shadow a newer task, nor an older task the
        # newer single job a restart submitted)
        return max((mtime, jid) for mtime, jid, _ in logs)[1]
    return max(logs, key=lambda x: int(x[1]))[1]


# What may follow <name> in the name of its own SLURM log.
_OWN_TASK_LOG = re.compile(r'-(\d+_\d+)\.o')
_OWN_JOB_LOGS = (re.compile(r'-(\d+)\.o'), re.compile(r'_(\d{6,})\.o'), re.compile(r'\.o(\d+)'))


def _log_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:  # a link whose log is gone
        try:
            return path.lstat().st_mtime
        except OSError:
            return 0.0


def _own_logs(parent: Path, base: str) -> List[Tuple[float, str, bool]]:
    """(mtime, job id, is array task) of each SLURM log of ``base`` itself."""
    import glob as _glob
    logs = []
    for f in parent.glob(f"{_glob.escape(base)}*.o*"):
        rest = f.name[len(base):]
        m = _OWN_TASK_LOG.fullmatch(rest)
        if m:
            logs.append((_log_mtime(f), m.group(1), True))
            continue
        for rx in _OWN_JOB_LOGS:
            m = rx.fullmatch(rest)
            if m:
                logs.append((_log_mtime(f), m.group(1), False))
                break
    return logs


def _slurm_job_identity(job_id: str) -> Optional[Tuple[str, str]]:
    """(job name, working directory) SLURM gives for a queued job: squeue,
    else scontrol. None when neither answers."""
    try:
        from mace.queue.slurm_ids import slurm_env
        env = slurm_env()
    except Exception:
        env = None
    try:
        res = subprocess.run(['squeue', '-h', '-j', job_id, '-o', '%j|%Z'],
                             capture_output=True, text=True, timeout=30, env=env)
        lines = [l for l in res.stdout.splitlines() if l.strip()] if res.returncode == 0 else []
        if len(lines) == 1 and '|' in lines[0]:
            name, workdir = lines[0].split('|', 1)
            return name.strip(), workdir.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    try:
        res = subprocess.run(['scontrol', 'show', 'job', '-o', job_id],
                             capture_output=True, text=True, timeout=30, env=env)
        if res.returncode == 0:
            name = re.search(r'\bJobName=(.*?)\s+\w+=', res.stdout)
            workdir = re.search(r'\bWorkDir=(\S+)', res.stdout)
            if name and workdir:
                return name.group(1).strip(), workdir.group(1)
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _same_dir(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def _array_member(out_path: Path, task_id: str) -> Optional[Dict]:
    """The submitted.json of the MACE job array that ran task A_T for this
    output, with the member, or None."""
    import json
    array_id, task = task_id.split('_', 1)
    base, folder = out_path.stem, out_path.parent
    for root in (folder, folder.parent):
        for f in root.glob('.mace_arrays/*/submitted.json'):
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if str(data.get('array_job_id')) != array_id:
                continue
            for m in data.get('members') or []:
                if (str(m.get('task')) == task and m.get('job') == base
                        and _same_dir(m.get('dir', ''), folder)):
                    return data
    return None


def confirm_own_job(job_id: str, out_path: Path) -> Optional[str]:
    """Why ``job_id`` must not be canceled as the job of ``out_path``, or
    None when SLURM confirms it is: a plain job named like the calculation,
    working in its folder; a MACE job array task of an array whose record
    lists this calculation as that task, working in the array's folder."""
    who = _slurm_job_identity(job_id)
    if who is None:
        return "SLURM did not say which job this is (squeue/scontrol gave no answer)"
    name, workdir = who
    if '_' in job_id:
        data = _array_member(out_path, job_id)
        if data is None:
            return f"no MACE job array record lists {out_path.stem} as task {job_id}"
        if not _same_dir(workdir, data.get('array_dir', '')):
            return f"SLURM runs {job_id} in {workdir}, not in the array folder {data.get('array_dir')}"
        return None
    if name != out_path.stem:
        return f"SLURM names job {job_id} '{name}', not '{out_path.stem}'"
    if not _same_dir(workdir, out_path.parent):
        return f"SLURM runs job {job_id} in {workdir}, not in {out_path.parent}"
    return None


def get_running_jobs() -> Set[str]:
    """
    Get set of currently running job IDs from squeue.

    Returns:
        Set of job ID strings that are currently running
    """
    try:
        from mace.queue.slurm_ids import expand_queue_id, slurm_env
        user = os.environ.get('USER', '')
        result = subprocess.run(
            ['squeue', '-u', user, '-h', '-o', '%i'],
            capture_output=True, text=True, timeout=30, env=slurm_env()
        )
        if result.returncode == 0 and result.stdout.strip():
            # The pending tasks of a job array come as one token
            # (A_[2-3%2]): expanded to A_2, A_3. A token cut short is kept
            # as it is (it matches nothing).
            jobs = set()
            for token in result.stdout.strip().split('\n'):
                ids = expand_queue_id(token)
                if isinstance(ids, list):
                    jobs.update(ids)
                else:
                    jobs.add(token)
            return jobs
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

def _cancel_confirmed(zombies: List[Dict]):
    """scancel the jobs of ``zombies`` that SLURM confirms are theirs
    (confirm_own_job); never a job it does not, nor a whole job array.
    Returns (scancel result or None, the ids canceled)."""
    from mace.queue.slurm_ids import safe_scancel_many
    ids, refused = [], []
    for z in zombies:
        why = confirm_own_job(z['job_id'], z['path'])
        if why:
            refused.append((z['job_id'], why))
        elif z['job_id'] not in ids:
            ids.append(z['job_id'])
    # Task ids (A_T) of a job array are canceled one task at a
    # time; a bare array id would cancel every task and is refused.
    result, more = (safe_scancel_many(ids, search_dirs=[z['path'].parent for z in zombies])
                    if ids else (None, []))
    for jid, reason in refused + more:
        print(f"\n✗ Not canceled {jid}: {reason}")
    canceled = [j for j in ids if j not in {r[0] for r in more}]
    return result, canceled


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

    # Show completed zombies first (these should definitely be canceled)
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

    # Combine completed and error zombies - both should be canceled
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
            result, canceled = _cancel_confirmed(definite_zombies)

            if result is not None and result.returncode == 0:
                print(f"\n✓ Canceled {len(canceled)} zombie job(s):")
                for jid in canceled:
                    print(f"  scancel {jid}")
            elif result is not None:
                print(f"\n✗ Error canceling jobs: {result.stderr}")
        else:
            print("\nNo zombie jobs canceled.")
            if definite_ids:
                print("To cancel manually:")
                for jid in definite_ids:
                    print(f"  scancel {jid}")

    # Optionally ask about ongoing jobs
    if ongoing_zombies:
        print()
        response = input(f"Also cancel {len(ongoing_zombies)} ongoing job(s) that may be stuck? [y/N]: ").strip().lower()

        if response == 'y':
            result, canceled = _cancel_confirmed(ongoing_zombies)

            if result is not None and result.returncode == 0:
                print(f"\n✓ Canceled {len(canceled)} ongoing job(s):")
                for jid in canceled:
                    print(f"  scancel {jid}")
            elif result is not None:
                print(f"\n✗ Error canceling jobs: {result.stderr}")
        else:
            print("\nNo ongoing jobs canceled.")

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

Status categories:
  Complete   finished (OPT END or TOTAL CPU TIME), sorted by calculation type
             (OPT, SP, FREQ, BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL)
  Errors     failed, sorted by the error found (time limit, SCF, memory, ...)
  Ongoing    no completion line and no error yet: still running or pending
  Unknown    an error line that matches no known pattern
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
                       help='Detect and cancel jobs that failed but are still running in SLURM '
                            "(only a calculation's own, SLURM-confirmed job)")

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
