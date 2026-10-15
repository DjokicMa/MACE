#!/usr/bin/env python3
"""
Module to populate database with completed calculations found in workflow output directories.
This is crucial for workflow progression in isolated context mode.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

from mace.completion_checker import CALC_TYPE_TO_BUCKET, categorize_output_file
from mace.database.materials import create_material_id_from_file


_BUCKET_TO_CALC_TYPE = {bucket: calc_type
                        for calc_type, bucket in CALC_TYPE_TO_BUCKET.items()}


def _completed_calc_type(category: str) -> str:
    """The calculation type behind a completion checker 'complete*' bucket."""
    return _BUCKET_TO_CALC_TYPE.get(category, 'OPT')


def scan_for_completed_calculations(base_dir: Path) -> List[Dict]:
    """
    Scan directory for completed CRYSTAL calculations.
    
    Args:
        base_dir: Base directory to scan
        
    Returns:
        List of calculation info dictionaries
    """
    completed_calcs = []
    
    # Look for .out files
    for out_file in base_dir.rglob("*.out"):
        # Skip if file is empty or doesn't exist
        if not out_file.exists() or out_file.stat().st_size == 0:
            continue
        # MACE's X_temp.out beside the run's X.out is a copy of the same run:
        # registered once, by X.out (the second record was a duplicate)
        from mace.utils.db_build import temp_copy_of
        if temp_copy_of(out_file) is not None:
            continue
            
        # Check if calculation completed — same validated detector as the
        # rest of the pipeline. The old test was `"TERMINATION" in content`,
        # which the MPI "BAD TERMINATION" failure banner itself satisfies:
        # an OOM-killed SP (0-byte fort.9) was adopted as completed and
        # BAND/DOSS were fanned out from its empty wavefunction.
        try:
            category, _ = categorize_output_file(out_file)
            if not category.startswith('complete'):
                continue
                    
            # Derive the material id with the SAME canonical function the workflow
            # engine/executor use (create_material_id_from_file). A local
            # trailing-suffix strip loop diverged from canonical for numbered and
            # continuation filenames (e.g. 'mat_opt2.out' -> 'mat_opt2', and
            # 'mat_opt_B3LYP-D3_optimized.out' -> the whole stem) instead of 'mat',
            # so scanned OPT->SP continuations were re-registered as brand-new
            # materials, silently duplicating material rows on every workflow scan.
            material_name = create_material_id_from_file(out_file.name)

            # The calculation type is the completion checker's own verdict
            # (the bucket it just sorted the file into), so this scan and
            # `mace check` agree. The old file-name substring test recorded a
            # name without a type token ("X.freq.out", "X_SLAB_OPT_FSI.out") as
            # OPT, and "X_opt_..._sp_..._optimized_freq_guessp" as SP because
            # "_sp" came before "_freq" in its test order.
            calc_type = _completed_calc_type(category)

            # Look for corresponding input file. Use the actual output stem (NOT
            # the canonical material id, which strips calc/functional tokens) so the
            # sibling .d12 is still found for continuation filenames.
            d12_file = out_file.with_suffix('.d12')
            if not d12_file.exists():
                d12_file = out_file.parent / f"{out_file.stem}.d12"
            if not d12_file.exists():
                d12_file = out_file.parent / f"{out_file.stem}_temp.d12"   # the deck MACE ran it from
                
            calc_info = {
                'material_id': material_name,
                'calc_type': calc_type,
                'output_file': str(out_file),
                'input_file': str(d12_file) if d12_file.exists() else None,
                'work_dir': str(out_file.parent),
                'completed': True,
                'has_termination': True
            }
            
            job_id = slurm_job_id_for(out_file)
            if job_id:
                calc_info['slurm_job_id'] = job_id

            completed_calcs.append(calc_info)
            
        except Exception as e:
            print(f"  Error scanning {out_file}: {e}")
            continue
            
    return completed_calcs


def slurm_job_id_for(out_file: Path) -> Optional[str]:
    """The SLURM job id of an output from the job logs beside it
    (<name>-<id>.o). A MACE job array task leaves <name>-<A>_<T>.o; then the
    id is A_T, and the newest log wins (completion_checker.find_slurm_job_id)."""
    out_file = Path(out_file)
    o_files = list(out_file.parent.glob(f"{out_file.stem}*.o*"))
    if any(re.search(r'-\d+_\d+\.o$', f.name) for f in o_files):
        from mace.completion_checker import find_slurm_job_id
        return find_slurm_job_id(out_file) or None
    for o_file in o_files:
        try:
            # Extract job ID from filename like material-12345.o
            parts = o_file.stem.split('-')
            if len(parts) >= 2 and parts[-1].split('.')[0].isdigit():
                return parts[-1].split('.')[0]
        except Exception:
            pass
    return None


def _job_owner(db, job_id: str) -> Optional[Dict]:
    """The calculation that has this SLURM job id, or None."""
    with db._get_connection() as conn:
        row = conn.execute("SELECT calc_id, output_file, work_dir FROM calculations WHERE slurm_job_id = ?",
                           (job_id,)).fetchone()
    return dict(row) if row else None


def _check_job_id(db, calc_info: Dict) -> Optional[str]:
    """The job id is unique in the calculations table. When another
    calculation (another folder, its own output) already has this output's
    job id: an identical output is a copy of that run - 'copy' (not
    registered again); a different one is registered without the job id
    (with a warning) - 'drop'. None otherwise."""
    job = calc_info.get('slurm_job_id')
    if not job:
        return None
    owner = _job_owner(db, job)
    if not owner or not owner.get('output_file'):
        return None
    from mace.utils.db_build import same_output, spellings
    out = calc_info.get('output_file')
    if not out or set(spellings(Path(owner['output_file']))) & set(spellings(Path(out))):
        return None
    if owner.get('work_dir') and str(owner['work_dir']).rstrip('/') == str(calc_info.get('work_dir') or '').rstrip('/'):
        return None      # the same folder: the usual record matching below decides
    if same_output(Path(owner['output_file']), Path(out)):
        print(f"  {out} is the same run as {owner['output_file']} (SLURM job {job}): not registered again")
        return 'copy'
    print(f"  ⚠️  SLURM job {job} is already recorded for {owner['output_file']}; "
          f"{out} is registered without the job id")
    return 'drop'


def _complete_left_pending(db, calc_info: Dict) -> None:
    """The row an earlier scan left for this copy when its job id clashed
    (created, then the status update failed): 'pending', no output file, no
    job id, the copy's folder and deck. It gets the copy as its output and
    the completed status (without the job id, which the run's first record
    has); nothing is deleted."""
    stem = Path(calc_info['output_file']).stem
    base = str(calc_info.get('calc_type') or '').rstrip('0123456789')
    with db._get_connection() as conn:
        rows = conn.execute(
            "SELECT calc_id, calc_type, input_file FROM calculations WHERE status = 'pending' "
            "AND output_file IS NULL AND slurm_job_id IS NULL AND rtrim(work_dir, '/') = ?",
            (str(calc_info.get('work_dir') or '').rstrip('/'),)).fetchall()
    for r in rows:
        if str(r['calc_type'] or '').rstrip('0123456789') == base and r['input_file'] \
                and Path(r['input_file']).stem == stem:
            db.update_calculation_status(r['calc_id'], 'completed', output_file=calc_info['output_file'])
            print(f"  Completed {r['calc_id']}: the row an earlier scan left pending for {calc_info['output_file']}")


def _structure_key(output_file, input_file=None) -> Optional[tuple]:
    """(reduced formula, space group) of a run, from its output (the deck
    as fallback); None when neither gives a formula."""
    try:
        from mace.utils.formula_extractor import material_info, extract_space_group_from_output
        info = material_info(output_file=output_file, d12_file=input_file)
        if not info.get('formula'):
            return None
        sg = extract_space_group_from_output(output_file) if output_file and Path(output_file).is_file() else None
        return (info['formula'], int(sg) if sg else None)
    except Exception:
        return None


def _material_structure(db, material_id: str, cache: Dict) -> Optional[tuple]:
    """(reduced formula, space group) of an existing material, from the
    first of its calculations whose output can be read (never the row's
    formula, which may be typed by hand); None when none can."""
    if material_id in cache:
        return cache[material_id]
    key = None
    with db._get_connection() as conn:
        calcs = conn.execute("SELECT output_file, input_file FROM calculations WHERE material_id = ? "
                             "AND output_file IS NOT NULL ORDER BY created_at", (material_id,)).fetchall()
    for out, inp in calcs:
        if out and Path(out).is_file():
            key = _structure_key(out, inp if inp and Path(inp).is_file() else None)
            if key:
                break
    cache[material_id] = key
    return key


def _differs(a: Optional[tuple], b: Optional[tuple]) -> bool:
    """Clearly different structures: another reduced formula, or the same
    formula in another space group (both known)."""
    if not a or not b:
        return False
    if a[0] != b[0]:
        return True
    return bool(a[1] and b[1] and a[1] != b[1])


def _recorded_material(db, output_file: str) -> Optional[str]:
    """The material of a calculation that already records this output."""
    from mace.utils.db_build import spellings
    outs = spellings(Path(output_file))
    with db._get_connection() as conn:
        r = conn.execute(f"SELECT material_id FROM calculations WHERE output_file IN ({','.join('?' * len(outs))}) "
                         "ORDER BY created_at LIMIT 1", outs).fetchone()
    return r[0] if r else None


def distinct_material_id(db, calc_info: Dict, cache: Dict) -> str:
    """The material a scanned calculation belongs to. Outputs of the same
    name in different folders (A/mat_opt.out, B/mat_opt.out) map to one id;
    when the material that id names holds a clearly different structure
    (material_info's reduced formula, or the space group), the calculation
    gets the id plus its folder, as Batch Setup prefixes same-name parents
    ("mat_B"; a short hash of the folder when that is taken by yet another
    structure), with a warning. The same structure in another folder (a
    copied project) stays the same material. A calculation already recorded
    keeps its material: existing rows are never renamed."""
    material_id = calc_info['material_id']
    out = calc_info.get('output_file')
    if not out:
        return material_id
    recorded = _recorded_material(db, out)
    if recorded:
        return recorded
    if not db.get_material(material_id):
        return material_id
    mine = _structure_key(out, calc_info.get('input_file'))
    if not _differs(_material_structure(db, material_id, cache), mine):
        return material_id
    import hashlib
    folder = re.sub(r'[^A-Za-z0-9_-]+', '_', Path(out).parent.name) or 'run'
    cands = [f"{material_id}_{folder}",
             f"{material_id}_{folder}_{hashlib.sha1(str(Path(out).parent.resolve()).encode()).hexdigest()[:6]}"]
    for cand in cands:
        if not db.get_material(cand) or not _differs(_material_structure(db, cand, cache), mine):
            print(f"  ⚠️  {out}: a different structure ({mine[0]}) from material {material_id}; "
                  f"recorded as material {cand}")
            cache.setdefault(cand, mine)
            return cand
    return cands[-1]


def _correct_calc_type(db, existing: Dict, scanned_type: str) -> None:
    """Give a record the type its output shows, when an older scan typed it
    from the file name (1.1.1 recorded an SP named "X_SLAB_OPT_FSI.out" as
    OPT). Only a plain base type the scan itself writes is corrected: numbered
    types (OPT2, SP2, ...) come from the workflow engine, which knows what it
    ran. The calc_id keeps its old name - other tables refer to it."""
    old = existing.get('calc_type')
    if old not in CALC_TYPE_TO_BUCKET or old == scanned_type:
        return
    try:
        with db._get_connection() as conn:
            conn.execute("UPDATE calculations SET calc_type = ? WHERE calc_id = ?",
                         (scanned_type, existing['calc_id']))
        print(f"  Corrected {existing['calc_id']}: calc type {old} -> {scanned_type} (from its output)")
    except Exception as e:
        print(f"  Could not correct the calc type of {existing['calc_id']}: {e}")


def _stale_for_array_task(existing: Dict, calc_info: Dict) -> bool:
    """True when ``existing`` is a job array task (slurm_job_id A_T) and the
    scanned output is not its own output written after its submission: the
    tasks of an array often share one directory, and a resubmitted deck
    keeps the output of its last run until the task writes a new one."""
    job_id = str(existing.get('slurm_job_id') or '')
    if not re.match(r'^\d+_\d+$', job_id):
        return False
    out = calc_info.get('output_file')
    stem = Path(existing['input_file']).stem if existing.get('input_file') else None
    if not out or not stem or Path(out).stem != stem:
        return True
    submitted = existing.get('submitted_at')
    if not submitted:
        return False
    try:
        return Path(out).stat().st_mtime < datetime.fromisoformat(submitted).timestamp() - 2
    except (OSError, ValueError):
        return True


def populate_database(completed_calcs: List[Dict], db) -> int:
    """
    Populate database with completed calculations.
    
    Args:
        completed_calcs: List of calculation info from scan_for_completed_calculations
        db: MaterialDatabase instance
        
    Returns:
        Number of calculations added
    """
    added_count = 0
    structures: Dict = {}       # material id -> (formula, space group), for distinct_material_id
    
    for calc_info in completed_calcs:
        try:
            duplicate = _check_job_id(db, calc_info)
            if duplicate == 'copy':
                _complete_left_pending(db, calc_info)
                continue
            if duplicate == 'drop':
                calc_info = dict(calc_info, slurm_job_id=None)
            material_id = distinct_material_id(db, calc_info, structures)
            calc_info = dict(calc_info, material_id=material_id)
            calc_type = calc_info['calc_type']
            
            # Check if material exists, create if not
            material = db.get_material(material_id)
            if not material:
                # Create material with minimal info; the dimensionality is
                # the one CRYSTAL prints (a molecule was stored as CRYSTAL)
                print(f"  Creating material: {material_id}")
                dimensionality = None
                if calc_info.get('output_file'):
                    try:
                        from mace.utils.formula_extractor import output_dimensionality
                        dimensionality = output_dimensionality(calc_info['output_file'])
                    except Exception:
                        dimensionality = None
                db.create_material(
                    material_id=material_id,
                    formula="Unknown",  # Will be updated later from output
                    space_group=1,      # Will be updated later
                    dimensionality=dimensionality or "CRYSTAL",
                    source_type="d12",
                    source_file=calc_info.get('input_file', 'unknown')
                )
                
            # Create calculation ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
            calc_id = f"{material_id}_{calc_type}_{timestamp}"
            
            # Check if this calculation already exists (by output file)
            try:
                existing_calcs = db.get_material_calculations(material_id)
            except AttributeError:
                # Fallback for older database interface
                existing_calcs = []
                all_calcs = db.get_all_calculations()
                for calc in all_calcs:
                    if calc.get('material_id') == material_id:
                        existing_calcs.append(calc)
            
            already_exists = False
            for existing in existing_calcs:
                # Match by output file, or -- for records whose stored output_file
                # (a submission-time guess set by the executor) differs from the
                # actual scanned filename -- by work_dir + base calc type. The scan
                # always emits a BASE calc type (OPT/SP/BAND/...), while the engine
                # persists NUMBERED steps (OPT2/SP2/BAND3), so compare on the
                # digit-stripped base or a second OPT/SP would dedup-miss.
                same_output = (
                    calc_info.get('output_file')
                    and existing.get('output_file') == calc_info.get('output_file')
                )
                same_workdir = (
                    existing.get('work_dir') and calc_info.get('work_dir')
                    and str(existing['work_dir']).rstrip('/') == str(calc_info['work_dir']).rstrip('/')
                    and str(existing.get('calc_type') or '').rstrip('0123456789')
                        == str(calc_info.get('calc_type') or '').rstrip('0123456789')
                )
                if same_output or same_workdir:
                    already_exists = True
                    if _stale_for_array_task(existing, calc_info):
                        # A job array task is judged from its own output
                        # written after its submission, never an older one.
                        break
                    _correct_calc_type(db, existing, calc_type)
                    # Update status to completed if needed
                    if existing.get('status') != 'completed':
                        job = calc_info.get('slurm_job_id')
                        owner = _job_owner(db, job) if job else None
                        if owner and owner['calc_id'] != existing['calc_id']:
                            job = None      # another calculation has it (unique)
                        db.update_calculation_status(
                            existing['calc_id'],
                            'completed',
                            slurm_job_id=job,
                            output_file=calc_info.get('output_file')
                        )
                        print(f"  Updated {existing['calc_id']} to completed status")
                    break
                    
            if not already_exists:
                # Create new calculation record
                calc_id = db.create_calculation(
                    material_id=material_id,
                    calc_type=calc_type,
                    input_file=calc_info.get('input_file'),
                    work_dir=calc_info.get('work_dir')
                )
                
                # Update with completion info; output_file MUST be stored on
                # the record — the dedup check above compares against it, so
                # leaving it NULL re-added every calculation on each scan.
                # A failed update takes the new row back out: a 'pending' row
                # with no output file was left behind, and the build filed
                # the output's properties under it.
                try:
                    db.update_calculation_status(
                        calc_id,
                        'completed',
                        slurm_job_id=calc_info.get('slurm_job_id'),
                        output_file=calc_info.get('output_file')
                    )
                except Exception:
                    with db._get_connection() as conn:
                        conn.execute("DELETE FROM calculations WHERE calc_id = ?", (calc_id,))
                    raise

                # Also keep the output file path in the settings JSON
                if calc_info.get('output_file'):
                    try:
                        # Get current settings
                        current_calc = db.get_calculation(calc_id)
                        if current_calc:
                            import json
                            settings = json.loads(current_calc.get('settings_json') or '{}')
                            settings['output_file'] = calc_info.get('output_file')
                            # Carry the workflow the job belongs to. This scan runs
                            # from the completion callback, where the job script has
                            # exported the authoritative id; without it the record
                            # looks workflow-less and progression cannot resolve the
                            # plan or the output directory from it.
                            env_wfid = os.environ.get('MACE_WORKFLOW_ID')
                            if env_wfid and not settings.get('workflow_id'):
                                settings['workflow_id'] = env_wfid
                            db.update_calculation_settings(calc_id, settings)
                    except Exception as e:
                        print(f"  Failed to update output file in settings: {e}")

                added_count += 1
                print(f"  Added completed calculation: {calc_id}")
                
        except Exception as e:
            print(f"  Error adding calculation to database: {e}")
            continue
            
    return added_count


def main():
    """CLI interface for testing."""
    import argparse
    from mace.database.materials import MaterialDatabase
    
    parser = argparse.ArgumentParser(description="Populate database with completed calculations")
    parser.add_argument("scan_dir", help="Directory to scan for completed calculations")
    parser.add_argument("--db", default="materials.db", help="Database path")
    
    args = parser.parse_args()
    
    # Initialize database
    db = MaterialDatabase(args.db)
    
    # Scan for completed calculations
    print(f"Scanning {args.scan_dir} for completed calculations...")
    completed_calcs = scan_for_completed_calculations(Path(args.scan_dir))
    print(f"Found {len(completed_calcs)} completed calculations")
    
    if completed_calcs:
        # Populate database
        added = populate_database(completed_calcs, db)
        print(f"Added {added} new calculations to database")
        
        # Show summary
        materials = db.get_all_materials()
        calcs = db.get_all_calculations()
        print(f"\nDatabase now contains:")
        print(f"  - {len(materials)} materials")
        print(f"  - {len(calcs)} calculations")


if __name__ == "__main__":
    main()