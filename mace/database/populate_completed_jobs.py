#!/usr/bin/env python3
"""
Module to populate database with completed calculations found in workflow output directories.
This is crucial for workflow progression in isolated context mode.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

from mace.completion_checker import categorize_output_file
from mace.database.materials import create_material_id_from_file


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

            # Determine calculation type (D3 property types before SP/FREQ so
            # their records dedup against engine-created ones instead of
            # being re-registered as OPT)
            calc_type = 'OPT'  # Default
            stem_lower = out_file.stem.lower()
            if '_doss' in stem_lower:
                calc_type = 'DOSS'
            elif '_band' in stem_lower:
                calc_type = 'BAND'
            elif '_transport' in stem_lower:
                calc_type = 'TRANSPORT'
            elif '_charge+potential' in stem_lower or '_charge_potential' in stem_lower:
                calc_type = 'CHARGE+POTENTIAL'
            elif '_sp' in stem_lower:
                calc_type = 'SP'
            elif '_freq' in stem_lower:
                calc_type = 'FREQ'
            elif out_file.parent.name.startswith('step_') and '_OPT' in out_file.parent.name:
                calc_type = 'OPT'
                
            # Look for corresponding input file. Use the actual output stem (NOT
            # the canonical material id, which strips calc/functional tokens) so the
            # sibling .d12 is still found for continuation filenames.
            d12_file = out_file.with_suffix('.d12')
            if not d12_file.exists():
                d12_file = out_file.parent / f"{out_file.stem}.d12"
                
            calc_info = {
                'material_id': material_name,
                'calc_type': calc_type,
                'output_file': str(out_file),
                'input_file': str(d12_file) if d12_file.exists() else None,
                'work_dir': str(out_file.parent),
                'completed': True,
                'has_termination': True
            }
            
            # Try to find SLURM job ID from .o files
            for o_file in out_file.parent.glob(f"{out_file.stem}*.o*"):
                try:
                    # Extract job ID from filename like material-12345.o
                    parts = o_file.stem.split('-')
                    if len(parts) >= 2 and parts[-1].split('.')[0].isdigit():
                        calc_info['slurm_job_id'] = parts[-1].split('.')[0]
                        break
                except:
                    pass
                    
            completed_calcs.append(calc_info)
            
        except Exception as e:
            print(f"  Error scanning {out_file}: {e}")
            continue
            
    return completed_calcs


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
    
    for calc_info in completed_calcs:
        try:
            material_id = calc_info['material_id']
            calc_type = calc_info['calc_type']
            
            # Check if material exists, create if not
            material = db.get_material(material_id)
            if not material:
                # Create material with minimal info
                print(f"  Creating material: {material_id}")
                db.create_material(
                    material_id=material_id,
                    formula="Unknown",  # Will be updated later from output
                    space_group=1,      # Will be updated later
                    dimensionality="CRYSTAL",
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
                    # Update status to completed if needed
                    if existing.get('status') != 'completed':
                        db.update_calculation_status(
                            existing['calc_id'],
                            'completed',
                            slurm_job_id=calc_info.get('slurm_job_id'),
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
                # leaving it NULL re-added every calculation on each scan
                db.update_calculation_status(
                    calc_id,
                    'completed',
                    slurm_job_id=calc_info.get('slurm_job_id'),
                    output_file=calc_info.get('output_file')
                )

                # Also keep the output file path in the settings JSON
                if calc_info.get('output_file'):
                    try:
                        # Get current settings
                        current_calc = db.get_calculation(calc_id)
                        if current_calc:
                            import json
                            settings = json.loads(current_calc.get('settings_json') or '{}')
                            settings['output_file'] = calc_info.get('output_file')
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