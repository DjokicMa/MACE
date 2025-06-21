#!/usr/bin/env python3
"""
Populate Database with Completed Jobs
------------------------------------
This script scans for completed CRYSTAL calculations and adds them to the 
materials database for workflow tracking.

Usage:
  python populate_completed_jobs.py [--base-dir DIR] [--db-path PATH]
"""

import os
import sys
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from material_database import MaterialDatabase, create_material_id_from_file
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"Make sure material_database.py is in the same directory as {__file__}")
    sys.exit(1)


def is_calculation_completed(out_file: Path) -> bool:
    """Check if a CRYSTAL calculation completed successfully."""
    if not out_file.exists():
        return False
    
    try:
        with open(out_file, 'r') as f:
            content = f.read()
            
        # Look for completion indicators
        completion_patterns = [
            r'OPT END - CONVERGED',
            r'GEOMETRY OPTIMIZATION COMPLETED',
            r'OPTGEOM.*CONVERGED',
            r'CALCULATION CONVERGED',
            r'EEEEEEEEEE TERMINATION',
            r'TTTTTTTTTT TERMINATION'
        ]
        
        for pattern in completion_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
                
        return False
        
    except Exception:
        return False


def extract_calc_type_from_output(out_file: Path) -> str:
    """Extract calculation type from CRYSTAL output file."""
    try:
        # First check directory context for better identification
        file_path_str = str(out_file)
        
        # Check directory-based context first (most reliable)
        if '/step_002_SP/' in file_path_str or '/SP/' in file_path_str:
            return 'SP'
        elif '/step_001_OPT/' in file_path_str or '/OPT/' in file_path_str:
            return 'OPT'
        elif '/BAND/' in file_path_str or '_band_' in file_path_str:
            return 'BAND'
        elif '/DOSS/' in file_path_str or '_doss_' in file_path_str:
            return 'DOSS'
        elif '/FREQ/' in file_path_str or '_freq_' in file_path_str:
            return 'FREQ'
        
        # Fallback to content-based detection
        with open(out_file, 'r') as f:
            content = f.read()
        
        # Check for specific calculation type keywords in input/output sections    
        if '_sp_' in out_file.name.lower():
            return 'SP'
        elif '_freq_' in out_file.name.lower():
            return 'FREQ'
        elif re.search(r'BAND.*STRUCTURE', content, re.IGNORECASE):
            return 'BAND'
        elif re.search(r'DENSITY.*STATES|NEWK.*DOSS', content, re.IGNORECASE):
            return 'DOSS'
        elif re.search(r'FREQUENCY|PHONON', content, re.IGNORECASE):
            return 'FREQ'
        elif re.search(r'OPTGEOM|FULLOPTG|GEOMETRY OPTIMIZATION', content, re.IGNORECASE):
            return 'OPT'
        elif re.search(r'SINGLE POINT|SCF.*CALCULATION', content, re.IGNORECASE):
            return 'SP'
        else:
            return 'OPT'  # Default assumption
            
    except Exception:
        return 'OPT'


def scan_for_completed_calculations(base_dir: Path) -> List[Dict]:
    """Scan directory for completed CRYSTAL calculations."""
    completed_calcs = []
    
    # Look for .out files in various locations
    search_patterns = [
        "**/*.out",
        "**/workflow_outputs/**/*.out", 
        "**/crystalouputs/**/*.out",
        "*.out"
    ]
    
    found_files = set()
    for pattern in search_patterns:
        for out_file in base_dir.glob(pattern):
            if out_file in found_files:
                continue
            found_files.add(out_file)
            
            if is_calculation_completed(out_file):
                # Look for corresponding .d12 and .f9 files
                base_name = out_file.stem
                d12_file = out_file.parent / f"{base_name}.d12"
                f9_file = out_file.parent / f"{base_name}.f9"
                
                # Extract material ID from filename
                material_id = create_material_id_from_file(str(out_file.name))
                calc_type = extract_calc_type_from_output(out_file)
                
                calc_info = {
                    'material_id': material_id,
                    'calc_type': calc_type,
                    'output_file': str(out_file),
                    'input_file': str(d12_file) if d12_file.exists() else None,
                    'wavefunction_file': str(f9_file) if f9_file.exists() else None,
                    'status': 'completed',
                    'completed_at': datetime.fromtimestamp(out_file.stat().st_mtime).isoformat()
                }
                
                completed_calcs.append(calc_info)
                print(f"Found completed {calc_type}: {material_id}")
    
    return completed_calcs


def populate_database(completed_calcs: List[Dict], db: MaterialDatabase) -> int:
    """Add completed calculations to the database."""
    added_count = 0
    
    for calc in completed_calcs:
        try:
            # Check if material exists, create if not
            material = db.get_material(calc['material_id'])
            if not material:
                print(f"Creating material record for: {calc['material_id']}")
                db.create_material(
                    material_id=calc['material_id'],
                    formula="Unknown",
                    source_file=calc['input_file'],
                    source_type='auto_detected',
                    metadata={'auto_populated': True, 'source': 'populate_completed_jobs.py'}
                )
            
            # First, check if calculation already exists as completed
            existing_completed_calcs = db.get_calculations_by_status('completed', calc['calc_type'], calc['material_id'])
            if any(c['output_file'] == calc['output_file'] for c in existing_completed_calcs):
                print(f"  Skipping {calc['calc_type']} for {calc['material_id']} - already in database as completed")
                continue
            
            # Look for existing submitted/failed calculations that match this completion
            work_dir = str(Path(calc['output_file']).parent)
            existing_calc = None
            
            # Check for calculations in the same work directory
            all_calcs = db.get_calculations_by_material(calc['material_id'])
            for c in all_calcs:
                if (c['calc_type'] == calc['calc_type'] and 
                    c['work_dir'] == work_dir and 
                    c['status'] in ['submitted', 'running', 'failed']):
                    existing_calc = c
                    break
            
            if existing_calc:
                # Update existing calculation to completed status
                print(f"  Updating existing {calc['calc_type']} calculation: {existing_calc['calc_id']} -> completed")
                db.update_calculation_status(
                    calc_id=existing_calc['calc_id'],
                    status='completed',
                    output_file=calc['output_file']
                )
                calc_id = existing_calc['calc_id']
            else:
                # Create new calculation record (for cases where submission wasn't tracked)
                print(f"  Creating new {calc['calc_type']} calculation for {calc['material_id']} (no existing submission found)")
                calc_id = db.create_calculation(
                    material_id=calc['material_id'],
                    calc_type=calc['calc_type'],
                    input_file=calc['input_file'],
                    work_dir=work_dir,
                    settings={'auto_populated': True}
                )
                
                # Update the calculation status to completed
                db.update_calculation_status(
                    calc_id=calc_id,
                    status='completed',
                    output_file=calc['output_file']
                )
            
            print(f"  Added {calc['calc_type']} calculation: {calc_id}")
            added_count += 1
            
        except Exception as e:
            print(f"  Error adding {calc['calc_type']} for {calc['material_id']}: {e}")
    
    return added_count


def main():
    parser = argparse.ArgumentParser(description="Populate database with completed CRYSTAL calculations")
    parser.add_argument("--base-dir", default=".", help="Base directory to scan for calculations")
    parser.add_argument("--db-path", default="materials.db", help="Path to materials database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be added without adding")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    print(f"Scanning for completed calculations in: {base_dir}")
    
    # Scan for completed calculations
    completed_calcs = scan_for_completed_calculations(base_dir)
    
    if not completed_calcs:
        print("No completed calculations found.")
        return
    
    print(f"\nFound {len(completed_calcs)} completed calculations:")
    for calc in completed_calcs:
        print(f"  {calc['material_id']}: {calc['calc_type']} ({calc['output_file']})")
    
    if args.dry_run:
        print("\nDry run - no changes made to database")
        return
    
    # Initialize database
    db = MaterialDatabase(args.db_path)
    
    # Populate database
    print(f"\nAdding to database: {args.db_path}")
    added_count = populate_database(completed_calcs, db)
    
    print(f"\nDatabase population complete!")
    print(f"  Added: {added_count} calculations")
    print(f"  Skipped: {len(completed_calcs) - added_count} (already in database)")


if __name__ == "__main__":
    main()