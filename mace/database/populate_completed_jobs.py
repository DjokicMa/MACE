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
            
        # Check if calculation completed
        try:
            with open(out_file, 'r') as f:
                content = f.read()
                if "TERMINATION" not in content:
                    continue
                    
            # Extract material name from file
            material_name = out_file.stem
            
            # Remove common suffixes
            for suffix in ['_opt', '_sp', '_freq']:
                if material_name.endswith(suffix):
                    material_name = material_name[:-len(suffix)]
                    break
                    
            # Determine calculation type
            calc_type = 'OPT'  # Default
            if '_sp' in out_file.stem:
                calc_type = 'SP'
            elif '_freq' in out_file.stem:
                calc_type = 'FREQ'
            elif out_file.parent.name.startswith('step_') and '_OPT' in out_file.parent.name:
                calc_type = 'OPT'
                
            # Look for corresponding input file
            d12_file = out_file.with_suffix('.d12')
            if not d12_file.exists():
                # Try without suffix
                d12_file = out_file.parent / f"{material_name}.d12"
                
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
                if existing.get('output_file') == calc_info.get('output_file'):
                    already_exists = True
                    # Update status to completed if needed
                    if existing.get('status') != 'completed':
                        db.update_calculation_status(
                            existing['calc_id'], 
                            'completed',
                            slurm_job_id=calc_info.get('slurm_job_id')
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
                
                # Update with completion info
                db.update_calculation_status(
                    calc_id,
                    'completed',
                    slurm_job_id=calc_info.get('slurm_job_id')
                )
                
                # Update file paths
                if calc_info.get('output_file'):
                    db.update_calculation_files(
                        calc_id,
                        output_file=calc_info.get('output_file')
                    )
                    
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