#!/usr/bin/env python3
"""
Populate Database with Existing Completed Calculations
-----------------------------------------------------
This script scans for existing completed CRYSTAL calculations and adds them 
to the materials.db database so that workflow_engine.py can find them.

This fixes the issue where workflow_engine.py can't find completed calculations
because they were run before the material tracking system was implemented.
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import our material database
from material_database import MaterialDatabase, create_material_id_from_file, extract_formula_from_d12

class CompletedCalculationImporter:
    """Import existing completed calculations into the database."""
    
    def __init__(self, db_path: str = "materials.db"):
        self.db = MaterialDatabase(db_path)
        
    def detect_calculation_type(self, file_path: Path) -> str:
        """Detect calculation type from filename or content."""
        filename = file_path.name.lower()
        
        # Check filename patterns
        if 'opt' in filename or 'optgeom' in filename:
            return 'OPT'
        elif 'sp' in filename or 'single' in filename:
            return 'SP'
        elif 'band' in filename:
            return 'BAND'
        elif 'dos' in filename or 'doss' in filename:
            return 'DOSS'
        elif 'freq' in filename:
            return 'FREQ'
        elif 'transport' in filename:
            return 'TRANSPORT'
        
        # If unclear from filename, try to determine from input file if it exists
        input_file = file_path.with_suffix('.d12')
        if input_file.exists():
            try:
                with open(input_file, 'r') as f:
                    content = f.read().upper()
                    if 'OPTGEOM' in content:
                        return 'OPT'
                    elif 'FREQCALC' in content:
                        return 'FREQ'
            except:
                pass
                
        # Default assumption for completed calculations
        return 'OPT'  # Most likely for bulk optimizations
        
    def check_if_completed(self, output_file: Path) -> Tuple[bool, str]:
        """
        Check if a calculation completed successfully.
        
        Returns:
            (is_completed, completion_status)
        """
        if not output_file.exists():
            return False, "no_output_file"
            
        try:
            with open(output_file, 'r') as f:
                content = f.read()
                
            # Check for successful completion indicators
            content_upper = content.upper()
            
            if "OPT END - CONVERGED" in content_upper:
                return True, "opt_converged"
            elif "CRYSTAL ENDS" in content_upper:
                return True, "crystal_ends"
            elif "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT END" in content_upper:
                return True, "normal_termination"
            elif "CALCULATION TERMINATED" in content_upper:
                return True, "terminated_normally"
            
            # Check for failure indicators
            failure_patterns = [
                "CRYSTAL STOPS",
                "FORTRAN STOP", 
                "SEGMENTATION FAULT",
                "OUT OF MEMORY",
                "DUE TO TIME LIMIT"
            ]
            
            for pattern in failure_patterns:
                if pattern in content_upper:
                    return False, f"failed_{pattern.lower().replace(' ', '_')}"
                    
            # If file exists but no clear completion/failure indicator
            # Check file size - very small files likely failed immediately
            if len(content) < 1000:
                return False, "output_too_small"
                
            # If we have substantial output but no clear completion marker,
            # it might be incomplete or the run was interrupted
            return False, "unclear_status"
            
        except Exception as e:
            return False, f"error_reading_file: {e}"
            
    def extract_completion_time(self, output_file: Path) -> Optional[str]:
        """Extract completion time from output file if possible."""
        try:
            # Use file modification time as completion time
            mtime = output_file.stat().st_mtime
            return datetime.fromtimestamp(mtime).isoformat()
        except:
            return None
            
    def import_completed_calculation(self, output_file: Path, input_file: Path = None) -> Optional[str]:
        """
        Import a single completed calculation into the database.
        
        Args:
            output_file: Path to .out file
            input_file: Path to corresponding .d12 file (if exists)
            
        Returns:
            calc_id if successful, None if failed
        """
        # Check if calculation completed
        is_completed, completion_status = self.check_if_completed(output_file)
        
        if not is_completed:
            print(f"Skipping {output_file.name}: {completion_status}")
            return None
            
        # Determine input file if not provided
        if input_file is None:
            input_file = output_file.with_suffix('.d12')
            
        # Extract material information
        if input_file.exists():
            material_id = create_material_id_from_file(input_file.name)
            try:
                formula = extract_formula_from_d12(str(input_file))
            except Exception as e:
                print(f"Warning: Could not extract formula from {input_file}: {e}")
                formula = "Unknown"
        else:
            # Create material ID from output filename
            material_id = create_material_id_from_file(output_file.name)
            formula = "Unknown"
            
        # Detect calculation type
        calc_type = self.detect_calculation_type(output_file)
        
        # Check if material exists, create if needed
        existing_material = self.db.get_material(material_id)
        if not existing_material:
            self.db.create_material(
                material_id=material_id,
                formula=formula,
                source_type='imported',
                source_file=str(input_file) if input_file.exists() else str(output_file),
                metadata={
                    'imported_from_existing': True,
                    'original_output_file': str(output_file),
                    'import_timestamp': datetime.now().isoformat(),
                    'completion_status': completion_status
                }
            )
            print(f"Created material record: {material_id}")
            
        # Check if calculation already exists
        existing_calcs = self.db.get_calculations_by_status(material_id=material_id)
        for calc in existing_calcs:
            if (calc['calc_type'] == calc_type and 
                (calc['output_file'] == str(output_file) or 
                 calc['input_file'] == str(input_file))):
                print(f"Calculation already exists: {calc['calc_id']}")
                return calc['calc_id']
                
        # Create calculation record
        completion_time = self.extract_completion_time(output_file)
        
        calc_id = self.db.create_calculation(
            material_id=material_id,
            calc_type=calc_type,
            input_file=str(input_file) if input_file.exists() else None,
            work_dir=str(output_file.parent),
            settings={
                'imported_from_existing': True,
                'completion_status': completion_status,
                'import_source': 'populate_database_script',
                'detected_calc_type': calc_type
            }
        )
        
        # Mark as completed with output file
        self.db.update_calculation_status(
            calc_id,
            'completed',
            output_file=str(output_file)
        )
        
        print(f"Imported completed {calc_type} calculation: {calc_id}")
        return calc_id
        
    def scan_directory_for_completed(self, directory: Path) -> int:
        """
        Scan directory for completed calculations and import them.
        
        Returns:
            Number of calculations imported
        """
        if not directory.exists():
            print(f"Directory does not exist: {directory}")
            return 0
            
        print(f"Scanning directory: {directory}")
        
        # Find all .out files
        out_files = list(directory.glob("*.out"))
        
        if not out_files:
            print(f"No .out files found in {directory}")
            return 0
            
        imported_count = 0
        
        for out_file in out_files:
            print(f"\nProcessing: {out_file.name}")
            
            # Look for corresponding .d12 file
            d12_file = out_file.with_suffix('.d12')
            
            calc_id = self.import_completed_calculation(out_file, d12_file)
            if calc_id:
                imported_count += 1
                
        return imported_count
        
    def scan_multiple_directories(self, directories: List[Path]) -> int:
        """Scan multiple directories for completed calculations."""
        total_imported = 0
        
        for directory in directories:
            count = self.scan_directory_for_completed(directory)
            total_imported += count
            
        return total_imported


def main():
    """Main function to populate database with existing completed calculations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Import existing completed CRYSTAL calculations into materials database"
    )
    parser.add_argument("--db", default="materials.db", 
                       help="Path to materials database")
    parser.add_argument("--directories", nargs="+", 
                       default=["../../crystalouputs"], 
                       help="Directories to scan for completed calculations")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be imported without actually importing")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made to database")
        # TODO: Implement dry run mode
        return
        
    # Initialize importer
    importer = CompletedCalculationImporter(args.db)
    
    # Convert directory paths
    directories = [Path(d).resolve() for d in args.directories]
    
    print(f"Importing completed calculations into: {args.db}")
    print(f"Scanning directories: {[str(d) for d in directories]}")
    
    # Import calculations
    total_imported = importer.scan_multiple_directories(directories)
    
    print(f"\n=== Import Summary ===")
    print(f"Total calculations imported: {total_imported}")
    
    # Show database statistics
    all_materials = importer.db.get_all_materials()
    all_calcs = importer.db.get_all_calculations()
    completed_calcs = importer.db.get_calculations_by_status('completed')
    
    print(f"Total materials in database: {len(all_materials)}")
    print(f"Total calculations in database: {len(all_calcs)}")
    print(f"Completed calculations: {len(completed_calcs)}")
    
    if completed_calcs:
        print("\nCompleted calculations by type:")
        calc_types = {}
        for calc in completed_calcs:
            calc_type = calc['calc_type']
            calc_types[calc_type] = calc_types.get(calc_type, 0) + 1
        
        for calc_type, count in sorted(calc_types.items()):
            print(f"  {calc_type}: {count}")


if __name__ == "__main__":
    main()