#!/usr/bin/env python3
"""
Update Effective Masses from BAND/DOSS Data
==========================================
Processes completed BAND and DOSS calculations to extract real effective masses
and advanced electronic properties, updating the materials database.

This script:
1. Finds all completed BAND/DOSS calculations
2. Runs advanced electronic analysis on them
3. Updates the database with real effective mass values
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import the advanced analyzer
try:
    from advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def find_band_doss_files(base_dir: Path) -> List[Tuple[Path, Optional[Path], str]]:
    """
    Find all BAND.DAT and DOSS.DAT files in the directory structure.
    
    Returns list of tuples: (band_file, doss_file, material_id)
    """
    files = []
    
    # Search for BAND.DAT files
    for band_file in base_dir.rglob("*.BAND.DAT"):
        # Extract material ID from path
        # Typical path: workflow_outputs/workflow_*/step_*_BAND/material_id/file.BAND.DAT
        parts = band_file.parts
        
        # Try to find material ID from parent directory name
        material_dir = band_file.parent.name
        if material_dir.endswith('_opt_band'):
            material_id = material_dir[:-9]  # Remove _opt_band suffix
        elif material_dir.endswith('_band'):
            material_id = material_dir[:-5]  # Remove _band suffix
        else:
            material_id = material_dir
            
        # Clean up material ID - remove _opt suffix if present
        if material_id.endswith('_opt'):
            material_id = material_id[:-4]
            
        # Look for corresponding DOSS file
        doss_file = None
        parent_dir = band_file.parent.parent.parent  # Go up to workflow step level
        
        # Check for DOSS in parallel step directory
        for doss_step in parent_dir.glob("step_*_DOSS"):
            potential_doss = doss_step / material_dir.replace('_band', '_doss') / f"{material_dir.replace('_band', '_doss')}.DOSS.DAT"
            if potential_doss.exists():
                doss_file = potential_doss
                break
            # Also try without suffix changes
            potential_doss = doss_step / material_id / f"{material_id}.DOSS.DAT"
            if potential_doss.exists():
                doss_file = potential_doss
                break
                
        files.append((band_file, doss_file, material_id))
        
    return files


def update_database_properties(conn: sqlite3.Connection, material_id: str, 
                             calc_id: str, properties: Dict):
    """Update database with extracted properties."""
    
    cursor = conn.cursor()
    from datetime import datetime
    
    # Properties to store with their types
    property_mappings = {
        'electronic_classification': ('electronic_classification_advanced', 'string', 'electronic'),
        'band_gap_eV': ('band_gap_advanced_eV', 'number', 'electronic'),
        'electron_effective_mass': ('electron_effective_mass_real', 'number', 'electronic'),
        'hole_effective_mass': ('hole_effective_mass_real', 'number', 'electronic'),
        'electron_mobility': ('electron_mobility_calculated', 'number', 'electronic'),
        'hole_mobility': ('hole_mobility_calculated', 'number', 'electronic'),
        'conductivity_type': ('conductivity_type_calculated', 'string', 'electronic'),
        'fermi_level_eV': ('fermi_level_eV', 'number', 'electronic'),
        'dos_at_fermi': ('dos_at_fermi_level', 'number', 'electronic'),
        'analysis_method': ('effective_mass_analysis_method', 'string', 'electronic'),
        'has_real_effective_mass': ('has_real_effective_mass', 'boolean', 'electronic'),
    }
    
    for prop_key, (db_name, prop_type, category) in property_mappings.items():
        if prop_key in properties and properties[prop_key] is not None:
            value = properties[prop_key]
            
            # Prepare values based on type
            if prop_type == 'number':
                prop_value = float(value)
                prop_value_text = None
                prop_unit = 'm_e' if 'effective_mass' in db_name else (
                           'eV' if '_eV' in db_name else (
                           'cm²/(V·s)' if 'mobility' in db_name else 
                           'dimensionless'))
            else:
                prop_value = None
                prop_value_text = str(value)
                prop_unit = None
                
            # Insert or update property
            cursor.execute("""
                INSERT OR REPLACE INTO properties 
                (material_id, calc_id, property_category, property_name, 
                 property_value, property_value_text, property_unit, 
                 confidence, extracted_at, extractor_script)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                material_id, calc_id, category, db_name,
                prop_value, prop_value_text, prop_unit,
                1.0,  # High confidence for direct calculations
                datetime.now().isoformat(),
                'update_effective_masses.py'
            ))
    
    conn.commit()


def main():
    """Main function to process all BAND/DOSS files and update database."""
    
    print("=" * 80)
    print("EFFECTIVE MASS DATABASE UPDATE")
    print("=" * 80)
    
    # Initialize components
    db = MaterialDatabase("materials.db")
    analyzer = AdvancedElectronicAnalyzer()
    
    # Find all BAND/DOSS files
    base_dir = Path(".")
    files = find_band_doss_files(base_dir)
    
    print(f"\nFound {len(files)} BAND files to process")
    
    # Track statistics
    stats = {
        'processed': 0,
        'with_effective_mass': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # Process each file
    for band_file, doss_file, material_id in files:
        print(f"\nProcessing: {material_id}")
        print(f"  BAND: {band_file}")
        if doss_file:
            print(f"  DOSS: {doss_file}")
        else:
            print("  DOSS: Not found")
            
        try:
            # Run analysis
            results = analyzer.analyze_material(band_file, doss_file)
            
            # Find the corresponding calculation in database
            # Look for BAND calculation for this material
            conn = sqlite3.connect("materials.db")
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT calc_id FROM calculations 
                WHERE material_id = ? AND calc_type LIKE '%BAND%'
                ORDER BY created_at DESC LIMIT 1
            """, (material_id,))
            
            calc_row = cursor.fetchone()
            if not calc_row:
                # Try without material ID suffix
                base_material_id = material_id.split('_')[0]
                cursor.execute("""
                    SELECT calc_id FROM calculations 
                    WHERE material_id = ? AND calc_type LIKE '%BAND%'
                    ORDER BY created_at DESC LIMIT 1
                """, (base_material_id,))
                calc_row = cursor.fetchone()
                
            if calc_row:
                calc_id = calc_row[0]
                
                # Update properties
                update_database_properties(conn, material_id, calc_id, results)
                
                stats['processed'] += 1
                if results.get('has_real_effective_mass'):
                    stats['with_effective_mass'] += 1
                    
                print(f"  ✓ Updated properties in database")
                if results.get('electron_effective_mass'):
                    print(f"    Electron effective mass: {results['electron_effective_mass']:.3f} m_e")
                if results.get('hole_effective_mass'):
                    print(f"    Hole effective mass: {results['hole_effective_mass']:.3f} m_e")
            else:
                print(f"  ⚠ No calculation found in database for {material_id}")
                stats['skipped'] += 1
                
            conn.close()
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            stats['failed'] += 1
            
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total BAND files found: {len(files)}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"With effective mass data: {stats['with_effective_mass']}")
    print(f"Skipped (not in database): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    
    # Show some statistics on the values
    if stats['with_effective_mass'] > 0:
        conn = sqlite3.connect("materials.db")
        cursor = conn.cursor()
        
        print("\nEffective Mass Statistics:")
        for mass_type in ['electron_effective_mass_real', 'hole_effective_mass_real']:
            cursor.execute("""
                SELECT 
                    COUNT(*) as count,
                    MIN(CAST(property_value AS REAL)) as min_val,
                    MAX(CAST(property_value AS REAL)) as max_val,
                    AVG(CAST(property_value AS REAL)) as avg_val
                FROM properties
                WHERE property_name = ?
                AND property_value IS NOT NULL
            """, (mass_type,))
            
            row = cursor.fetchone()
            if row[0] > 0:
                print(f"\n{mass_type}:")
                print(f"  Count: {row[0]}")
                print(f"  Range: {row[1]:.3f} - {row[2]:.3f} m_e")
                print(f"  Average: {row[3]:.3f} m_e")
                
        conn.close()
        
    print("\n✓ Database update complete!")


if __name__ == "__main__":
    main()