#!/usr/bin/env python3
"""
Fix Database Duplicate Entries
===============================
This script fixes the duplicate database entries issue where the same calculation
appears multiple times with different statuses (submitted vs completed).

It consolidates duplicate entries by:
1. Finding calculations with the same material_id, calc_type, and work_dir
2. Merging submission info (SLURM job ID) with completion info (output files)
3. Removing duplicate entries
4. Preserving the most complete information

Usage:
  python fix_database_duplicates.py [--db-path materials.db] [--dry-run]
"""

import os
import sys
import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"Error importing MaterialDatabase: {e}")
    sys.exit(1)

def find_duplicate_groups(db: MaterialDatabase) -> List[List[Dict]]:
    """Find groups of calculations that are duplicates."""
    print("🔍 Scanning for duplicate calculation entries...")
    
    # Get all calculations
    with db._get_connection() as conn:
        cursor = conn.execute("""
            SELECT calc_id, material_id, calc_type, work_dir, status, slurm_job_id, 
                   output_file, created_at, recovery_attempts, completion_type
            FROM calculations 
            ORDER BY material_id, calc_type, created_at
        """)
        all_calcs = [dict(row) for row in cursor.fetchall()]
    
    # Group by (material_id, calc_type, work_dir)
    groups = {}
    for calc in all_calcs:
        key = (calc['material_id'], calc['calc_type'], calc['work_dir'])
        if key not in groups:
            groups[key] = []
        groups[key].append(calc)
    
    # Find groups with duplicates
    duplicate_groups = []
    for key, group in groups.items():
        if len(group) > 1:
            duplicate_groups.append(group)
            print(f"  📋 Found {len(group)} duplicates for {key[0]} {key[1]}")
    
    print(f"📊 Found {len(duplicate_groups)} duplicate groups affecting {sum(len(g) for g in duplicate_groups)} entries")
    return duplicate_groups

def consolidate_duplicate_group(group: List[Dict]) -> Tuple[Dict, List[str]]:
    """
    Consolidate a group of duplicate calculations into one entry.
    
    Returns:
        Tuple of (consolidated_calc, calc_ids_to_delete)
    """
    # Sort by creation time to prefer earlier entries
    group.sort(key=lambda x: x['created_at'])
    
    # Find the best candidate for the master record
    # Prefer: completed > submitted > failed > pending
    status_priority = {'completed': 4, 'submitted': 3, 'failed': 2, 'pending': 1, 'resubmitted': 3}
    master = max(group, key=lambda x: (
        status_priority.get(x['status'], 0),
        bool(x['slurm_job_id']),  # Has SLURM job ID
        bool(x['output_file']),   # Has output file
        x['recovery_attempts'] or 0  # Recovery attempts
    ))
    
    # Collect the best information from all entries
    consolidated = master.copy()
    
    # Merge information from all entries
    for calc in group:
        # Preserve SLURM job ID if master doesn't have one
        if not consolidated['slurm_job_id'] and calc['slurm_job_id']:
            consolidated['slurm_job_id'] = calc['slurm_job_id']
        
        # Preserve output file if master doesn't have one
        if not consolidated['output_file'] and calc['output_file']:
            consolidated['output_file'] = calc['output_file']
        
        # Take the highest recovery attempts
        if calc['recovery_attempts'] and calc['recovery_attempts'] > (consolidated['recovery_attempts'] or 0):
            consolidated['recovery_attempts'] = calc['recovery_attempts']
        
        # Update status to completed if any entry is completed
        if calc['status'] == 'completed' and consolidated['status'] != 'completed':
            consolidated['status'] = 'completed'
    
    # Determine completion type
    recovery_attempts = consolidated['recovery_attempts'] or 0
    if recovery_attempts > 0:
        consolidated['completion_type'] = 'recovered'
    else:
        consolidated['completion_type'] = 'first_try'
    
    # Identify entries to delete (all except master)
    calc_ids_to_delete = [calc['calc_id'] for calc in group if calc['calc_id'] != master['calc_id']]
    
    return consolidated, calc_ids_to_delete

def fix_database_duplicates(db_path: str, dry_run: bool = False):
    """Fix duplicate entries in the database."""
    print("🔧 Fixing Database Duplicate Entries")
    print("=" * 50)
    
    db = MaterialDatabase(db_path)
    duplicate_groups = find_duplicate_groups(db)
    
    if not duplicate_groups:
        print("✅ No duplicate entries found! Database is clean.")
        return
    
    total_deletions = 0
    total_updates = 0
    
    for i, group in enumerate(duplicate_groups):
        print(f"\n📝 Processing duplicate group {i+1}/{len(duplicate_groups)}")
        print(f"   Material: {group[0]['material_id']}, Type: {group[0]['calc_type']}")
        
        # Show current entries
        for calc in group:
            status = calc['status']
            slurm_id = calc['slurm_job_id'] or 'None'
            output = 'Yes' if calc['output_file'] else 'No'
            recovery = calc['recovery_attempts'] or 0
            print(f"     • {calc['calc_id'][:20]:20} {status:12} SLURM:{slurm_id:12} Output:{output:3} Recovery:{recovery}")
        
        # Consolidate
        consolidated, to_delete = consolidate_duplicate_group(group)
        
        print(f"   ➡️  Consolidating to: {consolidated['calc_id'][:20]:20} {consolidated['status']:12}")
        print(f"   🗑️  Will delete: {len(to_delete)} duplicate entries")
        
        if not dry_run:
            try:
                with db._get_connection() as conn:
                    # Update the master record with consolidated information
                    conn.execute("""
                        UPDATE calculations 
                        SET status = ?, slurm_job_id = ?, output_file = ?, 
                            recovery_attempts = ?, completion_type = ?
                        WHERE calc_id = ?
                    """, (
                        consolidated['status'],
                        consolidated['slurm_job_id'],
                        consolidated['output_file'],
                        consolidated['recovery_attempts'],
                        consolidated['completion_type'],
                        consolidated['calc_id']
                    ))
                    
                    # Delete duplicate entries
                    for calc_id in to_delete:
                        conn.execute("DELETE FROM calculations WHERE calc_id = ?", (calc_id,))
                    
                    total_updates += 1
                    total_deletions += len(to_delete)
                    print(f"   ✅ Consolidated successfully")
                    
            except Exception as e:
                print(f"   ❌ Error consolidating group: {e}")
        else:
            print(f"   🔍 DRY RUN - No changes made")
            total_updates += 1
            total_deletions += len(to_delete)
    
    print(f"\n📊 Summary:")
    print(f"   Updated master records: {total_updates}")
    print(f"   Deleted duplicate entries: {total_deletions}")
    
    if dry_run:
        print(f"\n🔍 DRY RUN COMPLETE - Run without --dry-run to apply changes")
    else:
        print(f"\n✅ DATABASE FIXED - Duplicates removed and entries consolidated")
        
        # Show final stats
        with db._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM calculations")
            total_calcs = cursor.fetchone()[0]
            print(f"   Total calculations remaining: {total_calcs}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix database duplicate entries")
    parser.add_argument("--db-path", default="materials.db", help="Path to materials database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"❌ Database file not found: {db_path}")
        sys.exit(1)
    
    # Create backup
    if not args.dry_run:
        backup_path = db_path.with_suffix(f'.backup_{int(__import__("time").time())}')
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"💾 Created backup: {backup_path}")
    
    fix_database_duplicates(str(db_path), args.dry_run)

if __name__ == "__main__":
    main()