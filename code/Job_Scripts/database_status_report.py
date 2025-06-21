#!/usr/bin/env python3
"""
Database Status Report
======================
Shows the current status of the materials database, including any duplicate entries
and recovery statistics.

Usage:
  python database_status_report.py [--db-path materials.db]
"""

import os
import sys
import argparse
import sqlite3
from pathlib import Path
from collections import defaultdict

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"Error importing MaterialDatabase: {e}")
    sys.exit(1)

def analyze_database(db_path: str):
    """Analyze the database and provide a comprehensive report."""
    print("📊 Database Status Report")
    print("=" * 60)
    
    db = MaterialDatabase(db_path)
    
    with db._get_connection() as conn:
        # Basic statistics
        cursor = conn.execute("SELECT COUNT(*) FROM materials")
        total_materials = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM calculations")
        total_calculations = cursor.fetchone()[0]
        
        print(f"📋 Basic Statistics:")
        print(f"   Total materials: {total_materials}")
        print(f"   Total calculations: {total_calculations}")
        
        # Status breakdown
        print(f"\n📈 Calculation Status Breakdown:")
        cursor = conn.execute("""
            SELECT status, COUNT(*) as count 
            FROM calculations 
            GROUP BY status 
            ORDER BY count DESC
        """)
        
        for row in cursor:
            status, count = row
            print(f"   {status:12}: {count:4}")
        
        # Calculation type breakdown
        print(f"\n🧪 Calculation Type Breakdown:")
        cursor = conn.execute("""
            SELECT calc_type, COUNT(*) as count 
            FROM calculations 
            GROUP BY calc_type 
            ORDER BY count DESC
        """)
        
        for row in cursor:
            calc_type, count = row
            print(f"   {calc_type:12}: {count:4}")
        
        # Error recovery statistics (if column exists)
        try:
            cursor = conn.execute("""
                SELECT 
                    SUM(CASE WHEN recovery_attempts > 0 THEN 1 ELSE 0 END) as recovered_jobs,
                    SUM(recovery_attempts) as total_attempts,
                    MAX(recovery_attempts) as max_attempts
                FROM calculations
            """)
            row = cursor.fetchone()
            if row and any(row):
                recovered_jobs, total_attempts, max_attempts = row
                print(f"\n🔧 Error Recovery Statistics:")
                print(f"   Jobs with recovery attempts: {recovered_jobs or 0}")
                print(f"   Total recovery attempts: {total_attempts or 0}")
                print(f"   Maximum attempts for one job: {max_attempts or 0}")
        except sqlite3.OperationalError:
            print(f"\n🔧 Error Recovery: Column not available (older database)")
        
        # Completion type breakdown (if column exists)
        try:
            cursor = conn.execute("""
                SELECT completion_type, COUNT(*) as count 
                FROM calculations 
                WHERE completion_type IS NOT NULL
                GROUP BY completion_type 
                ORDER BY count DESC
            """)
            
            rows = cursor.fetchall()
            if rows:
                print(f"\n🏁 Completion Type Breakdown:")
                for row in rows:
                    comp_type, count = row
                    print(f"   {comp_type:15}: {count:4}")
        except sqlite3.OperationalError:
            print(f"\n🏁 Completion Type: Column not available (older database)")
        
        # Duplicate analysis
        print(f"\n🔍 Duplicate Analysis:")
        cursor = conn.execute("""
            SELECT material_id, calc_type, work_dir, COUNT(*) as count
            FROM calculations 
            GROUP BY material_id, calc_type, work_dir
            HAVING COUNT(*) > 1
            ORDER BY count DESC
        """)
        
        duplicate_groups = cursor.fetchall()
        if duplicate_groups:
            total_duplicates = sum(count - 1 for _, _, _, count in duplicate_groups)
            print(f"   Duplicate groups found: {len(duplicate_groups)}")
            print(f"   Extra entries (duplicates): {total_duplicates}")
            print(f"   Affected calculations: {sum(count for _, _, _, count in duplicate_groups)}")
            
            print(f"\n   Top duplicate groups:")
            for material_id, calc_type, work_dir, count in duplicate_groups[:5]:
                print(f"     • {material_id:20} {calc_type:6}: {count} entries")
        else:
            print(f"   ✅ No duplicate entries found!")
        
        # SLURM job tracking
        print(f"\n🎯 SLURM Job Tracking:")
        cursor = conn.execute("""
            SELECT 
                SUM(CASE WHEN slurm_job_id IS NOT NULL THEN 1 ELSE 0 END) as with_slurm_id,
                SUM(CASE WHEN slurm_job_id IS NULL THEN 1 ELSE 0 END) as without_slurm_id
            FROM calculations
        """)
        
        row = cursor.fetchone()
        if row:
            with_slurm, without_slurm = row
            print(f"   With SLURM job ID: {with_slurm}")
            print(f"   Without SLURM job ID: {without_slurm}")
        
        # Output file tracking
        cursor = conn.execute("""
            SELECT 
                SUM(CASE WHEN output_file IS NOT NULL THEN 1 ELSE 0 END) as with_output,
                SUM(CASE WHEN output_file IS NULL THEN 1 ELSE 0 END) as without_output
            FROM calculations
        """)
        
        row = cursor.fetchone()
        if row:
            with_output, without_output = row
            print(f"   With output file: {with_output}")
            print(f"   Without output file: {without_output}")
        
        # Recent activity
        print(f"\n📅 Recent Activity (Last 24 hours):")
        cursor = conn.execute("""
            SELECT calc_type, status, COUNT(*) as count
            FROM calculations 
            WHERE created_at >= datetime('now', '-1 day')
            GROUP BY calc_type, status
            ORDER BY count DESC
        """)
        
        recent_activity = cursor.fetchall()
        if recent_activity:
            for calc_type, status, count in recent_activity:
                print(f"   {calc_type:6} {status:12}: {count}")
        else:
            print(f"   No recent activity")
        
        # Material workflow progression
        print(f"\n🔄 Workflow Progression Analysis:")
        cursor = conn.execute("""
            SELECT 
                m.material_id,
                SUM(CASE WHEN c.calc_type = 'OPT' AND c.status = 'completed' THEN 1 ELSE 0 END) as opt_complete,
                SUM(CASE WHEN c.calc_type = 'SP' AND c.status = 'completed' THEN 1 ELSE 0 END) as sp_complete,
                SUM(CASE WHEN c.calc_type = 'BAND' AND c.status = 'completed' THEN 1 ELSE 0 END) as band_complete,
                SUM(CASE WHEN c.calc_type = 'DOSS' AND c.status = 'completed' THEN 1 ELSE 0 END) as doss_complete
            FROM materials m
            LEFT JOIN calculations c ON m.material_id = c.material_id
            GROUP BY m.material_id
            HAVING opt_complete > 0 OR sp_complete > 0 OR band_complete > 0 OR doss_complete > 0
        """)
        
        progression_data = cursor.fetchall()
        if progression_data:
            # Count materials at each stage
            stages = defaultdict(int)
            for material_id, opt, sp, band, doss in progression_data:
                if doss > 0:
                    stages['DOSS complete'] += 1
                elif band > 0:
                    stages['BAND complete'] += 1
                elif sp > 0:
                    stages['SP complete'] += 1
                elif opt > 0:
                    stages['OPT complete'] += 1
            
            for stage, count in stages.items():
                print(f"   {stage:15}: {count} materials")
        
        print(f"\n" + "=" * 60)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate database status report")
    parser.add_argument("--db-path", default="materials.db", help="Path to materials database")
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"❌ Database file not found: {db_path}")
        sys.exit(1)
    
    analyze_database(str(db_path))

if __name__ == "__main__":
    main()