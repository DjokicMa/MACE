#!/usr/bin/env python3
"""
Query Stored Files and Settings
===============================
Query and display stored calculation files and extracted settings.

This script demonstrates how to access the stored D12/D3 files and their
extracted settings to answer the user's question about storing input settings.

Usage:
  python query_stored_files.py --calc-id CALC_ID
  python query_stored_files.py --material-id MATERIAL_ID  
  python query_stored_files.py --list-all
  python query_stored_files.py --settings-summary
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from file_storage_manager import FileStorageManager
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def query_calculation_files(calc_id: str, db_path: str = "materials.db"):
    """Query and display files for a specific calculation."""
    print(f"📂 Files for Calculation: {calc_id}")
    print("=" * 60)
    
    storage_manager = FileStorageManager(db_path)
    
    # Get stored files
    files = storage_manager.list_stored_files(calc_id)
    
    if files:
        print(f"📁 Found {len(files)} stored files:")
        print(f"{'Type':12} | {'Name':25} | {'Size':10} | {'Created':20}")
        print("-" * 75)
        
        for file_info in files:
            print(f"{file_info['file_type']:12} | {file_info['file_name']:25} | "
                  f"{file_info['file_size']:8}B | {file_info['created_at'][:19]}")
        
        # Show extracted settings
        settings = storage_manager.get_calculation_settings(calc_id)
        if settings:
            print(f"\n⚙️  Extracted Settings:")
            for filename, file_settings in settings.items():
                print(f"\n📄 {filename}:")
                
                if 'crystal_keywords' in file_settings:
                    keywords = file_settings['crystal_keywords']
                    print(f"   CRYSTAL Keywords: {', '.join(keywords)}")
                
                if 'calculation_parameters' in file_settings:
                    params = file_settings['calculation_parameters']
                    print(f"   Parameters:")
                    for param, value in params.items():
                        print(f"     {param}: {value}")
                
                if 'basis_set_info' in file_settings:
                    basis = file_settings['basis_set_info']
                    print(f"   Basis Set: {basis}")
                
                if 'geometry_info' in file_settings:
                    geom = file_settings['geometry_info']
                    print(f"   Geometry: {geom}")
        
        # Verify file integrity
        print(f"\n🔍 File Integrity Check:")
        integrity = storage_manager.verify_file_integrity(calc_id)
        all_valid = True
        for filename, is_valid in integrity.items():
            status = "✅" if is_valid else "❌"
            print(f"   {status} {filename}")
            if not is_valid:
                all_valid = False
        
        if all_valid:
            print(f"\n✅ All files verified successfully")
        else:
            print(f"\n⚠️  Some files may be corrupted")
            
    else:
        print(f"❌ No files found for calculation {calc_id}")


def query_material_files(material_id: str, db_path: str = "materials.db"):
    """Query and display all files for a material."""
    print(f"📊 Files for Material: {material_id}")
    print("=" * 60)
    
    db = MaterialDatabase(db_path)
    storage_manager = FileStorageManager(db_path)
    
    # Get all calculations for this material
    with db._get_connection() as conn:
        cursor = conn.execute("""
            SELECT calc_id, calc_type, status, created_at 
            FROM calculations 
            WHERE material_id = ?
            ORDER BY created_at
        """, (material_id,))
        
        calculations = cursor.fetchall()
    
    if calculations:
        print(f"🔍 Found {len(calculations)} calculations:")
        print(f"{'Calc ID':25} | {'Type':8} | {'Status':12} | {'Created':20}")
        print("-" * 75)
        
        total_files = 0
        for calc in calculations:
            calc_id, calc_type, status, created_at = calc
            print(f"{calc_id:25} | {calc_type:8} | {status:12} | {created_at[:19]}")
            
            # Count files for this calculation
            files = storage_manager.list_stored_files(calc_id)
            total_files += len(files)
            
            if files:
                print(f"   📁 {len(files)} files stored")
        
        print(f"\n📈 Summary: {total_files} total files across {len(calculations)} calculations")
        
        # Show settings summary for this material
        print(f"\n⚙️  Settings Summary:")
        unique_keywords = set()
        unique_functionals = set()
        
        for calc in calculations:
            calc_id = calc[0]
            settings = storage_manager.get_calculation_settings(calc_id)
            
            for filename, file_settings in settings.items():
                if 'crystal_keywords' in file_settings:
                    unique_keywords.update(file_settings['crystal_keywords'])
                
                if 'calculation_parameters' in file_settings:
                    params = file_settings['calculation_parameters']
                    if 'exchange_functional' in params:
                        unique_functionals.add(params['exchange_functional'][0])
        
        if unique_keywords:
            print(f"   CRYSTAL Keywords Used: {', '.join(sorted(unique_keywords))}")
        if unique_functionals:
            print(f"   Exchange Functionals: {', '.join(sorted(unique_functionals))}")
            
    else:
        print(f"❌ No calculations found for material {material_id}")


def list_all_stored_files(db_path: str = "materials.db"):
    """List all stored files in the database."""
    print(f"📋 All Stored Files")
    print("=" * 60)
    
    db = MaterialDatabase(db_path)
    
    with db._get_connection() as conn:
        cursor = conn.execute("""
            SELECT f.calc_id, f.file_type, f.file_name, f.file_size, f.created_at,
                   c.material_id, c.calc_type, c.status
            FROM files f
            JOIN calculations c ON f.calc_id = c.calc_id
            ORDER BY f.created_at DESC
        """)
        
        files = cursor.fetchall()
    
    if files:
        print(f"📁 Found {len(files)} stored files:")
        print(f"{'Material':15} | {'Type':8} | {'File':20} | {'Size':8} | {'Created':12}")
        print("-" * 80)
        
        file_types = {}
        total_size = 0
        
        for file_info in files:
            calc_id, file_type, file_name, file_size, created_at, material_id, calc_type, status = file_info
            
            print(f"{material_id[:15]:15} | {file_type:8} | {file_name[:20]:20} | "
                  f"{file_size:8}B | {created_at[:12]:12}")
            
            # Statistics
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_size += file_size or 0
        
        print(f"\n📊 Storage Statistics:")
        print(f"   Total Files: {len(files)}")
        print(f"   Total Size: {total_size / (1024*1024):.2f} MB")
        print(f"   File Types:")
        for file_type, count in sorted(file_types.items()):
            print(f"     {file_type}: {count} files")
            
    else:
        print(f"❌ No stored files found")


def show_settings_summary(db_path: str = "materials.db"):
    """Show summary of all extracted settings."""
    print(f"⚙️  Settings Summary")
    print("=" * 60)
    
    db = MaterialDatabase(db_path)
    storage_manager = FileStorageManager(db_path)
    
    # Get all calculations with settings
    with db._get_connection() as conn:
        cursor = conn.execute("""
            SELECT calc_id, material_id, calc_type, settings_json 
            FROM calculations 
            WHERE settings_json IS NOT NULL AND settings_json != '{}'
            ORDER BY material_id, calc_type
        """)
        
        calculations = cursor.fetchall()
    
    if calculations:
        print(f"🔍 Found settings for {len(calculations)} calculations:")
        
        all_keywords = set()
        all_functionals = set()
        all_parameters = set()
        basis_types = set()
        
        for calc in calculations:
            calc_id, material_id, calc_type, settings_json = calc
            
            try:
                settings = json.loads(settings_json)
                
                print(f"\n📊 {material_id} ({calc_type}):")
                
                for filename, file_settings in settings.items():
                    print(f"   📄 {filename}:")
                    
                    if 'crystal_keywords' in file_settings:
                        keywords = file_settings['crystal_keywords']
                        all_keywords.update(keywords)
                        print(f"      Keywords: {', '.join(keywords)}")
                    
                    if 'calculation_parameters' in file_settings:
                        params = file_settings['calculation_parameters']
                        all_parameters.update(params.keys())
                        
                        for param, value in params.items():
                            print(f"      {param}: {value}")
                            
                            if param == 'exchange_functional':
                                all_functionals.add(value[0] if isinstance(value, (list, tuple)) else value)
                    
                    if 'basis_set_info' in file_settings:
                        basis = file_settings['basis_set_info']
                        if 'type' in basis:
                            basis_types.add(basis['type'])
                        print(f"      Basis: {basis}")
                        
            except json.JSONDecodeError:
                print(f"   ⚠️  Invalid JSON in settings")
        
        print(f"\n📈 Global Settings Statistics:")
        print(f"   CRYSTAL Keywords: {len(all_keywords)}")
        print(f"      {', '.join(sorted(all_keywords))}")
        print(f"   Parameters Tracked: {len(all_parameters)}")
        print(f"      {', '.join(sorted(all_parameters))}")
        print(f"   Exchange Functionals: {len(all_functionals)}")
        print(f"      {', '.join(sorted(all_functionals))}")
        print(f"   Basis Set Types: {', '.join(sorted(basis_types))}")
        
    else:
        print(f"❌ No calculations with extracted settings found")


def main():
    """Main query function."""
    parser = argparse.ArgumentParser(description="Query stored CRYSTAL files and settings")
    parser.add_argument("--calc-id", help="Show files for specific calculation")
    parser.add_argument("--material-id", help="Show files for specific material")
    parser.add_argument("--list-all", action="store_true", help="List all stored files")
    parser.add_argument("--settings-summary", action="store_true", help="Show settings summary")
    parser.add_argument("--db-path", default="materials.db", help="Database path")
    
    args = parser.parse_args()
    
    if args.calc_id:
        query_calculation_files(args.calc_id, args.db_path)
    elif args.material_id:
        query_material_files(args.material_id, args.db_path)
    elif args.list_all:
        list_all_stored_files(args.db_path)
    elif args.settings_summary:
        show_settings_summary(args.db_path)
    else:
        print("❌ Please specify one of: --calc-id, --material-id, --list-all, or --settings-summary")
        print("\nUsage examples:")
        print("  python query_stored_files.py --calc-id calc_diamond_opt_001")
        print("  python query_stored_files.py --material-id diamond")
        print("  python query_stored_files.py --list-all")
        print("  python query_stored_files.py --settings-summary")


if __name__ == "__main__":
    main()