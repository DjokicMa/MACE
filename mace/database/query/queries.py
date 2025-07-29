#!/usr/bin/env python3
"""
Query Input Settings from Materials Database
==========================================
Query and display input settings stored directly in the materials.db database.

This script shows how to access D12/D3 input settings that are now stored
directly in the calculations.settings_json field of the materials database.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

Usage:
  python query_input_settings.py --calc-id CALC_ID
  python query_input_settings.py --material-id MATERIAL_ID  
  python query_input_settings.py --list-all
  python query_input_settings.py --settings-summary
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
    from ..materials import MaterialDatabase
    # Try to import query_calculation_settings - this might be in utils
    try:
        from mace.utils.settings_extractor import query_calculation_settings
    except ImportError:
        # Fallback - define a simple version
        def query_calculation_settings(calc_id, db_path):
            db = MaterialDatabase(db_path)
            calc = db.get_calculation(calc_id)
            if calc and calc.get('settings_json'):
                import json
                return json.loads(calc['settings_json'])
            return None
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def query_calculation_input_settings(calc_id: str, db_path: str = "materials.db"):
    """Query and display input settings for a specific calculation."""
    print(f"⚙️  Input Settings for Calculation: {calc_id}")
    print("=" * 60)
    
    settings = query_calculation_settings(calc_id, db_path)
    
    if settings:
        print(f"📄 File: {settings.get('file_name', 'N/A')}")
        print(f"📅 Extracted: {settings.get('extraction_timestamp', 'N/A')[:19]}")
        print(f"🔧 Type: {settings.get('calculation_type', 'N/A')}")
        
        # Show CRYSTAL keywords
        if 'crystal_keywords' in settings and settings['crystal_keywords']:
            keywords = settings['crystal_keywords']
            print(f"\n📋 CRYSTAL Keywords ({len(keywords)}):")
            print(f"   {', '.join(keywords)}")
        
        # Show calculation parameters
        if 'calculation_parameters' in settings and settings['calculation_parameters']:
            params = settings['calculation_parameters']
            print(f"\n🔢 Calculation Parameters:")
            for param, value in params.items():
                if isinstance(value, dict):
                    print(f"   {param}: {value}")
                else:
                    print(f"   {param}: {value}")
        
        # Show SCF parameters
        if 'scf_parameters' in settings and settings['scf_parameters']:
            scf_params = settings['scf_parameters']
            print(f"\n🔄 SCF Parameters:")
            for param, value in scf_params.items():
                print(f"   {param}: {value}")
        
        # Show functional information
        if 'functional_info' in settings and settings['functional_info']:
            func_info = settings['functional_info']
            print(f"\n⚛️  Functional Information:")
            for key, value in func_info.items():
                print(f"   {key}: {value}")
        
        # Show basis set information
        if 'basis_set_info' in settings and settings['basis_set_info']:
            basis_info = settings['basis_set_info']
            print(f"\n📚 Basis Set:")
            for key, value in basis_info.items():
                print(f"   {key}: {value}")
        
        # Show geometry information
        if 'geometry_info' in settings and settings['geometry_info']:
            geom_info = settings['geometry_info']
            print(f"\n🔬 Geometry:")
            for key, value in geom_info.items():
                print(f"   {key}: {value}")
        
        # Show optimization parameters if present
        if 'optimization_parameters' in settings and settings['optimization_parameters']:
            opt_params = settings['optimization_parameters']
            print(f"\n📈 Optimization Parameters:")
            for key, value in opt_params.items():
                print(f"   {key}: {value}")
        
        # Show property parameters if present
        if 'property_parameters' in settings and settings['property_parameters']:
            prop_params = settings['property_parameters']
            print(f"\n📊 Property Parameters:")
            for key, value in prop_params.items():
                print(f"   {key}: {value}")
                
    else:
        print(f"❌ No input settings found for calculation {calc_id}")


def query_material_input_settings(material_id: str, db_path: str = "materials.db"):
    """Query and display input settings for all calculations of a material."""
    print(f"⚙️  Input Settings for Material: {material_id}")
    print("=" * 60)
    
    db = MaterialDatabase(db_path)
    
    # Get all calculations for this material with settings
    with db._get_connection() as conn:
        cursor = conn.execute("""
            SELECT calc_id, calc_type, status, created_at, settings_json
            FROM calculations 
            WHERE material_id = ? AND settings_json IS NOT NULL AND settings_json != '{}'
            ORDER BY created_at
        """, (material_id,))
        
        calculations = cursor.fetchall()
    
    if calculations:
        print(f"🔍 Found {len(calculations)} calculations with input settings:")
        
        for calc in calculations:
            calc_id, calc_type, status, created_at, settings_json = calc
            
            try:
                settings = json.loads(settings_json)
                
                print(f"\n📊 {calc_id} ({calc_type} - {status}):")
                print(f"   📅 Created: {created_at[:19]}")
                
                # Show key settings
                if 'crystal_keywords' in settings:
                    keywords = settings['crystal_keywords'][:5]  # First 5
                    extra = len(settings['crystal_keywords']) - 5
                    keywords_str = ', '.join(keywords)
                    if extra > 0:
                        keywords_str += f" (+{extra} more)"
                    print(f"   📋 Keywords: {keywords_str}")
                
                if 'functional_info' in settings:
                    func_info = settings['functional_info']
                    method = func_info.get('method', 'N/A')
                    exchange = func_info.get('exchange', 'N/A')
                    print(f"   ⚛️  Method: {method}, Exchange: {exchange}")
                
                if 'calculation_parameters' in settings:
                    params = settings['calculation_parameters']
                    if 'shrink_factor' in params:
                        shrink = params['shrink_factor']
                        print(f"   🔢 SHRINK: {shrink}")
                
            except json.JSONDecodeError:
                print(f"   ⚠️  Invalid JSON in settings")
                
    else:
        print(f"❌ No calculations with input settings found for material {material_id}")


def list_all_input_settings(db_path: str = "materials.db"):
    """List all calculations with stored input settings."""
    print(f"📋 All Stored Input Settings")
    print("=" * 60)
    
    db = MaterialDatabase(db_path)
    
    with db._get_connection() as conn:
        cursor = conn.execute("""
            SELECT material_id, calc_id, calc_type, status, created_at, settings_json
            FROM calculations 
            WHERE settings_json IS NOT NULL AND settings_json != '{}'
            ORDER BY created_at DESC
        """)
        
        calculations = cursor.fetchall()
    
    if calculations:
        print(f"📁 Found {len(calculations)} calculations with input settings:")
        print(f"{'Material':15} | {'Calc ID':25} | {'Type':8} | {'Status':12} | {'Created':12}")
        print("-" * 85)
        
        for calc in calculations:
            material_id, calc_id, calc_type, status, created_at, settings_json = calc
            
            print(f"{material_id[:15]:15} | {calc_id[:25]:25} | {calc_type:8} | "
                  f"{status:12} | {created_at[:12]:12}")
                  
    else:
        print(f"❌ No calculations with input settings found")


def show_settings_summary(db_path: str = "materials.db"):
    """Show summary of all stored input settings."""
    print(f"📊 Input Settings Summary")
    print("=" * 60)
    
    db = MaterialDatabase(db_path)
    
    with db._get_connection() as conn:
        cursor = conn.execute("""
            SELECT material_id, calc_id, calc_type, settings_json
            FROM calculations 
            WHERE settings_json IS NOT NULL AND settings_json != '{}'
            ORDER BY material_id, calc_type
        """)
        
        calculations = cursor.fetchall()
    
    if calculations:
        print(f"🔍 Analyzing {len(calculations)} calculations with input settings:")
        
        all_keywords = set()
        all_methods = set()
        all_exchanges = set()
        all_correlations = set()
        basis_types = set()
        shrink_factors = set()
        
        for calc in calculations:
            material_id, calc_id, calc_type, settings_json = calc
            
            try:
                settings = json.loads(settings_json)
                
                # Collect keywords
                if 'crystal_keywords' in settings:
                    all_keywords.update(settings['crystal_keywords'])
                
                # Collect functional info
                if 'functional_info' in settings:
                    func_info = settings['functional_info']
                    if 'method' in func_info:
                        all_methods.add(func_info['method'])
                    if 'exchange' in func_info:
                        all_exchanges.add(func_info['exchange'])
                    if 'correlation' in func_info:
                        all_correlations.add(func_info['correlation'])
                
                # Collect basis set info
                if 'basis_set_info' in settings:
                    basis_info = settings['basis_set_info']
                    if 'type' in basis_info:
                        basis_types.add(basis_info['type'])
                
                # Collect SHRINK factors
                if 'calculation_parameters' in settings:
                    params = settings['calculation_parameters']
                    if 'shrink_factor' in params:
                        shrink = params['shrink_factor']
                        if isinstance(shrink, dict):
                            shrink_factors.add(f"{shrink.get('k_points', 'N/A')}x{shrink.get('density', 'N/A')}")
                        else:
                            shrink_factors.add(str(shrink))
                            
            except json.JSONDecodeError:
                continue
        
        print(f"\n📈 Global Statistics:")
        print(f"   CRYSTAL Keywords ({len(all_keywords)}):")
        print(f"      {', '.join(sorted(all_keywords))}")
        
        print(f"   Methods ({len(all_methods)}):")
        print(f"      {', '.join(sorted(all_methods))}")
        
        if all_exchanges:
            print(f"   Exchange Functionals ({len(all_exchanges)}):")
            print(f"      {', '.join(sorted(all_exchanges))}")
        
        if all_correlations:
            print(f"   Correlation Functionals ({len(all_correlations)}):")
            print(f"      {', '.join(sorted(all_correlations))}")
        
        if basis_types:
            print(f"   Basis Set Types: {', '.join(sorted(basis_types))}")
        
        if shrink_factors:
            print(f"   SHRINK Factors ({len(shrink_factors)}):")
            print(f"      {', '.join(sorted(shrink_factors))}")
        
    else:
        print(f"❌ No calculations with input settings found")


def main():
    """Main query function."""
    parser = argparse.ArgumentParser(description="Query CRYSTAL input settings from materials.db")
    parser.add_argument("--calc-id", help="Show settings for specific calculation")
    parser.add_argument("--material-id", help="Show settings for specific material")
    parser.add_argument("--list-all", action="store_true", help="List all calculations with settings")
    parser.add_argument("--settings-summary", action="store_true", help="Show settings summary")
    parser.add_argument("--db-path", default="materials.db", help="Database path")
    
    args = parser.parse_args()
    
    if args.calc_id:
        query_calculation_input_settings(args.calc_id, args.db_path)
    elif args.material_id:
        query_material_input_settings(args.material_id, args.db_path)
    elif args.list_all:
        list_all_input_settings(args.db_path)
    elif args.settings_summary:
        show_settings_summary(args.db_path)
    else:
        print("❌ Please specify one of: --calc-id, --material-id, --list-all, or --settings-summary")
        print("\nUsage examples:")
        print("  python query_input_settings.py --calc-id calc_diamond_opt_001")
        print("  python query_input_settings.py --material-id diamond")
        print("  python query_input_settings.py --list-all")
        print("  python query_input_settings.py --settings-summary")


# Export functions for use in other modules
def query_materials(db, filters=None, limit=None):
    """Query materials from database with optional filters."""
    materials = db.get_all_materials()
    
    # Apply filters if provided
    if filters:
        # This is a simplified version - you can expand as needed
        filtered = []
        for mat in materials:
            # Add filter logic here
            filtered.append(mat)
        materials = filtered
    
    # Apply limit if specified
    if limit:
        materials = materials[:limit]
        
    return materials


def execute_custom_query(db, query_string):
    """Execute a custom SQL query on the database."""
    # This is a placeholder - implement actual query execution
    # based on your database implementation
    return []


if __name__ == "__main__":
    main()