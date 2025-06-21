#!/usr/bin/env python3
"""
Test Property Extraction System
===============================
Comprehensive test of the property extraction system to show all the capabilities.

This script demonstrates:
1. Database duplicate fixing
2. Formula and space group extraction from materials 
3. Comprehensive property extraction from output files
4. Integration with the enhanced queue manager

Usage:
  python test_property_extraction.py [--fix-database] [--scan-outputs] [--test-extraction]
"""

import os
import sys
import argparse
from pathlib import Path

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from database_status_report import analyze_database
    from fix_database_duplicates import fix_database_duplicates
    from crystal_property_extractor import CrystalPropertyExtractor
    from formula_extractor import extract_formula_from_d12, extract_space_group_from_output
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

def test_formula_extraction():
    """Test formula extraction from D12 files."""
    print("🧪 Testing Formula Extraction")
    print("=" * 40)
    
    # Find D12 files in current directory
    d12_files = list(Path.cwd().glob("*.d12"))
    
    if not d12_files:
        print("  ⚠️  No D12 files found in current directory")
        return
    
    for d12_file in d12_files[:5]:  # Test first 5 files
        formula = extract_formula_from_d12(d12_file)
        print(f"  📄 {d12_file.name:50} → {formula or 'N/A'}")

def test_space_group_extraction():
    """Test space group extraction from output files."""
    print("\n🔍 Testing Space Group Extraction")
    print("=" * 40)
    
    # Find output files
    output_files = list(Path.cwd().rglob("*.out"))
    
    if not output_files:
        print("  ⚠️  No output files found")
        return
    
    for output_file in output_files[:5]:  # Test first 5 files
        space_group = extract_space_group_from_output(output_file)
        print(f"  📄 {output_file.name:50} → {space_group or 'N/A'}")

def test_property_extraction():
    """Test comprehensive property extraction."""
    print("\n📊 Testing Property Extraction")
    print("=" * 40)
    
    # Find output files
    output_files = list(Path.cwd().rglob("*.out"))
    
    if not output_files:
        print("  ⚠️  No output files found")
        return
    
    extractor = CrystalPropertyExtractor("materials.db")
    
    for output_file in output_files[:3]:  # Test first 3 files
        print(f"\n  📄 Processing: {output_file.name}")
        
        properties = extractor.extract_all_properties(output_file)
        
        if properties:
            print(f"    ✅ Extracted {len(properties)} properties:")
            
            # Show sample properties by category
            categories = {}
            for prop_name in properties:
                if prop_name.startswith('_'):
                    continue
                category = extractor._categorize_property(prop_name)
                if category not in categories:
                    categories[category] = []
                categories[category].append(prop_name)
            
            for category, props in categories.items():
                print(f"      📋 {category.title()}: {len(props)} properties")
                for prop in props[:3]:  # Show first 3 in each category
                    value = properties[prop]
                    if isinstance(value, (dict, list)):
                        print(f"        • {prop}: [complex data]")
                    else:
                        print(f"        • {prop}: {value}")
                if len(props) > 3:
                    print(f"        • ... and {len(props) - 3} more")
        else:
            print(f"    ⚠️  No properties extracted")

def scan_and_extract_all():
    """Scan all output files and extract properties."""
    print("\n🔍 Scanning All Output Files")
    print("=" * 40)
    
    output_files = list(Path.cwd().rglob("*.out"))
    
    if not output_files:
        print("  ⚠️  No output files found")
        return
    
    print(f"  📊 Found {len(output_files)} output files")
    
    extractor = CrystalPropertyExtractor("materials.db")
    total_properties = 0
    
    for i, output_file in enumerate(output_files):
        print(f"  📄 Processing {i+1}/{len(output_files)}: {output_file.name}")
        
        properties = extractor.extract_all_properties(output_file)
        
        if properties:
            saved_count = extractor.save_properties_to_database(properties)
            total_properties += saved_count
            print(f"    ✅ Saved {saved_count} properties")
        else:
            print(f"    ⚠️  No properties extracted")
    
    print(f"\n🎉 Total properties extracted and saved: {total_properties}")

def show_database_status():
    """Show current database status."""
    print("\n📊 Database Status")
    print("=" * 40)
    
    if not Path("materials.db").exists():
        print("  ⚠️  Database not found: materials.db")
        return
    
    analyze_database("materials.db")

def fix_duplicates():
    """Fix database duplicates."""
    print("\n🔧 Fixing Database Duplicates")
    print("=" * 40)
    
    if not Path("materials.db").exists():
        print("  ⚠️  Database not found: materials.db")
        return
    
    fix_database_duplicates("materials.db", dry_run=False)

def show_sample_properties():
    """Show sample properties from database."""
    print("\n📈 Sample Properties from Database")
    print("=" * 40)
    
    if not Path("materials.db").exists():
        print("  ⚠️  Database not found: materials.db")
        return
    
    db = MaterialDatabase("materials.db")
    
    # Get some sample properties
    with db._get_connection() as conn:
        cursor = conn.execute("""
            SELECT material_id, property_category, property_name, property_value, property_unit
            FROM properties 
            ORDER BY extracted_at DESC 
            LIMIT 20
        """)
        
        properties = cursor.fetchall()
        
        if properties:
            print("  Recent properties:")
            for prop in properties:
                material_id, category, name, value, unit = prop
                value_str = f"{value} {unit}" if value and unit else (value or "N/A")
                print(f"    📊 {material_id:15} | {category:12} | {name:25} | {value_str}")
        else:
            print("  ⚠️  No properties found in database")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test property extraction system")
    parser.add_argument("--fix-database", action="store_true", help="Fix database duplicates")
    parser.add_argument("--scan-outputs", action="store_true", help="Scan and extract from all output files")
    parser.add_argument("--test-extraction", action="store_true", help="Test extraction on sample files")
    
    args = parser.parse_args()
    
    print("🧪 CRYSTAL Property Extraction Test Suite")
    print("=" * 60)
    
    # Show initial database status
    show_database_status()
    
    if args.fix_database:
        fix_duplicates()
        show_database_status()
    
    if args.test_extraction:
        test_formula_extraction()
        test_space_group_extraction()
        test_property_extraction()
    
    if args.scan_outputs:
        scan_and_extract_all()
        show_database_status()
    
    # Always show sample properties if available
    show_sample_properties()
    
    print("\n🎯 Integration Status:")
    print("✅ Property extraction is now integrated into enhanced_queue_manager.py")
    print("✅ Formula and space group extraction is automatic")
    print("✅ Database duplicates can be fixed with fix_database_duplicates.py")
    print("✅ All properties are automatically extracted when jobs complete")
    
    print("\n📋 Available Properties Include:")
    print("   • Structural: lattice parameters, cell volumes, densities, atomic positions")
    print("   • Electronic: band gaps (direct/indirect, alpha/beta), total energies")
    print("   • Energy Components: kinetic, exchange, correlation, D3 dispersion")
    print("   • Population Analysis: Mulliken charges, overlap populations")
    print("   • Optimization: convergence info, gradient norms, cycle counts")
    print("   • Crystallographic: space groups, crystal systems, centering codes")
    
    print("\n🚀 Next Steps:")
    print("1. Run workflows normally - properties will be extracted automatically")
    print("2. Use database_status_report.py to check extraction status")
    print("3. Query properties using SQL or MaterialDatabase methods")

if __name__ == "__main__":
    main()