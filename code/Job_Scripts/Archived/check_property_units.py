#!/usr/bin/env python3
"""Check property units for scientific accuracy."""

from material_database import MaterialDatabase

def check_property_units():
    db = MaterialDatabase("materials.db")
    
    print("🔍 Checking Property Units for Scientific Accuracy:")
    print("="*60)
    
    # Get all unique property names and their units
    with db._get_connection() as conn:
        cursor = conn.execute("""
            SELECT DISTINCT property_name, property_unit, property_category, COUNT(*) as count
            FROM properties 
            WHERE property_unit IS NOT NULL AND property_unit != ""
            GROUP BY property_name, property_unit 
            ORDER BY property_category, property_name
        """)
        
        current_category = None
        units_to_fix = []
        
        for row in cursor.fetchall():
            prop_name, unit, category, count = row
            
            if category != current_category:
                print(f"\n📊 {category.upper()} PROPERTIES:")
                current_category = category
            
            print(f"   {prop_name:<35} | {unit:<15} | {count} values")
            
            # Check for incorrect units
            if prop_name in ['optimization_cycles', 'optimization_converged'] and unit == 'Å':
                units_to_fix.append((prop_name, unit, 'cycles' if 'cycles' in prop_name else 'boolean'))
            elif prop_name == 'centering_code' and unit == 'Å':
                units_to_fix.append((prop_name, unit, 'dimensionless'))
    
    # Report units that need fixing
    if units_to_fix:
        print(f"\n❌ UNITS THAT NEED FIXING:")
        for prop_name, wrong_unit, correct_unit in units_to_fix:
            print(f"   {prop_name}: {wrong_unit} → {correct_unit}")
    else:
        print(f"\n✅ All property units appear correct!")
    
    return units_to_fix

if __name__ == "__main__":
    check_property_units()