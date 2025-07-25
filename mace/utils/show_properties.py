#!/usr/bin/env python3
from mace.database.materials import MaterialDatabase
import sys

material_id = sys.argv[1] if len(sys.argv) > 1 else '1_dia%'

db = MaterialDatabase('materials.db')
with db._get_connection() as conn:
    cursor = conn.execute('''
        SELECT material_id, property_category, property_name, property_value, property_unit 
        FROM properties 
        WHERE material_id LIKE ? 
        ORDER BY property_category, property_name
    ''', (material_id,))
    
    current_category = None
    for row in cursor:
        material_id, category, name, value, unit = row
        if category != current_category:
            print(f'\n📋 {category.upper()}:')
            current_category = category
        
        value_str = f'{value} {unit}' if value and unit else (str(value) if value else 'N/A')
        print(f'  • {name}: {value_str}')