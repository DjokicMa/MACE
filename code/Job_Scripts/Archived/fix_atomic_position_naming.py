#!/usr/bin/env python3
"""
Fix Atomic Position Naming Confusion
====================================
Clarify and fix the confusing atomic position property names to be more descriptive
and scientifically meaningful.

Current confusing naming:
- initial_initial_atomic_positions
- initial_atomic_positions  
- final_atomic_positions
- final_final_atomic_positions
- atomic_positions

Author: Generated for materials database project
"""

from material_database import MaterialDatabase
from typing import Dict, List, Tuple


class AtomicPositionNamingFixer:
    """Fix confusing atomic position property names."""
    
    def __init__(self, db_path: str = "materials.db"):
        self.db = MaterialDatabase(db_path)
        
        # Define clear naming scheme
        self.naming_corrections = {
            # Count properties - clarify what they count
            'initial_initial_atoms_count': 'input_geometry_atom_count',
            'initial_atoms_count': 'starting_geometry_atom_count', 
            'final_atoms_count': 'optimized_geometry_atom_count',
            'final_final_atoms_count': 'final_geometry_atom_count',
            
            # Position properties - clarify source and type
            'initial_initial_atomic_positions': 'input_file_atomic_positions',
            'initial_atomic_positions': 'starting_calculation_atomic_positions',
            'atomic_positions': 'current_atomic_positions',  # Most generic
            'final_atomic_positions': 'optimized_atomic_positions',
            'final_final_atomic_positions': 'final_output_atomic_positions'
        }
        
        # Add descriptions for clarity
        self.property_descriptions = {
            'input_geometry_atom_count': 'Number of atoms in input file geometry',
            'starting_geometry_atom_count': 'Number of atoms at calculation start',
            'optimized_geometry_atom_count': 'Number of atoms in optimized geometry',
            'final_geometry_atom_count': 'Number of atoms in final output geometry',
            
            'input_file_atomic_positions': 'Atomic positions from input file (CIF/D12)',
            'starting_calculation_atomic_positions': 'Atomic positions at calculation start',
            'current_atomic_positions': 'Current atomic positions (context-dependent)',
            'optimized_atomic_positions': 'Atomic positions after optimization',
            'final_output_atomic_positions': 'Final atomic positions in output file'
        }
    
    def analyze_position_properties(self) -> Dict[str, List]:
        """Analyze existing position properties to understand their context."""
        analysis = {
            'count_properties': [],
            'position_properties': [],
            'unclear_naming': [],
            'calc_type_distribution': {}
        }
        
        with self.db._get_connection() as conn:
            # Get all position-related properties
            cursor = conn.execute("""
                SELECT property_name, COUNT(*) as count,
                       GROUP_CONCAT(DISTINCT calc_id) as calc_ids
                FROM properties 
                WHERE property_name LIKE '%position%' 
                   OR property_name LIKE '%atoms_count'
                   OR property_name LIKE '%atomic_position%'
                GROUP BY property_name
                ORDER BY property_name
            """)
            
            for prop_name, count, calc_ids in cursor.fetchall():
                calc_id_list = calc_ids.split(',') if calc_ids else []
                
                prop_info = {
                    'name': prop_name,
                    'count': count,
                    'calc_ids': calc_id_list[:5]  # First 5 calc_ids
                }
                
                if 'count' in prop_name:
                    analysis['count_properties'].append(prop_info)
                elif 'position' in prop_name:
                    analysis['position_properties'].append(prop_info)
                
                # Check for unclear naming patterns
                if any(pattern in prop_name for pattern in ['initial_initial', 'final_final']):
                    analysis['unclear_naming'].append(prop_info)
                
                # Analyze calc_type distribution
                if calc_id_list:
                    for calc_id in calc_id_list[:3]:  # Check first 3
                        cursor_calc = conn.execute(
                            "SELECT calc_type FROM calculations WHERE calc_id = ?", 
                            (calc_id,)
                        )
                        result = cursor_calc.fetchone()
                        if result:
                            calc_type = result[0]
                            if prop_name not in analysis['calc_type_distribution']:
                                analysis['calc_type_distribution'][prop_name] = {}
                            if calc_type not in analysis['calc_type_distribution'][prop_name]:
                                analysis['calc_type_distribution'][prop_name][calc_type] = 0
                            analysis['calc_type_distribution'][prop_name][calc_type] += 1
        
        return analysis
    
    def suggest_naming_improvements(self, analysis: Dict) -> List[Tuple[str, str, str]]:
        """Suggest improved naming based on analysis."""
        suggestions = []
        
        for prop_info in analysis['unclear_naming']:
            prop_name = prop_info['name']
            if prop_name in self.naming_corrections:
                new_name = self.naming_corrections[prop_name]
                description = self.property_descriptions.get(new_name, "Improved naming")
                suggestions.append((prop_name, new_name, description))
        
        # Check calc_type distribution for better context
        for prop_name, calc_types in analysis['calc_type_distribution'].items():
            if prop_name not in self.naming_corrections:
                continue
                
            new_name = self.naming_corrections[prop_name]
            context = f"Found in: {', '.join(calc_types.keys())}"
            suggestions.append((prop_name, new_name, context))
        
        # Remove duplicates
        unique_suggestions = []
        seen = set()
        for old_name, new_name, desc in suggestions:
            if old_name not in seen:
                unique_suggestions.append((old_name, new_name, desc))
                seen.add(old_name)
        
        return unique_suggestions
    
    def fix_property_names(self, dry_run: bool = True) -> Dict[str, int]:
        """Fix confusing property names in the database."""
        suggestions = self.suggest_naming_improvements(self.analyze_position_properties())
        
        if not suggestions:
            print("✅ No confusing property names found!")
            return {}
        
        print(f"🔧 Found {len(suggestions)} properties with confusing names:")
        
        fixes_made = {}
        
        for old_name, new_name, description in suggestions:
            print(f"   {old_name}")
            print(f"   → {new_name}")
            print(f"     {description}")
            
            if not dry_run:
                with self.db._get_connection() as conn:
                    cursor = conn.execute("""
                        UPDATE properties 
                        SET property_name = ?
                        WHERE property_name = ?
                    """, (new_name, old_name))
                    
                    rows_updated = cursor.rowcount
                    fixes_made[old_name] = rows_updated
                    print(f"     ✅ Updated {rows_updated} records")
            
            print()
        
        if dry_run:
            print("🔍 This was a dry run. Use fix_property_names(dry_run=False) to apply changes.")
        else:
            print(f"✅ Fixed naming for {len(fixes_made)} properties!")
        
        return fixes_made
    
    def create_naming_documentation(self) -> str:
        """Create documentation for the improved naming scheme."""
        doc = """
# Atomic Position Property Naming Guide

## Clear Naming Scheme for Position-Related Properties

### Count Properties
- `input_geometry_atom_count`: Number of atoms in the original input file (CIF/D12)
- `starting_geometry_atom_count`: Number of atoms when calculation begins
- `optimized_geometry_atom_count`: Number of atoms after geometry optimization
- `final_geometry_atom_count`: Number of atoms in final calculation output

### Position Properties  
- `input_file_atomic_positions`: Original atomic coordinates from input file
- `starting_calculation_atomic_positions`: Atomic coordinates at calculation start
- `current_atomic_positions`: Current/general atomic positions (context-dependent)
- `optimized_atomic_positions`: Atomic coordinates after optimization
- `final_output_atomic_positions`: Final atomic coordinates in calculation output

### Context and Usage
- **Input File**: Original geometry from CIF or D12 file
- **Starting Calculation**: Geometry after any preprocessing/symmetry operations
- **Optimized**: Geometry after OPT calculation convergence
- **Final Output**: Geometry written to final output file

### Relationship to Primitive vs Crystallographic
- These naming schemes apply to both primitive and crystallographic cells
- Cell type should be indicated by separate properties (e.g., `primitive_` vs `crystallographic_` prefixes)
- Position properties store actual atomic coordinates regardless of cell choice

### Migration from Old Names
Old confusing names have been replaced:
- `initial_initial_*` → `input_file_*` (from original input)
- `initial_*` → `starting_calculation_*` (calculation start)
- `final_*` → `optimized_*` (after optimization)
- `final_final_*` → `final_output_*` (final output)
"""
        return doc


def analyze_primitive_vs_crystallographic():
    """Analyze primitive vs crystallographic cell properties."""
    db = MaterialDatabase("materials.db")
    
    print("🔍 Analyzing Primitive vs Crystallographic Cell Properties:")
    print("=" * 60)
    
    with db._get_connection() as conn:
        # Get primitive cell properties
        cursor = conn.execute("""
            SELECT DISTINCT property_name, COUNT(*) as count
            FROM properties 
            WHERE property_name LIKE '%primitive%'
            ORDER BY property_name
        """)
        
        primitive_props = cursor.fetchall()
        print(f"\n📐 PRIMITIVE CELL PROPERTIES ({len(primitive_props)}):")
        for prop_name, count in primitive_props:
            print(f"   {prop_name:<35} | {count} values")
        
        # Get crystallographic cell properties
        cursor = conn.execute("""
            SELECT DISTINCT property_name, COUNT(*) as count
            FROM properties 
            WHERE property_name LIKE '%crystallographic%'
            ORDER BY property_name
        """)
        
        crystallographic_props = cursor.fetchall()
        print(f"\n🏗️  CRYSTALLOGRAPHIC CELL PROPERTIES ({len(crystallographic_props)}):")
        for prop_name, count in crystallographic_props:
            print(f"   {prop_name:<35} | {count} values")
        
        # Check if we have both for same materials
        cursor = conn.execute("""
            SELECT material_id, 
                   SUM(CASE WHEN property_name LIKE '%primitive%' THEN 1 ELSE 0 END) as primitive_count,
                   SUM(CASE WHEN property_name LIKE '%crystallographic%' THEN 1 ELSE 0 END) as crystallographic_count
            FROM properties 
            WHERE property_name LIKE '%primitive%' OR property_name LIKE '%crystallographic%'
            GROUP BY material_id
            ORDER BY material_id
        """)
        
        print(f"\n📊 MATERIALS WITH BOTH CELL TYPES:")
        both_count = 0
        for material_id, prim_count, cryst_count in cursor.fetchall():
            if prim_count > 0 and cryst_count > 0:
                both_count += 1
                print(f"   {material_id}: {prim_count} primitive, {cryst_count} crystallographic")
        
        print(f"\n✅ {both_count} materials have both primitive and crystallographic data")


if __name__ == "__main__":
    print("🔧 Atomic Position Naming Clarification System")
    print("=" * 50)
    
    # Initialize fixer
    fixer = AtomicPositionNamingFixer()
    
    # Analyze current situation
    print("\n1. Analyzing current position properties:")
    analysis = fixer.analyze_position_properties()
    
    print(f"\n📍 Position Properties: {len(analysis['position_properties'])}")
    for prop in analysis['position_properties']:
        calc_types = analysis['calc_type_distribution'].get(prop['name'], {})
        calc_type_str = ', '.join(calc_types.keys()) if calc_types else 'Unknown'
        print(f"   {prop['name']:<40} | {prop['count']} values | {calc_type_str}")
    
    print(f"\n🔢 Count Properties: {len(analysis['count_properties'])}")
    for prop in analysis['count_properties']:
        calc_types = analysis['calc_type_distribution'].get(prop['name'], {})
        calc_type_str = ', '.join(calc_types.keys()) if calc_types else 'Unknown'
        print(f"   {prop['name']:<40} | {prop['count']} values | {calc_type_str}")
    
    print(f"\n❓ Unclear Naming: {len(analysis['unclear_naming'])}")
    for prop in analysis['unclear_naming']:
        print(f"   {prop['name']:<40} | {prop['count']} values")
    
    # Suggest improvements
    print(f"\n2. Suggesting naming improvements:")
    suggestions = fixer.suggest_naming_improvements(analysis)
    
    if suggestions:
        print(f"\nFound {len(suggestions)} properties that need clearer naming:")
        for old_name, new_name, desc in suggestions:
            print(f"   {old_name}")
            print(f"   → {new_name}")
            print(f"     {desc}")
            print()
        
        # Apply fixes
        print("3. Applying naming fixes:")
        fixes = fixer.fix_property_names(dry_run=False)
        
        if fixes:
            print(f"✅ Successfully renamed {len(fixes)} property types!")
            total_records = sum(fixes.values())
            print(f"   Total property records updated: {total_records}")
    else:
        print("✅ All property names are already clear!")
    
    # Analyze primitive vs crystallographic
    print(f"\n4. Analyzing primitive vs crystallographic distinction:")
    analyze_primitive_vs_crystallographic()
    
    # Create documentation
    print(f"\n5. Creating naming documentation:")
    doc = fixer.create_naming_documentation()
    print("✅ Documentation created for improved naming scheme")