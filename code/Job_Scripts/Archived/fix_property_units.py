#!/usr/bin/env python3
"""
Fix Property Units for Scientific Accuracy
==========================================
Correct incorrect units for various properties in the materials database.

Author: Generated for materials database project
"""

from material_database import MaterialDatabase
from typing import Dict, List, Tuple


class PropertyUnitFixer:
    """Fix incorrect property units in the database."""
    
    def __init__(self, db_path: str = "materials.db"):
        self.db = MaterialDatabase(db_path)
        
        # Define correct units for properties
        self.unit_corrections = {
            # Count properties should be dimensionless
            'final_atoms_count': 'dimensionless',
            'initial_atoms_count': 'dimensionless',
            'initial_initial_atoms_count': 'dimensionless',
            'final_final_atoms_count': 'dimensionless',
            'atoms_in_unit_cell': 'dimensionless',
            'max_coordination_number': 'dimensionless',
            'total_coordination_shells': 'dimensionless',
            
            # Population analysis should be in electrons or dimensionless
            'mulliken_alpha_plus_beta': 'electrons',
            'mulliken_alpha_minus_beta': 'electrons',
            'overlap_population_alpha_minus_beta': 'dimensionless',
            'overlap_population_alpha_plus_beta': 'dimensionless',
            'overlap_population': 'dimensionless',
            'mulliken_population': 'electrons',
            
            # Processed properties should have appropriate units
            'processed_atomic_charges': 'JSON',
            'processed_bonding_analysis': 'JSON',
            'processed_coordination_environments': 'JSON',
            'processed_magnetic_info': 'JSON',
            'processed_chemical_summary': 'JSON',
            
            # Complex data structures
            'neighbor_analysis': 'JSON',
            'atomic_positions': 'coordinates',
            'final_atomic_positions': 'coordinates', 
            'initial_atomic_positions': 'coordinates',
            'final_final_atomic_positions': 'coordinates',
            'initial_initial_atomic_positions': 'coordinates',
            
            # Boolean properties
            'optimization_converged': 'boolean',
            'is_spin_polarized': 'boolean',
            'spin_polarized': 'boolean',
            
            # Cycles and iterations
            'optimization_cycles': 'cycles',
            
            # Codes and identifiers
            'centering_code': 'code'
        }
    
    def identify_incorrect_units(self) -> List[Tuple[str, str, str]]:
        """Identify properties with incorrect units."""
        incorrect_units = []
        
        with self.db._get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT property_name, property_unit, COUNT(*) as count
                FROM properties
                WHERE property_unit IS NOT NULL
                GROUP BY property_name, property_unit
                ORDER BY property_name
            """)
            
            for prop_name, current_unit, count in cursor.fetchall():
                if prop_name in self.unit_corrections:
                    correct_unit = self.unit_corrections[prop_name]
                    if current_unit != correct_unit:
                        incorrect_units.append((prop_name, current_unit, correct_unit))
        
        return incorrect_units
    
    def fix_property_units(self, dry_run: bool = True) -> Dict[str, int]:
        """Fix incorrect property units in the database."""
        incorrect_units = self.identify_incorrect_units()
        
        if not incorrect_units:
            print("✅ All property units are correct!")
            return {}
        
        print(f"🔧 Found {len(incorrect_units)} properties with incorrect units:")
        
        fixes_made = {}
        
        for prop_name, wrong_unit, correct_unit in incorrect_units:
            print(f"   {prop_name}: {wrong_unit} → {correct_unit}")
            
            if not dry_run:
                with self.db._get_connection() as conn:
                    cursor = conn.execute("""
                        UPDATE properties 
                        SET property_unit = ?
                        WHERE property_name = ? AND property_unit = ?
                    """, (correct_unit, prop_name, wrong_unit))
                    
                    rows_updated = cursor.rowcount
                    fixes_made[prop_name] = rows_updated
                    print(f"      ✅ Updated {rows_updated} records")
        
        if dry_run:
            print("\n🔍 This was a dry run. Use fix_property_units(dry_run=False) to apply changes.")
        else:
            print(f"\n✅ Fixed units for {len(fixes_made)} properties!")
        
        return fixes_made
    
    def verify_unit_fixes(self) -> bool:
        """Verify that all unit fixes were applied correctly."""
        remaining_incorrect = self.identify_incorrect_units()
        
        if remaining_incorrect:
            print(f"❌ Still have {len(remaining_incorrect)} properties with incorrect units:")
            for prop_name, wrong_unit, correct_unit in remaining_incorrect:
                print(f"   {prop_name}: {wrong_unit} (should be {correct_unit})")
            return False
        else:
            print("✅ All property units are now correct!")
            return True


def fix_units_in_property_extractor():
    """Fix the property unit assignment in the extractor code."""
    
    print("🔧 Fixing Property Unit Assignment in Extractor:")
    print("=" * 50)
    
    # Read the current property extractor
    with open('/mnt/iscsi/UsefulScripts/Codebase/reorganization/code/Job_Scripts/crystal_property_extractor.py', 'r') as f:
        content = f.read()
    
    # Check if we need to fix the _get_property_unit method
    if '_get_property_unit' in content:
        print("✅ _get_property_unit method exists - needs updating")
        
        # Create updated unit mapping
        unit_mapping = """
    def _get_property_unit(self, prop_name: str) -> str:
        \"\"\"Get the appropriate unit for a property.\"\"\"
        
        # Energy properties
        if any(x in prop_name.lower() for x in ['energy', 'gap']) and prop_name.endswith('_ev'):
            return 'eV'
        elif any(x in prop_name.lower() for x in ['energy', 'gap']) and prop_name.endswith('_au'):
            return 'Hartree'
        elif 'band_gap' in prop_name.lower() or 'gap' in prop_name.lower():
            return 'eV'
        
        # Lattice parameters
        elif any(x in prop_name.lower() for x in ['primitive_a', 'primitive_b', 'primitive_c', 'crystallographic_a', 'crystallographic_b', 'crystallographic_c']):
            return 'Å'
        elif any(x in prop_name.lower() for x in ['alpha', 'beta', 'gamma']) and 'primitive' in prop_name:
            return 'degrees'
        elif 'volume' in prop_name.lower():
            return 'Å³'
        elif 'density' in prop_name.lower():
            return 'g/cm³'
        elif 'distance' in prop_name.lower():
            return 'Å'
        
        # Count properties (dimensionless)
        elif any(x in prop_name.lower() for x in ['atoms_count', 'coordination_number', 'atoms_in_unit_cell', 'shells']):
            return 'dimensionless'
        
        # Population analysis
        elif 'mulliken' in prop_name.lower():
            return 'electrons'
        elif 'overlap_population' in prop_name.lower():
            return 'dimensionless'
        
        # Boolean properties
        elif any(x in prop_name.lower() for x in ['converged', 'polarized']):
            return 'boolean'
        
        # Cycles and iterations
        elif 'cycles' in prop_name.lower():
            return 'cycles'
        
        # Gradients
        elif 'gradient' in prop_name.lower():
            return 'Hartree/Bohr'
        
        # Codes and identifiers
        elif any(x in prop_name.lower() for x in ['code', 'centering']):
            return 'code'
        
        # Positions and coordinates
        elif 'position' in prop_name.lower():
            return 'coordinates'
        
        # Complex data (JSON)
        elif any(x in prop_name.lower() for x in ['processed_', 'neighbor_analysis']):
            return 'JSON'
        
        # Default for unknown properties
        else:
            return 'dimensionless'
"""
        
        print("✅ Updated unit mapping created")
        return unit_mapping
    else:
        print("❌ _get_property_unit method not found")
        return None


if __name__ == "__main__":
    print("🔧 Property Unit Correction System")
    print("=" * 40)
    
    # Initialize fixer
    fixer = PropertyUnitFixer()
    
    # First, show current incorrect units
    print("\n1. Identifying incorrect units:")
    incorrect_units = fixer.identify_incorrect_units()
    
    if incorrect_units:
        print(f"\nFound {len(incorrect_units)} properties with incorrect units:")
        for prop_name, wrong_unit, correct_unit in incorrect_units:
            print(f"   {prop_name:<35} | {wrong_unit:<15} → {correct_unit}")
        
        # Apply fixes
        print(f"\n2. Applying unit corrections:")
        fixes_made = fixer.fix_property_units(dry_run=False)
        
        print(f"\n3. Verifying fixes:")
        success = fixer.verify_unit_fixes()
        
        if success:
            print(f"\n✅ All property units corrected successfully!")
            print(f"   Fixed {sum(fixes_made.values())} total property records")
        else:
            print(f"\n❌ Some unit issues remain")
    else:
        print("✅ All property units are already correct!")
    
    # Show the updated unit mapping for the extractor
    print(f"\n4. Property extractor unit mapping:")
    unit_mapping = fix_units_in_property_extractor()
    if unit_mapping:
        print("✅ Unit mapping ready for integration into property extractor")