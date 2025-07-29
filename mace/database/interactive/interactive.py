"""
Interactive Database Explorer
=============================
Interactive command-line interface for database exploration and queries.
"""

import cmd
import readline
import os
import sys
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Import database modules
from ..materials import MaterialDatabase
from ..query.filters import PropertyFilter
from ..analysis import (
    compare_materials, analyze_missing_data,
    calculate_property_correlations, analyze_property_distributions,
    track_workflow_progress
)
from ..export import export_materials
from ..utils import convert_units, get_property_units, validate_materials
from ..utils.history import PropertyHistory


class DatabaseExplorer(cmd.Cmd):
    """Interactive database explorer with command completion."""
    
    intro = """
╔══════════════════════════════════════════════════════════════╗
║              MACE Database Interactive Explorer              ║
╚══════════════════════════════════════════════════════════════╝

Type 'help' for available commands or 'help <command>' for details.
Type 'exit' or 'quit' to leave.

Current context: No material selected
"""
    
    prompt = 'mace-db> '
    
    def __init__(self, db_path: str = 'materials.db'):
        """Initialize the interactive explorer."""
        super().__init__()
        self.db = MaterialDatabase(db_path)
        self.history_manager = PropertyHistory(db_path)
        self.current_material = None
        self.last_results = []
        self.filter = PropertyFilter()
        
        # Set up command history
        self.history_file = os.path.expanduser('~/.mace_db_history')
        self._load_history()
        
    def _load_history(self):
        """Load command history."""
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            pass
            
    def _save_history(self):
        """Save command history."""
        readline.write_history_file(self.history_file)
        
    def postcmd(self, stop, line):
        """Update prompt after each command."""
        if self.current_material:
            self.prompt = f'mace-db [{self.current_material}]> '
        else:
            self.prompt = 'mace-db> '
        return stop
        
    def do_exit(self, arg):
        """Exit the interactive explorer."""
        self._save_history()
        print("\nGoodbye!")
        return True
        
    def do_quit(self, arg):
        """Exit the interactive explorer."""
        return self.do_exit(arg)
        
    def do_help(self, arg):
        """Show help for commands."""
        if arg:
            # Show help for specific command
            super().do_help(arg)
        else:
            print("\n=== Available Commands ===\n")
            print("Navigation:")
            print("  select <material_id>  - Select a material for detailed exploration")
            print("  list [limit]          - List materials (default: 20)")
            print("  search <query>        - Search materials by ID or formula")
            print("")
            print("Queries:")
            print("  filter <expression>   - Add property filter (e.g., 'band_gap > 3')")
            print("  clearfilter          - Clear all filters")
            print("  showfilter           - Show current filters")
            print("  query                - Execute query with current filters")
            print("")
            print("Properties:")
            print("  props [category]      - Show properties of current material")
            print("  value <property>      - Show specific property value")
            print("  convert <prop> <unit> - Convert property to different unit")
            print("  history [property]    - Show property history")
            print("")
            print("Analysis:")
            print("  compare <mat1,mat2..> - Compare multiple materials")
            print("  missing               - Analyze missing data")
            print("  correlate <p1> <p2>   - Correlate two properties")
            print("  distribution <prop>   - Show property distribution")
            print("  workflow [type]       - Track workflow progress")
            print("")
            print("Export:")
            print("  export <format> [file]- Export results (csv/json/excel)")
            print("  save                  - Save last results to file")
            print("")
            print("Utilities:")
            print("  stats                 - Show database statistics")
            print("  validate              - Validate current material")
            print("  units <property>      - Show available units")
            print("  clear                 - Clear screen")
            print("  help [command]        - Show help")
            print("  exit/quit            - Exit explorer")
            
    def do_clear(self, arg):
        """Clear the screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
    def do_select(self, arg):
        """Select a material for detailed exploration.
        Usage: select <material_id>"""
        if not arg:
            print("Usage: select <material_id>")
            return
            
        material = self.db.get_material(arg)
        if material:
            self.current_material = arg
            print(f"\nSelected material: {arg}")
            if material.get('formula'):
                print(f"Formula: {material['formula']}")
            if material.get('space_group'):
                print(f"Space Group: {material['space_group']}")
        else:
            print(f"Material '{arg}' not found")
            
    def complete_select(self, text, line, begidx, endidx):
        """Auto-complete material IDs."""
        materials = self.db.get_all_materials()
        mat_ids = [m['material_id'] for m in materials]
        return [mid for mid in mat_ids if mid.startswith(text)]
        
    def do_list(self, arg):
        """List materials in the database.
        Usage: list [limit]"""
        limit = 20
        if arg:
            try:
                limit = int(arg)
            except ValueError:
                print("Invalid limit. Using default: 20")
                
        materials = self.db.get_all_materials()[:limit]
        self.last_results = materials
        
        print(f"\n{'Material ID':20} {'Formula':15} {'Space Group':12} {'Properties':10}")
        print("-" * 60)
        
        for mat in materials:
            mat_id = mat['material_id'][:20]
            formula = (mat.get('formula') or 'N/A')[:15]
            sg = str(mat.get('space_group') or 'N/A')[:12]
            
            # Count properties
            props = self.db.get_material_properties(mat['material_id'])
            prop_count = len(props) if props else 0
            
            print(f"{mat_id:20} {formula:15} {sg:12} {prop_count:10}")
            
        if len(materials) < len(self.db.get_all_materials()):
            print(f"\nShowing {len(materials)} of {len(self.db.get_all_materials())} materials")
            
    def do_search(self, arg):
        """Search materials by ID or formula.
        Usage: search <query>"""
        if not arg:
            print("Usage: search <query>")
            return
            
        materials = self.db.get_all_materials()
        results = []
        
        query = arg.lower()
        for mat in materials:
            if (query in mat['material_id'].lower() or 
                (mat.get('formula') and query in mat['formula'].lower())):
                results.append(mat)
                
        self.last_results = results
        
        if results:
            print(f"\nFound {len(results)} materials:")
            print(f"\n{'Material ID':20} {'Formula':15} {'Space Group':12}")
            print("-" * 50)
            
            for mat in results[:20]:
                mat_id = mat['material_id'][:20]
                formula = (mat.get('formula') or 'N/A')[:15]
                sg = str(mat.get('space_group') or 'N/A')[:12]
                print(f"{mat_id:20} {formula:15} {sg:12}")
                
            if len(results) > 20:
                print(f"\n... and {len(results) - 20} more")
        else:
            print(f"No materials found matching '{arg}'")
            
    def do_filter(self, arg):
        """Add a property filter.
        Usage: filter <property> <operator> <value>
        Example: filter band_gap > 3.0"""
        if not arg:
            print("Usage: filter <property> <operator> <value>")
            print("Example: filter band_gap > 3.0")
            return
            
        # Parse filter expression
        parts = arg.split()
        if len(parts) < 3:
            print("Invalid filter format")
            return
            
        property_name = parts[0]
        operator = parts[1]
        value = ' '.join(parts[2:])
        
        # Try to convert value to appropriate type
        try:
            # Try float first
            value = float(value)
        except ValueError:
            # Try int
            try:
                value = int(value)
            except ValueError:
                # Keep as string
                pass
                
        try:
            self.filter.add_filter(property_name, operator, value)
            print(f"Added filter: {property_name} {operator} {value}")
        except ValueError as e:
            print(f"Invalid filter: {e}")
            
    def do_clearfilter(self, arg):
        """Clear all filters."""
        self.filter = PropertyFilter()
        print("All filters cleared")
        
    def do_showfilter(self, arg):
        """Show current filters."""
        filters = self.filter.get_filters()
        if filters:
            print("\nCurrent filters:")
            for f in filters:
                print(f"  {f['property']} {f['operator']} {f['value']}")
        else:
            print("No filters set")
            
    def do_query(self, arg):
        """Execute query with current filters."""
        if not self.filter.get_filters():
            print("No filters set. Use 'filter' command to add filters.")
            return
            
        # Apply filters
        materials = self.db.get_all_materials()
        results = []
        
        for material in materials:
            if self.filter.matches(material['material_id'], self.db):
                results.append(material)
                
        self.last_results = results
        
        print(f"\nFound {len(results)} materials matching filters:")
        
        if results:
            print(f"\n{'Material ID':20} {'Formula':15} {'Matched Properties':30}")
            print("-" * 70)
            
            for mat in results[:20]:
                mat_id = mat['material_id'][:20]
                formula = (mat.get('formula') or 'N/A')[:15]
                
                # Show which properties matched
                props = self.db.get_material_properties(mat['material_id'])
                matched = []
                for prop in props:
                    for f in self.filter.get_filters():
                        if prop['property_name'] == f['property']:
                            matched.append(f"{f['property']}={prop['property_value']}")
                            
                matched_str = ', '.join(matched[:2])
                if len(matched) > 2:
                    matched_str += f" (+{len(matched)-2})"
                    
                print(f"{mat_id:20} {formula:15} {matched_str:30}")
                
            if len(results) > 20:
                print(f"\n... and {len(results) - 20} more")
                
    def do_props(self, arg):
        """Show properties of current material.
        Usage: props [category]"""
        if not self.current_material:
            print("No material selected. Use 'select' command first.")
            return
            
        props = self.db.get_material_properties(self.current_material)
        
        if not props:
            print(f"No properties found for {self.current_material}")
            return
            
        # Group by category
        by_category = {}
        for prop in props:
            cat = prop.get('property_category', 'General')
            if arg and cat.lower() != arg.lower():
                continue
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(prop)
            
        if not by_category:
            print(f"No properties found in category '{arg}'")
            return
            
        for category, cat_props in sorted(by_category.items()):
            print(f"\n=== {category} ===")
            for prop in sorted(cat_props, key=lambda x: x['property_name']):
                name = prop['property_name']
                value = prop['property_value']
                unit = prop.get('property_unit', '')
                print(f"  {name:.<30} {value} {unit}")
                
    def do_value(self, arg):
        """Show specific property value.
        Usage: value <property_name>"""
        if not self.current_material:
            print("No material selected. Use 'select' command first.")
            return
            
        if not arg:
            print("Usage: value <property_name>")
            return
            
        props = self.db.get_material_properties(self.current_material)
        
        for prop in props:
            if prop['property_name'] == arg:
                print(f"\n{arg}: {prop['property_value']} {prop.get('property_unit', '')}")
                
                # Show metadata
                if prop.get('calc_id'):
                    calc = self.db.get_calculation(prop['calc_id'])
                    if calc:
                        print(f"  From: {calc.get('calculation_type', 'Unknown')} calculation")
                        print(f"  Status: {calc.get('status', 'Unknown')}")
                return
                
        print(f"Property '{arg}' not found")
        
    def complete_value(self, text, line, begidx, endidx):
        """Auto-complete property names."""
        if not self.current_material:
            return []
            
        props = self.db.get_material_properties(self.current_material)
        prop_names = [p['property_name'] for p in props]
        return [name for name in prop_names if name.startswith(text)]
        
    def do_convert(self, arg):
        """Convert property to different unit.
        Usage: convert <property> <unit>"""
        if not self.current_material:
            print("No material selected. Use 'select' command first.")
            return
            
        parts = arg.split()
        if len(parts) != 2:
            print("Usage: convert <property> <unit>")
            return
            
        prop_name, target_unit = parts
        
        props = self.db.get_material_properties(self.current_material)
        
        for prop in props:
            if prop['property_name'] == prop_name:
                try:
                    value = float(prop['property_value'])
                    from_unit = prop.get('property_unit', '')
                    
                    if from_unit:
                        converted = convert_units(value, from_unit, target_unit, prop_name)
                        print(f"\n{prop_name}:")
                        print(f"  Original: {value} {from_unit}")
                        print(f"  Converted: {converted:.6f} {target_unit}")
                    else:
                        print(f"No unit information for {prop_name}")
                        
                except Exception as e:
                    print(f"Conversion error: {e}")
                    
                return
                
        print(f"Property '{prop_name}' not found")
        
    def do_units(self, arg):
        """Show available units for a property.
        Usage: units <property>"""
        if not arg:
            print("Usage: units <property>")
            return
            
        units = get_property_units(arg)
        
        if units:
            print(f"\nAvailable units for '{arg}':")
            for unit in units:
                print(f"  {unit}")
        else:
            print(f"No unit information available for '{arg}'")
            
    def do_compare(self, arg):
        """Compare multiple materials.
        Usage: compare <material1,material2,...>"""
        if not arg:
            print("Usage: compare <material1,material2,...>")
            return
            
        material_ids = [m.strip() for m in arg.split(',')]
        
        try:
            result = compare_materials(self.db, material_ids, output_format='report')
            print(result)
        except Exception as e:
            print(f"Comparison error: {e}")
            
    def do_stats(self, arg):
        """Show database statistics."""
        materials = self.db.get_all_materials()
        all_props = self.db.get_all_properties()
        
        # Count properties by type
        prop_counts = {}
        for prop in all_props:
            name = prop['property_name']
            prop_counts[name] = prop_counts.get(name, 0) + 1
            
        print("\n=== Database Statistics ===")
        print(f"Total materials: {len(materials)}")
        print(f"Total properties: {len(all_props)}")
        print(f"Unique property types: {len(prop_counts)}")
        
        print("\n=== Top Properties ===")
        sorted_props = sorted(prop_counts.items(), key=lambda x: x[1], reverse=True)
        for prop, count in sorted_props[:10]:
            print(f"  {prop:.<30} {count}")
            
    def do_export(self, arg):
        """Export results to file.
        Usage: export <format> [filename]"""
        parts = arg.split()
        if not parts:
            print("Usage: export <format> [filename]")
            print("Formats: csv, json, excel")
            return
            
        format_type = parts[0]
        filename = parts[1] if len(parts) > 1 else None
        
        if not self.last_results:
            print("No results to export. Run a query first.")
            return
            
        # Convert results to material IDs
        material_ids = [m['material_id'] for m in self.last_results]
        
        try:
            exported_file = export_materials(
                self.db,
                format=format_type,
                output_file=filename,
                material_ids=material_ids
            )
            print(f"Exported to: {exported_file}")
        except Exception as e:
            print(f"Export error: {e}")
            
    def do_history(self, arg):
        """Show property history.
        Usage: history [property_name]"""
        if not self.current_material:
            print("No material selected. Use 'select' command first.")
            return
            
        report = self.history_manager.format_history_report(
            self.current_material, 
            arg if arg else None
        )
        print(report)
        
    def do_validate(self, arg):
        """Validate current material data."""
        if not self.current_material:
            print("No material selected. Use 'select' command first.")
            return
            
        result = validate_materials(
            self.db, 
            [self.current_material],
            output_format='report'
        )
        print(result)
        
    def do_missing(self, arg):
        """Analyze missing data for current material or all."""
        material_ids = None
        if self.current_material and not arg:
            material_ids = [self.current_material]
            
        result = analyze_missing_data(
            self.db,
            material_ids=material_ids,
            detail_level='detailed',
            output_format='report'
        )
        print(result)
        
    def do_workflow(self, arg):
        """Track workflow progress.
        Usage: workflow [workflow_type]"""
        workflow = arg if arg else 'full_electronic'
        material_ids = None
        
        if self.current_material:
            material_ids = [self.current_material]
            
        result = track_workflow_progress(
            self.db,
            material_ids=material_ids,
            workflow=workflow,
            output_format='report',
            detailed=True if material_ids else False
        )
        print(result)
        
    def do_correlate(self, arg):
        """Show correlation between two properties.
        Usage: correlate <property1> <property2>"""
        parts = arg.split()
        if len(parts) != 2:
            print("Usage: correlate <property1> <property2>")
            return
            
        prop1, prop2 = parts
        
        result = calculate_property_correlations(
            self.db,
            property_pairs=[(prop1, prop2)],
            output_format='report'
        )
        print(result)
        
    def do_distribution(self, arg):
        """Show property distribution.
        Usage: distribution <property>"""
        if not arg:
            print("Usage: distribution <property>")
            return
            
        result = analyze_property_distributions(
            self.db,
            properties=[arg],
            output_format='report'
        )
        print(result)


def run_interactive_explorer(db_path: str = 'materials.db'):
    """Run the interactive database explorer."""
    explorer = DatabaseExplorer(db_path)
    try:
        explorer.cmdloop()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Use 'exit' to quit properly.")
        explorer.cmdloop()