"""
Multi-format Export for MACE Database
=====================================
Supports exporting materials data to various formats:
- CSV (default)
- JSON
- Excel (.xlsx)
- LaTeX tables
- HTML tables

The writers live in mace.utils.db_export_formats (ExportFormatter), which the
GUI's Database view export calls too.
"""

from datetime import datetime
from typing import List

from mace.utils.db_export_formats import ExportFormatter, MACE_VERSION, EXTENSIONS  # noqa: F401


def export_materials(db, format: str = 'csv', output_file: str = None,
                    filters: List[str] = None, logic: str = 'AND',
                    properties_only: bool = False,
                    include_properties: List[str] = None,
                    include_structures: bool = False,
                    material_ids: List[str] = None) -> str:
    """
    Convenience function to export materials from database.
    
    Args:
        db: MaterialDatabase instance
        format: Export format
        output_file: Output file path (auto-generated if None)
        filters: Property filters to apply
        logic: Filter logic (AND/OR)
        properties_only: Export properties instead of materials
        include_properties: Property names to include (None = all, [] = none)
        include_structures: Whether to include structure data
        material_ids: Export only these materials (None = all); what the
            interactive explorer's `export` passes for its last results
        
    Returns:
        Path to exported file
    """
    formatter = ExportFormatter()
    
    # Generate output filename if not provided
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = EXTENSIONS.get(format, format)
        output_file = f"mace_export_{timestamp}.{extension}"
        
    # Get data based on options
    if properties_only:
        # Export properties
        data = db.get_all_properties()
        metadata = {'export_type': 'properties'}
    else:
        # Export materials
        if filters:
            # Check if it's a single advanced filter expression
            if len(filters) == 1 and any(op in filters[0] for op in ['(', ')', ' AND ', ' OR ', ' LIKE ', ' IN ', ' IS ']):
                # Use advanced filtering
                data = db.filter_materials_advanced(filters[0])
                metadata = {
                    'export_type': 'filtered_materials',
                    'filters': filters,
                    'filter_type': 'advanced'
                }
            else:
                # Use regular filtering
                data = db.filter_materials_by_properties(filters, logic)
                metadata = {
                    'export_type': 'filtered_materials',
                    'filters': filters,
                    'filter_logic': logic
                }
        else:
            data = db.get_all_materials()
            metadata = {'export_type': 'all_materials'}
        if material_ids is not None:
            wanted = set(material_ids)
            data = [m for m in data if m['material_id'] in wanted]
            metadata['material_ids'] = list(material_ids)
            
        # Add properties to the materials: all of them for None (the
        # default, and what `mace database --action export` passes), the
        # named ones for a list, none for an empty list. This read None as
        # 'none', so a plain export held the materials table only.
        # The values come from db_export_values, which the GUI's export
        # uses too (a text value, e.g. electronic_classification, included).
        if data and include_properties != []:
            from mace.utils.db_export_values import add_property_values
            with db._get_connection() as conn:
                add_property_values(conn, data, include_properties)
                
    # Export
    success = formatter.export(
        data,
        output_file,
        format=format,
        properties_filter=include_properties,
        include_structures=include_structures,
        metadata=metadata
    )
    
    return output_file if success else None

def empty_export_reason(db, properties_only: bool = False, filters: List[str] = None) -> str:
    """Why export_materials() wrote nothing (it returns None when there are
    no rows to write), for the message: an empty database, properties stored
    without materials (an extraction with --no-register), or filters that
    match nothing."""
    try:
        with db._get_connection() as conn:
            def count(table):
                try:
                    return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                except Exception:
                    return 0
            n_mat, n_prop = count("materials"), count("properties")
    except Exception:
        return "there were no rows to write"
    if properties_only:
        return "the database has no properties" if not n_prop else "there were no rows to write"
    if not n_mat and n_prop:
        return (f"the database has no materials, only {n_prop} properties of outputs without a "
                "calculation record; run `mace analyze --extract-properties DIR` again with "
                "--register (the default for a folder) to record them")
    if not n_mat:
        return ("the database has no materials yet; `mace analyze --extract-properties DIR` "
                "records a folder of outputs")
    if filters:
        return f"no material matches the filters ({n_mat} in the database)"
    return "there were no rows to write"
