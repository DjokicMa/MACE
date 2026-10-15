"""The property values of a materials export, shared by `mace database
--action export` and the GUI's Database view export (which never opens the
database read-write), so both write the same cells.

Standard library only: the GUI imports it without the mace.database package.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

# Large structure fields an export leaves out (ExportFormatter's list)
STRUCTURE_FIELDS = ('structure_json', 'structure_ase', 'atomic_positions',
                    'initial_atomic_positions', 'final_atomic_positions',
                    # doubled names earlier builds wrote (the same values as the
                    # names above); an existing database can still hold them
                    'initial_initial_atomic_positions', 'final_final_atomic_positions',
                    'initial_initial_atoms_count', 'final_final_atoms_count',
                    'final_final_atomic_position_axes')

_CHUNK = 500


def property_values(conn, material_ids: Iterable[str],
                    names: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, object]]:
    """{material_id: {property_name: value}} for ``material_ids`` from any
    sqlite3 connection. The value is property_value, or property_value_text
    when the number is NULL (a text property such as
    electronic_classification). Where several rows give one name (one per
    calculation), the last in (category, name, row id) order is the value,
    the same in every export. ``names`` limits the properties (None: all)."""
    ids = list(material_ids)
    out: Dict[str, Dict[str, object]] = {}
    wanted = set(names) if names is not None else None
    for k in range(0, len(ids), _CHUNK):
        part = ids[k:k + _CHUNK]
        rows = conn.execute(
            "SELECT material_id, property_name, property_value, property_value_text FROM properties "
            f"WHERE material_id IN ({','.join('?' * len(part))}) "
            "ORDER BY material_id, property_category, property_name, property_id", part)
        for mid, name, value, text in rows:
            if wanted is not None and name not in wanted:
                continue
            out.setdefault(mid, {})[name] = text if value is None and text else value
    return out
