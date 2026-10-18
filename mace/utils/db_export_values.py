"""The property values of a materials export, shared by `mace database
--action export` and the GUI's Database view export (which never opens the
database read-write), so both write the same cells.

Standard library only: the GUI imports it without the mace.database package.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

from mace.utils.property_keys import BLOB_KEYS, canonical, normalize_rows

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
    # a name renamed in 1.1.6 is asked for and written under its current
    # name; an old database's rows are read under it too (property_keys)
    wanted = {canonical(n) for n in names} if names is not None else None
    for k in range(0, len(ids), _CHUNK):
        part = ids[k:k + _CHUNK]
        cur = conn.execute(
            "SELECT material_id, calc_id, property_name, property_value, property_value_text FROM properties "
            f"WHERE material_id IN ({','.join('?' * len(part))}) "
            "ORDER BY material_id, property_category, property_name, property_id", part)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur]
        # the large JSON values (property_blobs, 1.1.6) only when asked for
        # by name: a neighbor table is no spreadsheet cell
        blob_names = sorted(wanted & BLOB_KEYS) if wanted else []
        if blob_names and conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'property_blobs'").fetchone():
            cur = conn.execute(
                "SELECT material_id, calc_id, property_name, NULL AS property_value, property_value_text "
                f"FROM property_blobs WHERE material_id IN ({','.join('?' * len(part))}) "
                f"AND property_name IN ({','.join('?' * len(blob_names))}) "
                "ORDER BY material_id, property_category, property_name, blob_id", [*part, *blob_names])
            rows += [dict(zip(cols, row)) for row in cur]
        for r in normalize_rows(rows):
            name, value, text = r['property_name'], r['property_value'], r['property_value_text']
            if wanted is not None and name not in wanted:
                continue
            out.setdefault(r['material_id'], {})[name] = text if value is None and text else value
    return out


def add_property_values(conn, records, names: Optional[Sequence[str]] = None) -> None:
    """Add each record's property values (property_values) to the materials
    rows ``records`` in place: what both exports do before writing."""
    values = property_values(conn, [r['material_id'] for r in records], names)
    for r in records:
        r.update(values.get(r['material_id'], {}))
