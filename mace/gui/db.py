"""Read-only view of a MACE materials.db, paged on the server (spec 8.4.7, plan B6).

``MaterialDatabase`` opens the file read-write (WAL mode, CREATE TABLE IF NOT
EXISTS, migrations, a structures.db beside it), so the browser uses sqlite3
directly with a read-only URI against the same schema instead.

A page is at most 200 materials, found by keyset on (sort column,
material_id) with LIMIT, never by reading the whole table: 100k materials
show their first page in milliseconds and stay under 30 KB. Counts, the KPI
strip and the property names are cached per (file mtime, size, filters).
Filters run on indexed columns only (formula, status, calc type and status,
property name), and the text search is a LIKE with its wildcards escaped.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from mace.utils import property_keys

PAGE_MAX = 200
# sort keys: the column, its NULL stand-in (keyset needs no NULLs), and whether it is computed
SORTS = {
    "material_id": ("m.material_id", None, False), "formula": ("m.formula", None, False),
    "space_group": ("m.space_group", "-1", False), "dimensionality": ("m.dimensionality", "''", False),
    "status": ("m.status", "''", False), "source_type": ("m.source_type", "''", False),
    "updated_at": ("m.updated_at", "''", False), "created_at": ("m.created_at", "''", False),
    "n_calcs": ("n_calcs", "0", True), "n_completed": ("n_completed", "0", True),
}
DIMS = ("CRYSTAL", "SLAB", "POLYMER", "MOLECULE")
_CACHE: "OrderedDict[Tuple, Any]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 256
QUERY_LOG: List[str] = []           # test hooks: every SQL statement run (the "always paged" check)


IMMUTABLE_OPENS: List[str] = []    # test hooks: every immutable=1 open (never while a writer may be active)
OPEN_BUSY_S = 5.0                   # how long a busy database is retried at open before DbBusy

# SQLite extended result codes (sqlite3.Error.sqlite_errorcode, Python >= 3.11)
SQLITE_CANTOPEN = 14
SQLITE_READONLY_RECOVERY = 264
SQLITE_READONLY_ROLLBACK = 776
SQLITE_READONLY_CANTINIT = 1288
SQLITE_READONLY_DIRECTORY = 1544
_WAL_INDEX_CODES = {SQLITE_CANTOPEN, SQLITE_READONLY_RECOVERY, SQLITE_READONLY_CANTINIT, SQLITE_READONLY_DIRECTORY}
INTERRUPTED_TEXT = ("A job was interrupted while writing the database; the next MACE command that writes it "
                    "repairs it.")
BUSY_TEXT = "Database busy (a job is recording its result) - retrying"


class DbBusy(Exception):
    """The database is being written (``reason`` "busy") or a writer was
    killed mid-transaction and left a hot journal ("interrupted"). The routes
    answer 503 {"busy": true}; the page keeps what it shows and retries."""

    def __init__(self, reason: str = "busy", detail: str = ""):
        self.reason = reason
        super().__init__(INTERRUPTED_TEXT if reason == "interrupted" else (detail or BUSY_TEXT))


class _Conn(sqlite3.Connection):
    immutable = False


def _code(exc) -> Optional[int]:
    return getattr(exc, "sqlite_errorcode", None)


def _is_busy(exc) -> bool:
    c = _code(exc)
    return c is not None and (c & 0xFF) in (5, 6)


def _journal(path: Path) -> Path:
    return Path(str(path) + "-journal")


def _open(uri: str) -> "_Conn":
    conn = sqlite3.connect(uri, uri=True, timeout=0.1, check_same_thread=False, factory=_Conn)
    conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
    return conn


def connect(path: Path) -> sqlite3.Connection:
    """A read-only connection. The fallback is chosen from SQLite's error
    code, never its message: busy -> retried for OPEN_BUSY_S, then DbBusy; a
    hot journal left by a killed writer (SQLITE_READONLY_ROLLBACK) ->
    DbBusy("interrupted"); the codes that only mean "cannot use the WAL
    index" (a folder we cannot write) -> ``immutable=1``, and only while no
    ``-journal`` exists. Anything else is raised as before."""
    path = Path(path)
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    t0 = time.monotonic()
    while True:
        try:
            conn = _open(uri)
            break
        except sqlite3.OperationalError as exc:
            code = _code(exc)
            if _is_busy(exc):
                if time.monotonic() - t0 < OPEN_BUSY_S:
                    time.sleep(0.1)
                    continue
                raise DbBusy("busy") from exc
            if code == SQLITE_READONLY_ROLLBACK:
                raise DbBusy("interrupted") from exc
            wal_index = code in _WAL_INDEX_CODES and (code != SQLITE_CANTOPEN or os.access(path, os.R_OK))
            if code is None:          # Python < 3.11: no codes; the old fallback, but never over a hot journal
                wal_index = True
            if not wal_index:
                raise
            if _journal(path).exists():
                raise DbBusy("interrupted") from exc
            # a WAL database in a directory we cannot write: read the file as is
            conn = sqlite3.connect(uri + "&immutable=1", uri=True, timeout=5, check_same_thread=False, factory=_Conn)
            conn.immutable = True
            if os.environ.get("MACE_GUI_TEST_HOOKS") == "1":
                IMMUTABLE_OPENS.append(str(path))
            break
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    if os.environ.get("MACE_GUI_TEST_HOOKS") == "1":
        conn.set_trace_callback(lambda s: (QUERY_LOG.append(s), QUERY_LOG.__delitem__(slice(0, -2000))))
    return conn


def _checked(conn, path: Path) -> bool:
    """Whether a read must be checked with stamps before and after it: an
    immutable read (SQLite ignores locks and the WAL), or a database on a
    mount without cluster-wide locks (dbsafety's degraded mode)."""
    if getattr(conn, "immutable", False):
        return True
    try:
        from .dbsafety_loader import dbsafety
        return dbsafety().journal_policy(path)[1]
    except Exception:
        return False


def read(path: Path, fn):
    """Run ``fn(conn)`` in one read transaction (a page and its count from one
    snapshot). Busy errors become DbBusy; a checked read whose file changed
    while it ran is retried once, then reported as DbBusy, never as a 500."""
    path = Path(path)
    for attempt in range(2):
        conn = connect(path)
        try:
            checked = _checked(conn, path)
            before = _stamp(path) if checked else None
            try:
                conn.execute("BEGIN")
                out = fn(conn)
                conn.execute("COMMIT")
            except sqlite3.OperationalError as exc:
                if _is_busy(exc):
                    raise DbBusy("busy") from exc
                if _code(exc) == SQLITE_READONLY_ROLLBACK:
                    raise DbBusy("interrupted") from exc
                raise
        finally:
            conn.close()
        if not checked or _stamp(path) == before:
            return out
    raise DbBusy("busy")


def db_stamp(path: Path) -> Dict[str, Any]:
    """Stat only (the database, -wal and -journal): what the live refresh polls."""
    st = _stamp(Path(path))
    return {"stamp": f"{st[1]}:{st[2]}:" + ",".join(f"{a}:{b}" for a, b in st[3])}


def _tables(conn) -> List[str]:
    return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def is_materials_db(path: Path) -> bool:
    try:
        conn = connect(path)
    except DbBusy:
        return True            # being written: shown as a database (it is opened when it is free)
    except sqlite3.Error:
        return False
    try:
        return {"materials", "calculations"} <= set(_tables(conn))
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _add_reduced(row: Dict[str, Any]) -> None:
    """The reduced formula and Z beside the stored cell formula, for display."""
    from .structures import reduce_formula_text
    reduced = reduce_formula_text(row.get("formula") or "")
    if reduced:
        row["reduced_formula"], row["z"] = reduced


def _stamp(path: Path) -> Tuple:
    st = os.stat(path)
    extra = []
    for suffix in ("-wal", "-journal"):
        w = Path(str(path) + suffix)
        try:
            ws = w.stat()
            extra.append((ws.st_mtime_ns, ws.st_size))
        except FileNotFoundError:
            extra.append((0, 0))
    return (str(Path(path).resolve()), st.st_mtime_ns, st.st_size, tuple(extra))


def _cached(key: Tuple, fn):
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    val = fn()
    with _CACHE_LOCK:
        _CACHE[key] = val
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return val


# ---------------------------------------------------------------- filters
class Filters:
    """What the filter bar sends: text, dimensionalities, a calc type and status present,
    a space-group range, a property present. Every value goes in as a parameter."""

    def __init__(self, q: str = "", dims=(), calc_type: str = "", calc_status: str = "", sg_min=None, sg_max=None,
                 prop: str = ""):
        self.q = (q or "").strip()[:100]
        self.dims = tuple(d for d in (dims or ()) if d in DIMS)
        self.calc_type = (calc_type or "")[:40]
        self.calc_status = (calc_status or "")[:40]
        self.sg_min = _int_or_none(sg_min)
        self.sg_max = _int_or_none(sg_max)
        self.prop = (prop or "")[:120]

    @classmethod
    def from_query(cls, get) -> "Filters":
        dims = [d for d in (get("dim") or "").split(",") if d]
        return cls(get("q") or "", dims, get("calc_type") or "", get("calc_status") or "", get("sg_min"), get("sg_max"),
                   get("prop") or "")

    def key(self) -> Tuple:
        return (self.q, self.dims, self.calc_type, self.calc_status, self.sg_min, self.sg_max, self.prop)

    def describe(self) -> List[str]:
        """The filters in words, for an export's metadata (empty: none)."""
        out = []
        if self.q:
            out.append(f"search: {self.q}")
        if self.dims:
            out.append("dimensionality: " + ", ".join(self.dims))
        if self.calc_type:
            out.append(f"calc_type: {self.calc_type}")
        if self.calc_status:
            out.append(f"calc_status: {self.calc_status}")
        if self.sg_min is not None:
            out.append(f"space_group >= {self.sg_min}")
        if self.sg_max is not None:
            out.append(f"space_group <= {self.sg_max}")
        if self.prop:
            out.append(f"property: {self.prop}")
        return out

    def where(self) -> Tuple[str, List[Any]]:
        parts, args = [], []
        if self.q:
            like = "%" + self.q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            parts.append("(m.material_id LIKE ? ESCAPE '\\' OR m.formula LIKE ? ESCAPE '\\')")
            args += [like, like]
        if self.dims:
            parts.append(f"m.dimensionality IN ({','.join('?' * len(self.dims))})")
            args += list(self.dims)
        if self.calc_type:
            parts.append("EXISTS (SELECT 1 FROM calculations c WHERE c.material_id = m.material_id AND c.calc_type = ?)")
            args.append(self.calc_type)
        if self.calc_status:
            parts.append("EXISTS (SELECT 1 FROM calculations c WHERE c.material_id = m.material_id AND c.status = ?)")
            args.append(self.calc_status)
        if self.sg_min is not None:
            parts.append("m.space_group >= ?")
            args.append(self.sg_min)
        if self.sg_max is not None:
            parts.append("m.space_group <= ?")
            args.append(self.sg_max)
        if self.prop:
            # a name renamed in 1.1.6 finds the rows of its old names too
            names = property_keys.query_names(self.prop, warn=False)
            parts.append("EXISTS (SELECT 1 FROM properties p WHERE p.material_id = m.material_id "
                         f"AND p.property_name IN ({','.join('?' * len(names))}))")
            args += names
        return (" AND ".join(parts) if parts else "1"), args


def _int_or_none(v):
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _require(conn) -> None:
    if "materials" not in _tables(conn):
        raise ValueError("not a MACE materials.db (no materials table)")


# ---------------------------------------------------------------- pages
def page(path: Path, sort: str = "material_id", direction: str = "asc", after: Optional[str] = None,
         limit: int = PAGE_MAX, filters: Optional[Filters] = None, with_total: bool = False) -> Dict[str, Any]:
    """One page of materials: keyset on (sort, material_id) for the table's columns
    (``after`` is the last row's [value, material_id]); LIMIT/OFFSET for the computed
    calculation counts (``after`` is the offset)."""
    if sort not in SORTS:
        raise ValueError(f"sort by one of {', '.join(SORTS)}")
    desc = str(direction).lower() == "desc"
    limit = max(1, min(int(limit), PAGE_MAX))
    filters = filters or Filters()
    col, null, computed = SORTS[sort]
    where, args = filters.where()

    def run(conn):
        _require(conn)
        expr = col if null is None else f"ifnull({col}, {null})"
        op, order = ("<", "DESC") if desc else (">", "ASC")
        offset = 0
        if computed:
            try:
                offset = max(0, int(after or 0))
            except ValueError:
                raise ValueError("after is the offset for a computed column")
            sql = (f"SELECT m.material_id, m.formula, m.space_group, m.dimensionality, m.source_type, m.status, m.updated_at, "
                   f"ifnull(c.n, 0) AS n_calcs, ifnull(c.done, 0) AS n_completed FROM materials m "
                   f"LEFT JOIN (SELECT material_id, COUNT(*) AS n, SUM(status = 'completed') AS done FROM calculations "
                   f"GROUP BY material_id) c ON c.material_id = m.material_id WHERE {where} "
                   f"ORDER BY {expr} {order}, m.material_id {order} LIMIT ? OFFSET ?")
            rows = [dict(r) for r in conn.execute(sql, args + [limit + 1, offset])]
        else:
            kw, kargs = where, list(args)
            if after:
                try:
                    v, mid = json.loads(after)
                except (ValueError, TypeError):
                    raise ValueError("after is the last row's [value, material_id]")
                kw = f"({where}) AND ({expr}, m.material_id) {op} (?, ?)"
                kargs = args + [v, mid]
            sql = (f"SELECT m.material_id, m.formula, m.space_group, m.dimensionality, m.source_type, m.status, m.updated_at "
                   f"FROM materials m WHERE {kw} ORDER BY {expr} {order}, m.material_id {order} LIMIT ?")
            rows = [dict(r) for r in conn.execute(sql, kargs + [limit + 1])]
            # the page's calculation counts: one indexed query for its ids
            ids = [r["material_id"] for r in rows[:limit]]
            counts = {}
            if ids:
                counts = {r[0]: (r[1], r[2] or 0) for r in conn.execute(
                    f"SELECT material_id, COUNT(*), SUM(status = 'completed') FROM calculations "
                    f"WHERE material_id IN ({','.join('?' * len(ids))}) GROUP BY material_id", ids)}
            for r in rows:
                r["n_calcs"], r["n_completed"] = counts.get(r["material_id"], (0, 0))
        more = len(rows) > limit
        rows = rows[:limit]
        for r in rows:
            _add_reduced(r)
        nxt = None
        if more and rows:
            last = rows[-1]
            if computed:
                nxt = str(offset + limit)
            else:
                v = last.get(sort)
                if v is None:
                    v = int(null) if null and null.lstrip("-").isdigit() else ""
                nxt = json.dumps([v, last["material_id"]])
        out = {"rows": rows, "next": nxt, "sort": sort, "dir": "desc" if desc else "asc", "limit": limit,
               "offset": offset if computed else None, "readonly_snapshot": bool(getattr(conn, "immutable", False))}
        if with_total:          # the count from the same snapshot as the page
            out["total"] = conn.execute(f"SELECT COUNT(*) FROM materials m WHERE {where}", args).fetchone()[0]
        return out
    return read(path, run)


def count(path: Path, filters: Optional[Filters] = None) -> int:
    filters = filters or Filters()

    def run():
        def q(conn):
            _require(conn)
            where, args = filters.where()
            return conn.execute(f"SELECT COUNT(*) FROM materials m WHERE {where}", args).fetchone()[0]
        return read(path, q)
    return _cached(("count", _stamp(path), filters.key()), run)


def offset_of(path: Path, sort: str, direction: str, index: int, filters: Optional[Filters] = None) -> Optional[str]:
    """The ``after`` cursor that starts a page at row ``index`` (a jump to the end, Home/End, a scrollbar drag)."""
    if sort not in SORTS:
        raise ValueError("unknown sort")
    col, null, computed = SORTS[sort]
    if computed or index <= 0:
        return str(max(0, index)) if computed else None
    filters = filters or Filters()
    where, args = filters.where()
    desc = str(direction).lower() == "desc"
    order = "DESC" if desc else "ASC"

    def run(conn):
        r = conn.execute(f"SELECT {col if null is None else f'ifnull({col}, {null})'} AS v, m.material_id FROM materials m WHERE {where} "
                         f"ORDER BY v {order}, m.material_id {order} LIMIT 1 OFFSET ?", args + [index - 1]).fetchone()
        return json.dumps([r[0], r[1]]) if r else None
    return read(path, run)


def stats(path: Path) -> Dict[str, Any]:
    def q(conn):
        _require(conn)
        tables = _tables(conn)
        status = dict(conn.execute("SELECT status, COUNT(*) FROM calculations GROUP BY status").fetchall())
        types = dict(conn.execute("SELECT calc_type, COUNT(*) FROM calculations GROUP BY calc_type").fetchall())
        n_props = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0] if "properties" in tables else 0
        n_mat = conn.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
        dims = dict(conn.execute("SELECT ifnull(dimensionality, ''), COUNT(*) FROM materials GROUP BY 1").fetchall())
        last = conn.execute("SELECT max(ifnull(completed_at, created_at)) FROM calculations").fetchone()[0]
        return {"n_materials": n_mat, "calc_status": status, "calc_types": types, "n_properties": n_props,
                "n_calculations": sum(status.values()), "last_calc_at": last,
                "dimensionality": dims, "readonly_snapshot": bool(getattr(conn, "immutable", False))}
    return _cached(("stats", _stamp(path)), lambda: read(path, q))


def property_names(path: Path) -> List[str]:
    def q(conn):
        if "properties" not in _tables(conn):
            return []
        names = {r[0] for r in conn.execute("SELECT DISTINCT property_name FROM properties")}
        if "property_blobs" in _tables(conn):
            names |= {r[0] for r in conn.execute("SELECT DISTINCT property_name FROM property_blobs")}
        # an old database's renamed names under their current ones
        return sorted({property_keys.ALIASES.get(n, n) for n in names})
    return _cached(("props", _stamp(path)), lambda: read(path, q))


def summary(path: Path) -> Dict[str, Any]:
    """Every material in one answer (the Phase 2 route, for scripts; the page uses page())."""
    return read(path, _summary)


def _summary(conn) -> Dict[str, Any]:
    tables = _tables(conn)
    if "materials" not in tables:
        raise ValueError("not a MACE materials.db (no materials table)")
    has_props = "properties" in tables
    rows = conn.execute("""
        SELECT m.material_id, m.formula, m.space_group, m.dimensionality,
               m.source_type, m.status, m.updated_at,
               ifnull(c.n, 0) AS n_calcs, ifnull(c.done, 0) AS n_completed
        FROM materials m LEFT JOIN (SELECT material_id, COUNT(*) AS n, SUM(status = 'completed') AS done
                                    FROM calculations GROUP BY material_id) c ON c.material_id = m.material_id
        ORDER BY m.material_id""").fetchall()
    status = dict(conn.execute("SELECT status, COUNT(*) FROM calculations GROUP BY status").fetchall())
    types = dict(conn.execute("SELECT calc_type, COUNT(*) FROM calculations GROUP BY calc_type").fetchall())
    n_props = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0] if has_props else 0
    materials = [dict(r) for r in rows]
    for m in materials:
        _add_reduced(m)
    return {"materials": materials, "calc_status": status, "calc_types": types, "n_properties": n_props}


def material(path: Path, material_id: str) -> Dict[str, Any]:
    return read(path, lambda conn: _material(conn, material_id))


def _material(conn, material_id: str) -> Dict[str, Any]:
    m = conn.execute("SELECT * FROM materials WHERE material_id = ?", (material_id,)).fetchone()
    if m is None:
        raise KeyError(material_id)
    calcs = [dict(r) for r in conn.execute(
        """SELECT calc_id, calc_type, calc_subtype, status, slurm_job_id,
                  created_at, completed_at, input_file, output_file, work_dir, settings_json,
                  error_type
           FROM calculations WHERE material_id = ? ORDER BY created_at""",
        (material_id,))]
    for c in calcs:
        try:
            c["settings"] = json.loads(c.pop("settings_json") or "null")
        except json.JSONDecodeError:
            c["settings"] = None
    props = []
    tables = _tables(conn)
    if "properties" in tables:
        # the large JSON values (1.1.6: the property_blobs table) with the
        # rest, as one list; the page shows them cut, with Copy JSON
        blobs = ("""
               UNION ALL
               SELECT property_category, property_name, NULL, property_value_text, property_unit, calc_id
               FROM property_blobs WHERE material_id = ?""" if "property_blobs" in tables else "")
        props = [dict(r) for r in conn.execute(
            f"""SELECT property_category, property_name, property_value,
                      property_value_text, property_unit, calc_id
               FROM properties WHERE material_id = ?{blobs}
               ORDER BY property_category, property_name""",
            (material_id, material_id) if blobs else (material_id,))]
        # names renamed in 1.1.6 shown under their current names
        props = property_keys.normalize_rows(props, scope=('calc_id',))
    out = dict(m)
    _add_reduced(out)
    try:
        out["metadata"] = json.loads(out.pop("metadata_json") or "null")
    except json.JSONDecodeError:
        out["metadata"] = None
    return {"material": out, "calculations": calcs, "properties": props}


# ---------------------------------------------------------------- export (plan B6; CLI/GUI parity 1.1.6)
EXPORT_TYPES = {"csv": "text/csv; charset=utf-8", "json": "application/json",
                "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "latex": "application/x-latex; charset=utf-8", "html": "text/html; charset=utf-8"}


def export(path: Path, fmt: str, sort: str = "material_id", direction: str = "asc",
           filters: Optional[Filters] = None) -> bytes:
    """The filtered, sorted materials as ``mace database --action export``
    writes them, in any of its formats (csv, json, excel, latex, html): the
    rows are read here (read-only: export_materials() needs a MaterialDatabase,
    which opens the file read-write), each the materials row with
    "metadata" read from metadata_json and the property values
    (db_export_values.add_property_values, as the CLI), and the CLI's own
    ExportFormatter writes them. Nothing is written on the server."""
    from mace.utils.db_export_formats import ExportFormatter
    from mace.utils.db_export_values import add_property_values
    if fmt not in EXPORT_TYPES:
        raise ValueError("format is " + ", ".join(EXPORT_TYPES))
    if sort not in SORTS:
        raise ValueError("unknown sort")
    filters = filters or Filters()
    where, args = filters.where()
    col, null, computed = SORTS[sort]
    order = "DESC" if str(direction).lower() == "desc" else "ASC"
    conn = connect(path)
    try:
        _require(conn)
        if computed:
            sql = (f"SELECT m.* FROM materials m LEFT JOIN (SELECT material_id, COUNT(*) AS n_calcs, "
                   f"SUM(status = 'completed') AS n_completed FROM calculations GROUP BY material_id) c "
                   f"ON c.material_id = m.material_id WHERE {where} ORDER BY ifnull({col}, {null}) {order}, m.material_id {order}")
        else:
            expr = col if null is None else f"ifnull({col}, {null})"
            sql = f"SELECT m.* FROM materials m WHERE {where} ORDER BY {expr} {order}, m.material_id {order}"
        data = [dict(r) for r in conn.execute(sql, args)]
        for r in data:
            if r.get("metadata_json"):
                try:
                    r["metadata"] = json.loads(r["metadata_json"])
                except json.JSONDecodeError:
                    pass
        if data and "properties" in _tables(conn):
            add_property_values(conn, data)
    finally:
        conn.close()
    described = filters.describe()
    meta = ({"export_type": "filtered_materials", "filters": described} if described
            else {"export_type": "all_materials"})
    out = ExportFormatter().render(data, fmt, metadata=meta)
    if out is None:
        raise ValueError("no materials to export" + (" (the filters match none)" if described else ""))
    return out
