"""Building or updating a materials database from a folder of outputs
(`mace analyze --extract-properties DIR --register --dry-run ...`, and the
GUI's "Create or Update Database from a Folder", which runs that command).
PLAN_analysis C9.

Change detection works only through tracked calculations
(``calculations.output_file``): each successful extraction of a tracked
calculation writes one row of the existing ``files`` table with
``file_type = 'output_extracted'``, the output's size and
``checksum = "mtime_ns=<n>;head_tail_sha1=<sha1 of the first and last
64 KiB>"``. A later run classifies each output as

* ``new``: no calculation, or a calculation with no ``output_extracted`` row;
* ``changed``: the size or the head/tail hash differs (an mtime alone never
  decides: ``cp -p``/``rsync -t`` copies keep it, and a ``touch`` changes
  nothing that matters);
* ``unchanged``;
* ``not tracked (always re-extracted)``: no calculation row and no
  ``--register``;
* ``unreadable``: an OS error, or not a CRYSTAL output;
* ``incomplete``: ``categorize_output_file`` says it is still running or
  failed (listed, and extracted as before).

Nothing here writes; the read-only database connection never creates a file.
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import quote
from typing import Any, Dict, Iterable, List, Optional, Tuple

EXTRACTED = "output_extracted"
EDGE = 64 * 1024
CLASSES = ("new", "changed", "unchanged", "not tracked (always re-extracted)", "unreadable", "incomplete")
NOT_TRACKED = "not tracked (always re-extracted)"
_BANNER = re.compile(rb"\*\s+CRYSTAL\d{2}\b")


def is_crystal_output(path: Path) -> bool:
    """CRYSTAL's banner (``*  CRYSTAL23  *``, also 14/17) in the first 64 KB;
    a job log or any other .out has none."""
    with open(path, "rb") as fh:
        head = fh.read(EDGE)
    return bool(_BANNER.search(head))


def fingerprint(path: Path) -> Tuple[int, int, str]:
    """(size, mtime_ns, sha1 of the first and last 64 KiB)."""
    st = os.stat(path)
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(EDGE))
        if st.st_size > EDGE:
            fh.seek(max(EDGE, st.st_size - EDGE))
            h.update(fh.read(EDGE))
    return st.st_size, st.st_mtime_ns, h.hexdigest()


def checksum_text(fp: Tuple[int, int, str]) -> str:
    return f"mtime_ns={fp[1]};head_tail_sha1={fp[2]}"


def parse_checksum(text: Optional[str]) -> Dict[str, str]:
    out = {}
    for part in (text or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def material_id_from_filename(output_file: Path) -> str:
    """The id an untracked output's rows are filed under (the extractor's
    validated rule)."""
    filename = Path(output_file).stem
    for suffix in ['_opt', '_sp', '_band', '_doss', '_freq']:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            break
    return filename


def spellings(path: Path) -> List[str]:
    forms = {str(path), os.path.abspath(str(path))}
    try:
        forms.add(str(Path(path).resolve()))
    except OSError:
        pass
    return sorted(forms)


def calc_for_output(conn, output_file: Path) -> Optional[str]:
    """The calc_id recorded for this output: by its path (every spelling),
    else in its directory by the input deck's name, else the directory's only
    calculation. ``conn`` is any sqlite3 connection to the database."""
    output_file = Path(output_file)
    outs, dirs = spellings(output_file), spellings(output_file.parent)
    marks = ",".join("?" * len(outs))
    row = conn.execute(
        f"SELECT calc_id FROM calculations WHERE output_file IN ({marks}) "
        "ORDER BY created_at DESC LIMIT 1", outs).fetchone()
    if row:
        return row[0]
    marks = ",".join("?" * len(dirs))
    rows = conn.execute(
        f"SELECT calc_id, input_file FROM calculations WHERE work_dir IN ({marks}) "
        "ORDER BY created_at DESC", dirs).fetchall()
    named = [r[0] for r in rows if r[1] and Path(r[1]).stem == output_file.stem]
    if named:
        return named[0]
    return rows[0][0] if len(rows) == 1 else None


def open_ro(db_path) -> Optional[sqlite3.Connection]:
    """A read-only connection, or None when the database does not exist."""
    p = Path(db_path)
    if not p.is_file():
        return None
    # the path is quoted: a "#" or "?" in a folder name would otherwise end the
    # path and open (or create) another file read-write
    conn = sqlite3.connect(f"file:{quote(str(p.resolve()))}?mode=ro", uri=True, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("SELECT 1 FROM calculations LIMIT 1")
    except sqlite3.Error:
        conn.close()
        return None
    return conn


def has_tracked(db_path) -> bool:
    conn = open_ro(db_path)
    if conn is None:
        return False
    try:
        return bool(conn.execute("SELECT 1 FROM calculations LIMIT 1").fetchone())
    finally:
        conn.close()


def extracted_row(conn, calc_id: str) -> Optional[Tuple[Any, ...]]:
    return conn.execute(
        "SELECT file_id, file_size, checksum FROM files WHERE calc_id = ? AND file_type = ? "
        "ORDER BY file_id DESC LIMIT 1", (calc_id, EXTRACTED)).fetchone()


def _calc_type(path: Path) -> Optional[str]:
    from mace.completion_checker import determine_completed_subtype
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    return determine_completed_subtype(Path(path), lines, has_opt_end=any("OPT END" in ln for ln in lines))


def _quiet():
    """The completion checker and the populate scan print as they go: kept
    off stdout (JSON lines go there)."""
    import contextlib
    import io
    return contextlib.redirect_stdout(io.StringIO())


def classify(outputs: Iterable[Path], db_path, register: bool = False,
             scan_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """One row per output: {file, class, material_id, calc_type, calc_id,
    size}. With ``register`` the outputs the populate scan would register
    count as tracked (their calculation would be created first)."""
    from mace.completion_checker import categorize_output_file
    outputs = [Path(p) for p in outputs]
    would = {}
    if register and scan_dir is not None:
        from mace.database.populate_completed_jobs import scan_for_completed_calculations
        with _quiet():
            for info in scan_for_completed_calculations(Path(scan_dir)):
                for sp in spellings(Path(info["output_file"])):
                    would[sp] = info
    conn = open_ro(db_path)
    rows = []
    try:
        for p in outputs:
            row: Dict[str, Any] = {"file": str(p), "class": None, "material_id": None, "calc_type": None,
                                   "calc_id": None, "size": None}
            try:
                if not is_crystal_output(p):
                    row["class"] = "unreadable"
                    row["reason"] = "not a CRYSTAL output"
                    rows.append(row)
                    continue
                fp = fingerprint(p)
                row["size"] = fp[0]
                with _quiet():
                    category, _ = categorize_output_file(p)
                complete = category.startswith("complete")
                try:
                    row["calc_type"] = _calc_type(p) if complete else None
                except OSError:
                    row["calc_type"] = None
            except OSError as e:
                row["class"] = "unreadable"
                row["reason"] = e.strerror or str(e)
                rows.append(row)
                continue
            calc_id = calc_for_output(conn, p) if conn is not None else None
            if calc_id is not None:
                mat = conn.execute("SELECT material_id, calc_type FROM calculations WHERE calc_id = ?",
                                   (calc_id,)).fetchone()
                row["calc_id"] = calc_id
                row["material_id"] = mat[0] if mat else None
                row["calc_type"] = row["calc_type"] or (mat[1] if mat else None)
                ex = extracted_row(conn, calc_id)
                if ex is None:
                    cls = "new"
                else:
                    stored = parse_checksum(ex[2])
                    cls = "changed" if (ex[1] != fp[0] or stored.get("head_tail_sha1") != fp[2]) else "unchanged"
            elif any(sp in would for sp in spellings(p)):
                info = next(would[sp] for sp in spellings(p) if sp in would)
                row["material_id"] = info["material_id"]
                row["calc_type"] = row["calc_type"] or info["calc_type"]
                cls = "new"
            else:
                row["material_id"] = material_id_from_filename(p)
                cls = NOT_TRACKED
            if not complete and cls != "unchanged":
                cls = "incomplete"
            row["class"] = cls
            rows.append(row)
    finally:
        if conn is not None:
            conn.close()
    return rows


def counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {c: 0 for c in CLASSES}
    for r in rows:
        out[r["class"]] = out.get(r["class"], 0) + 1
    return out


def record_extracted(db, calc_id: str, output_file: Path) -> None:
    """Write or update the calculation's ``output_extracted`` files row
    (``db`` is a MaterialDatabase). created_at stays as first written."""
    from datetime import datetime
    p = Path(output_file)
    fp = fingerprint(p)
    path = str(p.resolve())
    with db._get_connection() as conn:
        ex = extracted_row(conn, calc_id)
        if ex is None:
            conn.execute("INSERT INTO files (calc_id, file_type, file_name, file_path, file_size, created_at, checksum) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (calc_id, EXTRACTED, p.name, path, fp[0], datetime.now().isoformat(), checksum_text(fp)))
        else:
            conn.execute("UPDATE files SET file_name = ?, file_path = ?, file_size = ?, checksum = ? WHERE file_id = ?",
                         (p.name, path, fp[0], checksum_text(fp), ex[0]))


def table(rows: List[Dict[str, Any]]) -> str:
    """The human --dry-run listing."""
    out = []
    for c in CLASSES:
        sel = [r for r in rows if r["class"] == c]
        if not sel:
            continue
        out.append(f"{c} ({len(sel)}):")
        for r in sel:
            extra = f"  [{r.get('reason')}]" if r.get("reason") else ""
            out.append(f"  {Path(r['file']).name}  material {r['material_id'] or '-'}  type {r['calc_type'] or '-'}{extra}")
    return "\n".join(out)
