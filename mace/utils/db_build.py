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
  failed, and a calculation record has it (listed, and extracted as before);
* ``skipped``: not extracted by a build: with ``--register``, a failed or
  unfinished run that no calculation record has (registering records none, so
  its rows would sit under a material with no materials row), or MACE's ``X_temp.out`` copy of the run whose ``X.out``
  is beside it.

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
CLASSES = ("new", "changed", "unchanged", "not tracked (always re-extracted)", "unreadable", "incomplete",
           "skipped")
NOT_TRACKED = "not tracked (always re-extracted)"
SKIPPED = "skipped"
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


def same_output(a: Path, b: Path) -> bool:
    """Two outputs with the same size and the same first and last 64 KiB
    (the change detection's fingerprint): copies of one run."""
    try:
        fa, fb = fingerprint(Path(a)), fingerprint(Path(b))
    except OSError:
        return False
    return fa[0] == fb[0] and fa[2] == fb[2]


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
    """The id an untracked output's rows are filed under: the one its
    registration would give (materials.create_material_id_from_file, which
    the populate scan and the workflow use). A local one-suffix strip gave
    "1_dia_opt_rev1" for an OPT and "1_dia_opt_rev1_sp_..._optimized" for its
    SP, where a registration files both under "1_dia"."""
    from mace.database.materials import create_material_id_from_file
    return create_material_id_from_file(Path(output_file).name)


def temp_copy_of(path: Path) -> Optional[Path]:
    """``X.out`` when ``path`` is MACE's ``X_temp.out`` and ``X.out`` is
    beside it (the run's output kept under its deck's ``_temp`` name), else
    None. A lone ``X_temp.out`` is the run's only output and counts."""
    p = Path(path)
    if not p.stem.endswith("_temp"):
        return None
    run = p.with_name(p.stem[:-len("_temp")] + p.suffix)
    return run if run.is_file() else None


def file_candidates(name, db_dir: Path, work_dir=None) -> List[Path]:
    """Where a calculation's recorded input or output can be now, in order:
    the recorded path (a relative one under the database's folder, then
    under the recorded work_dir), then - for a project folder moved or copied
    with its database, whose records keep the old absolute paths - the end of
    the recorded path under the database's folder, longest first
    (``/old/proj/OPT/x.out`` -> ``<db dir>/OPT/x.out``), and a bare name
    under the recorded work_dir found that way. The caller takes the first
    that exists (and is where it may look)."""
    if not name:
        return []
    db_dir = Path(db_dir)
    out: List[Path] = []

    def tails(p: Path) -> List[Path]:
        parts = [x for x in p.parts if x not in (p.anchor, "")]
        return [db_dir.joinpath(*parts[k:]) for k in range(len(parts))]

    name_p = Path(str(name))
    if name_p.is_absolute():
        out.append(name_p)
        out += tails(name_p)
    else:
        out.append(db_dir / name_p)
        if work_dir:
            wd = Path(str(work_dir))
            out.append(wd / name_p)
            if wd.is_absolute():
                out += [t / name_p for t in tails(wd)]
    seen, uniq = set(), []
    for c in out:
        if str(c) not in seen:
            seen.add(str(c))
            uniq.append(c)
    return uniq


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
    # X.out of a run an earlier scan recorded by its X_temp.out copy
    temp = output_file.with_name(output_file.stem + "_temp" + output_file.suffix)
    if temp.is_file():
        outs += spellings(temp)
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
    if not named and temp.is_file():
        named = [r[0] for r in rows if r[1] and Path(r[1]).stem == temp.stem]
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
             scan_dir: Optional[Path] = None, registering: Optional[bool] = None) -> List[Dict[str, Any]]:
    """One row per output: {file, class, material_id, calc_type, calc_id,
    size}. With ``register`` the outputs the populate scan would register
    count as tracked (their calculation would be created first).
    ``registering`` (default: ``register``) says the build registers, after
    which a failed or unfinished output with no calculation is skipped."""
    if registering is None:
        registering = register
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
    first_of_job: Dict[str, Path] = {}     # job id -> the output registered with it in this scan
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
            copy_of = temp_copy_of(p)
            if copy_of is not None:
                cls = SKIPPED
                row["reason"] = f"a copy of {copy_of.name}"
            elif cls == "incomplete" and row["calc_id"] is None and registering:
                # registering records no calculation for it: its rows would sit
                # under a material with no materials row. Register off, it is
                # extracted as a plain run extracts it (with a materials row).
                cls = SKIPPED
                row["reason"] = "a failed or unfinished run with no calculation record"
            elif row["calc_id"] is None and complete:
                twin = _recorded_copy(conn, p, first_of_job, cls == "new")
                if twin is not None:
                    cls = SKIPPED
                    row["reason"] = f"the same run as {twin} (same SLURM job and content)"
            row["class"] = cls
            rows.append(row)
    finally:
        if conn is not None:
            conn.close()
    return rows


def _recorded_copy(conn, p: Path, first_of_job: Dict[str, Path], registering: bool) -> Optional[str]:
    """The output a calculation records (or this scan registers first) with
    the same SLURM job id as ``p`` and the same content, in another folder:
    ``p`` is a copy of that run, which the populate scan does not register
    again (populate_completed_jobs._check_job_id). None otherwise."""
    from mace.database.populate_completed_jobs import slurm_job_id_for
    try:
        job = slurm_job_id_for(p)
    except OSError:
        return None
    if not job:
        return None
    other = None
    if conn is not None:
        r = conn.execute("SELECT output_file FROM calculations WHERE slurm_job_id = ?", (job,)).fetchone()
        if r and r[0]:
            other = Path(r[0])
    if other is None:
        other = first_of_job.get(job)
        if other is None:
            if registering:
                first_of_job[job] = p
            return None
    if set(spellings(other)) & set(spellings(p)) or spellings(other.parent) == spellings(p.parent):
        return None
    return str(other) if same_output(other, p) else None


def counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {c: 0 for c in CLASSES}
    for r in rows:
        out[r["class"]] = out.get(r["class"], 0) + 1
    return out


def summarize(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The per-file progress records of a build (``--progress-json`` lines)
    counted for its summary: extracted outputs by class (``new`` counts the
    untracked and incomplete ones too), the outputs nothing was extracted
    from, and the skipped ones. The CLI and the GUI both count with this."""
    by_class: Dict[str, int] = {}
    for x in files:
        if x.get("status") == "extracted":
            k = x.get("class") or "extracted"
            by_class[k] = by_class.get(k, 0) + 1
    return {"new": sum(v for k, v in by_class.items() if k in ("new", NOT_TRACKED, "incomplete", "extracted")),
            "changed": by_class.get("changed", 0),
            # extracted again though unchanged (a build without --skip-unchanged)
            "reextracted": by_class.get("unchanged", 0),
            "failed": [x.get("rel") or x.get("file") for x in files if x.get("status") in ("failed", "no properties")],
            "skipped": sum(1 for x in files if x.get("status") == "skipped" and x.get("class") == SKIPPED)}


def summary_message(done: Dict[str, Any]) -> str:
    """The one-line summary of a build, from its ``done`` record (the CLI
    prints it, the GUI shows it in its notification)."""
    n_failed = done.get("failed_outputs", 0)
    reext = f", {done['reextracted']} re-extracted" if done.get("reextracted") else ""
    msg = (f"Database updated: {done.get('new', 0)} new, {done.get('changed', 0)} changed{reext}, "
           f"{n_failed} failed outputs; {done.get('materials', 0)} materials ({done.get('materials_new', 0)} new)")
    if done.get("skipped_outputs"):
        msg += f"; {done['skipped_outputs']} skipped"
    return msg


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
