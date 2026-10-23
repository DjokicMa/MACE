"""The bottom panel's Properties tab (1.1.6): what materials.db holds for the
run of the document in focus, the same values computed on the fly when the
run is not in a database, and Extract (that one run into the database).

GET  properties/run?path=REL[&db=REL][&scope=run|material]
        the run the document came from, the database that has it (or would),
        and its stored rows: {state: "db"|"not-in-db"|"no-db"|"no-run", run,
        db, calc, rows, units, derived, missing, extract}
GET  properties/compute?path=REL     -> task (pool job "properties.compute"):
        the run's values from the extractor and the derived-property registry,
        nothing written ({state: "computed", rows, units, derived})

Each row carries its unit facts (mace/properties/units.annotate: dimension,
stored unit, basis, extensive, the run's atoms and formula units; of keys
holding one quantity in several units, ``also`` on the one shown and
``merged_into`` on the others) and ``units`` is the catalog the tab converts
with (display only: the database is never changed).
GET  properties/value?db=REL&calc=ID&name=KEY[&material=ID]
        one stored value in full (a JSON curve, a table)
POST properties/extract {path, db, root_gen} -> task: mace analyze
        --extract-properties --output-file RUN --db-path DB --register
        --register-output-only --backup
GET  properties/references?path=REL[&db=REL]
        the Add reference… picker: the run's reference energies (formation,
        cohesive, surface, adsorption, reaction), each reference's saved and
        used runs, and the database's runs with whether each may be combined
        with the run (check: ok, warn, refuse, and why)
POST properties/references {path, db, property, references: {name: [CALC_ID |
        N*CALC_ID]}, clear} -> task: the Extract command with --reference
        NAME=CALC_ID for each (clear: NAME=auto, the database's choice again);
        the extraction stores the property and saves the choice with the run

Which database: the one the page names (the open Database document) when it
has the run; else the nearest materials.db from the run's folder up to the
served folder; else the workflow database of the run's folder or the served
folder (.mace_context_*). The first that exists is the one Extract writes
when none has the run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import clirun
from . import db as gui_db
from . import dbwriters
from . import tasks as gui_tasks

GET_ROUTES = {"properties/run", "properties/compute", "properties/value", "properties/references"}
POST_ROUTES = {"properties/extract", "properties/references"}
INLINE_JSON = 4096          # a JSON value up to this many characters comes with the rows
OWNER = "Extract to database"


def _q(query, name: str, default: Any = None):
    return (query.get(name) or [default])[0]


# ------------------------------------------------------------------ the run and its database
_STEM_TAILS = (".BAND.DAT", ".DOSS.DAT", "_IRSPEC.DAT", "_RAMSPEC.DAT", ".IRSPEC.DAT", ".RAMSPEC.DAT",
               "_SEEBECK.DAT", "_SIGMA.DAT", "_KAPPA.DAT", "_SIGMAS.DAT", ".SEEBECK.DAT", ".SIGMA.DAT", ".KAPPA.DAT")


def _job_log(path: Path) -> bool:
    """A job log named like an output (slurm-123.out, X-123.out: related._JOBLOG) without a CRYSTAL
    banner: never a run (a MOF-5.out with one stays a run)."""
    from .related import _JOBLOG
    if not _JOBLOG.match(path.name):
        return False
    from . import filekind
    try:
        h, t, _ = filekind.head_tail(path)
    except OSError:
        return False
    return filekind.role_of(path.name, h.decode("utf-8", "replace"),
                            t.decode("utf-8", "replace")) not in ("crystal_out", "properties_out")


def run_of(path: Path, root: Optional[Path] = None) -> Optional[Path]:
    """The CRYSTAL output a document belongs to: an .out itself (not a job
    log); else the .out of the same stem beside the file (a BAND.DAT, a
    spectrum, a cube, a deck, a run's f25 or job log), or one folder up.
    ``root``: only an output inside it (not one folder above the served
    folder, nor a link out of it)."""
    path = Path(path)
    if path.suffix.lower() == ".out":
        if not path.is_file():
            return None
        if not _job_log(path):
            return path
    from .related import stems
    names = list(dict.fromkeys(stems(path)))
    up = path.name.upper()
    for tail in _STEM_TAILS:
        if up.endswith(tail):
            names.insert(0, path.name[: -len(tail)])
    for d in (path.parent, path.parent.parent):
        for s in names:
            p = d / f"{s}.out"
            if p != path and p.is_file() and (root is None or _inside(root, p)) and not _job_log(p):
                return p
    return None


def _inside(root: Path, p: Path) -> bool:
    try:
        r = p.resolve()
    except (OSError, RuntimeError):
        return False
    return r == root or root in r.parents


def candidate_dbs(st, run: Path, hint: Optional[str]) -> List[Path]:
    """The databases to look in, in order (existing materials databases only)."""
    from .dbbuild import context_db
    out: List[Path] = []
    if hint:
        try:
            out.append(st.resolve(hint))
        except Exception:                             # noqa: BLE001 - a stale hint is skipped
            pass
    d = run.parent
    root = st.root
    while True:
        out.append(d / "materials.db")
        if d == root or root not in d.parents:
            break
        d = d.parent
    for folder in (run.parent, root):
        c = context_db(folder)
        if c is not None:
            out.append(c)
    seen, keep = set(), []
    for p in out:
        try:
            r = p.resolve()
        except (OSError, RuntimeError):
            continue
        if r in seen or not r.is_file() or not _inside(root, r) or not gui_db.is_materials_db(r):
            continue
        seen.add(r)
        keep.append(r)
    return keep


def _calc_of(conn, run: Path, db: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    from mace.utils import db_build
    # db: a run recorded by a path relative to the database's folder is found too
    cid = db_build.calc_for_output(conn, run, db.parent if db is not None else None)
    if not cid:
        return None
    row = conn.execute("SELECT calc_id, material_id, calc_type, status, output_file, completed_at "
                       "FROM calculations WHERE calc_id = ?", (cid,)).fetchone()
    if not row:
        return None
    return dict(zip(("calc_id", "material_id", "calc_type", "status", "output_file", "completed_at"), row))


def _untracked_count(conn, run: Path) -> Tuple[str, int]:
    from mace.utils import db_build
    mid = db_build.material_id_from_filename(run)
    n = conn.execute("SELECT count(*) FROM properties WHERE material_id = ? AND calc_id IS NULL", (mid,)).fetchone()[0]
    return mid, int(n)


# ------------------------------------------------------------------ rows
def _row(name: str, value, text, unit, category, script, extracted_at, calc: Dict[str, Any], blob: bool,
         provenance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    from mace.properties import display, registry
    d = display.describe(name)
    out: Dict[str, Any] = {
        "name": name, "label": d["label"], "label_plain": d["label_plain"], "unit": unit or d["unit"],
        "unit_text": display.unit_text(unit or d["unit"]), "group": d["group"], "category": category or d["category"],
        "precision": d["precision"], "description": d["description"], "derived": d["derived"],
        "version": d["version"], "interactive": d["interactive"], "script": script, "extracted_at": extracted_at,
        "calc_id": calc.get("calc_id"), "calc_type": calc.get("calc_type"), "material_id": calc.get("material_id"),
        "file": Path(str(calc.get("output_file") or "")).name or None, "blob": blob, "json": False}
    k_type = d.get("type")
    if k_type == "json" or blob:
        out["json"] = True
        size = len(text or "")
        out["size"] = size
        if size <= INLINE_JSON:
            out["value"] = registry.decode(value, text, name)
            out["display"] = display.format_value(out["value"])
        else:
            out["value"] = None
            out["display"] = f"JSON, {size / 1024:.0f} KB"
    else:
        from mace.properties import units
        v = registry.decode(value, text, name)
        out["value"] = v
        out["display"] = display.format_value(v, d["precision"], units.display_decimals(name, unit or d["unit"]))
    if d["derived"] and provenance:
        entry = (provenance.get("derived") or {}).get(d["derived"]) or {}
        out["inputs"] = entry.get("inputs")
    return out


#: the keys mace/properties/units.counts reads (atoms and formula units of a run)
_BASIS_KEYS = {"atoms_in_unit_cell", "atomic_positions", "initial_atomic_positions", "processed_atomic_charges",
               "formula_units"}


def _rows_of(conn, calc: Dict[str, Any], material_scope: bool = False) -> Tuple[List[Dict[str, Any]], Optional[Dict],
                                                                                 Dict[Any, Dict[str, Any]]]:
    from mace.utils import property_keys
    from mace.properties import registry
    registry.load()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    where, args = ("material_id = ?", (calc["material_id"],)) if material_scope else ("calc_id = ?", (calc["calc_id"],))
    raw = []
    for r in conn.execute(f"SELECT material_id, calc_id, property_name, property_value, property_value_text, property_unit, "
                          f"property_category, extractor_script, extracted_at FROM properties WHERE {where}", args):
        raw.append(dict(zip(("material_id", "calc_id", "property_name", "value", "text", "unit", "category", "script",
                             "extracted_at"), r), blob=False))
    if "property_blobs" in tables:
        for r in conn.execute(f"SELECT material_id, calc_id, property_name, property_value_text, property_unit, "
                              f"property_category, extractor_script, extracted_at FROM property_blobs WHERE {where}", args):
            raw.append(dict(zip(("material_id", "calc_id", "property_name", "text", "unit", "category", "script",
                                 "extracted_at"), r), value=None, blob=True))
    raw = property_keys.normalize_rows(raw)
    calcs = {calc["calc_id"]: calc}
    if material_scope:
        for c in conn.execute("SELECT calc_id, material_id, calc_type, status, output_file, completed_at "
                              "FROM calculations WHERE material_id = ?", (calc["material_id"],)):
            calcs[c[0]] = dict(zip(("calc_id", "material_id", "calc_type", "status", "output_file", "completed_at"), c))
    provs: Dict[Any, Dict] = {}
    for r in raw:
        if r["property_name"] == "extraction_provenance" and r.get("text"):
            try:
                provs[r["calc_id"]] = json.loads(r["text"])
            except ValueError:
                pass
    rows = []
    for r in raw:
        c = calcs.get(r["calc_id"]) or {"calc_id": r["calc_id"], "material_id": r["material_id"],
                                        "calc_type": None, "output_file": None}
        rows.append(_row(r["property_name"], r.get("value"), r.get("text"), r.get("unit"), r.get("category"),
                         r.get("script"), r.get("extracted_at"), c, r["blob"], provs.get(r["calc_id"])))
    # what per atom and per formula unit divide by, per run (mace/properties/units.counts)
    basis: Dict[Any, Dict[str, Any]] = {}
    if "materials" in tables:
        f = conn.execute("SELECT formula FROM materials WHERE material_id = ?", (calc["material_id"],)).fetchone()
        for cid in calcs:
            basis.setdefault(cid, {})["_cell_formula"] = f[0] if f else None
    for r in raw:
        if r["property_name"] in _BASIS_KEYS:
            basis.setdefault(r["calc_id"], {})[r["property_name"]] = registry.decode(r.get("value"), r.get("text"),
                                                                                     r["property_name"])
    return rows, provs.get(calc["calc_id"]), basis


def _derived_summary(provenance: Optional[Dict[str, Any]], calc_type: Optional[str]) -> Tuple[List[Dict], List[str]]:
    """(each derived property that applies: name, status, reason, version),
    and the names the stored provenance does not know (extracted by an
    earlier MACE or before the property existed: Extract computes them)."""
    from mace.properties import registry
    known = (provenance or {}).get("derived") or {}
    out, missing = [], []
    for p in registry.properties_for(calc_type, interactive=False):
        e = known.get(p.name)
        if e is None:
            missing.append(p.name)
            out.append({"name": p.name, "status": "not computed yet", "version": p.version, "reason": "",
                        "references": [r.name for r in p.references]})
            continue
        status = e.get("status")
        if status == "stored" and str(e.get("version")) != str(p.version):
            status = "stored (older version)"
            missing.append(p.name)
        out.append({"name": p.name, "status": status, "version": e.get("version"), "reason": e.get("reason", ""),
                    "inputs": e.get("inputs"), "references": [r.name for r in p.references]})
    return out, missing


def extract_argv(run_arg: str, db_arg: str) -> List[str]:
    # --register-output-only: the one run is recorded, never the rest of its folder tree
    return ["analyze", "--extract-properties", "--output-file", run_arg, "--db-path", db_arg,
            "--register", "--register-output-only", "--backup", "--keep-backups", "3"]


def run_view(st, path: Path, hint: Optional[str] = None, scope: str = "run") -> Dict[str, Any]:
    from .dbbuild import command_text, rel_or_abs
    run = run_of(path, st.root)
    base = {"path": st.rel(path) if _inside(st.root, path) else path.name}
    if run is None:
        return {**base, "state": "no-run", "rows": []}
    run_rel = st.rel(run)
    dbs = candidate_dbs(st, run, hint)
    for db in dbs:
        if dbwriters.owner(db):
            raise gui_db.DbBusy("building", f"Updating... ({dbwriters.owner(db).lower()} is in progress)")

        def look(conn):
            calc = _calc_of(conn, run, db)
            if not calc:
                return None
            rows, prov, basis = _rows_of(conn, calc, scope == "material")
            return calc, rows, prov, basis
        got = gui_db.read(db, look)
        if got:
            calc, rows, prov, basis = got
            derived, missing = _derived_summary(prov, calc.get("calc_type"))
            from mace.properties import units
            return {**base, "state": "db", "run": run_rel, "db": st.rel(db), "calc": calc, "scope": scope,
                    "rows": rows, "units": units.annotate(rows, basis),
                    "provenance": {k: v for k, v in (prov or {}).items() if k != "derived"},
                    "derived": derived, "missing": missing,
                    "extract": {"command": command_text(extract_argv(rel_or_abs(st.root, run), rel_or_abs(st.root, db)))}}
    out = {**base, "run": run_rel, "rows": [], "scope": "run"}
    if dbs:
        db = dbs[0]
        mid, n = gui_db.read(db, lambda conn: _untracked_count(conn, run))
        out.update(state="not-in-db", db=st.rel(db), untracked={"material_id": mid, "rows": n},
                   extract={"command": command_text(extract_argv(rel_or_abs(st.root, run), rel_or_abs(st.root, db)))})
    else:
        out.update(state="no-db", db=None, folder=st.rel(run.parent) if run.parent != st.root else "")
    return out


# ------------------------------------------------------------------ computed on the fly (a pool job)
def computed_values(path: Path, db: Optional[Path] = None) -> Dict[str, Any]:
    """{key: value} of a run as an extraction would store them (the extractor
    and the derived-property registry), nothing written: what the tab shows
    for a run not in a database. ``db``: a database the reference energies
    look their reference runs up in (read only). A run that database holds
    is computed as that run (its calc_id and material: the lookups leave it
    and its material out, and its saved reference choices apply), as an
    extraction into it does."""
    import contextlib
    import io
    import sqlite3
    from mace.properties import reference as R
    from mace.utils import db_build
    from mace.utils.property_extractor import CrystalPropertyExtractor
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    x.reference_db = str(db) if db else None
    calc_id = material_id = None
    if db and Path(db).is_file():
        conn = db_build.open_ro(db)
        if conn is not None:
            try:
                calc_id = R.tracked_calc(conn, Path(path))
                row = R.calc_row(conn, calc_id) if calc_id else None
                material_id = row[0] if row else None
            except sqlite3.Error:
                calc_id = material_id = None
            finally:
                conn.close()
    with contextlib.redirect_stdout(io.StringIO()):
        props = x.extract_all_properties(Path(path), material_id=material_id, calc_id=calc_id)
    return {k: v for k, v in props.items() if not k.startswith("_")}


def compute_job(ctx=None, path: str = "", db: Optional[str] = None, **_) -> Dict[str, Any]:
    """The run's values as an extraction would store them, nothing written."""
    p = Path(path)
    if ctx is not None:
        ctx.progress(0.1, f"Reading {p.name}")
    props = computed_values(p, Path(db) if db else None)
    if ctx is not None:
        ctx.progress(0.9, "Formatting")
    prov = props.get("extraction_provenance") or {}
    calc = {"calc_id": None, "material_id": None, "calc_type": props.get("calculation_type"), "output_file": str(p)}
    from mace.utils import property_keys
    from mace.properties import registry
    registry.load()
    rows = []
    for name, v in props.items():
        k = property_keys.key(name)
        if isinstance(v, (dict, list)):
            text, num = json.dumps(v), None
        elif isinstance(v, bool):
            text, num = str(v), float(v)
        elif isinstance(v, (int, float)):
            text, num = str(v), float(v)
        else:
            text, num = str(v), None
        rows.append(_row(name, num, text, k.unit if k else None, k.category if k else None, "computed now (not stored)",
                         None, calc, property_keys.is_blob(name), prov))
    derived, _ = _derived_summary(prov, calc["calc_type"])
    from mace.properties import units
    return {"state": "computed", "calc": calc, "rows": rows, "units": units.annotate(rows, {None: props}),
            "derived": derived}


# ------------------------------------------------------------------ Add reference… (reference energies)
#: references that take a stoichiometric coefficient (N*CALC_ID)
WEIGHTED = {"reactants", "products"}
CANDIDATES = 500


def _fits(role: str, run, cand) -> bool:
    """Whether a run is the kind a reference asks for (the picker lists those
    first; "Show all runs" lists the rest)."""
    from mace.properties import reference as R
    c, rc = cand.comp, run.comp
    if role == "elements":
        return bool(c) and len(c) == 1 and (not rc or set(c) <= set(rc))
    if role == "atoms":
        return cand.atoms == 1 and (not rc or set(c or {}) <= set(rc))
    if role == "bulk":
        return cand.dim == 3 and bool(c and rc and R.ratio(rc, c))
    if role == "surface":
        return cand.dim == run.dim and bool(c and rc and set(c) <= set(rc))
    if role == "adsorbate":
        return bool(c and rc and set(c) <= set(rc))
    return True


def _role_text(prop, ref) -> Dict[str, Any]:
    return {"name": ref.name, "many": ref.many, "required": ref.required, "description": ref.description,
            "weighted": ref.name in WEIGHTED}


def references_view(st, path: Path, hint: Optional[str] = None) -> Dict[str, Any]:
    """The picker's data: the run's reference energies and the database's runs."""
    from mace.properties import reference as R
    from mace.properties import registry
    from .dbbuild import command_text, rel_or_abs
    run = run_of(path, st.root)
    if run is None:
        raise ValueError(f"{Path(path).name}: no CRYSTAL output")
    registry.load()
    for db in candidate_dbs(st, run, hint):
        def look(conn):
            calc = _calc_of(conn, run, db)
            if not calc:
                return None
            cid = calc["calc_id"]
            vals = R.light_values(conn, [cid]).get(cid) or {}
            me = R.Part(run.name, vals, calc_id=cid, material_id=calc["material_id"], output_file=run,
                        calc_type=calc["calc_type"])
            r = conn.execute("SELECT property_value_text FROM properties WHERE calc_id = ? AND property_name = ?",
                             (cid, registry.PROVENANCE_KEY)).fetchone()
            try:
                prov = (json.loads(r[0]) if r and r[0] else {}).get("derived") or {}
            except ValueError:
                prov = {}
            saved = registry.saved_choices(conn, cid)
            props = []
            for p in registry.properties_for(calc["calc_type"], interactive=False):
                if not p.references:
                    continue
                e = prov.get(p.name) or {}
                props.append({"name": p.name, "description": p.description, "source": p.source,
                              "status": e.get("status") or "not computed yet", "reason": e.get("reason", ""),
                              "used": e.get("references") or {}, "saved": saved.get(p.name) or {},
                              "roles": [_role_text(p, r) for r in p.references]})
            cands = []
            for c in R.candidates(conn, exclude=[cid], limit=CANDIDATES):
                refuse, warn = R.check(me, c)
                cands.append({"calc_id": c.calc_id, "label": c.label, "calc_type": c.calc_type,
                              "material_id": c.material_id, "formula": R.formula(c.comp), "atoms": c.atoms,
                              "dim": c.dim, "functional": c.settings["functional"],
                              "basis_set": c.settings["basis_set"], "energy_au": c.energy,
                              "check": "refuse" if refuse else "warn" if warn else "ok", "reasons": refuse + warn,
                              "fits": [r["name"] for p in props for r in p["roles"] if _fits(r["name"], me, c)]})
            return calc, me, props, cands
        got = gui_db.read(db, look)
        if got:
            calc, me, props, cands = got
            return {"state": "db", "run": st.rel(run), "db": st.rel(db), "calc": calc,
                    "formula": R.formula(me.comp), "dim": me.dim, "functional": me.settings["functional"],
                    "basis_set": me.settings["basis_set"], "properties": props, "candidates": cands,
                    "command": command_text(extract_argv(rel_or_abs(st.root, run), rel_or_abs(st.root, db)))}
    return {"state": "not-in-db", "run": st.rel(run), "properties": [], "candidates": []}


def reference_argv(prop_name: str, refs: Dict[str, Any], clear: bool, known) -> List[str]:
    """The --reference arguments of a picker's choice: NAME=CALC_ID (N*CALC_ID
    with a coefficient) per run, NAME=auto per reference to forget. Only
    calc_ids the database has (``known``): no paths reach the command line."""
    import re
    from mace.properties import registry
    prop = registry.get(prop_name)
    if prop is None or not prop.references or prop.interactive:
        raise ValueError(f"{prop_name!r} is not a reference energy")
    names = {r.name: r for r in prop.references}
    out: List[str] = []
    if clear:
        for n in names:
            out += ["--reference", f"{n}={registry.AUTO}"]
        return out
    if not isinstance(refs, dict) or not any(refs.values()):
        raise ValueError("Choose at least one run")
    for n, vals in refs.items():
        if n not in names:
            raise ValueError(f"{prop_name} has no reference {n!r}")
        vals = vals if isinstance(vals, list) else [vals]
        if len(vals) > 1 and not names[n].many:
            raise ValueError(f"{n}: one run")
        for v in vals:
            m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\*)?([^\s=*]+)", str(v).strip())
            if not m or m.group(2) not in known:
                raise ValueError(f"{n}: {v!r} is not a calculation of the database")
            if m.group(1) and n not in WEIGHTED:
                raise ValueError(f"{n}: a coefficient is for the reactants and products of a reaction")
            out += ["--reference", f"{n}={str(v).strip()}"]
    return out


def _save_references(handler, body):
    st = handler.state
    p = st.resolve(str(body.get("path") or ""))
    run = run_of(p, st.root)
    if run is None:
        raise ValueError(f"{p.name}: no CRYSTAL output")
    if not body.get("db"):
        raise ValueError("Choose a database")
    db = st.resolve(str(body["db"]))
    if not _inside(st.root, db):
        from .dbbuild import OUTSIDE
        raise PermissionError(OUTSIDE)
    if not gui_db.is_materials_db(db):
        raise ValueError(f"{db.name} is not a MACE materials database")

    def look(conn):
        calc = _calc_of(conn, run, db)
        known = {r[0] for r in conn.execute("SELECT calc_id FROM calculations")}
        return calc, known
    calc, known = gui_db.read(db, look)
    if not calc:
        raise ValueError(f"{run.name} is not in {db.name}: Extract it first")
    extra = reference_argv(str(body.get("property") or ""), body.get("references") or {}, bool(body.get("clear")),
                           known)
    return _extract(handler, {**body, "path": st.rel(run)}, extra=extra,
                    title=f"{body.get('property')} of {run.name}")


# ------------------------------------------------------------------ routes
def _value(st, query) -> Dict[str, Any]:
    db = st.resolve(_q(query, "db", ""))
    if not db.is_file() or not gui_db.is_materials_db(db):
        raise ValueError(f"{db.name} is not a MACE materials database")
    calc_id, name = _q(query, "calc"), _q(query, "name", "")
    from mace.properties import registry

    def get(conn):
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        for t in ("properties", "property_blobs"):
            if t not in tables:
                continue
            num = "property_value" if t == "properties" else "NULL"
            r = conn.execute(f"SELECT {num}, property_value_text FROM {t} WHERE calc_id = ? AND property_name = ?",
                             (calc_id, name)).fetchone()
            if r:
                return registry.decode(r[0], r[1], name)
        raise FileNotFoundError(f"{name} is not stored for {calc_id}")
    return {"name": name, "calc_id": calc_id, "value": gui_db.read(db, get)}


def get(handler, route: str, query):
    st = handler.state
    if route == "properties/run":
        p = st.resolve(_q(query, "path", ""))
        scope = "material" if _q(query, "scope") == "material" else "run"
        return handler._json(run_view(st, p, _q(query, "db") or None, scope))
    if route == "properties/value":
        return handler._json(_value(st, query))
    if route == "properties/references":
        p = st.resolve(_q(query, "path", ""))
        return handler._json(references_view(st, p, _q(query, "db") or None))
    if route == "properties/compute":
        p = st.resolve(_q(query, "path", ""))
        run = run_of(p, st.root)
        if run is None:
            raise ValueError(f"{p.name}: no CRYSTAL output to read")
        from . import cache as gui_cache
        # the reference energies look their reference runs up in the run's database (read only)
        dbs = candidate_dbs(st, run, _q(query, "db") or None)
        ref_db = dbs[0] if dbs else None
        stamp = ref_db.stat().st_mtime_ns if ref_db else 0
        key = gui_cache.file_key(run, ("propcompute", 2, str(ref_db or ""), stamp))
        hit = st.parse_cache.get(key)

        def finish(res):
            st.parse_cache.put(key, res, 2048)
            # the job reads the run by its absolute path: the page gets it relative to the served folder
            calc = {**(res.get("calc") or {}), "output_file": st.rel(run)}
            return {**res, "calc": calc, "run": st.rel(run), "path": st.rel(p)}
        if hit is not None:
            return handler._json(finish(hit))
        t = st.tasks.submit("properties", f"Properties of {run.name}", "properties.compute",
                            {"path": str(run), "db": str(ref_db) if ref_db else None},
                            client=handler.headers.get("X-Mace-Client"), finish=finish,
                            supersede=("properties", st.rel(run)))
        return gui_tasks.answer(handler, st.tasks, t, gui_tasks.clamp_wait(_q(query, "wait", 0.05)))
    return handler._error(404, f"unknown API route {route}")


def _extract(handler, body, extra: Optional[List[str]] = None, title: Optional[str] = None):
    from .dbbuild import OUTSIDE, rel_or_abs
    from .mechanics_routes import _gen
    from .server import cli_task
    st = handler.state
    p = st.resolve(str(body.get("path") or ""))
    run = run_of(p, st.root)
    if run is None:
        raise ValueError(f"{p.name}: no CRYSTAL output to extract")
    if not body.get("db"):
        raise ValueError("Choose a database (Create Database… makes one)")
    db = st.resolve(str(body["db"]))
    if not _inside(st.root, db):
        raise PermissionError(OUTSIDE)
    if not gui_db.is_materials_db(db):
        raise ValueError(f"{db.name} is not a MACE materials database")
    st.check_write(db, allow=("database",))
    _gen(handler, body)
    argv = extract_argv(rel_or_abs(st.root, run), rel_or_abs(st.root, db)) + list(extra or [])
    if st.test_hooks:
        st.test_log.append({"properties_extract_argv": argv})
    token = dbwriters.acquire(db, OWNER)
    try:
        def job(ctx):
            ctx.progress(0.1, f"Extracting {run.name}")
            try:
                r = clirun.run(clirun.mace_argv("--no-banner", *argv), st.root, ctx, timeout=1800)
            finally:
                # given back before the task is done: the tab's next read must not find the database busy
                # (release_when_done below covers a task canceled before it ran)
                dbwriters.release(db, token)
            log = r.log.replace(str(st.root) + os.sep, "")
            if r.returncode != 0:
                raise RuntimeError("The extraction failed: " + ((r.stderr or log).strip().splitlines() or ["no output"])[-1][-500:])
            return {"code": r.returncode, "log": log[-8000:], "command": "mace " + " ".join(argv), "db": st.rel(db),
                    "run": st.rel(run)}
        t = cli_task(st, "propextract", title or f"Extract {run.name} into {db.name}", job,
                     client=handler.headers.get("X-Mace-Client"), supersede=("propextract", st.rel(run)))
    except BaseException:
        dbwriters.release(db, token)
        raise
    import threading

    def release_when_done():
        t.done.wait()
        dbwriters.release(db, token)
    threading.Thread(target=release_when_done, name=f"mace-propextract-{t.id}", daemon=True).start()
    return gui_tasks.answer(handler, st.tasks, t, gui_tasks.clamp_wait(body.get("wait", 0.3)))


def post(handler, route: str, body):
    if route == "properties/extract":
        return _extract(handler, body)
    if route == "properties/references":
        return _save_references(handler, body)
    return handler._error(404, f"unknown API route {route}")


def handle(handler, method: str, route: str, payload):
    from .submit import SubmitConflict
    try:
        if method == "GET" and route in GET_ROUTES:
            return get(handler, route, payload)
        if method == "POST" and route in POST_ROUTES:
            return post(handler, route, payload or {})
        return handler._error(404, f"unknown API route {route}")
    except gui_db.DbBusy as exc:
        return handler._json({"error": str(exc), "busy": True, "reason": exc.args[0] if exc.args else "busy"}, 503)
    except SubmitConflict as exc:
        return handler._json({"error": str(exc), "conflict": True}, 409)
    except dbwriters.WriterBusy as exc:
        return handler._json({"error": str(exc), "busy": True}, 409)
    except PermissionError as exc:
        return handler._error(403, str(exc))
    except FileNotFoundError as exc:
        return handler._error(404, str(exc))
    except ValueError as exc:
        return handler._error(400, str(exc))


# ------------------------------------------------------------------ interactive saves (the tau dialog)
def save_interactive(st, run: Path, prop_name: str, values: Dict[str, Any], who: str) -> Dict[str, Any]:
    """Store an interactive property's values with ``run`` in its database
    (candidate_dbs: the first that exists). The run's calculation when the
    database tracks it; else untracked, under the material id its file name
    gives. Raises ValueError when there is no database."""
    from mace.properties import registry
    from mace.utils import db_build
    dbs = candidate_dbs(st, run, None)
    if not dbs:
        raise ValueError(f"No materials.db for {run.name}: Create or Update Database from its folder first")
    db = dbs[0]
    st.check_write(db, allow=("database",))

    def look(conn):
        return _calc_of(conn, run, db)
    calc = gui_db.read(db, look)
    material_id = calc["material_id"] if calc else db_build.material_id_from_filename(run)
    calc_id = calc["calc_id"] if calc else None
    with dbwriters.holding(db, who):
        stored = registry.store_interactive(db, prop_name, values, material_id=material_id, calc_id=calc_id)
    return {"db": st.rel(db), "run": st.rel(run), "calc_id": calc_id, "material_id": material_id, "keys": stored}
