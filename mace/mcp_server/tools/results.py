"""Result tools: which runs finished, what a run says, every parsed
property, and materials.db queries.

They read with MACE's own parsers: the completion checker's categories
(``mace completion``), the convergence parser (``mace analyze
--convergence``), the property extractor (``mace analyze``) with tracking
off (nothing is written to any database), and the GUI's read-only
materials.db queries (``mace.gui.db``).
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any, Dict, List

HEADLINE = ("calculation_type", "total_energy_au", "total_energy_ev", "total_energy_corrected_au",
            "total_energy_corrected_ev", "band_gap", "direct_band_gap", "indirect_band_gap",
            "electronic_classification", "magnetic_classification", "spin_polarized", "space_group_number",
            "crystal_system", "atoms_in_unit_cell", "density", "primitive_cell_volume", "optimization_converged",
            "optimization_cycles", "final_gradient_norm", "scf_cycles", "cpu_time_total_s", "wall_time_s")
DIMS = ["CRYSTAL", "SLAB", "POLYMER", "MOLECULE"]
SORTS = ["material_id", "formula", "space_group", "dimensionality", "status", "source_type", "updated_at",
         "created_at", "n_calcs", "n_completed"]
_DB = {"type": "string", "default": "materials.db", "description": "the materials.db (relative to the served folder)"}


def register(reg) -> None:
    reg.tool("outputs_scan", (
        "Classify the CRYSTAL outputs (.out) in a folder like `mace completion`: finished (by calculation "
        "type), failed (by error: memory, SCF, walltime, geometry, basis...), still running."),
        {"properties": {"directory": {"type": "string", "default": "."},
                        "recursive": {"type": "boolean", "default": False}}},
        title="Scan outputs", group="results")(outputs_scan)

    reg.tool("output_summary", (
        "What one CRYSTAL output says: finished or failed (and why), calculation type, final energy, band gap, "
        "optimization steps and convergence, SCF cycles, timing."),
        {"properties": {"path": {"type": "string", "description": "a CRYSTAL .out"},
                        "energies": {"type": "boolean", "default": False,
                                     "description": "include the energy of every optimization step"}},
         "required": ["path"]},
        title="Output summary", group="results")(output_summary)

    reg.tool("output_properties", (
        "Every property MACE's extractor parses from a CRYSTAL output (energies, gaps, lattice, Mulliken "
        "charges and spins, SCF settings, phonons, transport...). Scalars by default; names picks properties "
        "(lists and tables included); full returns everything."),
        {"properties": {"path": {"type": "string"},
                        "names": {"type": "array", "items": {"type": "string"}, "maxItems": 200},
                        "full": {"type": "boolean", "default": False}},
         "required": ["path"]},
        title="Output properties", group="results")(output_properties)

    reg.tool("db_summary", (
        "A materials.db at a glance: materials, calculations by status and type, dimensionalities, property "
        "names."),
        {"properties": {"db": _DB}}, title="Database summary", group="results")(db_summary)

    reg.tool("db_search", (
        "Search the materials of a materials.db (read-only): text in id/formula, dimensionality, a calculation "
        "type or status, a space-group range, having a property. One page at a time (after = the page's "
        "'next' value)."),
        {"properties": {"db": _DB, "query": {"type": "string", "maxLength": 100},
                        "dimensionality": {"type": "array", "items": {"type": "string", "enum": DIMS}},
                        "calc_type": {"type": "string"}, "calc_status": {"type": "string"},
                        "sg_min": {"type": "integer", "minimum": 1, "maximum": 230},
                        "sg_max": {"type": "integer", "minimum": 1, "maximum": 230},
                        "property": {"type": "string"},
                        "sort": {"type": "string", "enum": SORTS, "default": "material_id"},
                        "direction": {"type": "string", "enum": ["asc", "desc"], "default": "asc"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                        "after": {"description": "the 'next' value of the previous page"}}},
        title="Search materials", group="results")(db_search)

    reg.tool("db_material", (
        "One material of a materials.db: its record, every calculation (type, status, files, settings) and "
        "every stored property."),
        {"properties": {"db": _DB, "material_id": {"type": "string"},
                        "max_properties": {"type": "integer", "minimum": 1, "maximum": 20000, "default": 2000}},
         "required": ["material_id"]},
        title="Material record", group="results")(db_material)


@contextlib.contextmanager
def _quiet():
    """The parsers print progress lines; they go to stderr, never the protocol stream."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def outputs_scan(ctx, args):
    top = ctx.path(args.get("directory") or ".", kind="dir")
    from mace.completion_checker import categorize_output_file
    files = sorted(top.glob("**/*.out" if args.get("recursive") else "*.out"))
    buckets: Dict[str, List[str]] = {}
    with _quiet():
        for i, f in enumerate(files):
            ctx.check()
            if len(files) > 20:
                ctx.progress(i / len(files), f.name)
            cat, _ = categorize_output_file(f)
            buckets.setdefault(cat, []).append(ctx.show(f))
    return {"directory": ctx.show(top), "n_outputs": len(files),
            "counts": {k: len(v) for k, v in sorted(buckets.items())}, "outputs": buckets}


def _extract(p: Path) -> Dict[str, Any]:
    from mace.properties import registry
    from mace.utils.property_extractor import CrystalPropertyExtractor
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    # no Bader partition or packing grid in a tool call (MACE_DERIVED_EXPENSIVE=1: computed)
    x.skip_derived = frozenset(registry.expensive())
    with _quiet():
        return x.extract_all_properties(p) or {}


def _scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def output_summary(ctx, args):
    p = ctx.path(args["path"], kind="file")
    from mace.completion_checker import categorize_output_file
    from mace.utils import out_convergence as oc
    with _quiet():
        status, _ = categorize_output_file(p)
        conv = oc.parse(p)
    out: Dict[str, Any] = {"path": ctx.show(p), "status": status, "complete": conv.get("complete"),
                           "terminated": conv.get("terminated"), "is_opt": conv.get("is_opt"),
                           "stop": oc.stop_sentence(conv)}
    pts = conv.get("points") or []
    if conv.get("is_opt"):
        last = pts[-1] if pts else {}
        out["optimization"] = {"steps": len(pts), "end": (conv.get("opt") or {}).get("end"),
                               "satisfied": (conv.get("opt") or {}).get("satisfied"),
                               "last_step": {k: last.get(k) for k in ("energy_au", "max_grad", "rms_grad", "max_disp",
                                                                      "rms_disp", "converged", "cell") if k in last},
                               "thresholds": {k: (v or {}).get("value") for k, v in (conv.get("thresholds") or {}).items()}}
        if args.get("energies"):
            out["optimization"]["energies_au"] = [q.get("energy_au") for q in pts]
    try:
        props = _extract(p)
    except Exception as exc:
        props = {}
        out["properties_error"] = f"{type(exc).__name__}: {exc}"
    head = {k: props[k] for k in HEADLINE if k in props}
    for k, v in props.items():
        if "gap" in k and _scalar(v) and k not in head:
            head[k] = v
    out["properties"] = head
    out["more"] = "output_properties lists everything the extractor parsed"
    return out


def output_properties(ctx, args):
    p = ctx.path(args["path"], kind="file")
    props = _extract(p)
    props.pop("_metadata", None)
    names = args.get("names")
    if names:
        missing = [n for n in names if n not in props]
        got = {n: props[n] for n in names if n in props}
        return {"path": ctx.show(p), "properties": got, "missing": missing,
                "available": sorted(props) if missing else None}
    if args.get("full"):
        return {"path": ctx.show(p), "properties": props}
    scal = {k: v for k, v in props.items() if _scalar(v)}
    rest = sorted(k for k in props if k not in scal)
    return {"path": ctx.show(p), "properties": scal, "tables": rest,
            "note": "tables (lists and objects) come with names: [...] or full: true"}


def _db(ctx, args) -> Path:
    p = ctx.path(args.get("db") or "materials.db", kind="file")
    from mace.gui import db as gdb
    if not gdb.is_materials_db(p):
        raise ValueError(f"{ctx.show(p)} is not a MACE materials.db")
    return p


def db_summary(ctx, args):
    p = _db(ctx, args)
    from mace.gui import db as gdb
    return {"db": ctx.show(p), **gdb.stats(p), "property_names": gdb.property_names(p)}


def db_search(ctx, args):
    p = _db(ctx, args)
    from mace.gui import db as gdb
    f = gdb.Filters(args.get("query") or "", args.get("dimensionality") or (), args.get("calc_type") or "",
                    args.get("calc_status") or "", args.get("sg_min"), args.get("sg_max"), args.get("property") or "")
    after = args.get("after")
    if isinstance(after, (list, tuple)):
        import json
        after = json.dumps(list(after))
    page = gdb.page(p, sort=args.get("sort") or "material_id", direction=args.get("direction") or "asc",
                    after=after, limit=int(args.get("limit") or 50), filters=f, with_total=True)
    return {"db": ctx.show(p), "filters": f.describe(), **page}


def db_material(ctx, args):
    p = _db(ctx, args)
    from mace.gui import db as gdb
    rec = gdb.material(p, str(args["material_id"]))
    cap = int(args.get("max_properties") or 2000)
    props = rec.get("properties") or []
    if len(props) > cap:
        rec["properties"] = props[:cap]
        rec["properties_truncated"] = len(props)
    return {"db": ctx.show(p), **rec}
