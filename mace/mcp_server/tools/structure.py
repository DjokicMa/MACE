"""Structure tools: read (any format MACE reads), supercells, standard
cells, slabs, Clean/preopt, and the ML potentials preopt can use.

They call the GUI's structure readers and builders (mace.gui.structures,
mace.gui.builders) and ``mace preopt``, so a file made here is the one the
GUI's Build menu or the CLI makes. Writes go through ctx.write_file (new
files directly; replacing a file needs confirm and keeps File History).
"""
from __future__ import annotations

import io
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

READ_SUFFIXES = ".cif, .d12, .out, .xyz, POSCAR/CONTCAR/.vasp, fort.34/.f34/.gui, .mol, .pdb, .gen"
WRITE_FORMATS = {".cif": "cif", ".xyz": "extxyz", ".extxyz": "extxyz", ".vasp": "vasp", ".poscar": "vasp"}
_PATH = {"type": "string", "description": f"structure file ({READ_SUFFIXES}), relative to the served folder"}
_FRAME = {"type": "integer", "default": -1, "description": "for a CRYSTAL .out: the geometry frame (-1 = last)"}
_OUTPUT = {"type": "string", "description": ("file to write (.cif, .xyz extended XYZ, .vasp/POSCAR); default: "
                                             "<input stem>_<operation>.cif beside the input (.xyz for a "
                                             "molecule, polymer or slab)")}
_SYMPREC = {"type": "number", "minimum": 1e-5, "maximum": 0.5, "default": 1e-3,
            "description": "symmetry tolerance in Angstrom"}


def register(reg) -> None:
    reg.tool("structure_inspect", (
        "Read a structure file (CIF, D12, CRYSTAL output frame, XYZ, POSCAR, fort.34, ...) and report formula, "
        "lattice, dimensionality, space group and, for small structures, every site."),
        {"properties": {"path": _PATH, "frame": _FRAME,
                        "sites": {"type": "boolean", "description": "list every site (default: up to 500 atoms)"},
                        "symprec": _SYMPREC},
         "required": ["path"]},
        title="Inspect a structure", group="structure")(structure_inspect)

    reg.tool("structure_supercell", (
        "Build a supercell and write it to a file. matrix is [na, nb, nc] or a 3x3 integer matrix (rows are the "
        "new vectors in terms of a, b, c). A slab or polymer repeats along its periodic directions only."),
        {"properties": {"path": _PATH, "matrix": {"type": "array", "minItems": 3, "maxItems": 3,
                                                  "description": "[na, nb, nc] or [[..],[..],[..]]"},
                        "frame": _FRAME, "output": _OUTPUT},
         "required": ["path", "matrix"]},
        title="Supercell", group="structure", writes=True, confirm="on_overwrite")(structure_supercell)

    reg.tool("structure_standardize", (
        "Transform a 3D crystal to spglib's primitive or conventional standard cell, its Niggli-reduced cell, or "
        "symmetrize it (refined, idealized cell at symprec), and write the result."),
        {"properties": {"path": _PATH, "kind": {"type": "string", "enum": ["primitive", "conventional", "niggli",
                                                                           "symmetrize"]},
                        "symprec": _SYMPREC, "frame": _FRAME, "output": _OUTPUT},
         "required": ["path", "kind"]},
        title="Standard cell", group="structure", writes=True, confirm="on_overwrite")(structure_standardize)

    reg.tool("structure_slab", (
        "Cut a slab from a 3D crystal along (hkl) with a number of atomic planes and vacuum, and write it. With "
        "terminations_only it only lists the distinct terminations (no file)."),
        {"properties": {"path": _PATH, "hkl": {"type": "array", "items": {"type": "integer"}, "minItems": 3,
                                               "maxItems": 3},
                        "layers": {"type": "integer", "minimum": 1, "maximum": 200, "default": 5},
                        "vacuum": {"type": "number", "minimum": 0, "maximum": 200, "default": 15.0,
                                   "description": "Angstrom between periodic images along c"},
                        "termination": {"type": "integer", "minimum": 0, "default": 0},
                        "symmetric": {"type": "boolean", "default": False},
                        "terminations_only": {"type": "boolean", "default": False},
                        "frame": _FRAME, "output": _OUTPUT},
         "required": ["path", "hkl"]},
        title="Slab", group="structure", writes=True, confirm="on_overwrite")(structure_slab)

    reg.tool("structure_clean", (
        "Clean/pre-optimize a geometry with `mace preopt` (springs, gfnff, xtb tiers, mlip:<id>, auto) and write "
        "<stem>_preopt.cif and its provenance JSON. Reads .cif and .d12. The result is a starting geometry, "
        "not a result: run a CRYSTAL OPT from it."),
        {"properties": {"path": {"type": "string", "description": "a .cif or .d12"},
                        "method": {"type": "string", "default": "springs",
                                   "description": "springs, gfnff, auto, or mlip:<model id> (see mlip_models)"},
                        "relax_cell": {"type": "boolean", "default": False},
                        "keep_symmetry": {"type": "boolean", "default": True},
                        "symprec": _SYMPREC,
                        "charge": {"type": "integer"},
                        "uhf": {"type": "integer", "minimum": 0, "default": 0,
                                "description": "unpaired electrons (GFN-FF/xTB tiers)"},
                        "steps": {"type": "integer", "minimum": 1, "maximum": 100000},
                        "time_limit": {"type": "number", "minimum": 1, "maximum": 3600,
                                       "description": "seconds"},
                        "threads": {"type": "integer", "minimum": 1, "maximum": 16},
                        "output_dir": {"type": "string", "description": "default: the input's folder"}},
         "required": ["path"]},
        title="Clean geometry", group="structure", writes=True, confirm="on_overwrite")(structure_clean)

    reg.tool("mlip_models", (
        "The ML interatomic potentials `mace preopt` can use (mlip:<id>), their licenses and whether they are "
        "ready on this machine."),
        {"properties": {}}, title="ML potentials", group="structure")(mlip_models)


# ---------------------------------------------------------------- helpers
def read(ctx, path: str, frame: int = -1) -> Tuple[Path, Dict[str, Any]]:
    p = ctx.path(path, kind="file")
    from mace.gui import structures as S
    return p, S.read_structure(p, int(frame))


def _np():
    import numpy as np
    return np


def summary(d: Dict[str, Any], ctx, p: Path, sites: Optional[bool] = None) -> Dict[str, Any]:
    n = len(d.get("numbers") or [])
    comp: Dict[str, int] = {}
    for s in d.get("symbols") or []:
        comp[s] = comp.get(s, 0) + 1
    sym = dict(d.get("symmetry") or {})
    for k in ("equivalent_atoms", "site_symmetry", "multiplicity"):
        if n > ctx.config.max_atoms_listed:
            sym.pop(k, None)
    out = {"path": ctx.show(p), "format": (d.get("source") or {}).get("kind"), "formula": d.get("formula"),
           "reduced_formula": d.get("reduced_formula"), "z": d.get("z"), "n_atoms": n, "composition": comp,
           "dimensionality": d.get("dimensionality"), "pbc": d.get("pbc"), "lattice": d.get("lattice"),
           "cell": d.get("cell"), "symmetry": sym}
    for k in ("n_frames", "frame_label", "energy_au", "opt_status"):
        if k in d:
            out[k] = d[k]
    want = (n <= ctx.config.max_atoms_listed) if sites is None else bool(sites)
    if want:
        out["sites"] = [{"index": i, "symbol": s, "frac": [round(x, 6) for x in f], "cart": [round(x, 6) for x in c]}
                        for i, (s, f, c) in enumerate(zip(d["symbols"], d["frac"], d["cart"]))]
    elif n:
        out["sites_note"] = f"{n} sites not listed (pass sites: true)"
    return out


def _default_output(p: Path, tag: str, pbc) -> Path:
    ext = ".cif" if all(pbc) else ".xyz"
    return p.with_name(f"{p.stem}_{tag}{ext}")


def write_structure(ctx, target: Path, cell, frac, numbers, pbc, reason: str) -> Dict[str, Any]:
    low = target.name.lower()
    fmt = WRITE_FORMATS.get(target.suffix.lower())
    if low in ("poscar", "contcar") or low.startswith(("poscar", "contcar")):
        fmt = "vasp"
    if fmt is None:
        raise ValueError(f"{target.name}: write .cif, .xyz (extended XYZ), .vasp or POSCAR")
    if fmt in ("cif", "vasp") and not all(pbc):
        raise ValueError(f"a {'CIF' if fmt == 'cif' else 'POSCAR'} is a 3D cell; write this "
                         f"{sum(map(bool, pbc))}-periodic structure as .xyz (extended XYZ keeps the periodicity)")
    np = _np()
    from ase import Atoms
    import ase.io
    atoms = Atoms(numbers=[int(z) for z in numbers], cell=np.asarray(cell, float).reshape(3, 3),
                  scaled_positions=np.asarray(frac, float).reshape(-1, 3), pbc=[bool(x) for x in pbc])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt == "cif":                    # ase's CIF writer writes bytes (as the GUI's cif_text)
            bbuf = io.BytesIO()
            ase.io.write(bbuf, atoms, format="cif")
            data = bbuf.getvalue()
        else:
            buf = io.StringIO()
            if fmt == "vasp":
                ase.io.write(buf, atoms, format="vasp", direct=True, sort=False)
            else:
                ase.io.write(buf, atoms, format=fmt)
            data = buf.getvalue().encode()
    return ctx.write_file(target, data, reason=reason)


def _arrays(d: Dict[str, Any]):
    return d["cell"], d["frac"], d["numbers"], tuple(bool(x) for x in d["pbc"])


def write_built(ctx, p: Path, built: Dict[str, Any], tag: str, output: Optional[str], reason: str) -> Dict[str, Any]:
    target = ctx.path(output, must_exist=False) if output else _default_output(p, tag, built["pbc"])
    if not ctx.inside(target) and not ctx.config.allow_outside_root:
        raise ValueError(f"{target} is outside the served folder")
    w = write_structure(ctx, target, built["cell"], built["frac"], built["numbers"], built["pbc"], reason)
    from mace.gui import structures as S
    from ase.geometry import cell_to_cellpar
    np = _np()
    cell = np.asarray(built["cell"], float).reshape(3, 3)
    numbers = [int(z) for z in built["numbers"]]
    par = [float(x) for x in cell_to_cellpar(cell)]
    lattice = dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), par), volume=float(abs(np.linalg.det(cell))))
    sym = S.symmetry(cell, np.asarray(built["frac"], float), numbers, built["pbc"]) if len(numbers) <= 20000 else {}
    formula, z = S.reduced_formula(numbers)
    return {"written": w, "n_atoms": len(numbers), "reduced_formula": formula, "z": z, "lattice": lattice,
            "pbc": [bool(x) for x in built["pbc"]],
            "space_group": {k: sym.get(k) for k in ("number", "international")} if sym.get("available") else None,
            "summary": built.get("summary"), "info": {k: v for k, v in (built.get("info") or {}).items()
                                                      if k not in ("vectors", "terminations")}}


def _need_3d(d: Dict[str, Any], what: str) -> None:
    if not all(d.get("pbc") or ()):
        raise ValueError(f"{what} needs a 3D crystal; this structure is a {str(d.get('dimensionality')).lower()}")


# ---------------------------------------------------------------- handlers
def structure_inspect(ctx, args):
    p, d = read(ctx, args["path"], args.get("frame", -1))
    symprec = args.get("symprec")
    if symprec is not None and abs(float(symprec) - 1e-3) > 1e-12:
        from mace.gui import structures as S
        np = _np()
        d["symmetry"] = S.symmetry(np.asarray(d["cell"], float), np.asarray(d["frac"], float),
                                   [int(z) for z in d["numbers"]], tuple(d["pbc"]), symprec=float(symprec))
    out = summary(d, ctx, p, args.get("sites"))
    try:
        from mace.gui.structures import dimensionality_hint
        hint = dimensionality_hint(d["cell"], _np().asarray(d["cart"], float), tuple(d["pbc"]))
        if hint:
            out["dimensionality_hint"] = hint
    except Exception:
        pass
    return out


def _matrix(raw) -> List[List[int]]:
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in raw):
        if any(float(x) != int(x) for x in raw):
            raise ValueError("matrix entries are integers")
        a, b, c = (int(x) for x in raw)
        return [[a, 0, 0], [0, b, 0], [0, 0, c]]
    if all(isinstance(r, (list, tuple)) and len(r) == 3 for r in raw):
        return [[int(x) for x in r] for r in raw]
    raise ValueError("matrix is [na, nb, nc] or a 3x3 list of integers")


def structure_supercell(ctx, args):
    p, d = read(ctx, args["path"], args.get("frame", -1))
    m = _matrix(args["matrix"])
    from mace.gui import builders
    cell, frac, numbers, pbc = _arrays(d)
    built = builders.supercell(ctx=ctx, cell=cell, frac=frac, numbers=numbers, pbc=pbc, matrix=m)
    diag = all(m[i][j] == 0 for i in range(3) for j in range(3) if i != j)
    tag = ("x".join(str(m[i][i]) for i in range(3)) if diag else "supercell")
    return write_built(ctx, p, built, tag, args.get("output"), f"Supercell {tag} of {p.name} (mace mcp)")


def structure_standardize(ctx, args):
    p, d = read(ctx, args["path"], args.get("frame", -1))
    kind = args["kind"]
    _need_3d(d, "a standard cell")
    from mace.gui import builders
    cell, frac, numbers, pbc = _arrays(d)
    symprec = float(args.get("symprec") or 1e-3)
    if kind == "symmetrize":
        built = builders.symmetrize(ctx=ctx, cell=cell, frac=frac, numbers=numbers, pbc=pbc, symprec=symprec)
    else:
        built = builders.standardize(ctx=ctx, cell=cell, frac=frac, numbers=numbers, pbc=pbc, kind=kind,
                                     symprec=symprec)
    return write_built(ctx, p, built, kind, args.get("output"), f"{kind.capitalize()} cell of {p.name} (mace mcp)")


def structure_slab(ctx, args):
    p, d = read(ctx, args["path"], args.get("frame", -1))
    _need_3d(d, "a slab cut")
    from mace.gui import builders
    cell, frac, numbers, pbc = _arrays(d)
    hkl = [int(x) for x in args["hkl"]]
    if args.get("terminations_only"):
        t = builders.slab_terminations(cell, frac, numbers, builders.check_hkl(hkl))
        for row in t.get("terminations") or []:
            row.pop("thumb", None)
        return {"path": ctx.show(p), "hkl": hkl, **t}
    built = builders.slab(ctx=ctx, cell=cell, frac=frac, numbers=numbers, pbc=pbc, hkl=hkl,
                          layers=int(args.get("layers", 5)), vacuum=float(args.get("vacuum", 15.0)),
                          termination=int(args.get("termination", 0)), symmetric=bool(args.get("symmetric")))
    tag = "slab_" + "".join(str(x) for x in hkl).replace("-", "m")
    out = write_built(ctx, p, built, tag, args.get("output"), f"Slab {hkl} of {p.name} (mace mcp)")
    out["note"] = ("The slab is written as a 3D cell with vacuum along c; for a CRYSTAL SLAB calculation set it up "
                   "as a 2D system (dimensionality SLAB) in the deck options.")
    return out


def structure_clean(ctx, args):
    p = ctx.path(args["path"], kind="file")
    if p.suffix.lower() not in (".cif", ".d12"):
        raise ValueError(f"{p.name}: mace preopt reads .cif and .d12 files")
    outdir = ctx.path(args["output_dir"], must_exist=False) if args.get("output_dir") else p.parent
    if ctx.config.read_only:
        raise ValueError("the server runs with --read-only")
    outs = [outdir / f"{p.stem}_preopt.cif", outdir / f"{p.stem}_preopt.json"]
    existing = [o for o in outs if o.exists()]
    if existing:
        ctx.require_confirm("replace " + " and ".join(ctx.show(o) for o in existing),
                            {"existing": [ctx.show(o) for o in existing]})
        from mace.gui import history
        if history.root() is None:
            history.configure(ctx.root)
        history.snapshot(existing, "Replaced by mace mcp structure_clean", route="mcp")
    outdir.mkdir(parents=True, exist_ok=True)
    argv = ["preopt", str(p), "--method", str(args.get("method") or "springs"), "-o", str(outdir), "--json"]
    if args.get("relax_cell"):
        argv.append("--relax-cell")
    if args.get("keep_symmetry") is False:
        argv.append("--no-symmetry")
    for key, flag in (("symprec", "--symprec"), ("charge", "--charge"), ("uhf", "--uhf"), ("steps", "--steps"),
                      ("time_limit", "--time-limit"), ("threads", "--threads")):
        if args.get(key) is not None:
            argv += [flag, str(args[key])]
    if existing:
        argv.append("--force")
    limit = float(args.get("time_limit") or 600) + 300
    r = ctx.run_cli(argv, cwd=p.parent, timeout=limit)
    prov = None
    try:
        prov = json.loads(r["stdout"]) if r["stdout"].strip() else None
    except json.JSONDecodeError:
        prov = None
    if r["code"] != 0:
        raise ValueError(("mace preopt failed: " + (r["stderr"] or r["stdout"]).strip())[-4000:])
    return {"cif": ctx.show(outs[0]), "provenance_json": ctx.show(outs[1]), "provenance": prov,
            "log": (r["stderr"] or "")[-3000:], "argv": r["argv"],
            "next": "A starting geometry, not a result: set up a CRYSTAL OPT from the CIF (setup_d12_preview)."}


def mlip_models(ctx, args):
    r = ctx.run_cli(["mlip", "list", "--json"], timeout=120)
    if r["code"] != 0:
        raise ValueError((r["stderr"] or r["stdout"]).strip()[-2000:] or "mace mlip list failed")
    return json.loads(r["stdout"])
