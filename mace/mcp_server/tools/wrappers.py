"""Typed wrappers that plugins.py exposes as tools, for MACE areas whose own
functions take arrays or server state rather than a file: vacuum supercells
(mace supercell), structure analysis (mace analyze) and the calculators
(mace calc). Each one only adapts arguments; the work is the area's own code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------- mace supercell --vacuum
def structure_vacuum(ctx, path: str, n1: Optional[int] = None, n2: Optional[int] = None, n3: Optional[int] = None,
                     vacuum: str = "bbox", output: Optional[str] = None) -> Dict[str, Any]:
    """Repeat a slab, polymer or molecule, also along its non-periodic directions, spaced by a vacuum mode
    (as mace supercell --vacuum and the GUI's Inspector > Display > Vacuum), and write the result. A
    non-periodic direction repeats only when its count is above 1; the result stays non-periodic there.

    Args:
        path: the structure file (CRYSTAL .out, P1 SLAB/POLYMER/MOLECULE .d12, extended XYZ, XYZ, CIF, ...)
        n1: repeats along a (default: 2 along a periodic direction, 1 along a non-periodic one)
        n2: repeats along b (same default)
        n3: repeats along c (same default)
        vacuum: bbox (the atoms' extent + 1 A on each side), file (the vacuum the file defines), crystal
            (CRYSTAL's 500 A), open (no repeats), or a number: the atoms' extent + that many A
        output: the file to write (.xyz keeps the periodicity; .cif or .vasp only for a 3D result); default
            <input stem>_<n1>x<n2>x<n3>.xyz beside the input (.cif for a 3D result)
    """
    from mace.utils import supercell_cli
    from . import structure as st
    p = ctx.path(path, kind="file")
    given = (n1, n2, n3)
    n = None if all(x is None for x in given) else [1 if x is None else int(x) for x in given]
    atoms, label, info = supercell_cli.build(p, n, None, str(vacuum))
    pbc = [bool(x) for x in atoms.pbc]
    target = ctx.path(output, must_exist=False) if output else \
        p.with_name(f"{p.stem}_{label}{'.cif' if all(pbc) else '.xyz'}")
    if not ctx.inside(target) and not ctx.config.allow_outside_root:
        raise ValueError(f"{target} is outside the served folder")
    w = st.write_structure(ctx, target, atoms.cell.array, atoms.get_scaled_positions(wrap=False),
                           atoms.numbers, pbc, f"supercell {label} of {p.name} (mace mcp)")
    return {"written": w, "n_atoms": len(atoms), "pbc": pbc, "repeats": info.get("repeats"),
            "vacuum_mode": info.get("mode"), "vacuum_vectors": info.get("vacuum")}


# ---------------------------------------------------------------- mace analyze
def _analysis(ctx, tool: str, path: Path, frame: int, params: Dict[str, Any]) -> Dict[str, Any]:
    from mace.analysis import structure as S
    from mace.gui import analysis_routes as A
    return A._clean(A.run_tool(tool, S.load(path, frame), params, ctx=ctx))


def analysis_porosity(ctx, path: Path, probe: str = "N2", radii: Literal["ccdc", "uff"] = "ccdc",
                      spacing: float = 0.15, frame: int = -1) -> Dict[str, Any]:
    """Pore analysis of a framework (mace analyze porosity): accessible surface area, void fraction, pore
    volume, largest included and free spheres, on a grid with a probe molecule.

    Args:
        path: the structure file
        probe: a probe molecule (N2, He, H2, H2O, CO2, Ar, CH4) or a radius in A
        radii: the atomic radii set
        spacing: the grid spacing in A (0.05 to 1)
        frame: for a CRYSTAL .out: the frame (-1: the last)
    """
    params: Dict[str, Any] = {"render": False, "radii": radii, "spacing": spacing}
    try:
        params["probe_radius"] = float(probe)
    except ValueError:
        params["probe"] = probe
    return _analysis(ctx, "porosity", path, frame, params)


def analysis_pxrd(ctx, path: Path, source: str = "CuKa", two_theta_min: float = 5.0, two_theta_max: float = 90.0,
                  step: float = 0.02, frame: int = -1) -> Dict[str, Any]:
    """A simulated powder X-ray diffraction pattern (mace analyze pxrd): the reflections (hkl, 2-theta,
    d, intensity) and the profile.

    Args:
        path: the structure file
        source: the X-ray source (CuKa, MoKa, CoKa, ... or a wavelength in A)
        two_theta_min: the first angle, degrees
        two_theta_max: the last angle, degrees
        step: the profile step, degrees
        frame: for a CRYSTAL .out: the frame (-1: the last)
    """
    return _analysis(ctx, "pxrd", path, frame, {"source": source, "min": two_theta_min, "max": two_theta_max,
                                                "step": step})


def analysis_topology(ctx, path: Path, mode: Literal["auto", "atoms", "sbu", "all"] = "auto",
                      tolerance: float = 1.15, frame: int = -1) -> Dict[str, Any]:
    """The net of a framework (mace analyze topology): its nodes, coordination sequences and point symbol,
    matched against the RCSR nets bundled with MACE (cite RCSR: the result's citation).

    Args:
        path: the structure file
        mode: the nodes: auto (sbu for a MOF, else atoms), atoms, sbu (secondary building units) or all
        tolerance: the bond tolerance (x the sum of covalent radii, 0.8 to 1.6)
        frame: for a CRYSTAL .out: the frame (-1: the last)
    """
    return _analysis(ctx, "topology", path, frame, {"mode": mode, "tolerance": tolerance})


# ---------------------------------------------------------------- mace calc
def _calc_json(res: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = json.loads(res.get("stdout") or "")
    except ValueError:
        data = None
    out: Dict[str, Any] = {"exit_code": res.get("code")}
    if data is not None:
        out["result"] = data
    else:
        out["output"] = (res.get("stdout") or "")[-4000:]
    if res.get("code") not in (0,):
        out["error"] = (res.get("stderr") or "")[-4000:]
    return out


def calc_backends(ctx, recheck: bool = False) -> Dict[str, Any]:
    """The calculators mace calc can use here (UFF/UFF4MOF, EQeq, xTB, DFTB+, ML potentials): each
    backend's methods and whether it is installed (mace calc list).

    Args:
        recheck: look for the external programs and ML environments again
    """
    return _calc_json(ctx.run_cli(["calc", "list", "--json"] + (["--recheck"] if recheck else [])))


# the mace calc options calc_run passes on, exactly as named (no abbreviations): "--name=value" for these
_CALC_VALUE_OPTIONS = ("--frame", "--threads", "--uhf", "--time-limit", "--fmax", "--steps", "--optimizer",
                       "--displacement", "--temperature", "--ensemble", "--timestep", "--md-steps", "--friction",
                       "--pressure", "--interval", "--seed", "--kpts", "--kpoint-density", "--engine",
                       "--uff-charges", "--cutoff", "--charge-centers", "--eqeq-method", "--gpu-mem-fraction",
                       "--dtype", "--option")
_CALC_FLAGS = ("--relax-cell", "--keep-symmetry")
# --option KEY=VALUE: the backend options a client may set (never skf_dir, never MACE's own _keys)
_CALC_OPTION_KEYS = ("kpts", "kpoint_density", "engine", "charges", "cutoff", "charge_centers", "eqeq_method",
                     "gpu_mem_fraction", "dtype", "lambda", "h_electron_affinity", "eta", "real_cells",
                     "recip_cells", "scc_tolerance", "max_scc", "no_third_order", "etemp", "per_call_timeout",
                     "compressibility_per_bar")
# what only the person running MACE decides: a model file and its license statement, a Slater-Koster folder
_CALC_REFUSED = {"--model-file": "a model file and its license statement come from you, not the client: run "
                                 "mace calc --model-file ... --model-license-ok yourself",
                 "--model-backend": "goes with --model-file, which this tool does not pass on",
                 "--model-license-ok": "the license statement comes from you, not the client",
                 "--skf-dir": "the Slater-Koster folder comes from MACE_DFTB_SKF (set it where the server starts)"}


def _calc_option(opt: Any) -> str:
    """One entry of calc_run's ``options``, checked against the allow-list."""
    if not isinstance(opt, str) or not opt.startswith("--"):
        raise ValueError(f"options: {opt!r} is not an option (write it as --name=value or --flag)")
    name, eq, value = opt.partition("=")
    if name in _CALC_REFUSED:
        raise ValueError(f"options: {name} is not passed on by this tool: {_CALC_REFUSED[name]}")
    if name in _CALC_FLAGS and not eq:
        return opt
    if name in _CALC_VALUE_OPTIONS and eq and value.strip():
        if name == "--option":
            key = value.partition("=")[0].strip()
            if "=" not in value or key not in _CALC_OPTION_KEYS:
                raise ValueError(f"options: --option {value!r}: the keys this tool passes on are "
                                 + ", ".join(_CALC_OPTION_KEYS))
        return opt
    raise ValueError(f"options: {opt!r} is not an option this tool passes on; it takes "
                     + ", ".join(f"{n}=..." for n in _CALC_VALUE_OPTIONS) + ", " + ", ".join(_CALC_FLAGS)
                     + " (use the path, output_dir, charge, device and confirm arguments for the others)")


def _calc_outputs(path: Path, task: str, backend: str, method: Optional[str], outdir: Path) -> List[Path]:
    """The files mace calc may write for this run (tasks.write_outputs' names, every variant), so that
    the existing ones can be kept in File History before --force replaces them."""
    from mace.calculators import base, registry, tasks
    try:
        name, m = registry.resolve(backend, method)
        m = registry.get(name).method(m).id
    except Exception:                       # an unknown backend or method: mace calc says so, writes nothing
        return []
    t = tasks.tag(base.CalcSpec(backend=name, method=m))
    stem = path.stem
    b = f"{stem}_{t}_{task}"
    names = [f"{b}.json", f"{b}.cif", f"{b}.extxyz", f"{b}.traj", f"{b}_log.csv", f"{b}_modes.extxyz",
             f"{stem}_{t}_charges.cif", f"{stem}_{t}_charges.extxyz"]
    return [outdir / n for n in names if (outdir / n).is_file()]


def calc_run(ctx, path: Path, task: Literal["sp", "opt", "freq", "md", "charges"], backend: str,
             method: Optional[str] = None, output_dir: Optional[Path] = None, charge: int = 0,
             device: str = "auto", options: Optional[List[str]] = None, confirm: bool = False) -> Dict[str, Any]:
    """Run a force field, charge model, tight-binding method or ML potential on a structure (mace calc):
    a single point, an optimization, frequencies, MD or partial charges. Writes the report (.json) and the
    task's files next to the input or into output_dir.

    Args:
        path: the structure file
        task: sp, opt, freq, md or charges
        backend: uff, eqeq, xtb, dftb or mlip (shorthands: uff4mof, gfn2, mlip:MODEL)
        method: the backend's method (calc_backends lists them)
        output_dir: where the files go (default: next to the input)
        charge: the total charge (xTB, DFTB+, EQeq)
        device: ML potentials: auto, cpu or cuda:N
        options: more mace calc options, each as --name=value or a bare flag, full names only (e.g.
            "--fmax=0.02", "--relax-cell", "--temperature=600"). Not passed on: --model-file and its license
            statement (only the user gives those) and --skf-dir (MACE_DFTB_SKF)
    """
    path = ctx.path(path, kind="file")
    outdir = ctx.path(output_dir, must_exist=False, kind="dir") if output_dir is not None else path.parent
    args = ["calc", task, str(path), "-b", backend, "--json", "--device", device]
    if method:
        args += ["-m", method]
    if output_dir is not None:
        args += ["-o", str(outdir)]
    if charge:
        args += ["--charge", str(int(charge))]
    for opt in options or []:
        args.append(_calc_option(opt))
    existing = _calc_outputs(path, task, backend, method, outdir)
    if existing:
        ctx.require_confirm("replace " + " and ".join(ctx.show(o) for o in existing),
                            {"existing": [ctx.show(o) for o in existing]})
        if ctx.config.read_only:
            raise ValueError("the server runs with --read-only")
        from mace.gui import history
        if history.root() is None:
            history.configure(ctx.root)
        history.snapshot(existing, "Replaced by mace mcp calc_run", route="mcp")
    if confirm:
        args.append("--force")
    return _calc_json(ctx.run_cli(args, timeout=None))
