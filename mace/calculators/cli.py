"""``mace calc``: force fields, tight binding, charges and ML potentials.

    mace calc list [--json] [--recheck]
    mace calc setup [uff|eqeq|xtb|dftb|mlip]
    mace calc sp|opt|freq|md INPUT -b BACKEND [--method M] [options]
    mace calc sp|opt|freq|md INPUT -b mlip --model-file PATH --model-backend B --model-license-ok
    mace calc charges INPUT [-b eqeq|xtb|dftb] [options]

Writes OUTDIR/<stem>_<backend>-<method>_<task>.json (the report) and, by
task: the optimized structure (.cif in the input's space group when it kept
it, else P1; .xyz for a molecule) and its trajectory; the MD trajectory
(.traj and .extxyz) and a CSV log; the normal modes (.extxyz); a P1 CIF
with ``_atom_site_charge`` for charges.

Exit codes: 0 done; 3 done but not converged or stopped early (the partial
result was written); 1 could not run or failed (nothing written); 2 usage.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

TASKS = ("sp", "opt", "freq", "md", "charges")
SETUP = {
    "uff": "UFF and UFF4MOF are built in: nothing to install.",
    "eqeq": "EQeq is built in: nothing to install.",
    "xtb": ("xTB (GFN2-xTB, GFN1-xTB, GFN-FF) runs the external xtb program, 6.7 or newer (LGPL-3.0):\n"
            '  conda install -c conda-forge "xtb>=6.7"\n'
            "or export MACE_XTB=/path/to/xtb (on a cluster, a wrapper around its xtb module: see docs/preopt.md).\n"
            "Optional, GFN1/GFN2 in-process: the tblite Python package (conda install -c conda-forge tblite-python),\n"
            "used with --engine tblite."),
    "dftb": ("DFTB+ runs the external dftb+ program (LGPL-3.0):\n"
             "  conda install -c conda-forge dftbplus      (or export MACE_DFTB=/path/to/dftb+)\n"
             "The skf method needs a Slater-Koster set; MACE never ships or downloads one. Get one from\n"
             "https://dftb.org/parameters (3ob, mio, matsci, pbc, ...; each has its own license and citation),\n"
             "unpack it and export MACE_DFTB_SKF=/path/to/set (or pass --skf-dir). The gfn2/gfn1 methods use\n"
             "DFTB+'s own xTB Hamiltonian (DFTB+ 22 or newer) and need no files."),
    "mlip": ("ML potentials run in their own Python environment, never in MACE's. MACE-MP is ACEsuit's\n"
             "mace-torch package, which is unrelated to this MACE tool.\n"
             "  mace mlip setup                  how to make the environment (mace-torch, orb-models, mattersim)\n"
             "  mace mlip setup --python DIR/bin/python\n"
             "  mace mlip list --all             the models, their licenses and what is missing\n"
             "  mace mlip fetch ID               download one model's weights (sha256-checked)\n"
             "GPU: --device auto (the default) uses GPU 0 when there is one, else the CPU; --device cuda:N or\n"
             "CUDA_VISIBLE_DEVICES picks another GPU, --device cpu never uses one."),
}


_USER_BACKENDS = ("mace-torch", "mattersim")     # mlip.USER_BACKENDS, without importing it for --help


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)        # --forc is not --force (mace mcp passes options on)
        super().__init__(*args, **kwargs)

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="mace calc", description="Force fields, tight binding, charges and ML potentials.")
    sub = p.add_subparsers(dest="cmd", parser_class=_Parser)
    ls = sub.add_parser("list", help="the backends, their methods and what is installed")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("--recheck", action="store_true", help="detect again (skip the caches)")
    st = sub.add_parser("setup", help="how to install a backend")
    st.add_argument("backend", nargs="?")
    for task in TASKS:
        t = sub.add_parser(task, help={"sp": "single point: energy, forces, stress",
                                       "opt": "geometry optimization (atoms, optionally the cell)",
                                       "freq": "frequencies by finite differences, harmonic thermochemistry",
                                       "md": "molecular dynamics (NVE, NVT Langevin, NPT Berendsen)",
                                       "charges": "partial charges (EQeq, or xTB/DFTB+ charges)"}[task])
        t.add_argument("input")
        t.add_argument("-b", "--backend", required=task != "charges", default="eqeq" if task == "charges" else None,
                       help="uff, eqeq, xtb, dftb, mlip (or uff4mof, gfn2, gfn1, gfnff, mlip:MODEL)")
        t.add_argument("-m", "--method", help="uff4mof|uff, gfn2|gfn1|gfnff, skf|gfn2|gfn1, an ML model id")
        t.add_argument("-o", "--output-dir", dest="outdir")
        t.add_argument("--force", action="store_true", help="overwrite existing output files")
        t.add_argument("--json", action="store_true", help="print the report as JSON on stdout")
        t.add_argument("--frame", type=int, default=-1, help="which frame of a CRYSTAL output or trajectory")
        t.add_argument("--device", default="auto", help="auto (GPU 0 when there is one), cpu or cuda:N (ML only)")
        t.add_argument("--threads", type=int)
        t.add_argument("--charge", type=int, default=None, help="total charge (default 0; OrbMol-v2 and UMA omol "
                                                                 "read the deck or output when it is not given)")
        t.add_argument("--uhf", type=int, default=None, help="unpaired electrons (xTB; ML: multiplicity - 1)")
        t.add_argument("--multiplicity", type=int, default=None, help="spin multiplicity 2S+1 (OrbMol-v2, UMA omol)")
        t.add_argument("--task-head", default=None, metavar="HEAD",
                       help="UMA's task head: omat, omol, oc20, odac, omc (default by dimensionality)")
        t.add_argument("--time-limit", type=float, help="seconds of wall clock for the whole run")
        t.add_argument("--debug", action="store_true", help="show the Python traceback when the calculation fails")
        if task == "opt":
            t.add_argument("--fmax", type=float, default=0.02, help="eV/Å (default 0.02)")
            t.add_argument("--steps", type=int, default=500)
            t.add_argument("--relax-cell", action="store_true")
            t.add_argument("--keep-symmetry", action="store_true", help="FixSymmetry: keep the space group")
            t.add_argument("--optimizer", default="fire", choices=("fire", "bfgs", "lbfgs"))
        if task == "freq":
            t.add_argument("--displacement", type=float, default=0.01, help="Å (default 0.01)")
        if task in ("freq", "md"):
            t.add_argument("--temperature", type=float, default=298.15 if task == "freq" else 300.0, help="K")
        if task == "md":
            t.add_argument("--ensemble", default="nvt", choices=("nve", "nvt", "npt"))
            t.add_argument("--timestep", type=float, default=1.0, help="fs (default 1)")
            t.add_argument("--md-steps", type=int, default=1000)
            t.add_argument("--friction", type=float, default=0.01, help="1/fs, Langevin (default 0.01)")
            t.add_argument("--pressure", type=float, default=1.01325, help="bar, NPT (default 1 atm)")
            t.add_argument("--interval", type=int, default=10, help="steps between saved frames")
            t.add_argument("--seed", type=int)
        g = t.add_argument_group("backend options")
        g.add_argument("--kpts", help="DFTB+: a k-point grid, e.g. 4,4,4 (default from --kpoint-density)")
        g.add_argument("--kpoint-density", type=float, help="DFTB+: n_i = ceil(D / |a_i|), D in Å (default 20)")
        g.add_argument("--skf-dir", help="DFTB+ skf: the Slater-Koster set (else MACE_DFTB_SKF)")
        g.add_argument("--engine", choices=("xtb", "tblite"), help="xTB: the xtb program (default) or tblite")
        g.add_argument("--uff-charges", choices=("none", "eqeq"), help="UFF: fixed point charges (default none)")
        g.add_argument("--cutoff", type=float, help="UFF: van der Waals and Coulomb cutoff, Å (default 12.5)")
        g.add_argument("--charge-centers", help="EQeq: e.g. Zn=2,Cu=1 (defaults: the paper's)")
        g.add_argument("--eqeq-method", choices=("auto", "ewald", "direct", "nonperiodic"))
        g.add_argument("--gpu-mem-fraction", type=float, help="ML: share of the GPU's memory (default 0.5)")
        g.add_argument("--dtype", help="ML: float32 or float64 (default: the model's)")
        if task != "charges":
            g.add_argument("--model-file", metavar="PATH",
                           help="ML: a model file you already have (with -b mlip); read where it is, never copied")
            g.add_argument("--model-backend", choices=tuple(_USER_BACKENDS),
                           help="ML: the package that loads --model-file")
            g.add_argument("--model-license-ok", action="store_true",
                           help="ML: you hold --model-file's license and it allows this use (MACE cannot check it)")
        g.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                       help="any other backend option (repeatable)")
    return p


def _err(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")


def _options(args) -> dict:
    o = {}
    for key, name in (("kpts", "kpts"), ("kpoint_density", "kpoint_density"), ("skf_dir", "skf_dir"),
                      ("engine", "engine"), ("uff_charges", "charges"), ("cutoff", "cutoff"),
                      ("charge_centers", "charge_centers"), ("eqeq_method", "eqeq_method"),
                      ("gpu_mem_fraction", "gpu_mem_fraction"), ("dtype", "dtype")):
        v = getattr(args, key, None)
        if v is not None:
            o[name] = v
    for kv in args.option or []:
        k, sep, v = kv.partition("=")
        if not sep or not k.strip():
            raise ValueError(f"--option {kv!r}: use KEY=VALUE")
        try:
            v = json.loads(v)
        except ValueError:
            pass
        o[k.strip()] = v
    return o


def cmd_list(args) -> int:
    from . import registry
    rows = [be.describe(recheck=args.recheck) for be in registry.all_backends()]
    from . import device
    gpus = device.gpus()
    if args.json:
        print(json.dumps({"backends": rows, "gpus": gpus, "device_default": device.default_text()}, indent=2,
                         default=str))
        return 0
    print("mace calc backends (the ML potentials are third-party models, unrelated to this MACE tool)")
    print(f"Device for ML potentials: {device.default_text()}")
    for r in rows:
        print()
        print(f"  {r['name']}: {r['label']}  [{', '.join(r['tasks'])}]")
        print(f"      license: {r['license']}")
        shown = set()
        for m in r["methods"]:
            mark = " (default)" if m.get("default") else ""
            st = "ready" if m.get("available") else (m.get("reason") or m.get("status") or "not available")
            print(f"    {r['name']} -m {m['id']}{mark}")
            print(f"        {m['label']}; {st}" + (f" ({m['version']})" if m.get("version") and m.get("available") else ""))
            if not m.get("available") and m.get("hint") and m["hint"] not in shown:
                shown.add(m["hint"])
                print(f"        next: {m['hint']}")
    return 0


def cmd_setup(args) -> int:
    if not args.backend:
        for k, v in SETUP.items():
            print(f"{k}:\n" + "\n".join("  " + ln for ln in v.splitlines()) + "\n")
        return 0
    from . import registry
    name, _ = registry.resolve(args.backend)
    if name not in SETUP:
        _err(f"mace calc setup: no backend {args.backend!r}; known: {', '.join(SETUP)}")
        return 2
    print(SETUP[name])
    return 0


def cmd_task(args) -> int:
    from . import base, registry, structio, tasks
    user_file = getattr(args, "model_file", None)
    if user_file is not None and not args.model_license_ok:
        _err("mace calc: error: --model-file needs --model-license-ok, your statement that you hold the model's "
             "license and that it allows this use. MACE cannot check a model it does not offer.")
        return 2
    try:
        name, method = registry.resolve(args.backend, args.method)
        if user_file is not None:
            if name != "mlip" or method:
                raise ValueError("--model-file runs with -b mlip and no -m/--method")
            if not args.model_backend:
                raise ValueError("--model-file needs --model-backend " + "|".join(_USER_BACKENDS)
                                 + " (the package that loads it)")
            from . import mlip
            m = mlip.user_model(user_file, args.model_backend)
            method = m["id"]
            _err(f"mace calc: user-supplied model {m['weights']['user_file']} (loaded by {args.model_backend}). "
                 "MACE does not offer, download or copy it; its license terms are your responsibility.")
        elif getattr(args, "model_backend", None) or getattr(args, "model_license_ok", False):
            raise ValueError("--model-backend and --model-license-ok go with --model-file")
        be = registry.get(name)
        if args.multiplicity is not None and args.uhf is not None and args.multiplicity != args.uhf + 1:
            raise ValueError("--multiplicity and --uhf disagree (multiplicity = uhf + 1)")
        uhf = args.uhf if args.uhf is not None else (args.multiplicity - 1 if args.multiplicity is not None else None)
        if uhf is not None and uhf < 0:
            raise ValueError("--multiplicity is 1 or more (--uhf 0 or more)")
        spec = base.CalcSpec(backend=name, method=method, task=args.cmd, device=args.device, threads=args.threads,
                             charge=args.charge or 0, uhf=uhf or 0, charge_set=args.charge is not None,
                             uhf_set=uhf is not None, time_limit=args.time_limit, options=_options(args))
        if args.task_head:
            spec.options["task_head"] = args.task_head
        be.method(spec.method)
    except (base.CalcError, ValueError) as exc:
        _err(f"mace calc: error: {exc}")
        return 2
    for k in ("fmax", "steps", "relax_cell", "keep_symmetry", "optimizer", "displacement", "temperature",
              "ensemble", "timestep", "md_steps", "friction", "pressure", "interval", "seed"):
        if hasattr(args, k) and getattr(args, k) is not None:
            setattr(spec, k, getattr(args, k))
    for k in ("fmax", "displacement", "timestep", "time_limit"):
        v = getattr(spec, k)
        if v is not None and not v > 0:
            _err(f"mace calc: error: --{k.replace('_', '-')} must be positive")
            return 2
    if spec.steps < 1 or spec.md_steps < 1 or spec.interval < 1 or (spec.threads is not None and spec.threads < 1):
        _err("mace calc: error: step counts and --threads must be at least 1")
        return 2
    if args.device != "auto":
        from mace.preopt import mlip_client
        try:
            mlip_client.device_mapping(args.device)
        except ValueError as exc:
            _err(f"mace calc: error: {exc}")
            return 2
    src = Path(args.input)
    try:
        atoms, sg, kinds = structio.read(src, args.frame)
    except base.CalcError as exc:
        _err(f"mace calc: {exc}")
        return 1
    outdir = Path(args.outdir) if args.outdir else src.resolve().parent
    stem = src.stem
    tty = sys.stderr.isatty()
    last = [0.0]

    def progress(frac, text):
        now = time.monotonic()
        if tty and (now - last[0] > 0.5 or frac >= 1.0):
            last[0] = now
            sys.stderr.write(f"\r  {text[:64]:64s} {100 * frac:5.1f}%")
            if frac >= 1.0:
                sys.stderr.write("\n")
            sys.stderr.flush()

    try:
        r = tasks.run(atoms, spec, progress=progress)
    except base.CalcError as exc:
        if tty:
            sys.stderr.write("\n")
        if getattr(args, "debug", False):
            import traceback
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        _err(f"mace calc: {exc}")
        if exc.hint:
            _err(f"next: {exc.hint}")
        return 2 if exc.kind == "usage" else 1
    except (KeyboardInterrupt, base.Canceled):
        _err("mace calc: interrupted; nothing was written")
        return 130
    try:
        files = tasks.write_outputs(r, outdir, stem, spec, force=args.force, spacegroup=sg, kinds=kinds)
    except FileExistsError as exc:
        _err(f"mace calc: {exc}")
        return 1
    say = sys.stderr.write if args.json else sys.stdout.write
    say(f"Ran: {r.summary()}\n")
    if r.report.get("device_text"):
        say(f"Device: {r.report['device_text']}\n")
    if r.task == "freq" and r.frequencies is not None:
        f = r.frequencies
        say("Frequencies (cm^-1): " + " ".join(f"{x:.1f}" for x in f[:12]) + (" ..." if len(f) > 12 else "") + "\n")
        if r.thermo.get("zpe_eV") is not None:
            say(f"ZPE {r.thermo['zpe_eV']:.4f} eV" + (f"; F_vib({r.thermo['temperature_K']:g} K) "
                                                       f"{r.thermo['vib_free_energy_eV']:.4f} eV"
                                                       if "vib_free_energy_eV" in r.thermo else "") + "\n")
    if r.charges is not None and len(r.charges) <= 40:
        say("Charges: " + " ".join(f"{s}{i + 1} {q:+.3f}" for i, (s, q) in
                                   enumerate(zip(r.atoms.get_chemical_symbols(), r.charges))) + "\n")
    for w in r.warnings:
        say(f"warning: {w}\n")
    for w in r.report.get("typing_warnings") or []:
        say(f"warning: UFF typing: {w}\n")
    say("Wrote " + ", ".join(str(outdir / v) for v in files.values()) + "\n")
    if r.task == "opt" and "structure" in files:
        say("This is a force-field/ML geometry: run a CRYSTAL OPT from it before SP, FREQ, BAND or DOSS.\n")
    cites = r.provenance.get("citation") or []
    if cites:
        say("Cite: " + "; ".join(c["text"] for c in cites) + "\n")
    if args.json:
        sys.stdout.write(json.dumps(r.as_dict(full=False), indent=2, default=tasks._json_default) + "\n")
    if r.stopped or (r.task == "opt" and not r.converged):
        return 3
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = _parser()
    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 2
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "setup":
        return cmd_setup(args)
    return cmd_task(args)


if __name__ == "__main__":
    sys.exit(main())
