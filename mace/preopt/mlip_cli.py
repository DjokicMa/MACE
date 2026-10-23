"""``mace mlip {list,setup,install,fetch}``: the ML-potential environment and weights.

``list`` shows the pre-optimization models (Clean, ``mace preopt``); ``list
--all`` adds the models ``mace calc`` offers (MACE-MP-0b3 from ACEsuit's
mace-torch, ORB-v3, MatterSim). ``fetch`` takes any of them.

MACE installs an ML environment and downloads weights only when asked:
``install`` (mlip_install.py, the GUI's installer too) makes the venv after
showing the plan, the licenses and the size; ``setup`` prints the same steps
to run by hand (in the default place, ``detect.default_env_dir()``, an
environment is found unrecorded) and records one that passes the probe;
``fetch`` downloads only after the license, size and citation are shown and
accepted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

HEADER = "ML potentials are third-party models, unrelated to this MACE tool."

SETUP_TEXT = """\
`mace mlip install` makes the ML environment for you: these same steps,
with the space and quota checked first, then recorded (and the GUI's
Settings > ML potentials runs it too). To make one by hand: in MACE's
default place MACE finds it with nothing to set or record:

     DIR = {default_dir}

An environment anywhere else works too: record it with
`mace mlip setup --python DIR/bin/python` (the record wins over the default).

1. A Python that already has torch (check: python -c "import torch"):

     python -m venv --system-site-packages DIR
     DIR/bin/python -m pip install --upgrade pip
     DIR/bin/pip install {mace_pin} {chgnet_pin}        (about 80 MB)

   For `mace calc`, the same environment can also hold ORB-v3 and MatterSim
   (they need torch 2.8 or newer and numpy 2):

     DIR/bin/pip install {orb_pin} {ms_pin}

2. No torch available: install a CPU-only torch first (several hundred MB;
   a plain `pip install torch` pulls a CUDA build of several GB):

     python3 -m venv DIR
     DIR/bin/python -m pip install --upgrade pip
     DIR/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
     DIR/bin/pip install {mace_pin} {chgnet_pin}

   In all about 1.5 GB and 46,000 files (half of each is torch): check a
   file-count quota too. The pip upgrade matters: an older pip (22.x, as some
   Python modules ship) fails on torch's current dependencies.

On a cluster whose Python comes from a module (module load Python/...):
run the pip lines after `unset PYTHONPATH`. The module's PYTHONPATH makes pip
skip packages the module already has, and MACE runs the environment without
PYTHONPATH, so they would be missing. DIR's python also needs that module's
libpython: `mace mlip setup --python` records its folder, and MACE then starts
DIR's python with it, also where the module is not loaded.

Put DIR on a filesystem with room (check with `df -h DIR`), then:

     mace mlip setup --python DIR/bin/python   (checks and records it; optional in the default place)
     mace mlip fetch {default_id}       (the MACE-MPA-0 weights, about 79 MB)
     mace mlip list                     (where the environment was found, what is ready)

CHGNet's weights ship inside the chgnet package: nothing to fetch.
ORB-v3 and OrbMol-v2 (orb-models 0.7) need Python 3.12 or newer.
UMA (Meta's fairchem-core) pins torch 2.13 and e3nn 0.5, against mace-torch:
it gets its own environment, {default_dir}-fairchem
(mace mlip install --backend fairchem-core; by hand, the steps above with
`DIR/bin/pip install 'torch==2.13.0' --index-url ...` and fairchem-core, then
mace mlip setup --python DIR/bin/python --env fairchem). Its weights are gated
on Hugging Face: accept the terms at https://huggingface.co/facebook/UMA with
your own account and log in (hf auth login, or HF_TOKEN) before
mace mlip fetch meta-uma-s-1p1.
"""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mace mlip", description=HEADER, allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd")
    ls = sub.add_parser("list", help="the models, their licenses and whether they are ready")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("--recheck", action="store_true", help="probe the ML environment again (skip the cache)")
    ls.add_argument("--all", action="store_true", help="also the models of mace calc (MACE-MP-0b3, ORB-v3, MatterSim, "
                                                     "UMA, OrbMol-v2) and the academic-only ones")
    ls.add_argument("--academic", action="store_true", help="also the academic-only models (MACE-OFF; opt-in)")
    st = sub.add_parser("setup", help="print how to make an ML environment, or record one")
    st.add_argument("--python", metavar="PATH", help="the ML environment's interpreter to record")
    st.add_argument("--env", default="default", metavar="NAME",
                    help="the environment it is (default; fairchem for UMA, whose pins conflict with mace-torch)")
    it = sub.add_parser("install", help="make the ML environment: a venv, CPU torch, the pinned packages, recorded")
    it.add_argument("--dir", metavar="DIR", help="where (default: $XDG_DATA_HOME/mace-tool/mlip/venv)")
    it.add_argument("--backend", default="mace-torch", metavar="LIST",
                    help="comma-separated packages: mace-torch (default), chgnet, orb-models, mattersim")
    it.add_argument("--gpu", action="store_true", help="torch with CUDA (only when an NVIDIA GPU answers)")
    it.add_argument("--python", metavar="PATH", help="the Python 3.10+ to build from (default: the one running MACE, "
                                                     "else the newest python3.1x on PATH)")
    g = it.add_mutually_exclusive_group()
    g.add_argument("--fetch", metavar="ID", help="then fetch this model's weights (default: MACE-MPA-0 with mace-torch)")
    g.add_argument("--no-fetch", action="store_true", help="do not fetch weights")
    it.add_argument("--yes", action="store_true", help="install, and accept the weights' license, without asking")
    it.add_argument("--accept-academic-license", action="store_true",
                    help="with --fetch of an academic-only model: acknowledge academic, non-commercial use")
    it.add_argument("--dry-run", action="store_true", help="show the plan and the equivalent commands; change nothing")
    ft = sub.add_parser("fetch", help="download a model's weights (after showing the license)")
    ft.add_argument("id")
    ft.add_argument("--yes", action="store_true", help="accept the license and download without asking")
    ft.add_argument("--dir", metavar="DIR", help="store the weights here (then export MACE_MLIP_DIR=DIR)")
    ft.add_argument("--accept-academic-license", action="store_true",
                    help="acknowledge academic, non-commercial use (academic-only models such as MACE-OFF; at a "
                         "terminal you also type ACADEMIC)")
    for q in sub.choices.values():
        q.allow_abbrev = False                          # full option names only
    return p


def cmd_list(args) -> int:
    from . import detect, registry
    rows = []
    wide = getattr(args, "all", False)
    for m in registry.load(scope=None if wide else "preopt", include_academic=wide or getattr(args, "academic", False)):
        st = detect.mlip_status(m, recheck=args.recheck)
        w = m["weights"]
        rows.append({"id": m["id"], "method": f"mlip:{m['id']}", "label": m["label"],
                     "code_license": m["code"]["license"], "weights_license": w["license"],
                     "status": st["status"], "reason": st.get("reason", ""), "hint": st.get("hint", ""),
                     "size_mb": round(w["size_bytes"] / 1e6, 1) if "size_bytes" in w else 0.0,
                     "default": bool(m.get("default")), "fallback": bool(m.get("fallback")),
                     "python": st.get("python"), "family": m.get("family"), "academic": bool(m.get("academic")),
                     "gated": bool(m.get("gated")), "env": st.get("env"),
                     "scopes": list(m.get("scopes", registry.SCOPES))})
    py, source = detect.python_source()
    if args.json:
        print(json.dumps({"note": HEADER, "python": py, "python_source": source,
                          "default_env_dir": str(detect.default_env_dir()), "models": rows}, indent=2))
        return 0
    print(HEADER)
    print()
    print(f"ML environment: {py} ({source})" if py else
          f"ML environment: none (MACE looks in {detect.default_env_dir()}; `mace mlip setup` prints how)")
    print()
    for r in rows:
        mark = " (default)" if r["default"] else " (fallback)" if r["fallback"] else ""
        if r["scopes"] == ["calc"]:
            mark += " [mace calc only]"
        if r["academic"]:
            mark += " [ACADEMIC, NON-COMMERCIAL USE ONLY: opt-in]"
        if r["gated"]:
            mark += " [gated: your own Hugging Face access]"
        size = f"{r['size_mb']:.0f} MB weights" if r["size_mb"] else "weights in the package"
        print(f"  mlip:{r['id']}{mark}")
        print(f"      {r['label']}; code {r['code_license']}, weights {r['weights_license']}; {size}")
        print(f"      status: {r['status']}" + (f" - {r['reason']}" if r["reason"] else ""))
        if r["hint"]:
            print(f"      next: {r['hint']}")
    return 0


def cmd_setup(args) -> int:
    from . import detect, registry
    if not args.python:
        mpa, chg = registry.get(registry.default_id()), registry.get(registry.fallback_id())
        orb = next((m for m in registry.load(scope=None) if m["backend"] == "orb"), None)
        ms = next((m for m in registry.load(scope=None) if m["backend"] == "mattersim"), None)
        print(SETUP_TEXT.format(mace_pin=mpa["code"]["pin"], chgnet_pin=chg["code"]["pin"],
                                default_id=mpa["id"], default_dir=detect.default_env_dir(),
                                orb_pin=orb["code"]["pin"] if orb else "orb-models",
                                ms_pin=ms["code"]["pin"] if ms else "mattersim"))
        return 0
    from . import mlip_install as MI
    py = os.path.abspath(os.path.expanduser(args.python))
    print(f"Probing {py} (up to {detect.PROBE_TIMEOUT_CLI:g} s on a cold start)...")
    r = MI.record_python(py, env=args.env)
    for name, ver, status in r["rows"]:
        print(f"  {name:11s} {ver:12s} {status}")
    if not r["ok"]:
        print(f"Not recorded: {r['reason']}")
        return 1
    print(f"Recorded in {r['path']}.")
    if r.get("library_dirs"):
        print(f"It needs {':'.join(r['library_dirs'])} on LD_LIBRARY_PATH (its Python's shared library): recorded, "
              "and MACE adds it whenever it starts this interpreter.")
    if r.get("library_note"):
        print(f"Note: {r['library_note']}.")
    if r["override"]:
        print(f"Note: {r['override']} names another interpreter and takes precedence over the recorded one.")
    return 0


def _print_plan(plan) -> None:
    print(f"Install into: {plan['dir']}" + {"existing": " (an environment already: repaired or upgraded in place)",
                                             "empty": " (an empty folder)", "new": ""}[plan["state"]])
    if plan["python"]:
        print(f"Built from:   {plan['python']} (Python {plan['python_version']}, {plan['python_why']})")
    print(f"torch:        {'CUDA build' if plan['torch'] == 'gpu' else 'CPU build'}"
          + (" (an NVIDIA GPU answers here: --gpu for the CUDA build)" if plan["torch"] == "cpu" and plan["gpu_present"] else ""))
    print("Packages:")
    for lic in plan["licenses"]:
        print(f"  {lic['pin']:24s} code {lic['license']}")
        for m in lic["models"]:
            print(f"      {m['label']}: weights {m['weights_license']}" + (" (academic use only)" if m["academic"] else ""))
    size = f"about {plan['estimate_mb'] / 1000:.1f} GB and {plan['estimate_files']:,} files"
    fb = plan["free_bytes"]
    free = "" if fb is None else (f"; {fb / 1e12:.0f} TB free" if fb >= 1e13 else f"; {fb / 1e9:.0f} GB free")
    q = plan["quota"]
    quota = (f"; quota {q['used_gb']:.0f}/{q['quota_gb']:.0f} GB, {q['files']:,}/{q['files_quota']:,} files"
             if q.get("available") else "")
    print(f"Size:         {size}{free}{quota}")
    for w in plan["warnings"]:
        print(f"warning: {w}")
    for pr in plan["problems"]:
        print(f"cannot install: {pr}")


def cmd_install(args) -> int:
    from . import mlip_install as MI, registry
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    try:
        plan = MI.make_plan(args.dir, backends, gpu=args.gpu, python=args.python)
    except MI.InstallError as exc:
        print(f"mace mlip install: {exc}", file=sys.stderr)
        return 2
    print(HEADER)
    print()
    _print_plan(plan)
    if plan["problems"]:
        return 1
    fetch_id = None
    if not args.no_fetch:
        fetch_id = args.fetch or (registry.default_id() if "mace-torch" in plan["backends"] else None)
    if fetch_id and registry.get(fetch_id, scope=None) is None:
        print(registry.unknown(fetch_id, scope=None), file=sys.stderr)
        return 2
    if fetch_id:
        print(f"Then:         fetch the weights of {fetch_id} (its license is shown first)")
    if args.dry_run:
        print("\nThe same steps by hand:")
        for c in plan["commands"]:
            print(f"  {c}")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("stdin is not a terminal: pass --yes to install.", file=sys.stderr)
            return 2
        if input("Install? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing was installed.")
            return 2
    print()
    try:
        res = MI.install(plan, log=lambda s: print(s, flush=True), fetch_id=fetch_id, accept_license=args.yes,
                         accept_academic=args.accept_academic_license)
    except MI.InstallCanceled as exc:
        print(f"mace mlip install: {exc}", file=sys.stderr)
        return 130
    except MI.InstallError as exc:
        print(f"mace mlip install: {exc}", file=sys.stderr)
        return 1
    print(f"\nThe ML environment is ready: {res['python']}. `mace mlip list` shows the models.")
    if res["fetch"] and res["fetch"]["rc"] != 0:
        print(f"The weights were not fetched; the environment is installed. Fetch them later: mace mlip fetch {fetch_id}")
        return 3
    return 0


def cmd_fetch(args) -> int:
    from . import fetch, registry
    m = registry.get(args.id, scope=None)
    if m is None:
        print(registry.unknown(args.id, scope=None), file=sys.stderr)
        return 2
    rc = fetch.fetch(m, yes=args.yes, directory=args.dir, accept_academic=args.accept_academic_license)
    if rc == 0 and args.dir:
        print(f"MACE looks for weights in MACE_MLIP_DIR: export MACE_MLIP_DIR={os.path.abspath(args.dir)}")
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    p = _parser()
    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 2
    return {"list": cmd_list, "setup": cmd_setup, "fetch": cmd_fetch, "install": cmd_install}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
