"""``mace mlip {list,setup,fetch}``: the ML-potential environment and weights.

MACE never installs an ML environment and never downloads weights on its
own: ``setup`` prints how to make an environment and records one that
passes the probe; ``fetch`` downloads only after the license, size and
citation are shown and accepted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

HEADER = "ML potentials are third-party models, unrelated to this MACE tool."

SETUP_TEXT = """\
MACE does not create or install the ML environment; make one yourself, then
record it with `mace mlip setup --python DIR/bin/python`.

1. A Python that already has torch (check: python -c "import torch"):

     python -m venv --system-site-packages DIR
     DIR/bin/pip install {mace_pin} {chgnet_pin}        (about 80 MB)

2. No torch available: install a CPU-only torch first (several hundred MB;
   a plain `pip install torch` pulls a CUDA build of several GB):

     python3 -m venv DIR
     DIR/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
     DIR/bin/pip install {mace_pin} {chgnet_pin}

Put DIR on a filesystem with room (check with `df -h DIR`), then:

     mace mlip setup --python DIR/bin/python
     mace mlip fetch {default_id}       (the MACE-MPA-0 weights, about 79 MB)

CHGNet's weights ship inside the chgnet package: nothing to fetch.
"""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mace mlip", description=HEADER)
    sub = p.add_subparsers(dest="cmd")
    ls = sub.add_parser("list", help="the models, their licenses and whether they are ready")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("--recheck", action="store_true", help="probe the ML environment again (skip the cache)")
    st = sub.add_parser("setup", help="print how to make an ML environment, or record one")
    st.add_argument("--python", metavar="PATH", help="the ML environment's interpreter to record")
    ft = sub.add_parser("fetch", help="download a model's weights (after showing the license)")
    ft.add_argument("id")
    ft.add_argument("--yes", action="store_true", help="accept the license and download without asking")
    ft.add_argument("--dir", metavar="DIR", help="store the weights here (then export MACE_MLIP_DIR=DIR)")
    return p


def cmd_list(args) -> int:
    from . import detect, registry
    rows = []
    for m in registry.load():
        st = detect.mlip_status(m, recheck=args.recheck)
        w = m["weights"]
        rows.append({"id": m["id"], "method": f"mlip:{m['id']}", "label": m["label"],
                     "code_license": m["code"]["license"], "weights_license": w["license"],
                     "status": st["status"], "reason": st.get("reason", ""), "hint": st.get("hint", ""),
                     "size_mb": round(w["size_bytes"] / 1e6, 1) if "size_bytes" in w else 0.0,
                     "default": bool(m.get("default")), "fallback": bool(m.get("fallback")),
                     "python": st.get("python")})
    if args.json:
        print(json.dumps({"note": HEADER, "models": rows}, indent=2))
        return 0
    print(HEADER)
    print()
    for r in rows:
        mark = " (default)" if r["default"] else " (fallback)" if r["fallback"] else ""
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
        print(SETUP_TEXT.format(mace_pin=mpa["code"]["pin"], chgnet_pin=chg["code"]["pin"],
                                default_id=mpa["id"]))
        return 0
    py = os.path.abspath(os.path.expanduser(args.python))
    print(f"Probing {py} (up to {detect.PROBE_TIMEOUT_CLI:g} s on a cold start)...")
    pr = detect.probe(py, timeout=detect.PROBE_TIMEOUT_CLI, recheck=True)
    if pr["status"] != "ok":
        print(f"Not recorded: {pr['reason']}")
        return 1
    pk = pr["data"].get("packages", {})
    for name in ("torch", "ase", "mace-torch", "chgnet"):
        r = pk.get(name, {})
        print(f"  {name:11s} {r.get('version') or '-':12s} {'ok' if r.get('ok') else r.get('error', 'missing')}")
    if not (pk.get("torch", {}).get("ok") and pk.get("ase", {}).get("ok")
            and (pk.get("mace-torch", {}).get("ok") or pk.get("chgnet", {}).get("ok"))):
        print("Not recorded: the environment needs torch, ase and mace-torch or chgnet "
              "(run `mace mlip setup` for the commands).")
        return 1
    path = detect.save_python(py)
    print(f"Recorded in {path}.")
    if os.environ.get("MACE_MLIP_PYTHON"):
        print("Note: MACE_MLIP_PYTHON is set and takes precedence over the recorded interpreter.")
    return 0


def cmd_fetch(args) -> int:
    from . import fetch, registry
    m = registry.get(args.id)
    if m is None:
        print(f"No model {args.id!r}; known: {', '.join(registry.ids())}", file=sys.stderr)
        return 2
    rc = fetch.fetch(m, yes=args.yes, directory=args.dir)
    if rc == 0 and args.dir:
        print(f"MACE looks for weights in MACE_MLIP_DIR: export MACE_MLIP_DIR={os.path.abspath(args.dir)}")
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    p = _parser()
    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 2
    return {"list": cmd_list, "setup": cmd_setup, "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
