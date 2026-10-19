"""Copy Diagnostics / Report a Problem / ``mace gui diagnostics`` (plan
PLAN_connect D20): one redacted bundle for a bug report.

The server part: MACE (and its git commit in a checkout), Python, the OS,
how the server runs (plain, through mace gui connect, in a SLURM job, in a
VS Code terminal, over ssh), cores, memory, pool workers, cache, uptime, the
CPU-time limit, package versions (read with importlib.metadata, never
imported), external codes (xtb and its source, CRYSTAL on PATH, the module
system), non-default settings (true/false, numbers and known choices only;
any free text shows as <text>), the last tasks (kind, state, time, first
error line; never titles, which name documents) and the last lines of the
server log. The page adds its own part (browser, WebGL, the last errors,
failed requests as route templates, notifications). Everything goes through
redact.py (rules 0-6) field by field and once more as a whole.

``mace gui diagnostics [--dir DIR]`` prints the server part with no server
running (standard library and importlib.metadata only).
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .redact import HEADER, Redactor, default_redactor

PACKAGES = ("numpy", "ase", "spglib", "seekpath", "plotly", "pandas", "pyarrow", "scipy", "phonopy", "torch",
            "mace-torch", "chgnet", "matplotlib", "rich", "selenium")
# the settings whose text values are fixed choices (anything else shows as <text>)
ENUMS = {
    "theme": {"system", "light", "dark", "hc-light", "hc-dark", "cvd", "sepia", "black",
              "catppuccin-mocha", "catppuccin-latte", "dracula", "gruvbox-dark", "nord", "one-dark",
              "solarized-dark", "solarized-light", "tokyo-night", "tokyo-night-storm"},
    "density": {"compact", "comfortable"}, "plots.surface": {"theme", "light", "dark"},
    "plots.style": {"default", "publication", "vesta", "presentation", "grayscale", "colorblind"},
    **{f"plots.style.{f}": {"same", "default", "publication", "vesta", "presentation", "grayscale", "colorblind"}
       for f in ("band", "spectra", "maps", "scf")},
    "plots.dosColors": {"auto", "total", "element", "orbital", "element-orbital"},
    "plots.spinDownDash": {"dash", "dot", "dashdot", "solid"}, "plots.spinDos": {"auto", "mirrored", "overlaid"},
    "plots.spinColors": {"auto", "mace", "blue-red", "black-vermilion", "custom"},
    "plots.bandStyle": {"auto", "single", "overlay", "ink"},
    "reduceMotion": {"system", "on", "off"}, "colorScheme": {"jmol", "vesta", "cvd", "legacy"},
    "radiusSet": {"atomic", "covalent", "ionic", "vdw"}, "bondColoring": {"split", "uniform"},
    "background": {"theme", "white", "black", "custom"}, "labels": {"off", "element", "index", "both", "wyckoff"},
    "projection": {"orthographic", "perspective"}, "viewport.defaultView": {"standard", "a", "c"},
    "quality": {"auto", "low", "medium", "high"}, "coordMode": {"frac", "cart"}, "lengthUnit": {"Å", "bohr"},
    "energyUnit": {"eV", "meV", "Ha", "kJ/mol"}, "symmetryModeDefault": {"sym", "p1"},
    "saveFormat": {"cif", "poscar", "xyz"}, "gpu_device": {"none", *[str(i) for i in range(16)]},
}


def settings_summary(values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (values or {}).items():
        if isinstance(v, bool) or isinstance(v, (int, float)) or v is None:
            out[k] = v
        elif isinstance(v, str):
            out[k] = v if v in ENUMS.get(k, ()) else "<text>"
        elif isinstance(v, list):
            out[k] = f"<list of {len(v)}>"
        else:
            out[k] = "<value>"
    return out


def _git(repo: Path) -> Optional[str]:
    if not (repo / ".git").exists():
        return None
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return (sha + ("+dirty" if dirty else "")) if sha else None


def _mem_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 2 ** 20, 1)
    except OSError:
        pass
    return None


def _os_line() -> str:
    s = f"{platform.system()} {platform.release()} {platform.machine()}"
    if sys.platform.startswith("linux"):
        libc = platform.libc_ver()
        if libc[0]:
            s += f", {libc[0]} {libc[1]}"
    elif sys.platform == "darwin":
        s += f", macOS {platform.mac_ver()[0]}"
    return s


def _mode(state=None) -> str:
    parts = []
    if state is not None and getattr(state, "managed", None) is not None:
        parts.append("managed (mace gui connect)")
    else:
        parts.append("plain mace gui" if state is not None else "no server (mace gui diagnostics)")
    if os.environ.get("SLURM_JOB_ID"):
        parts.append("in a SLURM job")
    if os.environ.get("TERM_PROGRAM") == "vscode" or os.environ.get("VSCODE_IPC_HOOK_CLI"):
        parts.append("VS Code terminal")
    parts.append("ssh session: " + ("yes" if os.environ.get("SSH_CONNECTION") else "no"))
    return "; ".join(parts)


def packages() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
        except Exception:
            out[name] = None
    return out


def external_codes(state=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    xt = None
    if state is not None:
        with state.clean_methods_lock:
            meth = (state.clean_methods or {}).get("methods") or []
        g = next((m for m in meth if m.get("id") == "gfnff"), None)
        if g:
            xt = {"status": g.get("status"), "version": g.get("version"), "source": g.get("source")}
    else:
        try:
            from mace.preopt import detect
            st = detect.xtb_status()
            xt = {"status": st.get("status"), "version": st.get("version"), "source": st.get("source")}
        except Exception as exc:
            xt = {"status": f"not checked ({type(exc).__name__})"}
    out["xtb"] = xt
    import shutil
    out["crystal on PATH"] = "yes" if any(shutil.which(n) for n in ("crystal", "Pcrystal", "MPPcrystal")) else "no"
    try:
        from mace.utils import envmodules
        out["module system"] = envmodules.module_system() or "none"
    except Exception:
        out["module system"] = "unknown"
    return out


def server_part(state=None, root: Optional[str] = None) -> Dict[str, Any]:
    from mace import __version__
    from . import settings as gui_settings
    repo = Path(__file__).resolve().parents[2]
    d: Dict[str, Any] = {
        "MACE": __version__ + (f" ({_git(repo)})" if _git(repo) else ""),
        "Python": platform.python_version(), "OS": _os_line(), "mode": _mode(state),
        "cores": f"{len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'} usable of "
                 f"{os.cpu_count()}",
        "memory GB": _mem_gb(),
    }
    try:
        from . import cpulimit
        lim = cpulimit.limit()
        d["CPU-time limit"] = f"{lim / 60:.0f} min per process" if lim else "none"
        if lim:
            d["CPU used (server)"] = f"{time.process_time() / 60:.1f} min"
    except Exception:
        pass
    if state is not None:
        d["uptime"] = f"{(time.time() - state.started) / 60:.1f} min"
        p = state.tasks.pool_if_started()
        d["pool"] = p.report() if p is not None else {"started": False}
        d["cache MB"] = f"{state.cache_used() / 2 ** 20:.0f} of {state.cache_limit / 2 ** 20:.0f}"
        tasks = []
        with state.tasks.lock:
            recs = [t.record() for t in list(state.tasks.tasks.values())[-20:]]
        for t in recs:
            err = (t.get("error") or "").splitlines()[0] if t.get("error") else ""
            tasks.append({"kind": t.get("kind"), "state": t.get("state"), "elapsed s": t.get("elapsed"),
                          "error": err})
        d["tasks"] = tasks
    d["packages"] = packages()
    d["external codes"] = external_codes(state)
    try:
        d["settings (non-default)"] = settings_summary(gui_settings.load())
    except Exception:
        d["settings (non-default)"] = {}
    from . import diaglog
    d["server log"] = diaglog.tail(200)
    return d


def _fmt(obj: Any, indent: int = 0) -> List[str]:
    pad = "  " * indent
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:")
                out += _fmt(v, indent + 1)
            else:
                out.append(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict):
                out.append(pad + "- " + ", ".join(f"{k}={x}" for k, x in v.items()))
            else:
                out.append(f"{pad}{v}")
    else:
        out.append(f"{pad}{obj}")
    return out


def _clean(obj: Any, r: Redactor) -> Any:
    if isinstance(obj, dict):
        return {r.text(str(k)): _clean(v, r) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v, r) for v in obj]
    if isinstance(obj, str):
        return r.text(obj)
    return obj


def bundle(server: Dict[str, Any], client: Optional[Dict[str, Any]], r: Redactor) -> Dict[str, Any]:
    data = {"server": _clean(server, r)}
    if client:
        data["browser"] = _clean(client, r)
    lines = [HEADER, ""] + ["== Server"] + _fmt(data["server"])
    if client:
        lines += ["", "== Browser"] + _fmt(data["browser"])
    text = "\n".join(lines) + "\n"
    text = r.text(text)
    text, n = r.final(text)
    if n:
        text += f"\n({n} more item{'s were' if n != 1 else ' was'} removed)\n"
    return {"text": text, "json": data}


def for_state(state, client: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    names = set(getattr(state, "names_seen", set()))
    try:
        from . import roots
        scratch = [str(p) for p in roots.scratch_roots()]
    except Exception:
        scratch = []
    secrets = [state.token] + list(state.web.table.keys()) + list(state.web.codes.keys())
    r = default_redactor(token=state.token, nonce=state.prefix.strip("/").split("/")[-1], root=str(state.root),
                         extra_secrets=secrets, names=names, cache=str(state.cache), scratch=scratch)
    if getattr(state, "start_dir", None):
        r.locations.append((str(state.start_dir), "<folder>"))
        r.locations.sort(key=lambda kv: len(kv[0]), reverse=True)
    return bundle(server_part(state), client, r)


def cli_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="mace gui diagnostics",
                                description="Print the MACE GUI diagnostics (versions, OS, packages, settings) for a "
                                            "bug report, with paths, file and folder names, tokens, user and host "
                                            "names removed. Works when the GUI cannot start.")
    p.add_argument("--dir", "-d", default=".", help="the folder you serve (its path and names are removed too)")
    p.add_argument("--json", action="store_true", help="print JSON")
    args = p.parse_args(argv)
    root = str(Path(args.dir).expanduser().resolve())
    names = set()
    try:
        for i, entry in enumerate(os.scandir(root)):
            if i > 2000:
                break
            names.add(entry.name)
    except OSError:
        pass
    r = default_redactor(root=root, names=names)
    out = bundle(server_part(None, root), None, r)
    sys.stdout.write(json.dumps(out["json"], indent=2) + "\n" if args.json else out["text"])
    return 0
