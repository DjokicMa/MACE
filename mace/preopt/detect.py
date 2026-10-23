"""What is installed: the ``xtb`` program and the ML-potential environment.

xtb
    ``MACE_XTB`` if set, else ``xtb`` on PATH, else the newest xtb module
    (6.7 or newer) of this machine's module system, loaded automatically
    (``mace/utils/envmodules.py``: its variables go to xtb's process only);
    ``xtb --version`` (10 s, a private working directory, the child
    environment of ``proc``) must say 6.7.0 or newer: periodic GFN-FF first
    shipped in 6.7.0. Status: ready, too_old, not_found or broken. Kept in
    memory, keyed by the program's path, real path, mtime, size and module
    environment; ``broken`` is never kept.

ML environment
    The interpreter is ``MACE_MLIP_PYTHON`` if set, else the one recorded by
    ``mace mlip setup --python`` in ``$XDG_CONFIG_HOME/mace/mlip.json``, else
    a venv in the default location ``$XDG_DATA_HOME/mace-tool/mlip/venv``
    (found with nothing set or recorded). A probe (``python -I -c ...``, no
    model loaded) imports torch, ase and the backend packages and prints
    their versions as JSON. Only a probe where everything installed imported
    is cached, in ``$XDG_CACHE_HOME/mace/preopt/
    detect.json`` for 7 days, keyed by the interpreter path as configured
    (not resolved: every venv on one base interpreter differs), the venv's
    ``pyvenv.cfg``, the versions of the relevant dist-info directories
    (an uninstall or upgrade changes the key) and the ``LD_LIBRARY_PATH`` the
    interpreter is started with. Failures and timeouts are never cached.
    ``recheck=True`` skips the cache.

Library folders
    An interpreter built from a cluster's module Python (and every venv made
    from one) finds its ``libpython3.x.so`` only through ``LD_LIBRARY_PATH``,
    which that module sets: it does not start in a shell without the module
    ("error while loading shared libraries"), and its standard library's
    extension modules may need more of the module's toolchain (``_ssl`` and
    OpenSSL's ``libssl``). ``find_library_dirs`` starts it with no
    ``LD_LIBRARY_PATH``, imports those extension modules and, while a missing
    library is named,
    looks for it in ``lib``/``lib64`` beside the interpreter's real path, beside
    the ``home`` of its ``pyvenv.cfg`` and in the folders of the current
    ``LD_LIBRARY_PATH``. The folders found are recorded per interpreter in
    mlip.json (``library_path``) by ``mace mlip install``, ``mace mlip setup
    --python`` and a probe that needed them, and every start of that
    interpreter (the probe, the worker, the gated-weights download, pip in an
    existing environment) puts them in front of the child's
    ``LD_LIBRARY_PATH`` (``interpreter_env``); MACE's own environment is never
    changed. Nothing is recorded for an interpreter that starts as it is.

Weights
    ``MACE_MLIP_DIR`` or ``$XDG_DATA_HOME/mace-tool/mlip/<id>/<file>`` (not
    ``~/.cache/mace``, which mace-torch uses for its own downloads). The
    sha256 is checked at fetch time and again whenever the file's size or
    mtime differs from ``<id>/ACCEPTED.json``.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import proc

XTB_MIN = (6, 7, 0)
# On MSU HPCC the xtb module (gfbf/2024a) cannot be loaded beside the Python module MACE
# runs on (gfbf/2023a): Lmod refuses. docs/preopt.md's HPCC notes make a wrapper for MACE_XTB.
XTB_HINT = ('install xtb 6.7 or newer: conda install -c conda-forge "xtb>=6.7", or export MACE_XTB=/path/to/xtb '
            "(on a cluster, a wrapper around its xtb module: see the HPCC notes in docs/preopt.md for an example)")
XTB_TIMEOUT = 10.0
PROBE_TIMEOUT_CLI = 60.0
PROBE_TIMEOUT_GUI = 20.0
CACHE_TTL_S = 7 * 24 * 3600
MIN_PYTHON = (3, 10)
_XTB_CACHE: Dict[Tuple, Dict[str, Any]] = {}

_VERSION = re.compile(r"xtb\s+version\s+(\d+)\.(\d+)\.(\d+)")


def parse_xtb_version(text: str) -> Optional[Tuple[int, int, int]]:
    m = _VERSION.search(text or "")
    return tuple(int(x) for x in m.groups()) if m else None


def xtb_path() -> Optional[str]:
    """MACE_XTB (as given), else xtb on PATH, else the binary of an xtb
    module this machine offers (loaded automatically; see xtb_status)."""
    env = os.environ.get("MACE_XTB")
    if env:
        p = os.path.abspath(os.path.expanduser(env))
        return p if os.path.isfile(p) and os.access(p, os.X_OK) else None
    found = shutil.which("xtb")
    if found:
        return found
    return xtb_status().get("path")


def xtb_hint(status: Optional[Dict[str, Any]] = None) -> str:
    """XTB_HINT, plus what this machine's module system has (computed, never
    a hard-coded module name; nothing extra without a module system)."""
    extra = (status or {}).get("module_hint") or ""
    return XTB_HINT + (f"; {extra}" if extra else "")


def _probe_version(path: str, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """{status, version, path, reason, hint} for one xtb binary (uncached)."""
    hint = XTB_HINT
    wd = proc.private_dir("xtbver")
    base = {**os.environ, **(extra_env or {})}
    try:
        p = subprocess.Popen([path, "--version"], cwd=str(wd), env=proc.child_env(base, OMP_NUM_THREADS="1"),
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            out, _ = p.communicate(timeout=XTB_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.stop(p)
            return {"status": "broken", "version": None, "path": path,
                    "reason": f"xtb --version gave no answer within {XTB_TIMEOUT:g} s", "hint": hint}
    except OSError as exc:
        return {"status": "broken", "version": None, "path": path, "reason": f"xtb could not start: {exc}",
                "hint": hint}
    finally:
        proc.remove(wd)
    text = out.decode(errors="replace")
    ver = parse_xtb_version(text)
    if p.returncode != 0 or ver is None:
        why = f"exit code {p.returncode}" if p.returncode != 0 else "no version in its output"
        return {"status": "broken", "version": None, "path": path, "reason": f"xtb --version failed ({why})",
                "hint": hint}
    vs = ".".join(map(str, ver))
    if ver < XTB_MIN:
        return {"status": "too_old", "version": vs, "path": path,
                "reason": f"Found xtb {vs}; GFN-FF for periodic cells needs 6.7+", "hint": hint}
    return {"status": "ready", "version": vs, "path": path, "reason": "", "hint": ""}


def _probe_cached(path: str, recheck: bool, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), os.path.realpath(path), st.st_mtime_ns, st.st_size,
               tuple(sorted((extra_env or {}).items())))
    except OSError:
        key = None
    if key and not recheck and key in _XTB_CACHE:
        return dict(_XTB_CACHE[key])
    res = _probe_version(path, extra_env)
    if key and res["status"] != "broken":
        _XTB_CACHE[key] = dict(res)
    return res


def xtb_status(recheck: bool = False) -> Dict[str, Any]:
    """{status, version, path, reason, hint, source, module, env}.

    MACE_XTB (always wins, never overridden) -> xtb on PATH when it is 6.7+
    -> the newest xtb module >= 6.7 this machine offers, loaded automatically
    (mace/utils/envmodules.py; its variables go to xtb's process only) -> the
    best found (too_old / broken / not_found, as before)."""
    if os.environ.get("MACE_XTB"):
        path = os.path.abspath(os.path.expanduser(os.environ["MACE_XTB"]))
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            return {"status": "not_found", "version": None, "path": None, "source": "MACE_XTB", "module": None,
                    "env": None, "reason": f"Not found: MACE_XTB={os.environ['MACE_XTB']} is not an executable file",
                    "hint": XTB_HINT}
        return {**_probe_cached(path, recheck), "source": "MACE_XTB", "module": None, "env": None}
    on_path = shutil.which("xtb")
    best: Optional[Dict[str, Any]] = None
    if on_path:
        best = {**_probe_cached(on_path, recheck), "source": "PATH", "module": None, "env": None}
        if best["status"] == "ready":
            return best
    from mace.utils import envmodules
    info: Dict[str, Any] = {}
    try:
        info = envmodules.discover("xtb", "xtb", XTB_MIN, use_cache=not recheck)
    except Exception:                       # discovery never breaks detection
        info = {}
    if info.get("binary"):
        res = {**_probe_cached(info["binary"], recheck, info.get("env")), "source": "module",
               "module": info["module"], "env": dict(info.get("env") or {})}
        if res["status"] == "ready":
            return res
        best = best or res
    module_hint = envmodules.hint("xtb", info, XTB_MIN, "GFN-FF for periodic cells") if info else ""
    if best is None:
        best = {"status": "not_found", "version": None, "path": None, "source": None, "module": None, "env": None,
                "reason": "Not found: xtb was not found on PATH" + (" or as a module" if info.get("system") else "")}
    if best["status"] != "ready":
        best["module_hint"] = module_hint
        best["hint"] = xtb_hint(best)
    return best


# --------------------------------------------------------------------------- ML environment

def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "mace" / "mlip.json"


def cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "mace" / "preopt" / "detect.json"


def weights_root() -> Path:
    env = os.environ.get("MACE_MLIP_DIR")
    if env:
        return Path(os.path.expanduser(env))
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "mace-tool" / "mlip"


DEFAULT_ENV = "default"


def env_name(model: Optional[Dict[str, Any]] = None) -> str:
    """The environment a model runs in: ``default`` (mace-torch, CHGNet, ORB-v3,
    MatterSim share it), or its registry ``env`` (e.g. ``fairchem``, whose
    pins conflict with mace-torch's)."""
    return str((model or {}).get("env") or DEFAULT_ENV)


def default_env_dir(env: str = DEFAULT_ENV) -> Path:
    """Where MACE looks for an ML environment when none is set or recorded:
    ``$XDG_DATA_HOME/mace-tool/mlip/venv`` (``venv-<env>`` for a named one),
    beside the default weights folder (MACE_MLIP_DIR moves the weights, not
    this)."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "mace-tool" / "mlip" / ("venv" if env == DEFAULT_ENV else f"venv-{env}")


def venv_python(directory) -> Optional[str]:
    """``directory``'s interpreter when it is there (not resolved: a venv's
    bin/python is a link to its base interpreter)."""
    d = Path(directory)
    for rel in (("bin", "python"), ("Scripts", "python.exe")):
        p = d.joinpath(*rel)
        if p.is_file() and os.access(p, os.X_OK):
            return os.path.abspath(str(p))
    return None


def default_python(env: str = DEFAULT_ENV) -> Optional[str]:
    """The default environment's interpreter, when one is there."""
    return venv_python(default_env_dir(env))


def _read_config() -> Dict[str, Any]:
    try:
        with open(config_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def python_source(env: str = DEFAULT_ENV) -> Tuple[Optional[str], str]:
    """(interpreter, where it came from): MACE_MLIP_PYTHON (the default
    environment) or MACE_MLIP_PYTHON_<ENV>, then the record of ``mace mlip
    setup --python`` (``python``, or ``envs.<env>``), then an environment in
    default_env_dir(env)."""
    var = "MACE_MLIP_PYTHON" if env == DEFAULT_ENV else f"MACE_MLIP_PYTHON_{env.upper()}"
    val = os.environ.get(var)
    if val:
        return os.path.abspath(os.path.expanduser(val)), var
    data = _read_config()
    p = data.get("python") if env == DEFAULT_ENV else (data.get("envs") or {}).get(env)
    if isinstance(p, str) and p:
        return os.path.abspath(p), f"recorded in {config_path()}"
    p = default_python(env)
    return (p, "the default location") if p else (None, "")


def configured_python(model: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The interpreter ``model`` runs in (no model: the default environment)."""
    return python_source(env_name(model))[0]


def _write_config(data: Dict[str, Any]) -> Path:
    cp = config_path()
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, cp)
    return cp


def save_python(path: str, env: str = DEFAULT_ENV) -> Path:
    data = _read_config()
    if env == DEFAULT_ENV:
        data["python"] = os.path.abspath(path)
    else:
        envs = data.get("envs") if isinstance(data.get("envs"), dict) else {}
        envs[env] = os.path.abspath(path)
        data["envs"] = envs
    return _write_config(data)


# --------------------------------------------------------------------------- library folders

_SO_MISSING = re.compile(r"([^\s:'\"]+\.so[^\s:'\"]*): cannot open shared object file")
LIB_TIMEOUT = 20.0
# the standard library's extension modules a model's packages import (ssl for downloads, ctypes, ...): a
# module Python's may need further folders of its toolchain (MSU HPCC's 3.11 module: OpenSSL 1.1's libssl)
STDLIB_CHECK = r"""
import importlib, sys
for name in ("_ssl", "_hashlib", "_ctypes", "_sqlite3", "_bz2", "_lzma", "zlib", "_uuid", "_decimal", "pyexpat"):
    try:
        importlib.import_module(name)
    except ImportError as exc:
        if "cannot open shared object file" in str(exc):
            sys.stderr.write("MACE_STDLIB_LIB " + str(exc).splitlines()[0] + "\n")
            sys.exit(3)
"""
#: folders found in this process (an unwritable config still gets them used here)
_LIBDIRS_MEM: Dict[str, List[str]] = {}


def recorded_library(python: str) -> Dict[str, Any]:
    """{dirs, module} recorded in mlip.json for this interpreter (as configured, not resolved), or {}."""
    rec = _read_config().get("library_path")
    ent = rec.get(os.path.abspath(python)) if isinstance(rec, dict) else None
    return ent if isinstance(ent, dict) else {}


def library_dirs(python: str) -> List[str]:
    """The folders put in front of LD_LIBRARY_PATH when MACE starts ``python``
    (those that exist): found in this process, else recorded."""
    py = os.path.abspath(python)
    dirs = _LIBDIRS_MEM.get(py)
    if dirs is None:
        dirs = recorded_library(py).get("dirs") or []
    return [d for d in dirs if isinstance(d, str) and os.path.isdir(d)]


def remember_library_dirs(python: str, dirs: List[str]) -> List[str]:
    """Use ``dirs`` for ``python`` in this process (not recorded); returns them."""
    _LIBDIRS_MEM[os.path.abspath(python)] = list(dirs)
    return list(dirs)


def with_library_dirs(env: Dict[str, str], dirs: List[str]) -> Dict[str, str]:
    """``env`` with ``dirs`` in front of its LD_LIBRARY_PATH (each once)."""
    if not dirs or os.name == "nt":
        return env
    old = [p for p in (env.get("LD_LIBRARY_PATH") or "").split(":") if p]
    env["LD_LIBRARY_PATH"] = ":".join(list(dict.fromkeys(dirs)) + [p for p in old if p not in dirs])
    return env


def interpreter_env(python: str, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The child environment (``env``, default ``proc.child_env()``) for
    starting ``python``: its library folders in front of LD_LIBRARY_PATH."""
    return with_library_dirs(dict(proc.child_env() if env is None else env), library_dirs(python))


def _starts(python: str, dirs: List[str], timeout: float) -> Tuple[bool, str]:
    """Does ``python`` start, and load its standard library's extension
    modules, with LD_LIBRARY_PATH set to ``dirs`` only (none: unset)?"""
    env = proc.child_env(LD_LIBRARY_PATH=":".join(dirs) if dirs else None)
    try:
        r = subprocess.run([python, "-I", "-S", "-c", STDLIB_CHECK], env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"it did not start within {timeout:g} s"
    except OSError as exc:
        return False, f"it could not start: {exc}"
    return r.returncode == 0, r.stderr or f"exit code {r.returncode}"


def _library_candidates(python: str) -> List[str]:
    """lib and lib64 beside the interpreter's real path and beside its venv's
    ``home``, then the folders of the current LD_LIBRARY_PATH."""
    py = os.path.abspath(python)
    homes = [os.path.dirname(os.path.realpath(py))]
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(py)), "pyvenv.cfg"), errors="replace") as fh:
            m = re.search(r"^home\s*=\s*(.+)$", fh.read(), re.M)
        if m:
            homes += [m.group(1).strip(), os.path.realpath(m.group(1).strip())]
    except OSError:
        pass
    out: List[str] = []
    for h in homes:
        for sub in ("lib", "lib64"):
            d = os.path.join(os.path.dirname(h), sub)
            if d not in out and os.path.isdir(d):
                out.append(d)
    for d in (os.environ.get("LD_LIBRARY_PATH") or "").split(":"):
        if d and d not in out and os.path.isdir(d):
            out.append(d)
    return out


def find_library_dirs(python: str, timeout: float = LIB_TIMEOUT) -> Dict[str, Any]:
    """{ok, dirs, reason[, missing]}: the folders ``python`` needs on
    LD_LIBRARY_PATH to start, and to load its standard library's extension
    modules, in a shell with no module loaded ([] when it does as it is).
    ``ok`` is False when it does not start and no folder helped; ``missing``
    names a library still not found (``ok`` stays True when only an
    extension module lacks it: the folders found are still needed)."""
    if os.name == "nt":
        return {"ok": True, "dirs": [], "reason": ""}
    dirs: List[str] = []
    cands = _library_candidates(python)
    for _ in range(12):
        ok, err = _starts(python, dirs, timeout)
        if ok:
            return {"ok": True, "dirs": dirs, "reason": ""}
        started = "MACE_STDLIB_LIB" in err
        m = _SO_MISSING.search(err)
        if not m:
            tail = " | ".join(err.strip().splitlines()[-2:])
            return {"ok": False, "dirs": dirs, "reason": f"{python} does not start: {tail}"}
        lib = m.group(1)
        hit = next((d for d in cands if d not in dirs and os.path.exists(os.path.join(d, lib))), None)
        if hit is None:
            where = ", ".join(cands) if cands else "a lib folder beside it"
            if started:
                return {"ok": True, "dirs": dirs, "missing": lib,
                        "reason": f"its standard library needs {lib}, which is not in {where}"}
            return {"ok": False, "dirs": dirs, "missing": lib, "reason": f"{python} needs {lib}, which is not in {where}"}
        dirs.append(hit)
    return {"ok": False, "dirs": dirs, "reason": f"{python} still does not start with {':'.join(dirs)}"}


def _module_of(python: str) -> Optional[str]:
    """The loaded module (EasyBuild's EBROOT<NAME> and LOADEDMODULES) whose
    root holds ``python``'s base interpreter, or None: for the hint only."""
    real = os.path.realpath(python)
    for k, v in os.environ.items():
        if not k.startswith("EBROOT") or not v:
            continue
        if not real.startswith(os.path.realpath(v).rstrip("/") + "/"):
            continue
        name, ver = k[len("EBROOT"):].lower(), os.environ.get("EBVERSION" + k[len("EBROOT"):], "")
        for mod in (os.environ.get("LOADEDMODULES") or "").split(":"):
            if mod.lower().startswith(name + "/") and ver in mod:
                return mod
    return None


def save_library_dirs(python: str, dirs: List[str], module: Optional[str] = None) -> Optional[Path]:
    """Record (``dirs``) or forget (no ``dirs``) an interpreter's library
    folders in mlip.json; None when there was nothing to change."""
    py = os.path.abspath(python)
    data = _read_config()
    rec = data.get("library_path") if isinstance(data.get("library_path"), dict) else {}
    if dirs:
        rec[py] = {"dirs": list(dirs), **({"module": module} if module else {})}
    elif py in rec:
        del rec[py]
    else:
        return None
    if rec:
        data["library_path"] = rec
    else:
        data.pop("library_path", None)
    return _write_config(data)


def record_library(python: str, timeout: float = LIB_TIMEOUT) -> Dict[str, Any]:
    """Find ``python``'s library folders and record them (or forget a stale
    record when it now starts as it is): find_library_dirs's result plus
    ``module`` and ``recorded`` (the mlip.json written, or None)."""
    py = os.path.abspath(python)
    res = find_library_dirs(py, timeout)
    res["recorded"] = None
    if not res["ok"]:
        return res
    old = recorded_library(py)
    _LIBDIRS_MEM[py] = list(res["dirs"])
    res["module"] = (_module_of(py) or old.get("module")) if res["dirs"] else None
    if old.get("dirs", []) == res["dirs"] and old.get("module") == res["module"]:
        return res
    try:
        path = save_library_dirs(py, res["dirs"], res["module"])
        res["recorded"] = str(path) if path else None
    except OSError:
        pass                               # an unwritable config: this process still uses them
    return res


PROBE = r"""
import json, sys, importlib, importlib.metadata as md
out = {"python": "%d.%d.%d" % sys.version_info[:3], "prefix": sys.prefix, "packages": {}}
def dist(name):
    try:
        d = md.distribution(name)
        return d.version, str(d.locate_file(""))
    except Exception:
        return None, None
for name, mod in (("torch", "torch"), ("ase", "ase"), ("mace-torch", "mace.calculators"), ("chgnet", "chgnet.model"),
                  ("orb-models", "orb_models.forcefield"), ("mattersim", "mattersim.forcefield"),
                  ("fairchem-core", "fairchem.core")):
    ver, loc = dist(name)
    rec = {"version": ver, "ok": False}
    if ver is None:
        rec["error"] = "not installed"
    else:
        try:
            m = importlib.import_module(mod)
            rec["ok"] = True
            rec["file"] = getattr(m, "__file__", None)
            if name == "mace-torch":
                files = {str(md.distribution("mace-torch").locate_file(f)) for f in (md.distribution("mace-torch").files or [])}
                rec["owned"] = rec["file"] in files
                if not rec["owned"]:
                    rec["ok"] = False
                    rec["error"] = "package name clash: 'mace' is not mace-torch's"
        except Exception as exc:
            rec["error"] = (type(exc).__name__ + ": " + str(exc)).splitlines()[0][:300]
    out["packages"][name] = rec
print("MACE_PROBE_JSON " + json.dumps(out))
"""

_NEEDS = {"mace_mp": ("torch", "ase", "mace-torch"), "chgnet": ("torch", "ase", "chgnet"),
          "orb": ("torch", "ase", "orb-models"),
          "mattersim": ("torch", "ase", "mattersim"), "fairchem": ("torch", "ase", "fairchem-core")}
_DIST_NAMES = ("mace_torch", "chgnet", "torch", "ase", "orb_models", "mattersim", "fairchem_core")


def _site_dirs(prefix: str) -> list:
    return sorted(glob.glob(os.path.join(prefix, "lib", "python3.*", "site-packages")))


def cache_key(python: str) -> Optional[str]:
    """The probe cache key, or None when nothing identifies the environment."""
    py = os.path.abspath(python)
    prefix = os.path.dirname(os.path.dirname(py))
    parts = [py]
    cfg = os.path.join(prefix, "pyvenv.cfg")
    sites = list(_site_dirs(prefix))
    if os.path.isfile(cfg):
        try:
            raw = open(cfg, "rb").read()
            parts.append(hashlib.sha256(raw).hexdigest())
            parts.append(str(os.stat(cfg).st_mtime_ns))
            text = raw.decode(errors="replace")
            if re.search(r"include-system-site-packages\s*=\s*true", text, re.I):
                m = re.search(r"^home\s*=\s*(.+)$", text, re.M)
                if m:
                    sites += _site_dirs(os.path.dirname(m.group(1).strip()))
        except OSError:
            pass
    found = set()
    for s in sites:
        for name in _DIST_NAMES:
            for d in glob.glob(os.path.join(s, f"{name}-*.dist-info")):
                base = os.path.basename(d)[:-len(".dist-info")]
                n, _, v = base.partition("-")
                found.add((s, n.lower(), v))
    if not found:
        return None
    parts += [f"{s}|{n}|{v}" for s, n, v in sorted(found)]
    # a module-built interpreter finds its libpython through the library path it starts with: a probe
    # with that module's folders says nothing about a start without them (a cluster gateway)
    parts.append("LD_LIBRARY_PATH=" + with_library_dirs({"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")},
                                                        library_dirs(py))["LD_LIBRARY_PATH"])
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _read_cache() -> Dict[str, Any]:
    try:
        with open(cache_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(data: Dict[str, Any]) -> None:
    try:
        cp = cache_path()
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, cp)
    except OSError:
        pass                       # unwritable cache: the result stays in memory only


_MEM: Dict[str, Dict[str, Any]] = {}


def probe(python: str, timeout: float = PROBE_TIMEOUT_CLI, recheck: bool = False) -> Dict[str, Any]:
    """{status: ok|missing|failed|timeout, data, reason}. ``ok`` means every
    package was looked at (some may be absent: see data.packages).

    A probe that fails on a missing shared library looks for the library's
    folder (find_library_dirs) and, when one makes the interpreter start,
    probes again with it and records it when that probe answers."""
    if not (os.path.isfile(python) and os.access(python, os.X_OK)):
        return {"status": "missing", "reason": f"{python} is not an executable file"}
    key = cache_key(python)
    if key and not recheck:
        if key in _MEM:
            return _MEM[key]
        ent = _read_cache().get(key)
        if isinstance(ent, dict) and time.time() - float(ent.get("t", 0)) < CACHE_TTL_S and "data" in ent:
            res = {"status": "ok", "data": ent["data"], "reason": "", "cached": True}
            _MEM[key] = res
            return res
    res = _probe_run(python, timeout, key)
    if res["status"] == "failed" and "shared librar" in res["reason"] and os.name != "nt":
        py = os.path.abspath(python)
        before = library_dirs(py)
        found = find_library_dirs(py)
        if found["ok"] and found["dirs"] and found["dirs"] != before:
            had = _LIBDIRS_MEM.get(py)
            _LIBDIRS_MEM[py] = list(found["dirs"])
            again = _probe_run(python, timeout, cache_key(python))
            if again["status"] == "ok":
                record_library(py)
                return dict(again, library_dirs=list(found["dirs"]))
            if had is None:
                _LIBDIRS_MEM.pop(py, None)
            else:
                _LIBDIRS_MEM[py] = had
        elif not found["ok"] and found.get("missing"):
            res["reason"] += f" ({found['reason']})"
    return res


def _probe_run(python: str, timeout: float, key: Optional[str]) -> Dict[str, Any]:
    wd = proc.private_dir("probe")
    try:
        try:
            p = subprocess.Popen([python, "-I", "-c", PROBE], cwd=str(wd),
                                 env=interpreter_env(python, proc.child_env(
                                     OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                                     TMPDIR=str(wd))),
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 start_new_session=True)
        except OSError as exc:
            return {"status": "failed", "reason": f"the ML interpreter could not start: {exc}"}
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.stop(p)
            return {"status": "timeout",
                    "reason": f"the ML environment did not answer within {timeout:g} s (a cold start on a slow "
                              "filesystem?); press Recheck or run `mace mlip list --recheck`"}
    finally:
        proc.remove(wd)
    text = out.decode(errors="replace")
    m = re.search(r"^MACE_PROBE_JSON (.+)$", text, re.M)
    if p.returncode != 0 or not m:
        tail = " | ".join(err.decode(errors="replace").strip().splitlines()[-3:])
        return {"status": "failed", "reason": f"the probe failed (exit code {p.returncode})" + (f": {tail}" if tail else "")}
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return {"status": "failed", "reason": "the probe printed something that is not JSON"}
    res = {"status": "ok", "data": data, "reason": ""}
    # cached when every installed package imported (one not installed is part of the key: installing it
    # adds its dist-info directory, which changes the key)
    if key and all(r.get("ok") or r.get("version") is None for r in data.get("packages", {}).values()):
        cache = _read_cache()
        cache[key] = {"t": time.time(), "data": data}
        _write_cache(cache)
        _MEM[key] = res
    return res


def weights_file(model: Dict[str, Any]) -> Optional[Path]:
    w = model.get("weights") or {}
    if "url" not in w:
        return None
    return weights_root() / model["id"] / w["filename"]


def record_dir(model: Dict[str, Any]) -> Path:
    """Where a model's license records live (ACCEPTED.json, ACADEMIC.json):
    beside its weights, also for gated weights that live in the Hugging Face
    cache."""
    return weights_root() / model["id"]


def academic_ack(model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The user's acknowledgment of academic, non-commercial use of an
    academic-only model (ACADEMIC.json), or None."""
    try:
        rec = json.loads((record_dir(model) / "ACADEMIC.json").read_text())
    except (OSError, ValueError):
        return None
    if isinstance(rec, dict) and rec.get("id") == model["id"] and rec.get("acknowledged_utc"):
        return rec
    return None


def _gated_status(model: Dict[str, Any]) -> Dict[str, Any]:
    """Gated weights (UMA): fetched into the Hugging Face cache with the
    user's own access; ACCEPTED.json records the terms accepted and the path."""
    w = model["weights"]
    try:
        rec = json.loads((record_dir(model) / "ACCEPTED.json").read_text())
    except (OSError, ValueError):
        rec = {}
    path = rec.get("path")
    if not rec or not path:
        return {"status": "needs_fetch", "path": None, "sha256": None,
                "reason": f"weights not downloaded (gated: accept the terms at {model['gated']['page']} with your "
                          f"Hugging Face account, then mace mlip fetch {model['id']})"}
    try:
        size = os.path.getsize(path)
    except OSError:
        return {"status": "needs_fetch", "path": path, "sha256": None,
                "reason": f"the weights are no longer in the Hugging Face cache: mace mlip fetch {model['id']}"}
    if size != int(w["size_bytes"]) or rec.get("sha256") != w["sha256"]:
        return {"status": "bad_hash", "path": path, "sha256": rec.get("sha256"),
                "reason": f"the cached weights are not the registry's: mace mlip fetch {model['id']}"}
    return {"status": "ready", "path": path, "sha256": w["sha256"]}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def weights_status(model: Dict[str, Any]) -> Dict[str, Any]:
    """{status: ready|needs_fetch|needs_ack|bad_hash|bundled|missing, path, sha256}."""
    user = (model.get("weights") or {}).get("user_file")
    if user:                                   # mace calc --model-file: read where it is, never fetched
        if not os.path.isfile(user):
            return {"status": "missing", "path": user, "sha256": None, "reason": f"the model file {user} is not there"}
        return {"status": "ready", "path": user, "sha256": model["weights"].get("sha256")}
    if model.get("academic") and academic_ack(model) is None:
        return {"status": "needs_ack", "path": None, "sha256": None,
                "reason": f"{model['label']} is for academic, non-commercial use only (its weights are under the "
                          f"{model['weights']['license']}); it is fetched and run only after you acknowledge that: "
                          f"mace mlip fetch {model['id']} --accept-academic-license"}
    if model.get("gated"):
        return _gated_status(model)
    path = weights_file(model)
    if path is None:
        return {"status": "bundled", "path": None, "sha256": None}
    w = model["weights"]
    if not path.is_file():
        return {"status": "needs_fetch", "path": str(path), "sha256": None,
                "reason": f"weights not downloaded: mace mlip fetch {model['id']}"}
    rec_path = path.parent / "ACCEPTED.json"
    st = path.stat()
    try:
        rec = json.loads(rec_path.read_text())
    except (OSError, ValueError):
        rec = {}
    if rec.get("sha256") == w["sha256"] and rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns:
        return {"status": "ready", "path": str(path), "sha256": w["sha256"]}
    digest = sha256_file(path)
    if digest != w["sha256"]:
        return {"status": "bad_hash", "path": str(path), "sha256": digest,
                "reason": f"the weights file's sha256 is {digest[:12]}..., not the registry's "
                          f"{w['sha256'][:12]}...: fetch it again (mace mlip fetch {model['id']})"}
    if rec:
        rec.update(size=st.st_size, mtime_ns=st.st_mtime_ns)
        try:
            rec_path.write_text(json.dumps(rec, indent=2))
        except OSError:
            pass
    return {"status": "ready", "path": str(path), "sha256": digest}


_COMMAND = re.compile(r'conda install -c conda-forge "xtb>=6\.7"|module load \S+?(?=,|;|$|\s)|export MACE_XTB=\S+?(?=,|;|$|\s)'
                      r'|mace mlip setup --python \S+?(?=,|;|$|\s)|mace mlip fetch \S+?(?=,|;|$|\s)'
                      r'|mace mlip setup(?= prints)|[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.]*')


def hint_commands(hint: str) -> List[str]:
    """The commands inside a hint, in order, each one a whole line to copy
    (the hint itself is prose around them); a pin becomes its pip install."""
    out = []
    for m in _COMMAND.finditer(hint or ""):
        c = m.group(0)
        if "==" in c and not c.startswith(("mace ", "conda ")):
            c = f"DIR/bin/pip install {c}"
        if c not in out:
            out.append(c)
    return out


def setup_hint(model: Dict[str, Any]) -> str:
    return ("set up an ML environment (mace mlip setup prints how), then record it: "
            "mace mlip setup --python DIR/bin/python" +
            (f"; then mace mlip fetch {model['id']}" if "url" in (model.get("weights") or {}) else ""))


def failed_probe_hint(python: str, reason: str) -> str:
    """The hint for an interpreter (set, recorded or in the default place)
    that did not run here: it is set up, so the setup steps are not the answer."""
    if "shared librar" in (reason or ""):
        mod = recorded_library(python).get("module")
        what = ("a shared library of its Python is missing here and was not found beside the interpreter: "
                "load the module its Python came from" + (f" ({mod})" if mod else
                                                           " (the Python module the environment was made with)")
                + f", then record it again: mace mlip setup --python {python}; that records its library "
                  "folder, and MACE then starts it without the module")
    else:
        what = "repair that environment (the reason says what failed)"
    return (f"the ML environment ({python}) did not run here; {what}, or record another: "
            "mace mlip setup --python DIR/bin/python")


def mlip_status(model: Dict[str, Any], recheck: bool = False, timeout: float = PROBE_TIMEOUT_CLI) -> Dict[str, Any]:
    """{status, reason, hint, python, versions, weights_path, weights_sha256}."""
    py = configured_python(model)
    base = {"python": py, "versions": {}, "weights_path": None, "weights_sha256": None, "env": env_name(model)}
    if model.get("academic") and academic_ack(model) is None:      # the opt-in comes first, before any setup
        ws = weights_status(model)
        return {**base, "status": "needs_ack", "reason": ws["reason"],
                "hint": f"mace mlip fetch {model['id']} --accept-academic-license"}
    if not py:
        return {**base, "status": "needs_setup", "reason": "no ML environment is set up",
                "hint": setup_hint(model)}
    pr = probe(py, timeout=timeout, recheck=recheck)
    if pr["status"] == "timeout":
        return {**base, "status": "probe_timeout", "reason": pr["reason"], "hint": setup_hint(model)}
    if pr["status"] == "failed":
        return {**base, "status": "import_failed", "reason": pr["reason"],
                "hint": failed_probe_hint(py, pr["reason"])}
    if pr["status"] != "ok":
        return {**base, "status": "needs_setup", "reason": pr["reason"], "hint": setup_hint(model)}
    data = pr["data"]
    pyv = tuple(int(x) for x in str(data.get("python", "0.0.0")).split(".")[:2])
    if pyv < MIN_PYTHON:
        return {**base, "status": "unsupported_python",
                "reason": f"the ML interpreter is Python {data.get('python')}; 3.10 or newer is needed",
                "hint": setup_hint(model)}
    pk = data.get("packages", {})
    versions = {k: v.get("version") for k, v in pk.items() if v.get("version")}
    versions["python"] = data.get("python")
    base["versions"] = versions
    for name in _NEEDS[model["backend"]]:
        rec = pk.get(name) or {}
        if not rec.get("ok"):
            reason = f"{name}: {rec.get('error') or 'not importable'}"
            if "cannot open shared object file" in reason:      # a library of the module's toolchain
                return {**base, "status": "import_failed", "reason": reason,
                        "hint": failed_probe_hint(py, "shared library: " + reason)}
            return {**base, "status": "import_failed", "reason": reason,
                    "hint": f"install it in the ML environment: {model['code']['pin']}"}
    ws = weights_status(model)
    base["weights_path"], base["weights_sha256"] = ws.get("path"), ws.get("sha256")
    if ws["status"] == "missing":
        return {**base, "status": "missing", "reason": ws["reason"], "hint": ""}
    if ws["status"] == "needs_ack":
        return {**base, "status": "needs_ack", "reason": ws["reason"],
                "hint": f"mace mlip fetch {model['id']} --accept-academic-license"}
    if ws["status"] in ("needs_fetch", "bad_hash"):
        return {**base, "status": ws["status"], "reason": ws["reason"],
                "hint": f"mace mlip fetch {model['id']}"}
    return {**base, "status": "ready", "reason": "", "hint": ""}


def python_version_ok() -> bool:
    return sys.version_info >= (3, 8)
