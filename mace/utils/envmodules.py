"""Find and load environment modules (Lmod or Tcl modules), generically.

MACE never hard-codes a module name with a version. Two users share this
file: GFN-FF's xtb discovery (``discover``, ``resolve``, ``hint``) and the
job-script generators' CRYSTAL module fallback (``candidates``,
``rank_candidates``, ``qualify``, ``cache_get``/``cache_put``).

When an external program (xtb, ...) is not on PATH, ``discover`` asks the
machine's own module system which versions exist, picks the newest one at or
above a minimum, loads it in a throwaway ``bash`` process and returns only
the variables the load added or changed (PATH, LD_LIBRARY_PATH, ...). The
caller passes those to the child process it starts; MACE's own environment
is never changed.

* No module system (no LMOD_CMD, MODULESHOME, MODULEPATH, exported ``module``
  function or ``/etc/profile.d`` init script; ``MACE_MODULES_NO_PROFILE=1``
  skips the init scripts): nothing runs, no subprocess.
* ``MACE_<CODE>_MODULE=<exact name>`` pins one module;
  ``MACE_<CODE>_MODULE=none`` turns discovery off for that code.
* Results are cached for 7 days in ``$XDG_CACHE_HOME/mace/modules.json``
  (one JSON object; each user writes its own keys, keyed by the code, a hash
  of MODULEPATH and the Lmod version); the binary is checked again before a
  cached entry is used. Failures are never cached.
* When a module cannot be loaded next to the modules already loaded (Lmod
  refuses mixed toolchains), the load is tried once more after ``module
  purge`` in that throwaway process only.
* ``qualify`` checks that a module (list) really provides given programs:
  under ``$EBROOTCRYSTAL/bin`` (EasyBuild, rule Q1) or, for modules that set
  no root, on ``PATH`` in one folder (rule Q2); ``rank_candidates`` puts
  candidates in the order a fallback tries them.

Every shell this module starts is counted (``PROBES``) and, when
``MACE_MODULES_SPY`` names a file, logged there one line per call, so tests
can tell how many module subprocesses ran, also in a child process.

Standard library only: the job-script generators import this on a cluster's
module Python, whose package set is small.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CACHE_TTL_S = 7 * 24 * 3600
AVAIL_TIMEOUT = 60.0
LOAD_TIMEOUT = 120.0
PROFILE_SCRIPTS = ("lmod.sh", "modules.sh", "z00_lmod.sh")
PROFILE_DIRS: Tuple[str, ...] = ("/etc/profile.d",)
_SPLIT = "\0MACE_ENV_SPLIT\0"
_DROP = {"PWD", "OLDPWD", "SHLVL", "_"}
_DROP_PREFIXES = ("BASH_FUNC_", "_ModuleTable", "__LMOD_REF_COUNT", "_LMFILES_", "LOADEDMODULES", "_ModuleTable")

# the last (module, environment) per code, so one process asks once
_MEM: Dict[str, Dict[str, Any]] = {}


def _env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def override_name(name: str) -> str:
    """MACE_<CODE>_MODULE for a module name ("xtb" -> MACE_XTB_MODULE)."""
    return "MACE_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper() + "_MODULE"


def module_system(env: Optional[Mapping[str, str]] = None,
                  profile_dirs: Sequence[str] = PROFILE_DIRS) -> Optional[str]:
    """"lmod", "tcl" or None, from markers only (never a subprocess)."""
    e = _env(env)
    if e.get("LMOD_CMD") or e.get("LMOD_VERSION"):
        return "lmod"
    if any(k.startswith("BASH_FUNC_module") for k in e) or e.get("MODULESHOME") or e.get("MODULEPATH"):
        return "lmod" if "lmod" in (e.get("MODULESHOME", "") + e.get("MODULEPATH", "")).lower() else "tcl"
    if e.get("MACE_MODULES_NO_PROFILE") == "1":
        return None
    for d in profile_dirs:
        for s in PROFILE_SCRIPTS:
            if os.path.isfile(os.path.join(d, s)):
                return "lmod" if "lmod" in s else "tcl"
    return None


def _bash_argv(script: str, env: Mapping[str, str]) -> List[str]:
    bash = shutil.which("bash", path=env.get("PATH")) or shutil.which("bash") or "/bin/bash"
    # an exported module function needs no login shell; otherwise the login
    # profile sets the module system up
    if any(k.startswith("BASH_FUNC_module") for k in env):
        return [bash, "-c", script, "mace-envmodules"]
    return [bash, "-lc", script, "mace-envmodules"]


PROBES = [0]


def _count(script: str, env: Mapping[str, str]) -> None:
    """One more module shell (PROBES), and a line in MACE_MODULES_SPY's file."""
    PROBES[0] += 1
    spy = env.get("MACE_MODULES_SPY")
    if spy:
        try:
            with open(spy, "a") as fh:
                fh.write(script.replace("\n", "; ")[:200] + "\n")
        except OSError:
            pass


def _run(script: str, args: Sequence[str], env: Mapping[str, str], timeout: float) -> Optional[Tuple[int, bytes]]:
    _count(script, env)
    argv = _bash_argv(script, env) + list(args)
    try:
        # a neutral working directory: a folder named like a Python package
        # in the cwd must never shadow anything the login scripts run
        p = subprocess.Popen(argv, env=dict(env), cwd=tempfile.gettempdir(), stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             start_new_session=(os.name == "posix"))
    except OSError:
        return None
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # a login script that waits for input or hangs: the whole group goes
        try:
            if os.name == "posix":
                os.killpg(p.pid, 9)
            else:
                p.kill()
        except OSError:
            pass
        try:
            p.communicate(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return None
    return p.returncode, out


_MARK = re.compile(r"\s*(\((?:default|D|L|[A-Za-z,]+)\)|<[A-Za-z,]+>)\s*$")


def parse_avail(text: str, name: str) -> List[str]:
    """Module names from ``module -t avail <name>`` (terse) output: header
    lines (ending in ':'), hidden modules and (default)-style markers dropped."""
    out: List[str] = []
    want = name.lower() + "/"
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.endswith(":"):
            continue
        line = _MARK.sub("", line).strip()
        if not line or line.startswith(".") or "/." in line:
            continue
        if not line.lower().startswith(want):
            continue
        if line not in out:
            out.append(line)
    return out


def version_key(module: str) -> Tuple[int, ...]:
    """The leading dotted number after ``name/`` as ints: "name/6.10.0" ->
    (6, 10, 0) (newer than 6.7.1); "name/23-intel-2023a" -> (23,)."""
    tail = module.split("/", 1)[1] if "/" in module else module
    m = re.match(r"v?(\d+(?:\.\d+)*)", tail)
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def sort_newest(modules: Iterable[str]) -> List[str]:
    """Newest first: by version, ties by the full name (descending)."""
    return sorted(modules, key=lambda m: (version_key(m), m), reverse=True)


def avail(name: str, *, env: Optional[Mapping[str, str]] = None, timeout: Optional[float] = None) -> Optional[List[str]]:
    """Every visible module of ``name``; None when the module system did not answer."""
    e = _env(env)
    r = _run('module -t avail "$1" 2>&1', [name], e, AVAIL_TIMEOUT if timeout is None else timeout)
    if r is None:
        return None
    return parse_avail(r[1].decode(errors="replace"), name)


def spider(name: str, *, env: Optional[Mapping[str, str]] = None, timeout: Optional[float] = None) -> List[str]:
    """Versions Lmod knows behind a hierarchy (``module -t spider``): for a hint only."""
    e = _env(env)
    r = _run('module -t spider "$1" 2>&1', [name], e, AVAIL_TIMEOUT if timeout is None else timeout)
    return parse_avail(r[1].decode(errors="replace"), name) if r else []


def newest(name: str, minimum: Sequence[int] = (), *, env: Optional[Mapping[str, str]] = None,
           modules: Optional[List[str]] = None) -> Optional[str]:
    mods = avail(name, env=env) if modules is None else modules
    ok = [m for m in (mods or []) if version_key(m) >= tuple(minimum)]
    return sort_newest(ok)[0] if ok else None


def _parse_env0(blob: bytes) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in blob.split(b"\0"):
        if b"=" not in item:
            continue
        k, _, v = item.partition(b"=")
        out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


def _wanted(k: str) -> bool:
    return k not in _DROP and not k.startswith(_DROP_PREFIXES)


def load_env(module: str, *, env: Optional[Mapping[str, str]] = None,
             timeout: Optional[float] = None) -> Optional[Dict[str, str]]:
    """The variables ``module load <module>`` adds or changes, or None when
    it does not load. Runs in a throwaway bash; retries once after ``module
    purge`` there (a toolchain conflict with the modules already loaded)."""
    e = _env(env)
    for purge in (False, True):
        pre = "module purge >/dev/null 2>&1; " if purge else ""
        script = ('env -0; printf "\\0MACE_ENV_SPLIT\\0"; ' + pre +
                  'module load "$1" >/dev/null 2>&1 || exit 3; env -0')
        r = _run(script, [module], e, LOAD_TIMEOUT if timeout is None else timeout)
        if r is None:
            return None                      # timeout: no second try
        rc, blob = r
        if rc != 0 or _SPLIT.encode() not in blob:
            continue
        before_b, _, after_b = blob.partition(_SPLIT.encode())
        before, after = _parse_env0(before_b), _parse_env0(after_b)
        changed = {k: v for k, v in after.items() if _wanted(k) and before.get(k) != v}
        if changed:
            return changed
    return None


def load_modules_env(modules, env: Optional[Mapping[str, str]] = None, *, purge: bool = False,
                     timeout: float = 120.0) -> Optional[Dict[str, str]]:
    """The variables that loading ``modules`` (one name or an ordered list)
    adds or changes, or None when a load fails. ``purge`` runs ``module
    purge`` first, as the job scripts do (the site settings' Python check)."""
    mods = [modules] if isinstance(modules, str) else [m for m in modules if m]
    if module_system(env) is None or not mods or not all(valid_name(m) for m in mods):
        return None
    loads = " && ".join(f"module load {m} >/dev/null 2>&1" for m in mods)
    script = ("printf 'MACE_ENV_BEGIN\\0'; env -0; printf 'MACE_ENV_SPLIT\\0'\n"
              + ("module purge >/dev/null 2>&1\n" if purge else "")
              + f"{loads} || exit 7\nenv -0\n")
    r = _bash(script, dict(env) if env is not None else None, timeout)
    if r is None or r.returncode != 0:
        return None
    try:
        before_txt, after_txt = r.stdout.split("MACE_ENV_SPLIT\0", 1)
        before_txt = before_txt.split("MACE_ENV_BEGIN\0", 1)[1]
    except (ValueError, IndexError):
        return None

    def parse(txt):
        d = {}
        for item in txt.split("\0"):
            if "=" in item:
                k, v = item.split("=", 1)
                d[k] = v
        return d
    before, after = parse(before_txt), parse(after_txt)
    return {k: v for k, v in after.items() if before.get(k) != v and _wanted(k)}


# --------------------------------------------------------------------- cache

def cache_path(env: Optional[Mapping[str, str]] = None) -> Path:
    e = _env(env)
    base = e.get("XDG_CACHE_HOME") or os.path.join(e.get("HOME") or os.path.expanduser("~"), ".cache")
    return Path(base) / "mace" / "modules.json"


def cache_key(name: str, *parts, env: Optional[Mapping[str, str]] = None) -> str:
    """The cache key of ``name`` (plus ``parts``: the CRYSTAL fallback's
    preferred modules and version range) on this module path."""
    e = _env(env)
    mp = hashlib.sha256((e.get("MODULEPATH") or "").encode()).hexdigest()[:16]
    if parts:
        return json.dumps([name, *parts, mp, e.get("LMOD_VERSION") or ""], default=str)
    return f"{name}|{mp}|{e.get('LMOD_VERSION', '')}"


def _read_cache(env) -> Dict[str, Any]:
    try:
        data = json.loads(cache_path(env).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(env, data: Dict[str, Any]) -> None:
    cp = cache_path(env)
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_name(f".{cp.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, cp)
    except OSError:
        pass


def which_in(binary: str, mod_env: Mapping[str, str], env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    path = mod_env.get("PATH") or _env(env).get("PATH")
    return shutil.which(binary, path=path)


def discover(name: str, binary: str, minimum: Sequence[int] = (), *, preferred: Optional[str] = None,
             env: Optional[Mapping[str, str]] = None, use_cache: bool = True) -> Dict[str, Any]:
    """Find, load and remember a module that provides ``binary``.

    Returns {module, env, binary, available, newest_any, spider, source,
    cached, reason}: ``module``/``env``/``binary`` are None when nothing
    suitable loads; ``available`` lists every visible version (for hints)."""
    e = _env(env)
    res: Dict[str, Any] = {"module": None, "env": None, "binary": None, "available": [], "newest_any": None,
                           "spider": [], "cached": False, "reason": "", "system": None}
    pinned = (e.get(override_name(name)) or "").strip()
    if pinned.lower() == "none":
        res["reason"] = f"{override_name(name)}=none"
        return res
    system = module_system(e)
    res["system"] = system
    if system is None:
        res["reason"] = "no module system here"
        return res
    key = cache_key(name, env=e)
    if use_cache and not pinned:
        hit = _MEM.get(key) or _read_cache(e).get(key)
        if isinstance(hit, dict) and time.time() - float(hit.get("t", 0)) < CACHE_TTL_S and hit.get("module"):
            b = hit.get("binary")
            if b and os.path.isfile(b) and os.access(b, os.X_OK) and \
                    (not minimum or version_key(hit["module"]) >= tuple(minimum)):
                _MEM[key] = hit
                return {**res, "module": hit["module"], "env": dict(hit.get("env") or {}), "binary": b,
                        "available": list(hit.get("available") or []), "cached": True}
    if pinned:
        candidates, mods = [pinned], [pinned]
    else:
        mods = avail(name, env=e)
        if mods is None:
            res["reason"] = "the module system did not answer"
            return res
        res["available"] = sort_newest(mods)
        res["newest_any"] = res["available"][0] if mods else None
        if not mods:
            res["spider"] = sort_newest(spider(name, env=e))
        ok = [m for m in sort_newest(mods) if version_key(m) >= tuple(minimum)]
        candidates = ([preferred] if preferred and preferred in mods else []) + [m for m in ok if m != preferred]
    for m in candidates[:3]:
        menv = load_env(m, env=e)
        if menv is None:
            continue
        b = which_in(binary, menv, e)
        if not b:
            continue
        res.update(module=m, env=menv, binary=b)
        if not pinned:
            rec = {"t": time.time(), "module": m, "env": menv, "binary": b, "available": res["available"]}
            _MEM[key] = rec
            data = _read_cache(e)
            data[key] = rec
            _write_cache(e, data)
        return res
    res["reason"] = (f"{override_name(name)}={pinned} did not load" if pinned else
                     "no module loaded" if candidates else "no suitable module")
    return res


def resolve(name: str, preferred: Optional[str] = None, minimum: Sequence[int] = (), *,
            env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``preferred`` when it loads, else the newest module >= minimum that loads."""
    e = _env(env)
    pinned = (e.get(override_name(name)) or "").strip()
    if pinned.lower() == "none" or module_system(e) is None:
        return None
    if pinned:
        return pinned if load_env(pinned, env=e) is not None else None
    if preferred and load_env(preferred, env=e) is not None:
        return preferred
    mods = avail(name, env=e) or []
    for m in sort_newest(m for m in mods if version_key(m) >= tuple(minimum)):
        if m != preferred and load_env(m, env=e) is not None:
            return m
    return None


def fmt_version(v: Sequence[int]) -> str:
    return ".".join(str(x) for x in v)


def hint(name: str, info: Mapping[str, Any], minimum: Sequence[int] = (), feature: str = "") -> str:
    """A computed sentence about this machine's modules (empty without a module system)."""
    if not info or info.get("system") is None:
        return ""
    need = f"{feature or name} needs {fmt_version(minimum)} or newer" if minimum else ""
    avail_ = list(info.get("available") or [])
    if info.get("reason", "").endswith("=none"):
        return ""
    if avail_:
        best = avail_[0]
        if minimum and version_key(best) < tuple(minimum):
            return f"the newest {name} module here is {best}; {need}"
        return f"the {name} module {best} did not load (try: module spider {best})"
    sp = list(info.get("spider") or [])
    if sp:
        return (f"{name} modules exist behind other modules here ({', '.join(sp[:3])}); "
                f"module spider {sp[0]} lists what to load first")
    return ""


# ------------------------------------------------- the CRYSTAL module fallback (submission/site.py)
_MODULE_RE = re.compile(r"^[A-Za-z0-9._+/-]{1,128}$")


def valid_name(name: str) -> bool:
    return bool(_MODULE_RE.fullmatch(name or ""))


def _neutral_cwd() -> str:
    tmp = tempfile.gettempdir()
    return tmp if os.path.isdir(tmp) else "/"


def _bash(script: str, env: Optional[Dict[str, str]] = None, timeout: float = 120.0) -> Optional[subprocess.CompletedProcess]:
    """Run ``script`` in a shell that has the ``module`` function: ``bash -c``
    when the environment already provides it (BASH_ENV or an exported
    function), else a login shell. None on a timeout or an OS error."""
    env = dict(os.environ if env is None else env)
    _count(script, env)
    login = not (env.get("BASH_ENV") or env.get("BASH_FUNC_module%%") or env.get("BASH_FUNC_module()"))
    argv = ["bash", "-lc" if login else "-c", script]
    try:
        return subprocess.run(argv, env=env, cwd=_neutral_cwd(), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_listing(text: str, name: str) -> List[str]:
    out: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.endswith(":"):
            continue
        line = re.sub(r"\s*\((default|D|L|S|E)\)$", "", line).strip()
        line = re.sub(r"\(@?[A-Za-z]+\)$", "", line).strip()
        if line.startswith(".") or "/." in line:
            continue
        if not line.startswith(name + "/") or not valid_name(line):
            continue
        if line not in out:
            out.append(line)
    return out


def _avail_listing(name: str, env: Optional[Dict[str, str]] = None, timeout: float = 60.0) -> List[str]:
    """The visible modules ``name/...`` (``module -t avail``), [] without an answer."""
    if module_system(env) is None or not valid_name(name):
        return []
    r = _bash(f"module -t avail {name} 2>&1", env, timeout)
    return _parse_listing((r.stdout + r.stderr) if r else "", name)


def avail_listing(name: str, env: Optional[Dict[str, str]] = None, timeout: float = 60.0) -> List[str]:
    """The visible modules ``name/...`` (``module -t avail``) only, no spider."""
    return _avail_listing(name, env, timeout)


def candidates(name: str, env: Optional[Dict[str, str]] = None, timeout: float = 60.0) -> List[str]:
    """``avail``, else the Lmod spider listing (modules behind a hierarchy)."""
    found = _avail_listing(name, env, timeout)
    if found or module_system(env) is None:
        return found
    r = _bash(f"module -t spider {name} 2>&1", env, timeout)
    return _parse_listing((r.stdout + r.stderr) if r else "", name)


_VARIANT = re.compile(r"-MPP-|_binary|-binary", re.I)


def _suffix(module: str) -> str:
    """The toolchain part after the version: ``<name>/23-intel-2023a`` -> ``intel-2023a``."""
    tail = module.split("/", 1)[1] if "/" in module else module
    m = re.match(r"\d+(?:\.\d+)*[-_]?(.*)$", tail)
    return (m.group(1) if m else tail).lower()


def _toolchain_year(module: str) -> Tuple[int, str]:
    m = re.findall(r"(20\d\d)([a-z]?)", module)
    return (int(m[-1][0]), m[-1][1]) if m else (0, "")


def rank_candidates(names: Iterable[str], preferred: Optional[str] = None) -> List[str]:
    """The order a fallback tries modules in: plain builds before MPP and
    vendor-binary builds, then the preferred module's toolchain, then the
    higher version, the newer toolchain, and the name (descending) last."""
    pref_suffix = _suffix(preferred) if preferred else None
    names = list(dict.fromkeys(names))
    # five stable sorts, least important first
    names.sort(reverse=True)
    names.sort(key=_toolchain_year, reverse=True)
    names.sort(key=version_key, reverse=True)
    names.sort(key=lambda n: 0 if pref_suffix is not None and _suffix(n) == pref_suffix else 1)
    names.sort(key=lambda n: 1 if _VARIANT.search(n) else 0)
    return names


def _as_list(modules) -> List[str]:
    if isinstance(modules, str):
        return [modules]
    return [m for m in modules if m]


def qualify(modules, binaries: Sequence[str], env: Optional[Dict[str, str]] = None, *, purge: bool = True,
            root_var: str = "EBROOTCRYSTAL", timeout: float = 120.0) -> Tuple[bool, Optional[str], bool, str]:
    """(ok, root, root_from_path, why) for ``modules`` (one name or an ordered
    list): after ``module purge`` (``purge``) and loading them in order,
    either ``$<root_var>/bin/<b>`` is executable for every binary (Q1), or
    ``root_var`` is unset and ``command -v <b>`` resolves every binary in one
    folder (Q2; the root is that folder's parent)."""
    mods = _as_list(modules)
    if module_system(env) is None:
        return False, None, False, "no module system here"
    if not mods or not all(valid_name(m) for m in mods):
        return False, None, False, "not a module name"
    lines = (["module purge >/dev/null 2>&1"] if purge else [])
    for m in mods:
        lines.append(f"module load {m} >/dev/null 2>&1 || {{ echo MACE_Q_NOLOAD={m}; exit 0; }}")
    lines.append(f"printf 'MACE_Q_ROOT=%s\\n' \"${root_var}\"")
    for b in binaries:
        lines.append(f"printf 'MACE_Q_BIN_{b}=%s\\n' \"$(command -v {b} 2>/dev/null)\"")
    r = _bash("\n".join(lines), env, timeout)
    if r is None:
        return False, None, False, "the module probe timed out"
    vals: Dict[str, str] = {}
    for line in (r.stdout or "").splitlines():
        if line.startswith("MACE_Q_") and "=" in line:
            k, v = line.split("=", 1)
            vals[k] = v.strip()
    if "MACE_Q_NOLOAD" in vals:
        return False, None, False, "does not load"
    if "MACE_Q_ROOT" not in vals:
        return False, None, False, "does not load"
    root = vals.get("MACE_Q_ROOT") or ""
    if root:
        missing = [b for b in binaries if not os.access(os.path.join(root, "bin", b), os.X_OK)]
        if missing:
            return False, root, False, f"no {', '.join(missing)} in ${root_var}/bin"
        return True, root, False, ""
    found = [vals.get(f"MACE_Q_BIN_{b}") or "" for b in binaries]
    if not all(found):
        return False, None, False, f"${root_var} not set and {binaries[0]} not on PATH" if not found[0] else \
            f"${root_var} not set and {', '.join(b for b, f in zip(binaries, found) if not f)} not on PATH"
    dirs = {os.path.dirname(f) for f in found}
    if len(dirs) != 1:
        return False, None, False, f"{', '.join(binaries)} are in different folders"
    return True, os.path.dirname(dirs.pop()), True, ""


def cache_get(key: str, env: Optional[Dict[str, str]] = None, binaries: Sequence[str] = ()) -> Optional[dict]:
    """A positive entry younger than 7 days whose ``<root>/bin/<binary>``
    files still exist (a stat, no subprocess), else None."""
    e = _read_cache(env).get(key)
    if not isinstance(e, dict) or time.time() - float(e.get("t", 0)) > CACHE_TTL_S:
        return None
    root = e.get("root")
    if binaries and (not root or not all(os.access(os.path.join(root, "bin", b), os.X_OK) for b in binaries)):
        return None
    return e


def cache_put(key: str, value: dict, env: Optional[Dict[str, str]] = None) -> None:
    """Write one positive entry atomically (temp file + rename); errors are ignored."""
    path = cache_path(env)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _read_cache(env)
        data[key] = {**value, "t": time.time()}
        fd, tmp = tempfile.mkstemp(prefix=".modules.", dir=str(path.parent))
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


def cache_forget(match, env: Optional[Dict[str, str]] = None) -> int:
    """Drop the entries for which ``match(key, entry)`` is true; how many."""
    path = cache_path(env)
    data = _read_cache(env)
    keep = {k: v for k, v in data.items() if not match(k, v)}
    if len(keep) == len(data):
        return 0
    try:
        fd, tmp = tempfile.mkstemp(prefix=".modules.", dir=str(path.parent))
        with os.fdopen(fd, "w") as fh:
            json.dump(keep, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        return 0
    return len(data) - len(keep)
