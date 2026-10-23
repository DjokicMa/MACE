"""Make an ML-potential environment: the one installer behind ``mace mlip
install`` and the GUI's Set up ML environment.

``make_plan`` says what would happen and checks it first: the folder
(``detect.default_env_dir(env)`` unless one is chosen), the Python it is
built from (3.10 or newer: the one that runs MACE, else the newest
python3.1x on PATH; a module's Python on a cluster), the packages (the
registry's pins and licenses, so only what the registry allows), the torch
wheel (CPU unless a GPU is asked for and present), the size and file
count, the free space and, where ``quota`` answers, the space and file
quota. ``install`` then runs the steps, each streamed line by line:

1. ``python -m venv DIR`` (skipped when DIR is already a venv: a second run
   repairs or upgrades it in place);
2. ``DIR/bin/python -m pip install --upgrade pip`` (a module's pip 22.x
   fails on torch's current dependencies);
3. torch from the CPU wheel index (or PyPI's CUDA build with ``gpu``);
4. the backends' pinned packages;
5. ``pip check``;
6. the probe MACE runs before every use, then the record in mlip.json
   (with the folder of the base Python's libpython when the environment's
   interpreter does not start without it: a module Python, see
   ``detect.find_library_dirs``; every step above already runs with it);
7. optionally the weights of one model, through ``fetch.fetch`` with its
   license shown and accepted (never silently).

Every child runs with ``proc.child_env``: no PYTHONPATH (a module's
PYTHONPATH makes pip skip packages that MACE, which runs the environment
without it, would then lack), no MACE_* and no VIRTUAL_ENV. A cancel stops
the running child; a failed or canceled run deletes the folder only when
this run created it, by its explicit path, and leaves an environment that
was there before as it was.
"""
from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import detect, proc, registry

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
#: (MB, files) measured on MSU HPCC (Python 3.11, October 2026); the
#: dependencies the backends share (numpy, scipy, ase, matplotlib, ...)
#: are counted once
EST_TORCH = {"cpu": (860, 23200), "gpu": (5200, 25000)}
EST_SHARED = (250, 9000)
EST_PACKAGE = {"mace-torch": (120, 4000), "chgnet": (270, 10000), "orb-models": (300, 6000),
               "mattersim": (150, 5000), "fairchem-core": (1500, 25000)}
QUOTA_SHARE = 0.9             # warn when a quota would be this full after the install
SPACE_MARGIN = 1.2            # refuse when the free space is below the estimate times this
STEP_TIMEOUT = 3600.0         # one pip call (a slow mirror, a CUDA torch)
DEFAULT_BACKENDS = ("mace-torch",)


class InstallError(RuntimeError):
    pass


class InstallCanceled(RuntimeError):
    pass


# ----------------------------------------------------------------------------- what can be installed

def packages(env: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """{package: {pin, license, license_url, env, models: [{id, label, weights_license, academic}]}}
    from the registry (every model, every scope): the registry's license rules
    are the only ones, so nothing outside them is offered."""
    out: Dict[str, Dict[str, Any]] = {}
    for m in registry.load(scope=None, include_academic=True):
        c = m["code"]
        e = detect.env_name(m)
        if env is not None and e != env:
            continue
        rec = out.setdefault(c["package"], {"package": c["package"], "pin": c.get("pin") or c["package"],
                                            "license": c["license"], "license_url": c.get("license_text_url"),
                                            "env": e, "models": [], "torch": c.get("torch"), "min_python": (3, 10)})
        rp = str(m.get("requires_python") or "")
        if rp.startswith(">="):
            want = tuple(int(x) for x in rp[2:].split(".")[:2])
            rec["min_python"] = max(rec["min_python"], want)
        rec["models"].append({"id": m["id"], "label": m["label"], "weights_license": m["weights"]["license"],
                              "academic": bool(m.get("academic")), "gated": bool(m.get("gated"))})
        for extra in m.get("requires") or []:          # a model's own extra package (installed with its backend)
            if extra not in rec.setdefault("extras", []):
                rec["extras"].append(extra)
    return out


def env_of(backends) -> str:
    envs = {packages()[b]["env"] for b in backends}
    if len(envs) != 1:
        raise InstallError("these packages need separate environments (their pins conflict): "
                           + ", ".join(f"{b} ({packages()[b]['env']})" for b in backends) + "; install them one "
                           "environment at a time")
    return envs.pop()


# ----------------------------------------------------------------------------- the host

def _version(python: str, timeout: float = 20.0, dirs=None) -> Optional[Tuple[int, int, int]]:
    try:
        r = subprocess.run([python, "-I", "-c", "import sys, venv, ensurepip; print('%d %d %d' % sys.version_info[:3])"],
                           env=detect.with_library_dirs(proc.child_env(), list(dirs or [])), capture_output=True,
                           text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    try:
        return tuple(int(x) for x in r.stdout.split()[:3])      # type: ignore[return-value]
    except ValueError:
        return None


def _version_dirs(python: str) -> Tuple[Optional[Tuple[int, int, int]], List[str]]:
    """(version, library folders): a module Python that does not start in
    this shell starts with the folder of its libpython (detect.find_library_dirs)."""
    v = _version(python)
    if v is not None:
        return v, []
    lib = detect.find_library_dirs(python)
    if lib["ok"] and lib["dirs"]:
        v = _version(python, dirs=lib["dirs"])
        if v is not None:
            return v, list(lib["dirs"])
    return None, []


def find_python(explicit: Optional[str] = None, minimum: Tuple[int, int] = (3, 10)) -> Dict[str, Any]:
    """{python, version, why, library_dirs} of the interpreter to build from,
    or {error}. ``explicit`` must work; otherwise the one that runs MACE, then
    the newest python3.1x on PATH (a cluster's module Python is the one on
    PATH), the first that is ``minimum`` or newer (orb-models 0.7 needs 3.12)."""
    minimum = max(tuple(minimum), detect.MIN_PYTHON)
    need = ".".join(map(str, minimum))
    if explicit:
        py = os.path.abspath(os.path.expanduser(explicit))
        v, dirs = _version_dirs(py)
        if v is None:
            return {"error": f"{py} does not run here, or has no venv/ensurepip module"}
        if v[:2] < minimum:
            return {"error": f"{py} is Python {'.'.join(map(str, v))}; {need} or newer is needed for these packages"}
        return {"python": py, "version": ".".join(map(str, v)), "why": "chosen", "library_dirs": dirs}
    tried = []
    cands = [(sys.executable, "the Python that runs MACE")]
    for minor in range(14, 9, -1):
        w = shutil.which(f"python3.{minor}")
        if w:
            cands.append((w, f"python3.{minor} on PATH"))
    w = shutil.which("python3")
    if w:
        cands.append((w, "python3 on PATH"))
    seen = set()
    for py, why in cands:
        if not py or py in seen:
            continue
        seen.add(py)
        v, dirs = _version_dirs(py)
        tried.append(f"{py} ({'.'.join(map(str, v)) if v else 'does not run, or no venv module'})")
        if v and v[:2] >= minimum:
            return {"python": py, "version": ".".join(map(str, v)), "why": why, "library_dirs": dirs}
    return {"error": f"no Python {need} or newer with the venv module was found (tried " + "; ".join(tried)
                     + f"); load one (e.g. module load Python/{need}) or pass its path"}


def gpu_present() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and "GPU" in r.stdout


def _existing_ancestor(path: Path) -> Path:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def free_bytes(path: Path) -> Optional[int]:
    try:
        return shutil.disk_usage(str(_existing_ancestor(path))).free
    except OSError:
        return None


def _mount_source(path: Path) -> Optional[str]:
    """The device or export (``host:/home``) /proc/mounts names for ``path``."""
    try:
        rp = str(_existing_ancestor(path).resolve())
        best, src = "", None
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and (rp == parts[1] or rp.startswith(parts[1].rstrip("/") + "/")) \
                        and len(parts[1]) > len(best):
                    best, src = parts[1], parts[0]
        return src
    except OSError:
        return None


def parse_quota(text: str) -> List[Dict[str, Any]]:
    """Rows of ``quota -w`` (Linux): {filesystem, used_kb, quota_kb, limit_kb,
    files, files_quota, files_limit}; a grace column, or a ``*`` past a soft
    limit, is skipped."""
    rows = []
    for line in text.splitlines():
        tok = line.split()
        if not tok or tok[0] in ("Disk", "Filesystem"):
            continue
        nums = []
        for t in tok[1:]:
            t = t.rstrip("*")
            if t.isdigit():
                nums.append(int(t))
        if len(nums) >= 6:
            rows.append({"filesystem": tok[0], "used_kb": nums[0], "quota_kb": nums[1], "limit_kb": nums[2],
                         "files": nums[3], "files_quota": nums[4], "files_limit": nums[5]})
    return rows


def quota_check(path: Path, add_mb: float, add_files: int,
                runner: Optional[Callable[[], Optional[str]]] = None) -> Dict[str, Any]:
    """{available, filesystem, used_gb, quota_gb, files, files_quota, warnings}
    for the file system ``path`` is on, when ``quota -w`` answers for it."""
    if runner is None:
        def runner():
            exe = shutil.which("quota")
            if not exe:
                return None
            try:
                r = subprocess.run([exe, "-w"], capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError):
                return None
            return r.stdout
    text = runner()
    if not text:
        return {"available": False, "warnings": []}
    src = _mount_source(path)
    rows = parse_quota(text)
    row = next((r for r in rows if src and r["filesystem"] == src), None)
    if row is None and src:
        # an export reached under another host name (hpcc-ib vs hpcc-dmz): match the path after the colon
        tail = src.split(":", 1)[-1]
        row = next((r for r in rows if ":" in r["filesystem"] and r["filesystem"].split(":", 1)[1] == tail), None)
    if row is None:
        return {"available": False, "warnings": []}
    out = {"available": True, "filesystem": row["filesystem"], "used_gb": row["used_kb"] / 1e6,
           "quota_gb": (row["quota_kb"] or row["limit_kb"]) / 1e6, "files": row["files"],
           "files_quota": row["files_quota"] or row["files_limit"], "warnings": []}
    q_kb = row["quota_kb"] or row["limit_kb"]
    if q_kb and row["used_kb"] + add_mb * 1000 > QUOTA_SHARE * q_kb:
        out["warnings"].append(f"the space quota on {row['filesystem']} would be {100 * (row['used_kb'] + add_mb * 1000) / q_kb:.0f} % "
                               f"full ({row['used_kb'] / 1e6:.1f} of {q_kb / 1e6:.1f} GB used, about {add_mb / 1000:.1f} GB more)")
    fq = row["files_quota"] or row["files_limit"]
    if fq and row["files"] + add_files > QUOTA_SHARE * fq:
        out["warnings"].append(f"the file quota on {row['filesystem']} would be {100 * (row['files'] + add_files) / fq:.0f} % "
                               f"full ({row['files']:,} of {fq:,} files used, about {add_files:,} more)")
    return out


# ----------------------------------------------------------------------------- the plan

def estimate(backends, gpu: bool) -> Tuple[int, int]:
    mb, files = EST_TORCH["gpu" if gpu else "cpu"]
    mb, files = mb + EST_SHARED[0], files + EST_SHARED[1]
    for b in backends:
        e = EST_PACKAGE.get(b, (200, 6000))
        mb, files = mb + e[0], files + e[1]
    return mb, files


def manual_commands(plan: Dict[str, Any]) -> List[str]:
    """The same steps as shell lines (``mace mlip setup`` prints these)."""
    d = plan["dir"]
    py = Path(d) / "bin" / "python"
    spec = plan.get("torch_spec") or "torch"
    torch = f"{py} -m pip install '{spec}'" if spec != "torch" else f"{py} -m pip install torch"
    torch += "" if plan["torch"] == "gpu" else f" --index-url {CPU_INDEX}"
    return ["unset PYTHONPATH", f"{plan['python']} -m venv {d}", f"{py} -m pip install --upgrade pip", torch,
            f"{py} -m pip install " + " ".join(plan["pins"]), f"{py} -m pip check",
            f"mace mlip setup --python {py}" + ("" if plan["env"] == detect.DEFAULT_ENV else f" --env {plan['env']}")]


def make_plan(directory: Optional[str] = None, backends=None, gpu: bool = False, python: Optional[str] = None,
              quota_runner: Optional[Callable[[], Optional[str]]] = None) -> Dict[str, Any]:
    """What ``install`` would do, with every check that can fail made first:
    ``problems`` (it would not start) and ``warnings`` (it would)."""
    avail = packages()
    backends = list(dict.fromkeys(backends or DEFAULT_BACKENDS))
    problems: List[str] = []
    warnings: List[str] = []
    unknown = [b for b in backends if b not in avail]
    if unknown:
        raise InstallError(f"unknown package(s) {', '.join(unknown)}; MACE installs " + ", ".join(sorted(avail)))
    env = env_of(backends)
    d = Path(directory).expanduser().absolute() if directory else detect.default_env_dir(env)
    state = "new"
    if (d / "pyvenv.cfg").is_file():
        state = "existing"
    elif d.exists() and (not d.is_dir() or any(d.iterdir())):
        problems.append(f"{d} exists and is not a Python environment: choose a new or empty folder")
    elif d.exists():
        state = "empty"
    minimum = max([avail[b]["min_python"] for b in backends] + [detect.MIN_PYTHON])
    base = find_python(python, minimum)
    torch_spec = next((avail[b]["torch"] for b in backends if avail[b].get("torch")), None) or "torch"
    if "error" in base:
        problems.append(base["error"])
    has_gpu = gpu_present()
    use_gpu = bool(gpu and has_gpu)
    if gpu and not has_gpu:
        warnings.append("a GPU build of torch was asked for, but no NVIDIA GPU answers here (nvidia-smi): "
                        "installing the CPU build")
    mb, files = estimate(backends, use_gpu)
    free = free_bytes(d)
    if free is not None and free < mb * 1e6 * SPACE_MARGIN and state != "existing":
        problems.append(f"only {free / 1e9:.1f} GB free under {_existing_ancestor(d)}; about {mb / 1000:.1f} GB "
                        "is needed")
    q = quota_check(d, mb, files, runner=quota_runner) if state != "existing" else {"available": False, "warnings": []}
    warnings += q["warnings"]
    lic = []
    for b in backends:
        p = avail[b]
        lic.append({"package": b, "pin": p["pin"], "license": p["license"], "license_url": p["license_url"],
                    "models": p["models"]})
        for m in p["models"]:
            if m["academic"]:
                warnings.append(f"{m['label']}: academic, non-commercial use only; it is fetched and run only "
                                "after you accept that (mace mlip fetch --accept-academic-license)")
    plan = {"dir": str(d), "env": env, "state": state, "python": base.get("python"),
            "python_version": base.get("version"), "python_why": base.get("why"), "backends": backends,
            "library_dirs": base.get("library_dirs") or [],
            "pins": [avail[b]["pin"] for b in backends], "licenses": lic, "torch": "gpu" if use_gpu else "cpu",
            "torch_spec": torch_spec, "min_python": ".".join(map(str, minimum)),
            "gpu_present": has_gpu, "estimate_mb": mb, "estimate_files": files, "free_bytes": free,
            "quota": q, "problems": problems, "warnings": warnings, "default_dir": str(detect.default_env_dir(env))}
    plan["commands"] = manual_commands(plan) if plan["python"] else []
    return plan


# ----------------------------------------------------------------------------- an environment made by hand

PROBED = ("torch", "ase", "mace-torch", "chgnet", "orb-models", "mattersim", "fairchem-core")


def record_python(python: str, env: str = detect.DEFAULT_ENV) -> Dict[str, Any]:
    """Probe an interpreter and record it when it can run a model of ``env``
    (``mace mlip setup --python`` and the GUI's Use an existing Python):
    {ok, rows: [(package, version, status)], reason, path, override}."""
    py = os.path.abspath(os.path.expanduser(python))
    pr = detect.probe(py, timeout=detect.PROBE_TIMEOUT_CLI, recheck=True)
    if pr["status"] != "ok":
        return {"ok": False, "python": py, "rows": [], "reason": pr["reason"]}
    pk = pr["data"].get("packages", {})
    rows = [(n, (pk.get(n) or {}).get("version") or "-",
             "ok" if (pk.get(n) or {}).get("ok") else (pk.get(n) or {}).get("error", "missing")) for n in PROBED]
    wanted = [p for p in packages(env) if p in PROBED] or list(PROBED[2:])
    if not ((pk.get("torch") or {}).get("ok") and (pk.get("ase") or {}).get("ok")
            and any((pk.get(n) or {}).get("ok") for n in wanted)):
        return {"ok": False, "python": py, "rows": rows,
                "reason": "the environment needs torch, ase and one of " + ", ".join(wanted)
                          + " (mace mlip install makes one; mace mlip setup prints the commands)"}
    path = detect.save_python(py, env)
    lib = detect.record_library(py)
    src = detect.python_source(env)
    return {"ok": True, "python": py, "rows": rows, "reason": "", "path": str(path),
            "override": src[1] if src[0] != py else None, "library_dirs": lib.get("dirs") or [],
            "library_note": lib["reason"] if (not lib["ok"] or lib.get("missing")) else ""}


# ----------------------------------------------------------------------------- running it

def _env(dirs=None) -> Dict[str, str]:
    return detect.with_library_dirs(proc.child_env(PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1",
                                                   PYTHONUNBUFFERED="1"), list(dirs or []))


def run_step(argv: List[str], log: Callable[[str], None], canceled: Callable[[], bool],
             timeout: float = STEP_TIMEOUT, dirs=None) -> int:
    """Run one child, its output (stdout and stderr) to ``log`` line by line;
    a cancel stops its process group. ``dirs``: library folders the
    interpreter needs in front of LD_LIBRARY_PATH."""
    log("$ " + " ".join(argv))
    try:
        p = subprocess.Popen(argv, env=_env(dirs), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                             start_new_session=True)
    except OSError as exc:
        raise InstallError(f"could not run {argv[0]}: {exc}")
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def pump():
        for line in p.stdout:
            lines.put(line.rstrip("\n"))
        lines.put(None)
    threading.Thread(target=pump, daemon=True).start()
    end = time.monotonic() + timeout
    try:
        while True:
            if canceled():
                proc.stop(p)
                raise InstallCanceled("canceled")
            if time.monotonic() > end:
                proc.stop(p)
                raise InstallError(f"{' '.join(argv[:4])} ... took more than {timeout:g} s")
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.strip():
                log("  " + line)
        return p.wait()
    except KeyboardInterrupt:              # Ctrl+C in the CLI: the child runs in its own session, stop it too
        proc.stop(p)
        raise InstallCanceled("interrupted")


def _rollback(d: Path, created: str, log) -> bool:
    """Undo a failed or canceled run: the folder this run created (``new``)
    goes, by its explicit path; one that was empty is emptied; an existing
    environment stays."""
    if created == "new" and d.is_dir():
        shutil.rmtree(str(d), ignore_errors=True)
        log(f"Removed {d} (this run created it).")
        return True
    if created == "empty" and d.is_dir():
        for child in list(d.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(str(child), ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        log(f"Emptied {d} again (it was empty before this run).")
        return True
    if created == "existing":
        log(f"{d} was an environment before this run and is left in place (pip may have changed some packages; "
            "running the install again repairs it).")
    return False


def install(plan: Dict[str, Any], *, log: Callable[[str], None] = print,
            progress: Optional[Callable[[float, str], None]] = None, canceled: Callable[[], bool] = lambda: False,
            fetch_id: Optional[str] = None, accept_license: bool = False,
            ask: Optional[Callable[[str], str]] = None, isatty: Optional[bool] = None,
            accept_academic: bool = False) -> Dict[str, Any]:
    """Run the plan. Returns {ok, python, recorded, probe, fetch, log}; raises
    InstallError or InstallCanceled after the rollback."""
    if plan["problems"]:
        raise InstallError("; ".join(plan["problems"]))
    d, state, env = Path(plan["dir"]), plan["state"], plan["env"]
    lines: List[str] = []

    def say(s: str) -> None:
        for part in str(s).rstrip("\n").splitlines() or [""]:
            lines.append(part)
            log(part)

    def step(frac: float, text: str) -> None:
        say(f"== {text}")
        if progress:
            progress(frac, text)

    def check(rc: int, what: str) -> None:
        if rc != 0:
            raise InstallError(f"{what} failed (exit code {rc}); the log above says why")

    py = str(Path(d) / "bin" / "python")
    try:
        if state == "existing":
            step(0.03, f"Using the environment in {d} (repairs and upgrades it in place)")
            vp = detect.venv_python(d)
            if not vp:
                raise InstallError(f"{d} has a pyvenv.cfg but no bin/python: delete it, or choose another folder")
            py = vp
        else:
            step(0.03, f"Creating the environment in {d} from {plan['python']} (Python {plan['python_version']}, "
                       f"{plan['python_why']})")
            check(run_step([plan["python"], "-m", "venv", str(d)], say, canceled, timeout=600,
                           dirs=plan.get("library_dirs")), "python -m venv")
            vp = detect.venv_python(d)
            if not vp:
                raise InstallError(f"python -m venv made no interpreter in {d}")
            py = vp
        # a venv of a module Python needs that module's libpython folder wherever it runs: found here
        # (with no LD_LIBRARY_PATH), used for every step and recorded with the interpreter below
        lib = detect.find_library_dirs(py)
        if not lib["ok"]:
            raise InstallError(f"the environment's python does not start: {lib['reason']}")
        dirs = detect.remember_library_dirs(py, lib["dirs"])
        if lib.get("missing"):
            say(f"warning: {lib['reason']}; load the module that provides it and run this install again")
        if dirs:
            say(f"  {py} needs {':'.join(dirs)} on LD_LIBRARY_PATH (its Python's shared library); MACE adds it "
                "whenever it starts this interpreter")
        step(0.08, "Upgrading pip (an older pip fails on torch's dependencies)")
        check(run_step([py, "-m", "pip", "install", "--upgrade", "pip"], say, canceled, dirs=dirs), "the pip upgrade")
        spec = plan.get("torch_spec") or "torch"
        if plan["torch"] == "gpu":
            step(0.12, f"Installing {spec} with CUDA (several GB)")
            check(run_step([py, "-m", "pip", "install", spec], say, canceled, dirs=dirs), "the torch install")
        else:
            step(0.12, f"Installing the CPU build of {spec} (about 0.9 GB)")
            check(run_step([py, "-m", "pip", "install", spec, "--index-url", CPU_INDEX], say, canceled, dirs=dirs),
                  "the torch install")
        step(0.55, "Installing " + ", ".join(plan["pins"]))
        check(run_step([py, "-m", "pip", "install", *plan["pins"]], say, canceled, dirs=dirs), "the package install")
        step(0.85, "Checking the installed packages (pip check)")
        if run_step([py, "-m", "pip", "check"], say, canceled, timeout=300, dirs=dirs) != 0:
            say("warning: pip check found a conflict (above); the probe below decides whether MACE can use it")
        step(0.9, "Probing the environment as MACE does before every use")
        pr = detect.probe(py, timeout=detect.PROBE_TIMEOUT_CLI, recheck=True)
        if pr["status"] != "ok":
            raise InstallError(f"the probe failed: {pr['reason']}")
        pk = pr["data"].get("packages", {})
        for name in ["torch", "ase"] + list(plan["backends"]):
            r = pk.get(name)
            if r is not None and not r.get("ok"):
                raise InstallError(f"{name} does not import: {r.get('error')}")
            say(f"  {name:13s} {(r or {}).get('version') or '-'}")
    except InstallCanceled:
        say("Canceled.")
        rolled = _rollback(d, state, say)
        exc = InstallCanceled("canceled" + ("; the half-built environment was removed" if rolled else ""))
        exc.log = lines                                   # type: ignore[attr-defined]
        raise exc
    except Exception as exc:
        say(f"Failed: {exc}")
        rolled = _rollback(d, state, say)
        err = InstallError(f"{exc}" + ("; the half-built environment was removed" if rolled else ""))
        err.log = lines                                   # type: ignore[attr-defined]
        raise err from exc
    path = detect.save_python(py, env)
    say(f"Recorded {py} in {path}.")
    rl = detect.record_library(py)
    if rl.get("dirs"):
        say(f"Recorded its library folder ({':'.join(rl['dirs'])}" + (f", from the module {rl['module']}"
            if rl.get("module") else "") + "): it starts from MACE without that module loaded.")
    src = detect.python_source(env)
    if src[0] != os.path.abspath(py):
        say(f"Note: {src[1]} names {src[0]}, which MACE uses before this record (unset it to use this one).")
    res = {"ok": True, "python": py, "env": env, "recorded": str(path), "probe": pr["data"], "fetch": None}
    if fetch_id:
        step(0.95, f"Fetching the weights of {fetch_id}")
        res["fetch"] = fetch_weights(fetch_id, log=say, yes=accept_license, ask=ask, isatty=isatty,
                                     accept_academic=accept_academic)
    if progress:
        progress(1.0, "done")
    res["log"] = lines
    return res


def fetch_weights(model_id: str, *, log: Callable[[str], None], yes: bool, ask=None, isatty=None,
                  accept_academic: bool = False, academic_how: Optional[str] = None,
                  consent_how: Optional[str] = None) -> Dict[str, Any]:
    """{rc, status}: the existing fetch (license shown, consent asked or given)."""
    from . import fetch
    m = registry.get(model_id, scope=None)
    if m is None:
        log(registry.unknown(model_id, scope=None))
        return {"rc": 2, "status": "unknown"}
    ws = detect.weights_status(m)
    if ws["status"] in ("ready", "bundled"):
        log(f"{m['label']}: the weights are already there ({ws['status']}).")
        return {"rc": 0, "status": ws["status"]}
    rc = fetch.fetch(m, yes=yes, out=lambda s: log(s.rstrip("\n")), ask=ask, isatty=isatty,
                     accept_academic=accept_academic, academic_how=academic_how, consent_how=consent_how)
    return {"rc": rc, "status": detect.weights_status(m)["status"]}
