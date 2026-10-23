"""The phonopy routes: Calculation Setup > Phonopy (generate) and
Analysis > Phonons with Phonopy... (the set, collect, the plots).

Every step is the CLI's own command (``mace opt2d12 --calc-type PHONOPY``,
``mace analyze phonopy``), run as a cancelable task in the parent's folder,
so the GUI and the CLI write the same files:

GET  phonopy/info                           phonopy / seekpath installed?
GET  phonopy/sets?dir=REL                   the displacement sets (folders with a
                                            mace_phonopy.json) under REL, two levels deep
GET  phonopy/status?dir=REL                 one set: its manifest, each displaced job's
                                            state (no output / running / done / failed),
                                            and the result files collect wrote
GET  phonopy/data?dir=REL                   band.yaml, total_dos.dat and
                                            thermal_properties.yaml as plot data (THz)
POST phonopy/generate {parent, d12?, dim, amplitude?, pm?, cry_sym?, keep_shrink?,
                       output_dir?, force?, root_gen}
                                            -> task: mace opt2d12 --calc-type PHONOPY
                                            (the set's folder goes in output_dir,
                                            default the parent's)
POST phonopy/collect  {dir, band?, band_labels?, band_points?, mesh?, tmin?, tmax?,
                       tstep?, nac_from?, check?, root_gen}
                                            -> task: mace analyze phonopy

The force evaluations themselves are ordinary CRYSTAL jobs: the dialog
hands the set's folder to Submit (with the walltime generate suggested).
Reading routes read only the set's own files; the writing routes carry the
page's root_gen.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List

from . import clirun
from . import tasks as gui_tasks

GET_ROUTES = {"phonopy/info", "phonopy/sets", "phonopy/status", "phonopy/data"}
POST_ROUTES = {"phonopy/generate", "phonopy/collect"}

MANIFEST = "mace_phonopy.json"
RESULT_FILES = ("FORCE_SETS", "phonopy_params.yaml", "band.yaml", "mesh.yaml", "total_dos.dat",
                "thermal_properties.yaml", "BORN", "band_dos.pdf")
_DIM_RE = re.compile(r"^\s*\d+(\s+-?\d+){2}(\s+-?\d+){0,6}\s*$")
_TAIL = 400_000          # bytes of an output read for its termination and forces
_RUNNING_S = 600         # an unfinished output this recent is still being written


class PhonopyRouteError(ValueError):
    """A request the dialog must correct (400)."""


def _q(query, name: str, default: Any = None):
    return (query.get(name) or [default])[0]


def _gen(handler, body) -> None:
    want = body.get("root_gen", handler.headers.get("X-Mace-Root-Gen"))
    if want is None:
        raise PhonopyRouteError("root_gen is required (reload the page)")
    if str(want) != str(getattr(handler.state, "root_gen", 0)):
        from .submit import SubmitConflict
        raise SubmitConflict("The folder changed; reload")


def info() -> Dict[str, Any]:
    out: Dict[str, Any] = {"phonopy": importlib.util.find_spec("phonopy") is not None,
                           "seekpath": importlib.util.find_spec("seekpath") is not None,
                           "install": "pip install 'phonopy<5'  (and seekpath for automatic band paths)"}
    if out["phonopy"]:
        try:
            from importlib.metadata import version
            out["version"] = version("phonopy")
        except Exception:
            out["version"] = None
    return out


def _job_state(outdir: Path, name: str) -> Dict[str, Any]:
    deck = outdir / f"{name}.d12"
    out = outdir / f"{name}.out"
    rec: Dict[str, Any] = {"name": name, "deck": deck.is_file(), "state": "not run"}
    if not out.is_file():
        return rec
    st = out.stat()
    rec["size"] = st.st_size
    with open(out, "rb") as fh:
        if st.st_size > _TAIL:
            fh.seek(st.st_size - _TAIL)
        tail = fh.read().decode("utf-8", "replace")
    if "EEEEEEEEEE TERMINATION" in tail:
        rec["state"] = "done" if "CARTESIAN FORCES IN HARTREE/BOHR" in tail else "failed"
        if rec["state"] == "failed":
            rec["reason"] = "terminated without a force block"
    elif re.search(r"ERROR \*{4}", tail):
        rec["state"] = "failed"
        m = re.search(r"ERROR \*{4}.*", tail)
        rec["reason"] = " ".join(m.group(0).split())[:200] if m else "CRYSTAL error"
    elif time.time() - st.st_mtime < _RUNNING_S:
        rec["state"] = "running"
    else:
        rec["state"] = "failed"
        rec["reason"] = "the output stops without CRYSTAL's termination line (killed, or out of walltime)"
    return rec


def status(st, outdir: Path) -> Dict[str, Any]:
    mf = outdir / MANIFEST
    if not mf.is_file():
        raise FileNotFoundError(f"{st.rel(outdir) or '.'} has no {MANIFEST} (not a displacement set of "
                                "mace opt2d12 --calc-type PHONOPY)")
    man = json.loads(mf.read_text())
    if not isinstance(man, dict) or not isinstance(man.get("jobs", []), list):
        raise PhonopyRouteError(f"{st.rel(mf)} is not the manifest of a phonopy displacement set")
    jobs = [_job_state(outdir, str(n)) for n in man.get("jobs", [])]
    counts: Dict[str, int] = {}
    for j in jobs:
        counts[j["state"]] = counts.get(j["state"], 0) + 1

    def rel_or_name(p):
        try:
            return st.rel(Path(p))
        except Exception:
            return Path(p).name if p else None
    return {"dir": st.rel(outdir), "parent_out": rel_or_name(man.get("parent_out")),
            "parent_d12": rel_or_name(man.get("parent_d12")), "dim": man.get("dim"),
            "amplitude": man.get("amplitude"), "n_atoms_supercell": man.get("n_atoms_supercell"),
            "walltime": man.get("walltime"), "cry_sym": man.get("cry_sym"), "toldee": man.get("toldee"),
            "warnings": man.get("warnings", []), "edits": man.get("edits", []),
            "phonopy_version": man.get("phonopy_version"), "jobs": jobs, "counts": counts,
            "ready": bool(jobs) and counts.get("done", 0) == len(jobs),
            "results": [f for f in RESULT_FILES if (outdir / f).is_file()]}


def sets(st, base: Path, max_dirs: int = 400) -> List[Dict[str, Any]]:
    """Folders under ``base`` (two levels) holding a mace_phonopy.json."""
    from .filekind import rel_inside
    found: List[Dict[str, Any]] = []
    seen = 0
    for d in [base, *sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))]:
        if rel_inside(st, d) is None:                 # a symlink out of the folder is not listed
            continue
        for cand in ([d] if d == base else [d, *sorted(p for p in d.iterdir() if p.is_dir())]):
            seen += 1
            if seen > max_dirs:
                return found
            rel = rel_inside(st, cand)
            if rel is not None and (cand / MANIFEST).is_file():
                try:
                    man = json.loads((cand / MANIFEST).read_text())
                except (OSError, ValueError):
                    continue
                if not isinstance(man, dict) or not isinstance(man.get("jobs", []), list):
                    continue
                found.append({"dir": rel, "dim": man.get("dim"),
                              "jobs": len(man.get("jobs", [])),
                              "parent": Path(man.get("parent_out") or "").name,
                              "collected": (cand / "FORCE_SETS").is_file()})
    return found


# ------------------------------------------------------------------ plot data
def read_band_yaml(path: Path) -> Dict[str, Any]:
    """distances, frequencies [band][point] (THz), segment ends and labels of a phonopy band.yaml."""
    dist: List[float] = []
    freqs: List[List[float]] = []
    cur: List[float] = []
    seg_n: List[int] = []
    labels: List[List[str]] = []
    mode = None
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            s = line.strip()
            if s.startswith("segment_nqpoint:"):
                mode = "seg"
                continue
            if s.startswith("labels:"):
                mode = "lab"
                continue
            if mode == "seg":
                if s.startswith("- ") and s[2:].strip().isdigit():
                    seg_n.append(int(s[2:]))
                    continue
                mode = None
            if mode == "lab":
                m = re.match(r"-\s*\[\s*'([^']*)'\s*,\s*'([^']*)'\s*\]", s)
                if m:
                    labels.append([m.group(1), m.group(2)])
                    continue
                mode = None
            if s.startswith("- q-position:"):
                if cur:
                    freqs.append(cur)
                cur = []
            elif s.startswith("distance:"):
                dist.append(float(s.split(":", 1)[1]))
            elif s.startswith("frequency:"):
                cur.append(float(s.split(":", 1)[1]))
    if cur:
        freqs.append(cur)
    if not dist or len(freqs) != len(dist):
        raise PhonopyRouteError(f"{path.name}: not a phonopy band.yaml")
    nb = min(len(f) for f in freqs)
    bands = [[freqs[i][b] for i in range(len(freqs))] for b in range(nb)]
    ticks = []
    if seg_n:
        i = 0
        for k, n in enumerate(seg_n):
            start, end = i, i + n - 1
            lab = labels[k] if k < len(labels) else ["", ""]
            if not ticks or abs(ticks[-1]["x"] - dist[start]) > 1e-9:
                ticks.append({"x": dist[start], "label": lab[0]})
            elif lab[0] and ticks[-1]["label"] != lab[0]:
                ticks[-1]["label"] = f"{ticks[-1]['label']}|{lab[0]}" if ticks[-1]["label"] else lab[0]
            ticks.append({"x": dist[end], "label": lab[1]})
            i = end + 1
    return {"distances": dist, "bands": bands, "ticks": ticks}


def read_total_dos(path: Path) -> Dict[str, Any]:
    f, d = [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            f.append(float(parts[0]))
            d.append(float(parts[1]))
    return {"frequency": f, "dos": d}


def read_thermal(path: Path) -> Dict[str, Any]:
    keys = ("temperature", "free_energy", "entropy", "heat_capacity", "energy")
    out: Dict[str, List[float]] = {k: [] for k in keys}
    text = path.read_text()
    start = text.find("thermal_properties:")       # after the unit block ("temperature: K")
    for line in (text[start:] if start >= 0 else "").splitlines():
        m = re.match(r"\s*-?\s*(temperature|free_energy|entropy|heat_capacity|energy):\s*(\S+)", line)
        if m:
            out[m.group(1)].append(float(m.group(2)))
    n = len(out["temperature"])
    if not n or any(len(v) != n for v in out.values()):
        return {}
    zpe = re.search(r"zero_point_energy:\s*(\S+)", text)
    return {**out, "zero_point_energy": float(zpe.group(1)) if zpe else None,
            "units": {"temperature": "K", "free_energy": "kJ/mol", "entropy": "J/(K mol)",
                      "heat_capacity": "J/(K mol)", "energy": "kJ/mol"}}


def data(outdir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"unit": "THz", "cm1_per_thz": 33.35641}
    if (outdir / "band.yaml").is_file():
        out["band"] = read_band_yaml(outdir / "band.yaml")
    if (outdir / "total_dos.dat").is_file():
        out["dos"] = read_total_dos(outdir / "total_dos.dat")
    if (outdir / "thermal_properties.yaml").is_file():
        out["thermal"] = read_thermal(outdir / "thermal_properties.yaml")
    if len(out) == 2:
        raise FileNotFoundError("no band.yaml, total_dos.dat or thermal_properties.yaml yet: run Collect first")
    return out


# ------------------------------------------------------------------ commands
def generate_argv(st, body) -> (List[str], Path, Path):
    """mace opt2d12 --calc-type PHONOPY on the parent: (argv, its folder, the folder the set goes in)."""
    parent = st.resolve(str(body.get("parent") or ""))
    if parent.suffix.lower() != ".out":
        raise PhonopyRouteError("the parent is a CRYSTAL output (.out) of a finished 3D OPT or SP run")
    dim = str(body.get("dim") or "").strip()
    if not _DIM_RE.match(dim) or len(dim.split()) not in (3, 9):
        raise PhonopyRouteError('dim is "N N N" or 9 integers')
    argv = ["--no-banner", "opt2d12", "--out-file", str(parent), "--calc-type", "PHONOPY",
            "--non-interactive", "--dim", *dim.split()]
    if body.get("d12"):
        argv += ["--d12-file", str(st.resolve(str(body["d12"])))]
    where = parent.parent
    if body.get("output_dir"):
        name = str(body["output_dir"])
        where = st.out_dir((parent.parent / name) if not os.path.isabs(name) else Path(name))
        argv += ["--output-dir", str(where)]
    if body.get("amplitude") not in (None, ""):
        amp = float(body["amplitude"])
        if not 0 < amp < 0.2:
            raise PhonopyRouteError("amplitude is in Angstrom, between 0 and 0.2 (phonopy's default 0.01)")
        argv += ["--amplitude", repr(amp)]
    pm = str(body.get("pm") or "auto")
    if pm not in ("auto", "yes", "no"):
        raise PhonopyRouteError("pm is auto, yes or no")
    argv += ["--pm", pm]
    for k, flag in (("cry_sym", "--cry-sym"), ("keep_shrink", "--keep-shrink"), ("force", "--force")):
        if body.get(k):
            argv.append(flag)
    return argv, parent.parent, where


def collect_argv(st, body) -> (List[str], Path):
    outdir = st.resolve(str(body.get("dir") or ""))
    if not (outdir / MANIFEST).is_file():
        raise PhonopyRouteError(f"{body.get('dir')} has no {MANIFEST}")
    argv = ["--no-banner", "analyze", "phonopy", str(outdir)]
    if body.get("check"):
        argv.append("--check")
    band = str(body.get("band") or "auto").strip()
    argv += ["--band", band]
    if body.get("band_labels"):
        argv += ["--band-labels", str(body["band_labels"])]
    for k, flag, kind in (("band_points", "--band-points", int), ("tmin", "--tmin", float),
                          ("tmax", "--tmax", float), ("tstep", "--tstep", float)):
        if body.get(k) not in (None, ""):
            argv += [flag, str(kind(body[k]))]
    if body.get("mesh") not in (None, ""):
        mesh = str(body["mesh"]).strip()
        if not re.fullmatch(r"\d+(\.\d*)?|\d+\s+\d+\s+\d+", mesh):
            raise PhonopyRouteError('mesh is a length (e.g. 100.0) or three integers "a b c"')
        argv += ["--mesh", mesh]
    if body.get("nac_from"):
        argv += ["--nac-from", str(st.resolve(str(body["nac_from"])))]
    return argv, outdir


def _shown(argv: List[str], root: Path) -> str:
    out = []
    for a in argv[1:]:
        try:
            p = Path(a)
            if p.is_absolute():
                a = os.path.relpath(a, root)
        except ValueError:
            pass
        out.append(shlex.quote(a))
    return "mace " + " ".join(out)


def _run(st, argv: List[str], cwd: Path, ctx, timeout: float) -> Dict[str, Any]:
    if ctx is not None:
        ctx.progress(0.1, _shown(argv, st.root))
    r = clirun.run(clirun.mace_argv(*argv), cwd, ctx, timeout=timeout)
    log = r.log.replace(str(st.root) + os.sep, "")
    return {"code": r.returncode, "log": log[-20000:], "command": _shown(argv, st.root)}


def get(handler, route: str, query):
    st = handler.state
    if route == "phonopy/info":
        return handler._json(info())
    if route == "phonopy/sets":
        base = st.resolve(_q(query, "dir", "") or ".")
        if base.is_file():
            base = base.parent
        return handler._json({"sets": sets(st, base)})
    outdir = st.resolve(_q(query, "dir", "") or "")
    if route == "phonopy/status":
        return handler._json(status(st, outdir))
    if route == "phonopy/data":
        return handler._json(data(outdir))
    return handler._error(404, f"unknown API route {route}")


def post(handler, route: str, body):
    st = handler.state
    from .server import cli_task
    _gen(handler, body)
    client = handler.headers.get("X-Mace-Client")
    if route == "phonopy/generate":
        argv, cwd, where = generate_argv(st, body)

        def job(ctx):
            res = _run(st, argv, cwd, ctx, 900)
            m = re.search(r"Walltime per job[^\n]*?(\d+(?:-\d\d)?:\d\d:\d\d)", res["log"])
            res["walltime"] = m.group(1) if m else None
            # the set's folder is named by opt2d12 (<stem>_ph<dim>_<functional>_optimized)
            m = re.search(r"Wrote \d+ deck\(s\) \+ \.ext to (.+?)/?\s*$", res["log"], re.M)
            outdir = where / Path(m.group(1)).name if res["code"] == 0 and m else None
            res["dir"] = st.rel(outdir) if outdir and outdir.is_dir() else None
            return res
        t = cli_task(st, "phonopy", "Phonopy: displaced supercells (mace opt2d12 --calc-type PHONOPY)", job,
                     doc="phonopy", client=client, supersede=("phonopy", "generate"))
        return gui_tasks.answer(handler, st.tasks, t, gui_tasks.clamp_wait(body.get("wait", 2)))
    if route == "phonopy/collect":
        argv, outdir = collect_argv(st, body)

        def job(ctx):
            res = _run(st, argv, outdir, ctx, 3600)
            res["dir"] = st.rel(outdir)
            res["results"] = [f for f in RESULT_FILES if (outdir / f).is_file()]
            return res
        t = cli_task(st, "phonopy", "Phonopy: forces to phonons (mace analyze phonopy)", job,
                     doc="phonopy", client=client, supersede=("phonopy", "collect"))
        return gui_tasks.answer(handler, st.tasks, t, gui_tasks.clamp_wait(body.get("wait", 2)))
    return handler._error(404, f"unknown API route {route}")


def handle(handler, method: str, route: str, payload):
    """Errors as answers: 400 a request to correct, 403 outside the folder,
    404 a missing file, 409 the folder changed."""
    from .submit import SubmitConflict
    try:
        if method == "GET" and route in GET_ROUTES:
            return get(handler, route, payload)
        if method == "POST" and route in POST_ROUTES:
            return post(handler, route, payload or {})
        return handler._error(404, f"unknown API route {route}")
    except SubmitConflict as exc:
        return handler._json({"error": str(exc), "conflict": True}, 409)
    except PermissionError as exc:
        return handler._error(403, str(exc))
    except FileNotFoundError as exc:
        return handler._error(404, str(exc))
    except ValueError as exc:
        return handler._error(400, str(exc))
