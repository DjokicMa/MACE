"""What a file in the served folder is: by name first (cheap, for the
explorer's listing), then by content for outputs and logs (lazy, in a pool
worker, cached by the file's stamp).

``kind_by_name`` gives every name the explorer listed before this module
existed the same kind as before (``.out`` and ``.o`` stay ``out``), and adds
kinds only for files that were hidden: cubes, molecular formats, fort.25,
transport data, CRYSTAL's restart data, wave functions, job scripts and job
logs. ``refine`` reads the head (256 KB) and the tail (64 KB) of an output
or log and decides its ``role``: a CRYSTAL output, a properties output, a
job log or plain text; for an output the calculation type comes from
``completion_checker.determine_completed_subtype``, the CLI's own
content-first classifier (no regex is copied out of it).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

HEAD = 256 * 1024
TAIL = 64 * 1024
FULL_READ_MAX = 64 * 1024 * 1024     # an output whose tells are not in the head or tail is read whole up to this

_BAND_SUFFIXES = (".BAND.DAT", ".band.band.dat", "_band.dat")
_DOS_SUFFIXES = (".DOSS.DAT", "_doss.dat")
# the names CRYSTAL itself gives its cube files (submit_prop.sh renames only the first three)
_CUBE_DAT = re.compile(r"(DENS_CUBE|POT_CUBE|SPIN_CUBE|_WAN_CUBE|\.CUBE)\.DAT$", re.I)
_TRANSPORT = re.compile(r"\.(SEEBECK|SIGMAS|SIGMA|KAPPA|TDF)\.dat$", re.I)
_CRYSTAL_DATA = ("FREQINFO.DAT", "BORN.DAT", "HESSOPT.DAT", "IRREFR.DAT", "IRDIEL.DAT", "OPTINFO.DAT",
                 "SCFOUT.LOG", "POTC.DAT")
_LOG_SUFFIX = re.compile(r"\.(log|e|err|o\d+|e\d+)$", re.I)


class KindConflict(ValueError):
    """A route asked for something this kind of file does not hold (a job log's geometry)."""


NO_GEOMETRY = "No geometry in this output: CRYSTAL stopped before printing one (see the end of the output)"


class NoGeometry(KindConflict):
    """A CRYSTAL output with no geometry block (a run that died early): 409, never a 500."""

    def __init__(self, msg: str = NO_GEOMETRY):
        super().__init__(msg)


def kind_by_name(name: str) -> str:
    """The explorer kind of a file name; "" hides it. Unchanged for every
    name the explorer listed before (cif, d12, d3, band, dos, ir, raman, out,
    db, json, image, html)."""
    low = name.lower()
    up = name.upper()
    if low.endswith(".cif"):
        return "cif"
    if low.endswith(".d12"):
        return "d12"
    if low.endswith(".d3"):
        return "d3"
    if name.endswith(_BAND_SUFFIXES):
        return "band"
    if name.endswith(_DOS_SUFFIXES):
        return "dos"
    if up.endswith("IRSPEC.DAT"):
        return "ir"
    if up.endswith("RAMSPEC.DAT"):
        return "raman"
    if low.endswith((".out", ".o")):
        return "out"
    if low.endswith(".db"):
        return "db"
    if low.endswith(".json"):
        return "json"
    if low.endswith((".png", ".svg", ".jpg", ".jpeg", ".gif")):
        return "image"
    if low.endswith((".html", ".htm")):
        return "html"
    # ---- kinds hidden before (new values only, so no consumer of an old kind changes)
    if low.endswith(".cube") or _CUBE_DAT.search(name):
        return "cube"
    if low.endswith((".xyz", ".extxyz")):
        return "xyz"
    if low.endswith((".vasp", ".poscar")) or re.match(r"^(POSCAR|CONTCAR)([._-]|$)", name):
        return "poscar"
    if low.endswith((".mol", ".sdf")):
        return "mol"
    if low.endswith(".pdb"):
        return "pdb"
    if low.endswith(".gen"):
        return "gen"
    if low == "fort.34" or low.endswith((".gui", ".f34")):
        return "fort34"
    if low == "fort.25" or low.endswith(".f25"):
        return "f25"
    if _TRANSPORT.search(name):
        return "transport"
    if up.endswith(_CRYSTAL_DATA):
        return "crystaldata"
    if low in ("fort.9", "fort.98") or low.endswith((".f9", ".f98")):
        return "wavefunction"
    if low.endswith(".sh"):
        return "script"
    if _LOG_SUFFIX.search(name):
        return "log"
    return ""


# ------------------------------------------------------------------ cubes
def cube_header_ok(path: Path) -> bool:
    """A Gaussian cube header in the first 4 KB: two comment lines, then
    ``natoms ox oy oz`` and three ``n vx vy vz`` voxel lines."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    lines = head.decode("utf-8", "replace").split("\n")
    if len(lines) < 7:
        return False
    try:
        for ln in lines[2:6]:
            parts = ln.split()
            if len(parts) < 4:
                return False
            int(parts[0])
            [float(x) for x in parts[1:4]]
    except ValueError:
        return False
    return True


def cube_grid_shape(path: Path):
    """The grid's point counts (n1, n2, n3) from a cube's header, or None."""
    try:
        with open(path, "rb") as fh:
            lines = fh.read(4096).decode("utf-8", "replace").split("\n")
        return tuple(abs(int(lines[k].split()[0])) for k in (3, 4, 5))
    except (OSError, ValueError, IndexError):
        return None


def cube_type(name: str) -> str:
    """density / potential / spin / orbital from the name token CRYSTAL and
    MACE use; "" when the name does not tell (the cube's comment decides)."""
    up = Path(name).name.upper()
    # CRYSTAL23's LOCALWF/PRINTPLO writes one <k>_WAN.CUBE per Wannier function (the manual says
    # <input>_WAN_CUBE.DAT; both names are taken)
    if "_WAN_CUBE" in up or up.endswith("WAN_CUBE.DAT") or re.search(r"(^|[_.])WAN\.CUBE$", up):
        return "orbital"
    for token, kind in (("SPIN", "spin"), ("POT", "potential"), ("DENS", "density")):
        if re.search(rf"(^|[_.]){token}(_CUBE)?[_.]", up + "."):
            return kind
    return ""


# ------------------------------------------------------------------ content
_BANNER = re.compile(r"EEEEEEEEEE STARTING|^\s*\*\s+CRYSTAL\d\d\s+\*|^\s*\*{5,}\s*CRYSTAL\d\d", re.M)
_PROPERTIES = "CRYSTAL - PROPERTIES"
_QUEUE_MARKERS = ("submit directory", "slurmstepd", "srun:", "Queue Manager Callback", "SLURM_JOB_ID",
                  "Starting job execution", "sbatch:", "Traceback (most recent call last)", "mace run:", "# batch done")
# job log names: SLURM's slurm-<id>.out, MACE's <stem>-<id>.o, <stem>.o<id>, array logs <tag>-<A>_<T>.o/.e,
# other sites' <name>_<id>.out
_LOG_NAME = re.compile(r"(^slurm-\d+(_\d+)?\.out$|-\d+(_\d+)?\.(o|e)$|\.(o|e)\d+$|[_-]\d{5,}\.(out|err|e|o|log)$|\.(e|err)$)",
                       re.I)


def head_tail(path: Path, head: int = HEAD, tail: int = TAIL):
    """(head bytes, tail bytes, size): at most head + tail bytes are read."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        h = fh.read(head)
        if size > head:
            fh.seek(max(head, size - tail))
            t = fh.read(tail)
        else:
            t = b""
    return h, t, size


def role_of(name: str, head: str, tail: str) -> str:
    """crystal_out | properties_out | joblog | text, from the text (the name only breaks ties)."""
    if _PROPERTIES in head:
        return "properties_out"
    if _BANNER.search(head):
        return "crystal_out"
    if _LOG_NAME.search(name) or any(m in head for m in _QUEUE_MARKERS):
        return "joblog"
    if Path(name).suffix.lower() in (".e", ".err") or _LOG_SUFFIX.search(name):
        return "joblog"
    return "text"


def _calc(path: Path, lines, has_opt_end: bool, role: str) -> str:
    from mace.completion_checker import determine_completed_subtype
    if role == "properties_out":
        return determine_completed_subtype(path, lines, has_opt_end)
    # a CRYSTAL output whose stem also has a .d3 (an SCF and a properties run sharing a stem): the
    # .d3 must not decide (its step 1), so the same name is passed under a folder that does not exist
    # and the content decides, exactly as the CLI's steps 2-4 do
    return determine_completed_subtype(Path("/nonexistent-mace-gui") / path.name, lines, has_opt_end)


def refine(path: Path) -> Dict[str, Any]:
    """{role, calc, dim, refined: True} for an output or log; reads the head
    and the tail, and a whole output (up to 64 MB) only when the head and
    tail say "SP" but the file is longer than both (a tell could sit in the
    middle)."""
    path = Path(path)
    h, t, size = head_tail(path)
    head = h.decode("utf-8", "replace")
    tail = t.decode("utf-8", "replace")
    role = role_of(path.name, head, tail)
    out: Dict[str, Any] = {"role": role, "calc": None, "dim": None, "refined": True}
    if role not in ("crystal_out", "properties_out"):
        return out
    lines = (head + "\n" + tail).splitlines(True)
    has_opt_end = "OPT END" in tail
    calc = _calc(path, lines, has_opt_end, role)
    if role == "crystal_out" and calc == "SP" and size > HEAD + TAIL and size <= FULL_READ_MAX:
        with open(path, "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
        calc = _calc(path, text.splitlines(True), "OPT END" in text, role)
    out["calc"] = calc
    if role == "crystal_out":
        out["dim"] = dim_of_head(head.splitlines())
    return out


def dim_of_head(lines) -> int:
    """structures._out_dimensionality on the first 400 lines; when those hold
    neither a banner nor a dimensionality line (a long input echo), the first
    dimensionality line further on."""
    from . import structures as S
    dim = S._out_dimensionality(lines)
    if dim != 3:
        return dim
    first = lines[:400]
    if any(b in x and "PCRYSTAL" not in x for x in first for b, _ in S._DIM_BANNERS) or \
            any("DIMENSIONALITY OF THE SYSTEM" in x for x in first):
        return dim
    for x in lines[400:]:
        if "DIMENSIONALITY OF THE SYSTEM" in x:
            m = S._DIM_LINE.search(x)
            if m:
                return int(m.group(1))
    return dim


def role_only(st, path: Path) -> str:
    """The role from the head and tail only (cheap: a request thread may ask), cached by stamp."""
    from . import cache as gui_cache
    key = gui_cache.file_key(path, ("filerole", 1))
    hit = st.parse_cache.get(key)
    if hit is not None:
        return hit
    h, t, _ = head_tail(path)
    r = role_of(path.name, h.decode("utf-8", "replace"), t.decode("utf-8", "replace"))
    st.parse_cache.put(key, r, 64)
    return r


def refine_cached(st, path: Path) -> Dict[str, Any]:
    """refine() through the server's parse cache (keyed by the file's stamp)."""
    from . import cache as gui_cache
    key = gui_cache.file_key(path, ("filekind", 1))
    hit = st.parse_cache.get(key)
    if hit is not None:
        return hit
    r = refine(path)
    st.parse_cache.put(key, r, 256)
    return r


def is_refinable(name: str) -> bool:
    """Files whose role is decided by content: outputs and logs."""
    return kind_by_name(name) in ("out", "log")


def assert_not_joblog(st, path: Path):
    """Raise KindConflict for a job log (structure/FREQ/trajectory routes answer 409)."""
    if not is_refinable(path.name):
        return
    if role_only(st, path) == "joblog":
        raise KindConflict("This is a job log, not a CRYSTAL output")


def kind_payload(st, path: Path) -> Dict[str, Any]:
    """GET /api/filekind: the name kind plus the content refinement."""
    kind = kind_by_name(path.name)
    out: Dict[str, Any] = {"kind": kind}
    if kind == "cube":
        out["cube_type"] = cube_type(path.name)
        out["cube_ok"] = cube_header_ok(path)
    if is_refinable(path.name):
        out.update(refine_cached(st, path))
    return out


def info_payload(path: Path, kind: str) -> Dict[str, Any]:
    """What the info document shows for a file MACE does not open (wave
    functions, CRYSTAL's restart data): its size and the run's output beside
    it. Never reads more than 4 KB."""
    size = path.stat().st_size
    stem = path.name
    for suffix in (".f98", ".f9", ".DAT", ".dat", ".LOG", ".log"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    sibling = None
    for cand in (path.with_name(stem + ".out"), path.with_suffix(".out")):
        if cand.is_file() and cand != path:
            sibling = cand
            break
    out = {"kind": kind, "size": size, "sibling": sibling}
    if kind == "f25":
        # fort.25: "-%-0BAND", "-%-1DOSS" or "-%-0MAPN" opens each block (plan E7)
        try:
            with open(path, "rb") as fh:
                head = fh.read(4096).decode("ascii", "replace")
        except OSError:
            head = ""
        m = re.search(r"-%-\d(BAND|DOSS|MAPN)", head)
        out["f25_kind"] = m.group(1) if m else None
        stem = path.name[:-4] if path.name.lower().endswith(".f25") else path.stem
        for suffix, k in ((".BAND.DAT", "BAND"), (".DOSS.DAT", "DOSS")):
            cand = path.with_name(stem + suffix)
            if out["f25_kind"] == k and cand.is_file():
                out["dat"] = cand
    return out


# ------------------------------------------------------------------ properties outputs
_PROPS_SKIP = ("script", "wavefunction", "log", "d3", "d12")
_PROPS_STEP_SUFFIXES = ("_band", "_doss", "_charge+potential", "_transport", "_BAND", "_DOSS", "_TRANSPORT",
                        "_CHARGE+POTENTIAL", ".band", ".doss")


def rel_inside(st, path: Optional[Path]) -> Optional[str]:
    """``path`` relative to the served root, or None when it is missing or leads out of the root
    (a symlink): a file beside another is offered only when the explorer would list it."""
    if path is None:
        return None
    try:
        r = Path(path).resolve()
    except OSError:
        return None
    if r != st.root and st.root not in r.parents or not r.exists():
        return None
    return r.relative_to(st.root).as_posix()


def props_payload(st, path: Path) -> Dict[str, Any]:
    """The properties document (plan E2.5): the run's type (content), its
    status, the data files it wrote (same stem) and the SCF output it ran
    from, when beside it. Head and tail only."""
    info = refine_cached(st, path)
    h, t, size = head_tail(path)
    tail = t.decode("utf-8", "replace") if t else h.decode("utf-8", "replace")
    status = "Finished" if "EEEEEEEEEE TERMINATION" in tail else "No termination line (stopped early or still running)"
    errors = [ln.strip() for ln in tail.splitlines() if "ERROR" in ln and "**" in ln][:3]
    stem = path.stem
    files = []
    for f in sorted(path.parent.iterdir()):
        if f == path or not f.is_file():
            continue
        if not (f.name.startswith(stem + ".") or f.name.startswith(stem + "_")):
            continue
        k = kind_by_name(f.name)
        if not k or k in _PROPS_SKIP or k == "out":
            continue
        rel = rel_inside(st, f)
        if rel is None:
            continue
        files.append({"path": rel, "name": f.name, "kind": k,
                      "cube_type": cube_type(f.name) if k == "cube" else None})
    parent = None
    for suffix in _PROPS_STEP_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            cand = path.with_name(stem[: -len(suffix)] + ".out")
            if cand.is_file() and cand != path and rel_inside(st, cand):
                parent = rel_inside(st, cand)
                break
    return {"path": st.rel(path), "calc": info.get("calc"), "role": info.get("role"), "status": status,
            "errors": errors, "files": files, "parent": parent, "size": size}
