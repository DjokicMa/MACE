"""Saving an edited structure into the served folder (spec §1.8, §2.6).

One path for every save the GUI makes (Save, Save As, Save All and the
classic panel's Save CIF button), with the same rules:

* the target is inside ``--dir`` (a subfolder is fine; ``..``, absolute paths
  and symlinks that leave the folder are refused);
* the format matches the extension: CIF (``.cif``, every atom or
  symmetrized), POSCAR (``POSCAR*``, ``CONTCAR*``, ``.vasp``, ``.poscar``) or
  XYZ (``.xyz``, written as extended XYZ so the cell is kept);
* an existing file is replaced only with ``overwrite``, and never when it
  changed on disk since the document was opened (``expected``, the
  ``{mtime_ns, size}`` the browser was given): then the user decides;
* the first save over a file keeps the original as ``<file>.bak``
  (``MgO.cif.bak``), created only when none exists: an older backup is never
  replaced, so it is always the oldest original MACE saw;
* the write is atomic: a temporary file in the same folder, then rename;
* calculation outputs, wavefunctions, data files, databases, MACE job
  scripts and the decks of queued or running jobs are never written
  (``protected_reason``, the one list every GUI write checks);
* the previous version of a replaced file goes to File History first
  (history.py), and the target's stamp is checked again right before the
  rename (a change in between is a conflict, never an overwrite).
"""
from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from urllib.parse import quote
from typing import Any, Dict, Optional

FORMATS = ("cif", "poscar", "xyz")
_POSCAR_NAME = re.compile(r"^(POSCAR|CONTCAR)([._-].*)?$")


class ProtectedOutput(PermissionError):
    """409: the GUI never writes this file (an output, or a queued job's deck)."""


# ------------------------------------------------------------------ protected kinds (PLAN_analysis C19)
_PROTECTED_NAMES = [
    (re.compile(r".+\.o\d+$"), "a SLURM job log"), (re.compile(r".+\.e\d+$"), "a SLURM job log"),
    (re.compile(r"^slurm-.*\.out$", re.I), "a SLURM job log"),
    (re.compile(r".+\.(f9|f98)$", re.I), "a wavefunction"), (re.compile(r"^fort\.(9|98|20)$"), "a wavefunction"),
    (re.compile(r".+\.dat$", re.I), "a CRYSTAL data file"), (re.compile(r"^fort\.25$"), "a CRYSTAL data file"),
    (re.compile(r".+\.f25$", re.I), "a CRYSTAL data file"),
    (re.compile(r".+\.cube$", re.I), "a CRYSTAL grid file"), (re.compile(r".*(ECH3|POT3|SPIN3?)\b.*\.(dat|cube)$", re.I), "a CRYSTAL grid file"),
    (re.compile(r"^SCFOUT\.LOG$", re.I), "CRYSTAL's SCF log"),
    (re.compile(r".+\.(db|sqlite|sqlite3)$", re.I), "a database"),
    (re.compile(r".+-(wal|shm|journal)$"), "a database file"), (re.compile(r".+\.mace-writer\.lock$"), "a database lock"),
]
# a deck whose job is queued or running: PLAN_submit's jobs.py sets this to
# its cached SLURM reader (one squeue per minute at most); without it the
# tracked rows of materials.db decide
JOB_STATE_PROVIDER = None


def is_job_script(path: Path) -> bool:
    """A MACE job script or its generator template (submitcrystal23.sh /
    submit_prop.sh, any version): a ``#!/bin/bash`` first line, a
    ``#SBATCH -J`` line (written or echoed) and the completion callback,
    read from the first 64 KB."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(65536).decode("utf-8", "replace")
    except OSError:
        return False
    first = head.split("\n", 1)[0].strip()
    # the generated <deck>.sh has the line itself; the generator templates echo it
    return (first.startswith("#!/bin/bash") and re.search(r"#SBATCH\s+-J\b", head) is not None
            and "--callback-mode completion" in head)


def _queued_job_reason(path: Path) -> Optional[str]:
    if path.suffix.lower() not in (".d12", ".d3"):
        return None
    if JOB_STATE_PROVIDER is not None:
        try:
            r = JOB_STATE_PROVIDER(path)
            if r:
                return r
        except Exception:
            pass
    # tracked rows: the databases MACE writes beside the decks or above them
    import sqlite3
    forms = {str(path), os.path.abspath(str(path))}
    try:
        forms.add(str(path.resolve()))
    except OSError:
        pass
    seen = set()
    for d in [path.parent, *list(path.parents)[:4]]:
        for db in [d / "materials.db", *sorted(d.glob(".mace_context_*/materials.db"))]:
            if db in seen or not db.is_file():
                continue
            seen.add(db)
            try:
                c = sqlite3.connect(f"file:{quote(str(db.resolve()))}?mode=ro", uri=True, timeout=1)
                try:
                    row = c.execute(f"SELECT calc_id, status, slurm_job_id FROM calculations WHERE input_file IN "
                                    f"({','.join('?' * len(forms))}) AND status IN ('pending', 'submitted', 'running') "
                                    "LIMIT 1", sorted(forms)).fetchone()
                finally:
                    c.close()
            except sqlite3.Error:
                continue
            if row:
                state = "running" if row[1] == "running" else "queued"
                return (f"{path.stem} is {state}; it reads this deck when it starts. Save As a new name, "
                        "or cancel the job first.") if state == "queued" else \
                       (f"{path.stem} is running with this deck. Save As a new name, or cancel the job first.")
    return None


def protected_reason(path: Path) -> Optional[str]:
    """Why the GUI must never write ``path``, or None (PLAN_analysis C19).
    By name (outputs, job logs, wavefunctions, data and grid files,
    databases), by content (an existing .out, a MACE job script) and by
    state (the deck of a queued or running job)."""
    path = Path(path)
    name = path.name
    if path.suffix.lower() == ".out" and path.exists():
        return f"{name} is a calculation output; MACE never writes over outputs."
    for rx, what in _PROTECTED_NAMES:
        if rx.match(name):
            return f"{name} is {what}; MACE never writes over it."
    if path.suffix.lower() == ".sh" and path.is_file() and is_job_script(path):
        return (f"{name} is a MACE job script; it is rewritten only by the command line "
                "(mace submit --overwrite-sh).")
    return _queued_job_reason(path)


class SaveConflict(Exception):
    """409: the caller must decide (overwrite, or the file changed on disk)."""

    def __init__(self, message: str, details: Dict[str, Any]):
        super().__init__(message)
        self.details = details


def file_stamp(path: Path) -> Optional[Dict[str, Any]]:
    """The file as the browser saw it: modification time in ns (a string:
    a JavaScript number cannot hold it exactly) and size."""
    try:
        st = path.stat()
    except OSError:
        return None
    return {"mtime_ns": str(int(st.st_mtime_ns)), "size": int(st.st_size)}


def same_stamp(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    if not a or not b:
        return a == b
    try:
        return str(int(a["mtime_ns"])) == str(int(b["mtime_ns"])) and int(a["size"]) == int(b["size"])
    except (KeyError, TypeError, ValueError):
        return False


def format_problem(name: str, fmt: str) -> Optional[str]:
    low = name.lower()
    if fmt == "cif":
        return None if low.endswith(".cif") else "a CIF file name ends in .cif"
    if fmt == "poscar":
        return None if (_POSCAR_NAME.match(name) or low.endswith((".vasp", ".poscar"))) else \
            "a POSCAR file is named POSCAR or CONTCAR, or ends in .vasp or .poscar"
    if fmt == "xyz":
        return None if low.endswith(".xyz") else "an XYZ file name ends in .xyz"
    return f"format must be one of {', '.join(FORMATS)}"


def render(sess, fmt: str, cif_mode: str = "p1") -> bytes:
    """The session's structure in a file format, as bytes."""
    if fmt == "cif":
        text = sess.cif_text(cif_mode)
        pre = (getattr(sess, "doc_meta", None) or {}).get("preopt")
        if isinstance(pre, dict) and pre.get("method_ran") not in (None, "", "springs"):
            # a document accepted from a GFN-FF/ML Clean: the same marker line as
            # `mace preopt`, so `mace convert` and Setup warn on a non-OPT deck
            from mace.preopt import marker
            text = marker.line(pre["method_ran"], pre.get("model")) + "\n" + text
        return text.encode()
    import ase.io
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt == "poscar":
            ase.io.write(buf, sess.atoms, format="vasp", direct=True, sort=False)
        elif fmt == "plainxyz":       # Export ▸ XYZ: symbols and Cartesian positions only
            try:
                ase.io.write(buf, sess.atoms, format="xyz", plain=True)     # ase routes "xyz" to extxyz
            except TypeError:
                buf = io.StringIO()
                ase.io.write(buf, sess.atoms, format="xyz")                 # an ase whose "xyz" is plain already
        else:                         # "xyz" (Save As keeps the cell) and "extxyz"
            ase.io.write(buf, sess.atoms, format="extxyz")
    return buf.getvalue().encode()


def _write_bytes(fh, data: bytes) -> None:
    """Separate so a test can make the write fail half-way."""
    fh.write(data)


def _umask() -> int:
    """The process umask, read without changing it (os.umask sets it, which
    is not safe with the request threads); 0o022 where /proc is missing."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    return 0o022


_UNSET = object()


def write_atomic(target: Path, data: bytes, expected: Any = _UNSET, reason: str = "Saved from MACE GUI",
                 route: Optional[str] = None, kind: str = "input", doc_title: Optional[str] = None) -> Optional[Any]:
    """Write ``data`` to ``target`` through a temporary file in the same
    folder: a failure leaves the old file as it was, and no partial file.

    An existing target's current version goes to File History first (a
    store that cannot be written refuses the write). With ``expected`` (the
    {mtime_ns, size} the browser was shown, or None for "did not exist"),
    the target is checked again right before the rename; a change in
    between raises SaveConflict and the temporary file is removed. Returns
    the history entry (or None)."""
    from . import history
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    entry = None
    try:
        with os.fdopen(fd, "wb") as fh:
            _write_bytes(fh, data)
            fh.flush()
            os.fsync(fh.fileno())
        existed = target.exists()
        if existed:
            shutil.copymode(target, tmp)
            entry = history.keep(target, reason, route, kind, doc_title)
        else:
            # mkstemp makes the file 0600; a new file gets the usual mode
            # (0666 less the umask), like any file the user's tools write
            os.chmod(tmp, 0o666 & ~_umask())
        if expected is not _UNSET:
            now = file_stamp(target)
            if not same_stamp(now, expected):
                raise SaveConflict(f"{target.name} changed on disk while it was being saved",
                                   {"changed_on_disk": True, **(now or {"mtime_ns": None, "size": None})})
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if not existed:
        history.record_created(target, reason, route)
    return entry


def keep_backup(target: Path) -> Dict[str, Any]:
    """``<file>.bak``, created only if there is none (exclusive create, so a
    backup made at the same moment by another save is not replaced either)."""
    bak = target.with_name(target.name + ".bak")
    created = False
    if not bak.exists():
        try:
            with open(bak, "xb") as out, open(target, "rb") as src:
                shutil.copyfileobj(src, out)
            shutil.copystat(target, bak)
            created = True
        except FileExistsError:
            pass
    st = bak.stat()
    return {"path": bak, "created": created, "t": st.st_mtime}


def save(state, sess, rel: str, fmt: str = "cif", cif_mode: str = "p1", overwrite: bool = False,
         backup: bool = True, expected: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Save the session's structure to ``rel`` (relative to the served folder)."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    if cif_mode not in ("p1", "symmetrized"):
        raise ValueError("CIF mode is p1 or symmetrized")
    target = state.resolve(rel, must_exist=False)
    if target == state.root or target.is_dir():
        raise ValueError(f"{rel} is a folder")
    if hasattr(state, "check_write"):
        state.check_write(Path(os.path.normpath(os.path.join(str(state.root), rel))))  # no hidden folders
    if not target.parent.is_dir():
        raise ValueError(f"the folder {Path(rel).parent.as_posix()} does not exist")
    if hasattr(state, "check_write"):
        state.check_write(target)            # root, symbolic links, protected kinds (before the name rules)
    else:
        prot = protected_reason(target)
        if prot:
            raise ProtectedOutput(prot)
    why = format_problem(target.name, fmt)
    if why:
        raise ValueError(why)
    exists = target.exists()
    if expected is not None:
        now = file_stamp(target)
        if not same_stamp(now, expected):
            raise SaveConflict(f"{target.name} changed on disk since it was opened",
                               {"changed_on_disk": True, **(now or {"mtime_ns": None, "size": None})})
    if exists and not overwrite:
        raise SaveConflict(f"{target.name} already exists", {"needs_overwrite": True})
    data = render(sess, fmt, cif_mode)
    bak = keep_backup(target) if (exists and backup) else None
    title = getattr(sess, "name", None) or getattr(sess, "title", None)
    hist = write_atomic(target, data, expected=(expected if expected is not None else (None if not exists else _UNSET)),
                        reason=f"Saved from {title}" if title else "Saved from the editor", route="save",
                        doc_title=title if isinstance(title, str) else None)
    meta = getattr(sess, "doc_meta", None)
    if meta:
        # where it was built from, and its charge hint: kept by the saved bytes'
        # hash in the user's cache folder (cache.py), never next to the file
        from .cache import write_sidecar
        write_sidecar(data, meta)
    out = {"saved": state.rel(target), "rev": sess.rev, "file": file_stamp(target),
           "bak": state.rel(bak["path"]) if bak else None, "bak_created": bool(bak and bak["created"]),
           "bak_t": bak["t"] if bak else None, "format": fmt, "t": time.time(),
           "history": hist.record() if hist is not None else None}
    return out
