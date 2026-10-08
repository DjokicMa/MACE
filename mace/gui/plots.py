"""Results plots for the GUI, made by ``mace plotting`` itself.

The band, DOS and spectra plotters chdir into the data file's directory and
write their PNG/SVG beside it. The GUI never writes into the browsed
directory, so it copies the data file (and the small files of the same run
the DOS plotter reads: .d3, .out) into a folder of its own cache directory
and runs ``mace plotting --band|--dos|--ir|--raman`` there, as a cancelable
``plot`` task (clirun: nice 10, one CPU slot, SIGTERM then SIGKILL). Those
folders count against the one memory budget (cache.DiskCache): the cache
directory is under /tmp, often RAM.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from . import clirun

# kind -> (mace plotting flag, data-file suffixes that name the run)
KINDS = {
    "band": ("--band", (".BAND.DAT", ".band.band.dat", "_band.dat")),
    "dos": ("--dos", (".DOSS.DAT", "_doss.dat")),
    "ir": ("--ir", ("_IRSPEC.DAT", ".IRSPEC.DAT", "IRSPEC.DAT")),
    "raman": ("--raman", ("_RAMSPEC.DAT", ".RAMSPEC.DAT", "RAMSPEC.DAT")),
}
FORMATS = ("svg", "png", "pdf")
_SIBLING_SUFFIXES = (".d3", ".out", ".d12")
_SIBLING_MAX = 20 * 1024 * 1024


def kind_of(path: Path) -> Optional[str]:
    """'band' / 'dos' / 'ir' / 'raman' when mace plotting's registry classifies the file so."""
    try:
        from mace.plotting import handlers  # noqa: F401  (registers the plotters)
        from mace.plotting.detect import classify_file
        kind = classify_file(str(path))
    except Exception:
        kind = None
    if kind is None:
        return None
    return {"band": "band", "dos": "dos", "ir": "ir", "raman": "raman"}.get(kind.value)


def _run_stem(name: str, kind: str) -> str:
    for suffix in KINDS[kind][1]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def render_key(path: Path, kind: str, fmt: str = "") -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{kind}|{fmt}".encode()).hexdigest()[:16]


def command(path: Path, kind: str, fmt: str = "") -> str:
    return f"mace plotting {KINDS[kind][0]} {path.name}" + (f" --format {fmt}" if fmt else "")


def render(path: Path, kind: str, disk, ctx: Any = None, fmt: str = "") -> Dict[str, Any]:
    """Plot ``path`` with mace plotting into the disk cache; the image files it made."""
    if kind not in KINDS:
        raise ValueError(f"cannot render {kind!r} plots")
    if fmt and fmt not in FORMATS:
        raise ValueError(f"format is one of {', '.join(FORMATS)}")
    key = render_key(path, kind, fmt)
    work = disk.acquire(key)
    ok = False
    try:
        done = work / ".done"
        if not done.exists():
            for p in list(work.iterdir()):
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
            if ctx is not None:
                ctx.progress(0.1, "Copying the data files")
            shutil.copy2(path, work / path.name)
            stem = _run_stem(path.name, kind)
            for sib in path.parent.iterdir():
                if (sib.is_file() and sib.name.startswith(stem) and sib != path
                        and sib.suffix in _SIBLING_SUFFIXES
                        and sib.stat().st_size < _SIBLING_MAX):
                    shutil.copy2(sib, work / sib.name)
            if ctx is not None:
                ctx.progress(0.3, f"Running {command(path, kind, fmt)}")
            args = ["--no-banner", "plotting", KINDS[kind][0], str(work / path.name)]
            if fmt:
                args += ["--format", fmt]
            result = clirun.run(clirun.mace_argv(*args), work, ctx, timeout=300)
            (work / "log.txt").write_text(result.log)
            done.write_text(str(result.returncode))
        if ctx is not None:
            ctx.progress(0.9, "Reading the images")
        images = sorted(p.name for p in work.iterdir() if p.suffix.lower() in (".svg", ".png", ".pdf"))
        log = (work / "log.txt").read_text() if (work / "log.txt").exists() else ""
        ok = True
        return {"key": key, "images": images, "log": log.replace(str(work) + "/", "")[-4000:],
                "command": command(path, kind, fmt)}
    finally:
        # a canceled or failed render leaves nothing half-made behind
        disk.release(key, remove=not ok)


def cached_file(cache: Path, key: str, name: str) -> Optional[Path]:
    """A file render() produced, or None (no path tricks)."""
    if not key.isalnum() or "/" in name or "\\" in name or name.startswith("."):
        return None
    target = cache / key / name
    return target if target.is_file() else None
