"""The command line of phonopy's collect step, ``mace analyze phonopy DIR``
(mace/analysis/cli.py registers it), and the walltime opt2d12 suggests for a
displacement set (calculation type PHONOPY, Crystal_d12/d12_calc_phonopy.py)."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

INSTALL_HINT = ("phonopy is not installed. Install the optional dependency with\n"
                "    pip install 'phonopy<5'\n"
                "(see requirements-optional.txt).")

COLLECT_EPILOG = """
Checks every output first (present, normal termination, SCF converged, one
force row per atom, printed geometry == its displaced cell) and stops on any
problem; --check stops there. Then writes FORCE_SETS, phonopy_params.yaml,
band.yaml, mesh.yaml, total_dos.dat and thermal_properties.yaml (--plot adds
band_dos.pdf). Frequencies are in THz (x 33.35641 for cm-1).
"""


# Walltime estimate for one displaced-supercell job at mace submit's default 32
# tasks. Calibrated on the heaviest case measured on CRYSTAL23: diamond,
# B3LYP-D3 + SPIN + XLGRID, 2x2x2 (16 atoms) in P1 with SHRINK 6 (216 k-points
# before time reversal) took 3160 s on 64 tasks, i.e. about 6300 s on 32. The
# cost is taken as k-points x atoms^3; 3x3x3 at SHRINK 4 (54 atoms) did not
# finish its SCF within 2 h on 64 tasks, which this puts at about 20 h on 32
# (30 h with the margin below).
_SECONDS_PER_K_ATOM3 = 2 * 3160.0 / (216 * 16 ** 3)
# Margin over that calibration: the 32-task time is extrapolated from the
# 64-task run, and a job killed at its walltime leaves no forces, so a low
# estimate costs the whole set while a high one only queue time (mace submit's
# own default is 7 days). Lighter methods and --cry-sym (measured: Si PBE0
# 4x4x4 at 1.7 h, --cry-sym about 3x cheaper) are not shortened for the same
# reason.
WALLTIME_MARGIN = 1.5
MIN_WALLTIME_HOURS = 2
MAX_WALLTIME_HOURS = 7 * 24          # mace submit's default (submitcrystal23.sh)
FALLBACK_WALLTIME = "1-00:00:00"


def format_walltime(hours: int) -> str:
    """SLURM -t value as mace submit --walltime passes it on: H:MM:SS or D-HH:MM:SS."""
    if hours < 24:
        return f"{hours}:00:00"
    return f"{hours // 24}-{hours % 24:02d}:00:00"


def suggest_walltime(n_kpoints: int, n_atoms: int):
    """(walltime, capped) for one job; capped when the estimate exceeds 7 days."""
    seconds = WALLTIME_MARGIN * _SECONDS_PER_K_ATOM3 * n_kpoints * n_atoms ** 3
    hours = max(MIN_WALLTIME_HOURS, math.ceil(seconds / 3600))
    capped = hours > MAX_WALLTIME_HOURS
    return format_walltime(min(hours, MAX_WALLTIME_HOURS)), capped


def add_collect_arguments(c: argparse.ArgumentParser) -> None:
    """The options of ``mace analyze phonopy`` (the collect step)."""
    c.add_argument("directory", help="the displacement set opt2d12 wrote (--calc-type PHONOPY)")
    c.add_argument("--check", action="store_true", help="only check the outputs")
    c.add_argument("--band", default="auto",
                   help='"auto" (SeeK-path, needs seekpath) or phonopy BAND points, e.g. '
                        '"0 0 0  1/2 0 1/2, 1/2 1/2 1/2  0 0 0"; "none" to skip')
    c.add_argument("--band-labels", help="labels for --band points, space separated")
    c.add_argument("--band-points", type=int, default=51)
    c.add_argument("--mesh", default="100.0", help='length (default 100.0) or "a b c"')
    c.add_argument("--tmin", type=float, default=0.0)
    c.add_argument("--tmax", type=float, default=1000.0)
    c.add_argument("--tstep", type=float, default=10.0)
    c.add_argument("--nac-from", help="Gamma FREQCALC output with INTENS + INTCPHF")
    c.add_argument("--plot", action="store_true", help="write band_dos.pdf")


def _require_phonopy() -> bool:
    try:
        import phonopy  # noqa: F401
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return False
    return True


def _kpoints(lines: List[str]) -> Optional[List[int]]:
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "SHRINK" and i + 1 < len(lines):
            try:
                first = [int(x) for x in lines[i + 1].split()[:2]]
                if first[0] == 0:
                    return [int(x) for x in lines[i + 2].split()[:3]]
                return [first[0]] * 3
            except (ValueError, IndexError):
                return None
    return None


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def cmd_collect(a) -> int:
    from . import collect as C
    from .nac import NacError

    d = Path(a.directory)
    if not (d / "mace_phonopy.json").is_file():
        print(f"Error: {d} has no mace_phonopy.json (not a displacement set written by "
              "mace opt2d12 --calc-type PHONOPY)", file=sys.stderr)
        return 1
    band = None if a.band.lower() == "none" else a.band
    if band == "auto":
        try:
            import seekpath  # noqa: F401
        except ImportError:
            print("Error: --band auto needs seekpath (pip install seekpath); give the path "
                  "with --band or skip it with --band none", file=sys.stderr)
            return 1
    try:
        res = C.collect(d, check_only=a.check, band=band, band_labels=a.band_labels,
                        band_points=a.band_points, mesh=a.mesh, tmin=a.tmin, tmax=a.tmax,
                        tstep=a.tstep, nac_from=a.nac_from, plot=a.plot)
    except NacError as exc:
        print(f"Error: NAC refused: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    bad = [c for c in res.checks if c.problems]
    for c in res.checks:
        state = "ok" if not c.problems else "PROBLEM"
        print(f"  {c.name}.out: {state}")
        for p in c.problems:
            print(f"      - {p}")
    if bad:
        print(f"{len(bad)} of {len(res.checks)} output(s) not usable; nothing written.",
              file=sys.stderr)
        return 1
    if a.check:
        print(f"All {len(res.checks)} output(s) usable.")
        return 0
    for ln in res.drifts:
        print(f"  {ln}")
    print("Wrote: " + ", ".join(res.files))
    if res.min_frequency is not None:
        print(f"Lowest mesh frequency: {res.min_frequency:.4f} THz "
              f"({res.min_frequency * C.CM1_PER_THZ:.1f} cm-1)")
    if res.imaginary:
        worst = min(res.imaginary, key=lambda t: t[1])
        print(f"WARNING: {len(res.imaginary)} q-point(s) away from Gamma with a mode below "
              f"{C.IMAGINARY_THRESHOLD} THz (lowest {worst[1]:.3f} THz at q={worst[0]}): "
              "the structure is dynamically unstable or the supercell is too small. Thermal "
              "properties skip imaginary modes.")
    return 0


def analyze(a) -> int:
    """``mace analyze phonopy DIR``: check the outputs, then forces -> phonons."""
    if not _require_phonopy():
        return 2
    return cmd_collect(a)


#: made when a completion callback starts a set's analysis: the callbacks of
#: jobs that end together run it once
AUTO_MARKER = ".mace_phonopy_analysis"
_ENDED = b"EEEEEEEEEE TERMINATION"


def _ended(out: Path) -> bool:
    try:
        with open(out, "rb") as fh:
            size = out.stat().st_size
            if size > 400_000:
                fh.seek(size - 400_000)
            return _ENDED in fh.read()
    except OSError:
        return False


def analyze_when_complete(directory: Optional[str], job: Optional[str]) -> Optional[int]:
    """The completion callback's step for a phonopy set: when ``job`` is one of
    the set's displaced jobs in ``directory`` and every job of the set has
    ended (CRYSTAL's termination line), run ``mace analyze phonopy DIR`` with
    its defaults (--band auto with seekpath, else none), once. Returns its exit
    status, or None when there was nothing to do. A set already analyzed
    (FORCE_SETS) is left alone; without phonopy where the job ends, it says
    to run the command by hand."""
    if not directory or not job:
        return None
    d = Path(directory)
    if not (d / "mace_phonopy.json").is_file() or (d / "FORCE_SETS").exists():
        return None
    from .writer import read_manifest
    try:
        names = [str(n) for n in read_manifest(d).get("jobs") or []]
    except (OSError, ValueError):
        return None
    if job not in names or not all(_ended(d / f"{n}.out") for n in names):
        return None
    try:
        fd = os.open(str(d / AUTO_MARKER), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return None
    try:
        import phonopy  # noqa: F401
    except ImportError:
        print(f"  phonopy set {d}: every job has ended, but phonopy is not installed where this "
              f"job ends; run: mace analyze phonopy {d}")
        (d / AUTO_MARKER).unlink()
        return None
    p = argparse.ArgumentParser(prog="mace analyze phonopy")
    add_collect_arguments(p)
    try:
        import seekpath  # noqa: F401
        argv = [str(d)]
    except ImportError:
        argv = [str(d), "--band", "none"]
    print(f"  phonopy set {d}: every job has ended; mace analyze phonopy {' '.join(argv)}")
    rc = cmd_collect(p.parse_args(argv))
    if rc != 0:
        # an output to fix (a resubmitted job's callback tries again)
        (d / AUTO_MARKER).unlink()
    return rc
