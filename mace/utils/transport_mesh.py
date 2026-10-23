"""Is a BOLTZTRA run's k mesh too coarse? Read from its own files (1.1.6).

CRYSTAL sums band velocities over the NEWK k points into the transport distribution (TDF, manual
eq. 14.4, p.314) on 0.01 eV bins. Too few k points leave the TDF spiky, and σ, S, κ and the Lorenz
number L = κₑ/(σT) follow the spikes. On the HPCC (2026-10-23) the example suite's five materials
converged (σ within 5%, S within 5 µV/K at 300 K) only at meshes far finer than their SCF's: the
Standard spacings of Crystal_d3/d3_config (TRANSPORT_K_STANDARD). Two signs, either one flags a run:

* its NEWK mesh (the .out's last "CAPPA:IS1 n;IS2 n;IS3 n") is below COARSE_FRACTION of the Standard
  IS of its cell (the .out's direct lattice vectors, periodic directions only). At 0.6 it flags every
  suite run more than 6% off in σ (diamond IS 24, h-BN 96, graphene 96, polyacetylene 144) and none of
  the converged ones (SOD IS 10 of 16, h-BN 128 of 194);
* at 300 K, at the chemical potentials inside a band (σ at least 1e-3 of its peak, |S| below
  DEGENERATE_S_UV_K), the median L is outside L0 × (1 ± L_TOLERANCE). The converged runs gave
  1.80 (SOD, a framework's flat bands) to 2.42; diamond's SCF mesh (IS 12) gave 0.94, graphene's 0.01.

The TDF's share of empty bins does not tell them apart (a converged polymer's TDF is still 85% empty:
a 1D density of states is spiky by nature), so it is not used. The note is the transport plot's (GUI,
mace plotting --transport) and the extractor's (transport_mesh_note in materials.db).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

L0 = 2.443                   # Sommerfeld's Lorenz number, 1e-8 W Ohm/K^2
DEGENERATE_S_UV_K = 150.0
L_TOLERANCE = 0.3
COARSE_FRACTION = 0.6
T_CHECK_K = 300.0
_CAPPA = re.compile(r"CAPPA:IS1\s*(\d+);IS2\s*(\d+);IS3\s*(\d+)")


def _d3_config():
    d3 = str(Path(__file__).resolve().parents[2] / "Crystal_d3")
    if d3 not in sys.path:
        sys.path.insert(0, d3)
    import d3_config
    return d3_config


def run_mesh(out_file: Path) -> Tuple[Optional[int], Optional[int]]:
    """(the run's NEWK IS over its periodic directions, the Standard IS of its cell) from its .out."""
    try:
        text = Path(out_file).read_text(errors="replace")
    except OSError:
        return None, None
    hits = _CAPPA.findall(text)
    if not hits:
        return None, None
    C = _d3_config()
    recip = C.reciprocal_lengths(str(out_file))
    if not recip:
        return None, None
    used = max(int(x) for x in hits[-1][:len(recip)])
    return used, C.transport_standard_is(recip)


def band_lorenz(run_file: Path, T_k: float = T_CHECK_K) -> Optional[float]:
    """The median Lorenz number (1e-8 W Ohm/K^2) at T_k over the chemical potentials inside a band."""
    from mace.utils import transport_plot as T
    run_file = Path(run_file)

    def avg(q):
        b = next((b for b in T.quantity_data(run_file, q)["blocks"] if b["T"] == T_k), None)
        return None if b is None else np.array([np.nan if v is None else v for v in b["avg"]], float)
    try:
        sig, S, L = avg("SIGMA"), avg("SEEBECK"), avg("lorenz")
    except Exception:
        return None
    if sig is None or S is None or L is None or not np.isfinite(sig).any():
        return None
    band = np.isfinite(L) & (sig >= 1e-3 * np.nanmax(sig)) & (np.abs(S) < DEGENERATE_S_UV_K)
    return float(np.median(L[band])) if band.sum() >= 5 else None


def run_out(run_file: Path) -> Optional[Path]:
    """The run's .out beside one of its transport files (<stem>.out)."""
    from mace.utils import transport_plot as T
    m = T.NAME_RE.match(Path(run_file).name)
    stem = m.group("stem") if m else Path(run_file).name.split(".")[0]
    out = Path(run_file).with_name(stem + ".out")
    return out if out.is_file() else None


def mesh_note(run_file: Path) -> Optional[str]:
    """Why the run's k mesh looks too coarse for transport, or None."""
    from mace.utils import transport_plot as T
    run_file = Path(run_file)
    try:
        files = T.run_files(run_file)
    except Exception:
        return None
    why = []
    out = run_out(run_file)
    used, std = run_mesh(out) if out is not None else (None, None)
    if used is not None and std is not None and used < COARSE_FRACTION * std:
        why.append(f"its NEWK mesh is IS {used}, well below the Standard mesh of this cell (IS {std})")
    if files.get("SIGMA") is not None and files.get("KAPPA") is not None:
        lm = band_lorenz(run_file)
        if lm is not None and abs(lm / L0 - 1) > L_TOLERANCE:
            why.append(f"the {T_CHECK_K:g} K Lorenz number inside the bands is {lm:.2g} × 1e-8 W Ω/K² (median), "
                       f"where a degenerate band gives Sommerfeld's {L0:.2f}")
    if not why:
        return None
    return ("the k mesh looks too coarse for transport: " + "; ".join(why)
            + ". Rerun the TRANSPORT deck with a finer mesh (mace opt2d3 --transport-k standard or fine)")


_CACHE: dict = {}


def cached_note(run_file: Path) -> Optional[str]:
    """mesh_note, once per run file's state (the GUI asks for every quantity it draws)."""
    p = Path(run_file)
    try:
        key = (str(p.resolve()), p.stat().st_mtime_ns)
    except OSError:
        return None
    if key not in _CACHE:
        try:
            _CACHE[key] = mesh_note(p)
        except Exception:
            _CACHE[key] = None
    return _CACHE[key]


def add_note(data: dict, path: Path) -> dict:
    """The transport plot's data with the mesh note among its notes (and as mesh_note)."""
    note = cached_note(path)
    data["mesh_note"] = note
    if note and note not in data.get("notes", []):
        data.setdefault("notes", []).append(note)
    return data
