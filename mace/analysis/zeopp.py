"""Zeo++ (Willems, Rycroft, Kazi, Meza, Haranczyk, Microporous Mesoporous
Mater. 149, 134 (2012)) as an optional second porosity engine.

MACE never bundles or installs Zeo++: it runs the ``network`` program when
one is found (the MACE_ZEOPP environment variable naming the binary, else
``network`` on PATH, checked by its usage text so an unrelated program of
the same name is not used). Zeo++ is distributed by LBNL under a BSD-style
license; build it from http://www.zeoplusplus.org.

The structure goes in as a P1 .cssr file and the radii as a .rad file, so
Zeo++ uses exactly the radii the native run used. Results are reported in
the same keys as porosity.analyze.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .structure import Structure

INSTALL_HINT = ("Zeo++ is not installed: build it from http://www.zeoplusplus.org (make in voro++/src, then make), "
                "then put `network` on PATH or set MACE_ZEOPP=/path/to/network")
_CACHE: Dict[str, Optional[str]] = {}


def find() -> Optional[str]:
    """The Zeo++ `network` binary, or None."""
    key = os.environ.get("MACE_ZEOPP", "") + "|" + os.environ.get("PATH", "")
    if key in _CACHE:
        return _CACHE[key]
    cands = []
    env = os.environ.get("MACE_ZEOPP")
    if env:
        cands.append(env)
    for name in ("network", "zeo++", "zeopp"):
        p = shutil.which(name)
        if p:
            cands.append(p)
    found = None
    for c in cands:
        try:
            r = subprocess.run([c], capture_output=True, text=True, timeout=20)
            if "invocation syntax" in (r.stdout + r.stderr) or "-ha" in (r.stdout + r.stderr):
                found = c
                break
        except (OSError, subprocess.SubprocessError):
            continue
    _CACHE[key] = found
    return found


def write_cssr(st: Structure, path: Path) -> None:
    L = st.lattice()
    lines = [f"\t\t\t\t{L['a']:.6f}  {L['b']:.6f}  {L['c']:.6f}",
             f"\t\t{L['alpha']:.6f}  {L['beta']:.6f}  {L['gamma']:.6f}  SPGR =  1 P 1\t\t OPT = 1",
             f"{st.n}   0 ", f"0 {st.name or 'mace'}\t: {st.name or 'mace'}"]
    # Zeo++ builds its own cell from a, b, c, alpha, beta, gamma: fractional coordinates carry over
    f = st.wrapped().frac
    for i, (s, x) in enumerate(zip(st.symbols, f)):
        lines.append(f" {i + 1} {s} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}  0  0  0  0  0  0  0  0  0.000")
    path.write_text("\n".join(lines) + "\n")


def _floats_after(text: str, key: str) -> Optional[float]:
    m = re.search(re.escape(key) + r"\s*(-?[0-9.eE+-]+)", text)
    return float(m.group(1)) if m else None


def analyze(st: Structure, probe="N2", radii="ccdc", radius_overrides=None, samples: int = 50000,
            surface_samples: int = 2000, high_accuracy: bool = True, binary: Optional[str] = None,
            timeout: float = 3600) -> Dict:
    """Run Zeo++ (-resex, -chan, -sa, -vol, -volpo) for one probe."""
    from .porosity import atom_radii, probe_radius
    exe = binary or find()
    if not exe:
        raise RuntimeError(INSTALL_HINT)
    st.require_3d("Zeo++")
    r_p, label = probe_radius(probe)
    R = atom_radii(st, radii, radius_overrides)
    with tempfile.TemporaryDirectory(prefix="mace-zeopp-") as td:
        t = Path(td)
        write_cssr(st, t / "s.cssr")
        rad = {}
        for s, r in zip(st.symbols, R):
            rad[s] = r
        (t / "s.rad").write_text("".join(f"{k} {v:.6f}\n" for k, v in rad.items()))
        cmd = [exe] + (["-ha"] if high_accuracy else []) + ["-r", "s.rad",
               "-resex", "o.res", "-chan", f"{r_p}", "o.chan",
               "-sa", f"{r_p}", f"{r_p}", str(int(surface_samples)), "o.sa",
               "-vol", f"{r_p}", f"{r_p}", str(int(samples)), "o.vol",
               "-volpo", f"{r_p}", f"{r_p}", str(int(samples)), "o.volpo", "s.cssr"]
        r = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not (t / "o.res").is_file():
            raise RuntimeError(f"Zeo++ failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
        res = (t / "o.res").read_text().split()
        sa = (t / "o.sa").read_text()
        vol = (t / "o.vol").read_text()
        volpo = (t / "o.volpo").read_text()
        chan = (t / "o.chan").read_text()
    nums = [float(x) for x in res[1:10]]
    V = _floats_after(vol, "Unitcell_volume:")
    m = re.search(r"(\d+)\s+channels identified of dimensionality\s+(\S+)", chan)
    dims = [int(x) for x in re.findall(r"dimensionality\s+(\d+)", chan)]
    out = {
        "method": "zeo++", "binary": exe, "command": " ".join(cmd),
        "probe": {"name": label, "radius": r_p},
        "cell_volume_A3": V, "density_g_cm3": _floats_after(vol, "Density:"),
        "Di": nums[0], "Df": nums[1], "Dif": nums[2], "Df_axes": nums[3:6],
        "AV": {"A3": _floats_after(vol, "AV_A^3:"), "fraction": _floats_after(vol, "AV_Volume_fraction:"),
               "cm3_g": _floats_after(vol, "AV_cm^3/g:")},
        "NAV": {"A3": _floats_after(vol, "NAV_A^3:"), "fraction": _floats_after(vol, "NAV_Volume_fraction:"),
                "cm3_g": _floats_after(vol, "NAV_cm^3/g:")},
        "POAV": {"A3": _floats_after(volpo, "POAV_A^3:"), "fraction": _floats_after(volpo, "POAV_Volume_fraction:"),
                 "cm3_g": _floats_after(volpo, "POAV_cm^3/g:")},
        "PONAV": {"A3": _floats_after(volpo, "PONAV_A^3:"), "fraction": _floats_after(volpo, "PONAV_Volume_fraction:"),
                  "cm3_g": _floats_after(volpo, "PONAV_cm^3/g:")},
        "ASA": {"A2": _floats_after(sa, "ASA_A^2:"), "m2_cm3": _floats_after(sa, "ASA_m^2/cm^3:"),
                "m2_g": _floats_after(sa, "ASA_m^2/g:")},
        "NASA": {"A2": _floats_after(sa, "NASA_A^2:"), "m2_cm3": _floats_after(sa, "NASA_m^2/cm^3:"),
                 "m2_g": _floats_after(sa, "NASA_m^2/g:")},
        "n_channels": int(m.group(1)) if m else 0,
        "channel_dimensionality": max(dims) if dims else 0,
        "n_pockets": int(_floats_after(sa, "Number_of_pockets:") or 0),
    }
    return out


def compare(native: Dict, zeo: Dict) -> Dict[str, Dict[str, float]]:
    """Side-by-side numbers of a native and a Zeo++ run."""
    rows = {}
    for k in ("Di", "Df", "Dif"):
        rows[k] = {"native": native[k], "zeo++": zeo[k], "diff": native[k] - zeo[k]}
    for k in ("AV", "NAV", "POAV", "PONAV"):
        a, b = native[k]["fraction"], zeo[k]["fraction"] or 0.0
        rows[k + "_fraction"] = {"native": a, "zeo++": b, "diff": a - b}
    for k in ("ASA", "NASA"):
        a, b = native[k]["m2_cm3"], zeo[k]["m2_cm3"] or 0.0
        rows[k + "_m2_cm3"] = {"native": a, "zeo++": b, "diff": a - b}
    return rows
