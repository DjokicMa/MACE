"""FREQ normal modes for the GUI (plan B5, T9), read without reading the whole output.

``mace.plotting.detect.is_freq_output`` and ``vibmode_viewer.Crystal23FreqParser``
both read the whole file. Here the output's byte-offset index (outindex
version 2) gives the FREQ section's range; a pool worker copies only that
range into a small temporary file (1 MB at a time) and runs the unchanged
``Crystal23FreqParser`` on it. The section is capped at 128 MB. The result is
kept in the parse cache (eigenvectors as float32) and the page asks for one
mode at a time.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

SECTION_MAX = 128 * 1024 * 1024
CHUNK = 1 << 20
# A phonon dispersion (FREQCALC with DISPERSI): modes away from Gamma, whose periodic
# images move with phases; the page then draws one cell only.
OFF_GAMMA = re.compile(rb"PHONON DISPERSION|DISPERSION K POINT|PHONON BANDS")
# CRYSTAL prints each mode at the classical turning point of its vibrational ground state,
# x = sqrt(hbar / (omega mu)) (CRYSTAL23 manual, SCANMODE): a physical amplitude, in bohr
CLASSICAL = b"NORMALIZED TO CLASSICAL AMPLITUDES"
# INTENS + INTCPHF: the Born charges, eps_0 and LO-TO open as the output's dielectric results (1.1.6)
BORN = b"ATOMIC BORN CHARGE TENSOR"


class TooBig(ValueError):
    pass


def parse_job(ctx=None, path: str = "", start: int = 0, end: int = 0, tmpdir: Optional[str] = None, **_) -> Dict[str, Any]:
    """In a pool worker: the FREQ section of ``path`` [start, end) parsed by mace plotting's parser."""
    import numpy as np
    if end - start > SECTION_MAX:
        raise TooBig("FREQ section too large to animate here; use mace plotting --freq")
    fd, tmp = tempfile.mkstemp(prefix="freq_", suffix=".out", dir=tmpdir)
    try:
        with os.fdopen(fd, "wb") as out, open(path, "rb") as src:
            src.seek(start)
            left = end - start
            done = 0
            tail = b""
            gamma = True
            classical = False
            born = False
            while left > 0:
                chunk = src.read(min(CHUNK, left))
                if not chunk:
                    break
                if gamma and OFF_GAMMA.search(tail + chunk):
                    gamma = False
                if not classical and CLASSICAL in tail + chunk:
                    classical = True
                if not born and BORN in tail + chunk:
                    born = True
                tail = chunk[-64:]
                out.write(chunk)
                left -= len(chunk)
                done += len(chunk)
                if ctx is not None:
                    ctx.check()
                    ctx.progress(0.6 * done / max(1, end - start), "Reading the FREQ section")
        if ctx is not None:
            ctx.progress(0.7, "Parsing the normal modes")
        from mace.plotting.engines.vibmode_viewer import Crystal23FreqParser
        p = Crystal23FreqParser(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    modes = [{k: m[k] for k in ("mode", "freq", "irrep", "imaginary", "degeneracy", "ir_active", "ir_intens",
                                "raman_active", "raman_intens")} for m in p.modes]
    disp = {int(k): np.asarray(v, dtype=np.float32) for k, v in p.displacements.items()}
    return {"modes": modes, "displacements": disp, "section_bytes": end - start, "gamma": gamma, "classical": classical,
            "born": born}


from mace.constants import BOHR_TO_ANGSTROM as BOHR  # noqa: E402


def frame_rotation(block_lines, cell) -> Optional[Any]:
    """The rotation that takes CRYSTAL's Cartesian frame (its DIRECT LATTICE VECTORS
    CARTESIAN COMPONENTS, printed with the geometry) to the frame of ``cell`` (the
    lattice as the GUI builds it from the cell parameters): R = A_crystal^-1 A_gui."""
    import numpy as np
    for i, line in enumerate(block_lines):
        if "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS" in line:
            rows = []
            for ln in block_lines[i + 1:i + 6]:
                parts = ln.split()
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue
                if len(vals) == 3:
                    rows.append(vals)
                if len(rows) == 3:
                    break
            if len(rows) != 3:
                return None
            A_c, A_g = np.asarray(rows, float), np.asarray(cell, float)
            if abs(np.linalg.det(A_c)) < 1e-8 or abs(np.linalg.det(A_g)) < 1e-8:
                return None
            R = np.linalg.solve(A_c, A_g)
            # the same lattice in two orientations: R is a rotation
            if np.allclose(R @ R.T, np.eye(3), atol=2e-3):
                return R
            # a slab or a polymer: the padding vector may differ in length (CRYSTAL prints 500 Å, the
            # GUI its own vacuum), so match a and b and the unit normal to them instead
            def frame(A):
                n = np.cross(A[0], A[1])
                ln = np.linalg.norm(n)
                return None if ln < 1e-8 else np.vstack([A[0], A[1], n / ln])
            F_c, F_g = frame(A_c), frame(A_g)
            if F_c is None or F_g is None:
                return None
            R = np.linalg.solve(F_c, F_g)
            return R if np.allclose(R @ R.T, np.eye(3), atol=2e-3) else None
    return None


def mode_payload(parsed: Dict[str, Any], k: int) -> Dict[str, Any]:
    """One mode: its row and its displacement per atom in Å, in the GUI's Cartesian frame.
    ``classical``: the output printed the modes normalized to classical amplitudes (the zero-point
    turning point, a physical displacement); False when it did not say so."""
    import numpy as np
    m = next((x for x in parsed["modes"] if x["mode"] == k), None)
    if m is None:
        raise KeyError(f"mode {k}")
    d = parsed["displacements"].get(k)
    if d is not None:
        d = np.asarray(d, float) * BOHR
        R = parsed.get("rotation")
        if R is not None:
            d = d @ np.asarray(R, float)
    return {**m, "displacement": None if d is None else [[round(float(v), 6) for v in row] for row in d],
            "units": "angstrom", "frame": "gui" if parsed.get("rotation") is not None else "crystal",
            "classical": bool(parsed.get("classical", True))}


def nbytes(parsed: Dict[str, Any]) -> int:
    return 512 * len(parsed["modes"]) + sum(int(v.nbytes) for v in parsed["displacements"].values())
