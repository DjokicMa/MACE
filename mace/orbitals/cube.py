"""Gaussian cube files for orbital grids, plus the job-side cube cache.

Cubes are standard Gaussian cubes (bohr), so ``CubeFile``, ``mace plotting``
and the GUI volume view read them unchanged. The first comment line names the
orbital; the second carries ``MACE ORBITALS`` and key=value fields (k, spin,
MO, energy, occupation, part) that :func:`cube_meta` reads back.

Cached cubes live in ``<job>_orbitals/`` next to the job, one file per request:
``<root>_k<n>_<spin>_MO<a>[-<b>...]_<part>[_<cells>][_<quality>].cube`` (no
cells part for 1x1x1, no quality part for the normal grid). More than six MOs
are named ``MO<first>to<last>h<hash>``, the hash taken over the full MO list so
two different sets never share a file. The ``conv`` field records the basis
convention (:data:`mace.orbitals.basis.CONVENTION`); a cached cube with another
convention or another MO list is recomputed (:func:`cached_cube_matches`).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .grid import GridSpec

MARK = "MACE ORBITALS"
_SPIN_TAG = {"ALPHA+BETA": "ab", "ALPHA": "alpha", "BETA": "beta"}


def write_cube(path, grid: GridSpec, data: np.ndarray, Z: Sequence[int], pos_bohr: np.ndarray,
               title: str, meta: Optional[Dict] = None) -> Path:
    path = Path(path)
    data = np.asarray(data, float).reshape(grid.shape)
    fields = ""
    if meta:
        fields = " ".join(f"{k}={str(v).replace(' ', '')}" for k, v in meta.items())
    nz = grid.shape[2]
    full, rem = divmod(nz, 6)
    row_fmt = ("%13.5E" * 6 + "\n") * full + (("%13.5E" * rem + "\n") if rem else "")
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as f:
        f.write(title.replace("\n", " ")[:200] + "\n")
        f.write(f"{MARK} {fields}".rstrip()[:400] + "\n")
        o = grid.origin
        f.write(f"{len(Z):5d} {o[0]:12.6f} {o[1]:12.6f} {o[2]:12.6f}\n")
        for i in range(3):
            v = grid.vecs[i]
            f.write(f"{grid.shape[i]:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}\n")
        for z, p in zip(Z, pos_bohr):
            f.write(f"{int(z):5d} {float(z):12.6f} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
        flat = data.reshape(-1, nz)
        for row in flat:
            f.write(row_fmt % tuple(row))
    tmp.replace(path)
    return path


def read_cube(path):
    """(GridSpec, data, Z, pos_bohr, comment1, comment2) of a Gaussian cube (single dataset)."""
    with open(path) as f:
        c1 = f.readline().rstrip("\n")
        c2 = f.readline().rstrip("\n")
        t = f.readline().split()
        nat = int(t[0])
        origin = np.array([float(x) for x in t[1:4]])
        shape, vecs = [], []
        for _ in range(3):
            t = f.readline().split()
            shape.append(int(t[0]))
            vecs.append([float(x) for x in t[1:4]])
        Z, pos = [], []
        for _ in range(abs(nat)):
            t = f.readline().split()
            Z.append(int(float(t[0])))
            pos.append([float(x) for x in t[2:5]])
        if nat < 0:
            f.readline()
        data = np.array(f.read().split(), float)
    shape = tuple(shape)
    data = data[: int(np.prod(shape))].reshape(shape)
    return GridSpec(origin, np.array(vecs), shape), data, np.array(Z), np.array(pos), c1, c2


def cube_meta(comment2: str) -> Optional[Dict[str, str]]:
    """key=value fields of a MACE orbital cube's second comment line, or None."""
    if not comment2.startswith(MARK):
        return None
    return dict(re.findall(r"(\w+)=(\S+)", comment2))


def cache_dir_for(job_dir, root: str) -> Path:
    """``<job>_orbitals/`` next to the job (root ``<job>_orb`` -> ``<job>_orbitals``).

    MACE names an ORBITALS step ``<material>_orbitals`` (root
    ``<material>_orbitals_orb``); its folder is ``<material>_orbitals/``, not
    ``..._orbitals_orbitals/``.
    """
    job = root[:-4] if root.endswith("_orb") else root
    if job.lower().endswith("_orbitals"):
        return Path(job_dir) / job
    return Path(job_dir) / f"{job}_orbitals"


def cube_name(root: str, k_index: int, spin: str, mos: Sequence[int], part: str,
              cells: Sequence[int] = (1, 1, 1), molecule: bool = False, quality: str = "normal") -> str:
    sp = _SPIN_TAG.get(spin, spin.lower())
    mo = "-".join(str(m) for m in mos)
    if len(mos) > 6:
        mo = f"{mos[0]}to{mos[-1]}h{hashlib.sha1(mo.encode()).hexdigest()[:8]}"
    k = "" if molecule else f"_k{k_index}"
    sc = "" if tuple(cells) == (1, 1, 1) else "_" + "x".join(str(c) for c in cells)
    q = "" if quality in ("normal", None) else f"_{quality}"
    return f"{root}{k}_{sp}_MO{mo}_{part}{sc}{q}.cube"


def read_cube_header_meta(path) -> Optional[Dict[str, str]]:
    """The MACE fields of a cube's second comment line (None for a foreign cube)."""
    try:
        with open(path) as f:
            f.readline()
            return cube_meta(f.readline().rstrip("\n"))
    except (OSError, UnicodeDecodeError):
        return None


def cached_cube_matches(path, mos: Sequence[int], convention: int) -> bool:
    """True when a cached MACE orbital cube holds exactly ``mos`` under ``convention``.

    A cube MACE did not write (no ``MACE ORBITALS`` line) counts as matching, so
    it is reused and never written over.
    """
    meta = read_cube_header_meta(path)
    if meta is None:
        return True
    if meta.get("conv") != str(convention):
        return False
    return meta.get("mo") == "-".join(str(m) for m in mos)
