"""The structure every analysis tool reads: a cell, fractional coordinates,
atomic numbers and which directions are periodic.

``load(path)`` reads any file the GUI opens (CIF, D12, CRYSTAL output,
POSCAR, XYZ, fort.34, ...) through mace.gui.structures, so the CLI and the
GUI see the same atoms. ``from_atoms`` takes an ase.Atoms, ``from_payload``
the GUI's structure payload.

Periodic neighbors come from one cKDTree over the atoms and the images of
the cell within the cutoff, which is exact for any triclinic cell (a
fractional-coordinate box query would not be).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

AVOGADRO = 6.02214076e23


def _ase_data():
    from ase import data
    return data


def symbols_of(numbers: Sequence[int]) -> List[str]:
    sym = _ase_data().chemical_symbols
    return [sym[int(z)] for z in numbers]


@dataclass
class Structure:
    cell: np.ndarray                     # rows are the lattice vectors, Angstrom
    frac: np.ndarray                     # (n, 3) fractional coordinates
    numbers: np.ndarray                  # (n,) atomic numbers
    pbc: Tuple[bool, bool, bool] = (True, True, True)
    name: str = "structure"
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.cell = np.asarray(self.cell, float).reshape(3, 3)
        self.frac = np.asarray(self.frac, float).reshape(-1, 3)
        self.numbers = np.asarray(self.numbers, int).reshape(-1)
        self.pbc = tuple(bool(x) for x in self.pbc)
        if len(self.frac) != len(self.numbers):
            raise ValueError("coordinates and atomic numbers differ in length")

    # -- basic properties ---------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.numbers)

    @property
    def cart(self) -> np.ndarray:
        return self.frac @ self.cell

    @property
    def symbols(self) -> List[str]:
        return symbols_of(self.numbers)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    @property
    def dim(self) -> int:
        return int(sum(self.pbc))

    @property
    def mass_amu(self) -> float:
        m = _ase_data().atomic_masses
        return float(sum(m[z] for z in self.numbers))

    @property
    def density(self) -> float:
        """g/cm^3 (only meaningful for a 3D cell)."""
        return self.mass_amu / AVOGADRO / (self.volume * 1e-24)

    def lattice(self) -> Dict[str, float]:
        a, b, c = (np.linalg.norm(v) for v in self.cell)
        ang = lambda u, v: float(np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
        return {"a": float(a), "b": float(b), "c": float(c), "alpha": ang(self.cell[1], self.cell[2]),
                "beta": ang(self.cell[0], self.cell[2]), "gamma": ang(self.cell[0], self.cell[1])}

    def formula(self) -> str:
        from collections import Counter
        c = Counter(self.symbols)
        return "".join(f"{k}{v if v > 1 else ''}" for k, v in sorted(c.items(), key=lambda kv: (kv[0] != "C", kv[0] != "H", kv[0])))

    def wrapped(self) -> "Structure":
        f = self.frac.copy()
        for i in range(3):
            if self.pbc[i]:
                f[:, i] -= np.floor(f[:, i])
                # -1e-17 - floor(-1e-17) rounds to exactly 1.0: keep [0, 1) so wrapping twice
                # gives the same coordinates (bonds() wraps again and its image shifts must match)
                f[f[:, i] >= 1.0, i] = 0.0
        return Structure(self.cell, f, self.numbers, self.pbc, self.name, dict(self.meta))

    def require_3d(self, what: str) -> None:
        if self.dim != 3:
            raise ValueError(f"{what} needs a 3D periodic crystal; this structure is periodic in {self.dim} direction(s)")

    def to_atoms(self):
        from ase import Atoms
        return Atoms(numbers=self.numbers, cell=self.cell, scaled_positions=self.frac, pbc=self.pbc)

    # -- periodic images ------------------------------------------------------
    def image_range(self, cutoff: float) -> List[np.ndarray]:
        """Integer image ranges needed to see every neighbor within cutoff."""
        inv = np.linalg.inv(self.cell)
        # distance between opposite faces = 1 / |row of inv^T|
        widths = 1.0 / np.linalg.norm(inv.T, axis=1)
        out = []
        for i in range(3):
            m = int(np.ceil(cutoff / widths[i])) if self.pbc[i] else 0
            out.append(np.arange(-m, m + 1))
        return out

    def images(self, cutoff: float, frac: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cart, index, shift) of the atoms and every image that can lie
        within ``cutoff`` of the home cell."""
        f = self.wrapped().frac if frac is None else frac
        rng = self.image_range(cutoff)
        shifts = np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T
        n = len(f)
        allf = (f[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        idx = np.tile(np.arange(n), len(shifts))
        sh = np.repeat(shifts, n, axis=0)
        cart = allf @ self.cell
        # keep only images whose fractional position is within the padded box
        inv = np.linalg.inv(self.cell)
        widths = 1.0 / np.linalg.norm(inv.T, axis=1)
        pad = np.where(self.pbc, cutoff / widths, 0.0) + 1e-9
        keep = np.all((allf >= -pad) & (allf <= 1 + pad), axis=1) | np.all(sh == 0, axis=1)
        return cart[keep], idx[keep], sh[keep]

    def neighbor_pairs(self, cutoff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every pair (i, j, shift, distance) with |r_j + shift - r_i| <= cutoff,
        i in the home cell. Each unordered pair appears twice (i->j and j->i)."""
        from scipy.spatial import cKDTree
        w = self.wrapped()
        cart, idx, sh = w.images(cutoff)
        home = w.cart
        tree = cKDTree(cart)
        pairs = tree.query_ball_point(home, cutoff)
        ii, jj = [], []
        for i, lst in enumerate(pairs):
            ii.extend([i] * len(lst))
            jj.extend(lst)
        ii = np.asarray(ii, int)
        jj = np.asarray(jj, int)
        d = np.linalg.norm(cart[jj] - home[ii], axis=1) if len(ii) else np.zeros(0)
        real = d > 1e-6
        return ii[real], idx[jj][real], sh[jj][real], d[real]


def from_atoms(atoms, name: str = "structure") -> Structure:
    pbc = tuple(bool(x) for x in atoms.pbc)
    cell = np.asarray(atoms.cell.array, float)
    if not all(pbc) and abs(np.linalg.det(cell)) < 1e-6:
        # a molecule without a box: a bounding box with 10 Angstrom of vacuum
        pos = atoms.positions
        lo, hi = pos.min(0) - 5.0, pos.max(0) + 5.0
        cell = np.diag(np.maximum(hi - lo, 1.0))
        frac = (pos - lo) / np.diag(cell)
        return Structure(cell, frac, atoms.numbers, pbc, name, {"origin": lo.tolist()})
    return Structure(cell, atoms.get_scaled_positions(wrap=False), atoms.numbers, pbc, name)


def from_payload(p: Dict, name: str = "structure") -> Structure:
    """The GUI's structure payload (structures._payload)."""
    pbc = tuple(p.get("pbc", (True, True, True)))
    cell = np.asarray(p["cell"], float)
    if "frac" in p and p["frac"] is not None and abs(np.linalg.det(cell)) > 1e-6:
        frac = np.asarray(p["frac"], float)
    else:
        frac = np.linalg.solve(cell.T, np.asarray(p["cart"], float).T).T
    return Structure(cell, frac, [int(z) for z in p["numbers"]], pbc, name)


def load(path, frame: int = -1) -> Structure:
    """Any structure file MACE opens."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    from mace.gui import structures as S
    payload = S.read_structure(p, frame)
    return from_payload(payload, p.stem)
