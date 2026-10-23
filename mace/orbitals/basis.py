"""Gaussian basis of a Molden file: angular functions, analytic overlap, self-check.

Conventions (established on CRYSTAL23 files by the C^H S C = 1 test):

* contraction coefficients multiply raw primitives ``r^l exp(-a r^2)`` times a
  Cartesian monomial; s and p contractions are normalized as written;
* spherical d and f are real solid harmonics in Molden order
  (``d0 d+1 d-1 d+2 d-2``; ``f0 f+1 f-1 f+2 f-2 f+3 f-3``). A d function is
  normalized like the ``xy`` monomial, which is what the contraction coefficient
  normalizes. CRYSTAL writes an f coefficient of ``-N_xyz/sqrt(40)`` (bcc Fe,
  exp 1.598: -2.588 = -16.37/sqrt(40)), so each f function is normalized to
  norm^2 40 in ``xyz`` units (the raw ``4xz^2 - x^3 - xy^2`` of f+1) with the
  opposite sign; all-MO ``C^H S C - 1`` on Fe drops from 0.975 to 2e-4;
* Cartesian shells (no ``[5D]``) use the monomials with coefficient 1.

The analytic overlap ``S(k) = sum_R exp(+2 pi i k.n) S(R)`` (Obara-Saika) is
the load-time self-check: ``C^H S(k) C = 1`` over the occupied orbitals (plus
two virtuals) of each spin channel, and an on-site norm of 1 for every basis
function; each orbital that is drawn is checked again on its own. Good files pass to 1e-6..1e-4; CRYSTAL's broken outputs (complex k
in 2D/3D, ILOC 1) fail at 0.1..1.
"""
from __future__ import annotations

import itertools
import math
from typing import List, Optional

import numpy as np

from mace.constants import BOHR_TO_ANGSTROM

from .molden import MoldenData, MoldenError

CART = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    3: [(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 0), (2, 1, 0), (2, 0, 1),
        (1, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1)],
}

_D = [  # d0 d+1 d-1 d+2 d-2
    {(0, 0, 2): 2.0, (2, 0, 0): -1.0, (0, 2, 0): -1.0},
    {(1, 0, 1): 1.0},
    {(0, 1, 1): 1.0},
    {(2, 0, 0): 1.0, (0, 2, 0): -1.0},
    {(1, 1, 0): 1.0},
]
_F = [  # f0 f+1 f-1 f+2 f-2 f+3 f-3
    {(0, 0, 3): 2.0, (2, 0, 1): -3.0, (0, 2, 1): -3.0},
    {(1, 0, 2): 4.0, (3, 0, 0): -1.0, (1, 2, 0): -1.0},
    {(0, 1, 2): 4.0, (2, 1, 0): -1.0, (0, 3, 0): -1.0},
    {(2, 0, 1): 1.0, (0, 2, 1): -1.0},
    {(1, 1, 1): 1.0},
    {(3, 0, 0): 1.0, (1, 2, 0): -3.0},
    {(2, 1, 0): 3.0, (0, 3, 0): -1.0},
]


def _dfact(n: int) -> float:
    return 1.0 if n <= 0 else n * _dfact(n - 2)


def _poly_norm2(poly) -> float:
    """Angular norm^2 of a polynomial in units of the Gaussian moments (2a-1)!!(2b-1)!!(2c-1)!!."""
    s = 0.0
    for (l1, c1), (l2, c2) in itertools.product(poly.items(), repeat=2):
        tot = [a + b for a, b in zip(l1, l2)]
        if any(t % 2 for t in tot):
            continue
        v = 1.0
        for t in tot:
            v *= _dfact(t - 1)
        s += c1 * c2 * v
    return s


def _normalized(poly, ref):
    n = math.sqrt(_poly_norm2(poly) / _poly_norm2(ref))
    return {k: v / n for k, v in poly.items()}


# f: norm^2 40 xyz units and the opposite sign, matching CRYSTAL's -N_xyz/sqrt(40) coefficient
_F_SCALE = -math.sqrt(40.0)
_SPH = {2: [_normalized(p, {(1, 1, 0): 1.0}) for p in _D],
        3: [{k: _F_SCALE * v for k, v in _normalized(p, {(1, 1, 1): 1.0}).items()} for p in _F]}

# Bumped whenever the evaluated psi changes for the same file (cube meta + cache key)
CONVENTION = 2


def transform(l: int, spherical: bool) -> np.ndarray:
    """Matrix (nfunc, ncart) from the Cartesian monomials CART[l] to the shell's functions."""
    if l > 3:
        raise MoldenError(f"angular momentum l={l} (g or higher) is not supported")
    carts = CART[l]
    if l < 2 or not spherical:
        return np.eye(len(carts))
    T = np.zeros((2 * l + 1, len(carts)))
    for i, poly in enumerate(_SPH[l]):
        for lmn, c in poly.items():
            T[i, carts.index(lmn)] = c
    return T


class Basis:
    """The basis functions of a Molden file placed at given atom positions (bohr)."""

    def __init__(self, md: MoldenData, positions_bohr: Optional[np.ndarray] = None):
        self.md = md
        self.pos = md.positions_bohr() if positions_bohr is None else np.asarray(positions_bohr, float)
        self.periodic = md.periodic_dims
        self.axes = (md.axes_ang / BOHR_TO_ANGSTROM) if md.axes_ang is not None else None  # bohr rows
        self.shells = []
        off = 0
        for sh in md.shells:
            sph = (sh.l == 2 and md.spherical_d) or (sh.l == 3 and md.spherical_f)
            T = transform(sh.l, sph)
            self.shells.append({"atom": sh.atom, "l": sh.l, "exps": sh.exps, "coefs": sh.coefs,
                                "T": T, "off": off, "n": T.shape[0]})
            off += T.shape[0]
        self.nbf = off
        if self.nbf != md.nbf:
            raise MoldenError("basis function count mismatch")

    # ---------------- lattice ----------------
    def lattice_axes(self) -> np.ndarray:
        """3x3 rows (bohr); zero rows along non-periodic directions."""
        return np.zeros((3, 3)) if self.axes is None else self.axes

    def cells_within(self, rmax: float) -> np.ndarray:
        """Integer cell indices n (periodic directions only) with |n.A| < rmax."""
        if not self.periodic:
            return np.zeros((1, 3), int)
        A = self.lattice_axes()
        rng = []
        for i in range(3):
            if i in self.periodic:
                # distance between planes of the lattice along i bounds the index range
                others = [A[j] for j in self.periodic if j != i]
                if len(others) == 2:
                    nrm = np.cross(others[0], others[1])
                    h = abs(A[i] @ nrm) / np.linalg.norm(nrm)
                elif len(others) == 1:
                    o = others[0] / np.linalg.norm(others[0])
                    h = np.linalg.norm(A[i] - (A[i] @ o) * o)
                else:
                    h = np.linalg.norm(A[i])
                N = int(math.ceil(rmax / h)) + 1
                rng.append(np.arange(-N, N + 1))
            else:
                rng.append(np.array([0]))
        ns = np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T
        R = ns @ A
        return ns[np.linalg.norm(R, axis=1) < rmax]

    # ---------------- analytic overlap ----------------
    def overlap_cells(self, tol: float = 1e-17):
        """(ns, S_R) with S_R[c, mu, nu] = <phi_mu(r - tau_mu) | phi_nu(r - tau_nu - R_c)>."""
        amin = min(float(sh["exps"].min()) for sh in self.shells)
        pos = self.pos
        span = 0.0
        if len(pos) > 1:
            span = float(np.max(np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)))
        rmax = math.sqrt(math.log(1.0 / tol) / (amin / 2.0)) + span
        ns = self.cells_within(rmax)
        R = ns @ self.lattice_axes()
        nR = len(ns)
        S = np.zeros((nR, self.nbf, self.nbf))
        for A, B in itertools.product(self.shells, repeat=2):
            if B["off"] < A["off"]:
                continue
            AB = pos[A["atom"]][None, :] - (pos[B["atom"]][None, :] + R)      # (nR, 3)
            blk = self._shell_pair_overlap(A, B, AB)                         # (nR, na, nb)
            S[:, A["off"]:A["off"] + A["n"], B["off"]:B["off"] + B["n"]] = blk
        # S_R(-R)[nu, mu] = S_R(R)[mu, nu]: fill the lower shell blocks from the mirrored cell
        index = {tuple(n): i for i, n in enumerate(ns)}
        mirror = np.array([index.get(tuple(-n), -1) for n in ns])
        for A, B in itertools.product(self.shells, repeat=2):
            if B["off"] >= A["off"]:
                continue
            ok = mirror >= 0
            sa = slice(A["off"], A["off"] + A["n"])
            sb = slice(B["off"], B["off"] + B["n"])
            S[ok, sa, sb] = np.transpose(S[mirror[ok], sb, sa], (0, 2, 1))
        return ns, S

    @staticmethod
    def _shell_pair_overlap(A, B, AB):
        la, lb = A["l"], B["l"]
        a = A["exps"][:, None, None]
        b = B["exps"][None, :, None]
        p = a + b
        mu = a * b / p
        cc = (A["coefs"][:, None] * B["coefs"][None, :])[:, :, None]
        tables = []
        for d in range(3):
            X = AB[:, d][None, None, :]
            PA = -b / p * X
            PB = a / p * X
            T = [[None] * (lb + 1) for _ in range(la + 1)]
            T[0][0] = np.sqrt(np.pi / p) * np.exp(-mu * X * X)
            for i in range(la + 1):
                for j in range(lb + 1):
                    if i == 0 and j == 0:
                        continue
                    if i > 0:
                        v = PA * T[i - 1][j]
                        if i > 1:
                            v = v + (i - 1) / (2 * p) * T[i - 2][j]
                        if j > 0:
                            v = v + j / (2 * p) * T[i - 1][j - 1]
                    else:
                        v = PB * T[i][j - 1]
                        if j > 1:
                            v = v + (j - 1) / (2 * p) * T[i][j - 2]
                    T[i][j] = v
            tables.append(T)
        ca, cb = CART[la], CART[lb]
        Sc = np.zeros((AB.shape[0], len(ca), len(cb)))
        for i, m1 in enumerate(ca):
            for j, m2 in enumerate(cb):
                prod = cc * tables[0][m1[0]][m2[0]] * tables[1][m1[1]][m2[1]] * tables[2][m1[2]][m2[2]]
                Sc[:, i, j] = prod.sum(axis=(0, 1))
        return np.einsum("ia,rab,jb->rij", A["T"], Sc, B["T"])


def bloch_overlap(ns: np.ndarray, S_R: np.ndarray, kfrac) -> np.ndarray:
    """S(k) = sum_R exp(+2 pi i k.n) S(R)."""
    ph = np.exp(2j * math.pi * (ns @ np.asarray(kfrac, float)))
    return np.tensordot(ph, S_R, axes=(0, 0))


def orthonormality_error(C: np.ndarray, S: np.ndarray) -> float:
    """max |C^H S C - 1| for the columns of C."""
    if C.shape[1] == 0:
        return 0.0
    M = C.conj().T @ S @ C
    return float(np.abs(M - np.eye(M.shape[0])).max())
