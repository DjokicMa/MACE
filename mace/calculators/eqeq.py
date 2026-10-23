"""EQeq: extended charge equilibration for periodic structures, native.

C. E. Wilmer, K. C. Kim and R. Q. Snurr, An extended charge equilibration
method, J. Phys. Chem. Lett. 3, 2506 (2012). Point charges in one linear
solve, no iteration, from measured ionization energies:

* each atom's electronegativity and hardness come from the two ionization
  energies around its charge center C (its common oxidation state; 0 for
  most elements): J = IP(C+1) - IP(C), X = (IP(C+1) + IP(C))/2 - C J, with
  IP(0) the electron affinity. Hydrogen uses an electron affinity of
  -2.0 eV (the paper's choice);
* the pair interaction is lambda (k/2) (1/R + an orbital-overlap term
  exp(-a^2 R^2)(2a - a^2 R - 1/R)), a = sqrt(Ji Jj)/k, k = 14.4 eV Å,
  lambda = 1.2 (dielectric screening);
* equal electronegativity everywhere, X_i + sum_j H_ij q_j = mu, with the
  total charge fixed, gives the charges.

The periodic sums follow the published method exactly, including its
defaults (``method="ewald"``: Ewald splitting eta = 50 Å, the real-space sum
over -2..2 cells along each axis and the reciprocal sum over -2..2 reciprocal
vectors, whatever the cell), so the charges are the published method's:
tests/calculators/test_eqeq.py checks IRMOF-1 against the authors' reference
program to 1e-4 e. ``method="direct"`` sums 1/R over the images along the
periodic axes only (slabs, wires); a molecule (no periodic axis) uses the
plain pair sums.

The ionization data are physical constants (``eqeq_data``); MACE's code is
its own (the authors' reference program is GPL and is not used).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import base, eqeq_data as D

K_COULOMB = 14.4
LAMBDA = 1.2
ETA = 50.0
CELLS = 2
CITATIONS = [{"text": "C. E. Wilmer, K. C. Kim and R. Q. Snurr, An extended charge equilibration method, J. Phys. "
                      "Chem. Lett. 3, 2506 (2012)", "doi": "10.1021/jz3008485"}]


class EQeqError(ValueError):
    pass


def atomic_params(symbols: List[str], charge_centers: Optional[Dict[str, int]] = None,
                  h_ea: float = D.H_EA) -> Dict[str, Any]:
    """(X, J, centers, warnings) per atom."""
    centers = dict(D.CHARGE_CENTERS)
    centers.update({k: int(v) for k, v in (charge_centers or {}).items()})
    X, J, cc, warn = [], [], [], []
    seen = set()
    for s in symbols:
        if s == "H":
            j = D.H_IP - h_ea
            X.append(0.5 * (D.H_IP + h_ea))
            J.append(j)
            cc.append(0)
            continue
        ion = D.IONIZATION.get(s)
        c = centers.get(s, 0)
        if ion is None:
            raise EQeqError(f"EQeq has no ionization data for {s} (only H to Po)")
        if c < 0 or c + 1 >= len(ion) or ion[c] is None or ion[c + 1] is None:
            raise EQeqError(f"EQeq has no ionization energies for {s} around charge center {c}")
        j = ion[c + 1] - ion[c]
        X.append(0.5 * (ion[c + 1] + ion[c]) - c * j)
        J.append(j)
        cc.append(c)
        if s in D.UNCERTAIN and s not in seen:
            seen.add(s)
            warn.append(f"{s}: the ionization data are partly estimates ({D.UNCERTAIN[s]})")
    return {"X": np.array(X), "J": np.array(J), "centers": cc, "warnings": warn}


def _overlap(R, a):
    return np.exp(-(a * a * R * R)) * (2 * a - a * a * R - 1.0 / R)


def _images(cell, mask, m):
    rng = [range(-m, m + 1) if p else range(0, 1) for p in mask]
    return np.array([(u, v, w) for u in rng[0] for v in rng[1] for w in rng[2]], float)


def matrix(atoms, J, *, method: str = "ewald", lam: float = LAMBDA, eta: float = ETA, cells: int = CELLS,
           recip_cells: int = CELLS, canceled: Optional[Callable[[], bool]] = None) -> np.ndarray:
    pos = atoms.positions
    n = len(atoms)
    cell = atoms.cell.array
    pref = lam * K_COULOMB / 2.0
    A = np.sqrt(np.outer(J, J)) / K_COULOMB
    dr = pos[:, None, :] - pos[None, :, :]
    H = np.zeros((n, n))
    off = ~np.eye(n, dtype=bool)
    if method == "nonperiodic":
        R = np.linalg.norm(dr, axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            H[off] = pref * (1.0 / R[off] + _overlap(R[off], A[off]))
        H[np.diag_indices(n)] = J
        return H
    mask = list(atoms.pbc) if method == "direct" else [True, True, True]
    T = _images(cell, mask, cells) @ cell
    zero = np.all(T == 0, axis=1)
    diag = np.zeros(n)
    aii = J / K_COULOMB
    from scipy.special import erfc
    for t, z in zip(T, zero):
        if canceled is not None and canceled():
            raise base.Canceled("canceled")
        R = np.linalg.norm(dr + t, axis=2)
        if method == "ewald":
            if z:
                H[off] += erfc(R[off] / eta) / R[off] + _overlap(R[off], A[off])
            else:
                H += erfc(R / eta) / R + _overlap(R, A)
                Rt = float(np.linalg.norm(t))
                diag += erfc(Rt / eta) / Rt + _overlap(Rt, aii)
        else:
            if z:
                H[off] += 1.0 / R[off] + _overlap(R[off], A[off])
            else:
                H += 1.0 / R + _overlap(R, A)
                Rt = float(np.linalg.norm(t))
                diag += 1.0 / Rt + _overlap(Rt, aii)
    if method == "ewald":
        rec = 2 * np.pi * np.linalg.inv(cell).T
        V = abs(np.linalg.det(cell))
        Gs = _images(cell, [True] * 3, recip_cells) @ rec
        beta_self = 0.0
        for g in Gs:
            h = float(np.linalg.norm(g))
            if h == 0.0:
                continue
            w = np.exp(-(0.5 * h * eta) ** 2) / (h * h)
            if w < 1e-300:
                continue
            H += 4 * np.pi / V * w * np.cos(dr @ g)
            beta_self += 4 * np.pi / V * w
        H[np.diag_indices(n)] = 0.0
        diag += beta_self - 2.0 / (eta * np.sqrt(np.pi))
    H *= pref
    H[np.diag_indices(n)] = J + pref * diag
    return H


def solve(H, X, total_charge: float = 0.0) -> np.ndarray:
    n = len(X)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = H
    A[:n, n] = -1.0
    A[n, :n] = 1.0
    b = np.concatenate([-np.asarray(X, float), [float(total_charge)]])
    sol = np.linalg.solve(A, b)
    return sol[:n]


def default_method(atoms) -> str:
    pbc = list(atoms.pbc)
    if all(pbc) and atoms.cell.rank == 3:
        return "ewald"
    return "direct" if any(pbc) else "nonperiodic"


def charges(atoms, *, method: str = "auto", lam: float = LAMBDA, h_ea: float = D.H_EA, eta: float = ETA,
            cells: int = CELLS, recip_cells: int = CELLS, charge_centers: Optional[Dict[str, int]] = None,
            total_charge: float = 0.0, canceled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """{charges, method, params, warnings, electronegativity}: EQeq charges of ``atoms``."""
    if len(atoms) == 0:
        raise EQeqError("no atoms")
    m = default_method(atoms) if method == "auto" else method
    if m not in ("ewald", "direct", "nonperiodic"):
        raise EQeqError(f"method is auto, ewald, direct or nonperiodic, not {method!r}")
    if m == "ewald" and not (all(atoms.pbc) and atoms.cell.rank == 3):
        raise EQeqError("the Ewald sums need a cell periodic in 3D; use direct or nonperiodic")
    p = atomic_params(atoms.get_chemical_symbols(), charge_centers, h_ea)
    H = matrix(atoms, p["J"], method=m, lam=lam, eta=eta, cells=cells, recip_cells=recip_cells, canceled=canceled)
    q = solve(H, p["X"], total_charge)
    mu = float(np.mean(p["X"] + H @ q))
    return {"charges": q, "method": m, "electronegativity_eV": mu, "warnings": p["warnings"],
            "params": {"lambda": lam, "h_electron_affinity": h_ea, "eta": eta, "real_cells": cells,
                       "recip_cells": recip_cells, "total_charge": total_charge,
                       "charge_centers": {s: c for s, c in zip(atoms.get_chemical_symbols(), p["centers"]) if c}}}


class EQeqBackend(base.Backend):
    name = "eqeq"
    label = "EQeq charges"
    family = "charges"
    license = "built into MACE (method: Wilmer, Kim and Snurr 2012)"
    capabilities = base.Capabilities(frozenset({"charges"}), frozenset({0, 1, 2, 3}), frozenset({"charges"}),
                                     needs_charge_input=True)
    citation = CITATIONS

    def methods(self) -> List[base.Method]:
        return [base.Method("eqeq", "EQeq (Ewald sums as published; direct sums for slabs and wires)",
                            default=True, license=self.license, citation=list(CITATIONS))]

    def charges(self, atoms, spec: base.CalcSpec, canceled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        o = spec.options or {}
        cc = o.get("charge_centers") or {}
        if isinstance(cc, str):
            cc = dict((k.strip(), int(v)) for k, v in (x.split("=") for x in cc.split(",") if x.strip()))
        try:
            return charges(atoms, method=str(o.get("eqeq_method", "auto")), lam=float(o.get("lambda", LAMBDA)),
                           h_ea=float(o.get("h_electron_affinity", D.H_EA)), eta=float(o.get("eta", ETA)),
                           cells=int(o.get("real_cells", CELLS)), recip_cells=int(o.get("recip_cells", CELLS)),
                           charge_centers=cc, total_charge=float(spec.charge), canceled=canceled)
        except EQeqError as exc:
            raise base.CalcError(str(exc), kind="usage")
