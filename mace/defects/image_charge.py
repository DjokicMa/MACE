"""Image-charge term for charged 3D supercells.

The Makov-Payne monopole term with the Lany-Zunger scaling:

    E_MP(q) = -f * q**2 * v_M / (2 * eps),   f = 1 - (1 - 1/eps) / 3

v_M is the Ewald Madelung potential of the actual supercell lattice: the
potential (atomic units, per unit charge) at a point charge from all its
periodic images plus the neutralising background, self term removed. For a
simple cubic lattice of edge L it is -2.837297/L. E_MP is added to the
formation energy of the charge state. For near-cubic cells v_M < 0 and E_MP is
positive (= f q^2 |v_M| / 2 eps); a cell elongated or flattened by a factor
of about 4 or more has v_M > 0, and E_MP keeps the sign of -v_M.

eps is the dielectric constant that screens the defect charge: eps_inf
(electronic only) for defects at the pristine positions (unrelaxed, vertical),
eps_0 (electronic + ionic) for relaxed defects. Measured on CRYSTAL23 for the
MgO oxygen vacancy, this term with eps_inf puts eps(+/0) of the 53, 63, 127
and 249 atom cells at 2.85-2.88 eV (3.30-3.57 eV without it).
"""

import itertools
import math
from typing import Sequence

import numpy as np

from mace.defects import DefectError

#: erfc(x) and exp(-x**2) are below 1e-14 beyond this many Ewald widths.
_EWALD_CUTOFF = 5.8


def _erfc(x: np.ndarray) -> np.ndarray:
    return np.frompyfunc(math.erfc, 1, 1)(x).astype(float)


def _shell(vectors: np.ndarray, reach: float) -> np.ndarray:
    """Integer combinations of the rows of ``vectors`` covering a sphere of radius ``reach``, origin excluded."""
    dual = np.linalg.inv(vectors).T                    # rows b_i with a_i . b_j = delta_ij
    n = [int(math.ceil(reach * np.linalg.norm(b))) for b in dual]
    idx = np.stack(np.meshgrid(*[np.arange(-k, k + 1) for k in n], indexing="ij"), -1).reshape(-1, 3)
    idx = idx[np.any(idx != 0, axis=1)]
    return idx @ vectors


def madelung(lattice: Sequence[Sequence[float]]) -> float:
    """Ewald Madelung potential of a point-charge lattice (rows = lattice vectors).

    In units of 1/length of the lattice vectors: give them in bohr for an
    energy in Ha. Simple cubic with unit edge: -2.837297.
    """
    a = np.asarray(lattice, float)
    if a.shape != (3, 3):
        raise DefectError("the Madelung potential needs three lattice vectors")
    volume = abs(float(np.linalg.det(a)))
    if volume < 1e-10:
        raise DefectError("the lattice vectors are linearly dependent")
    eta = math.sqrt(math.pi) / volume ** (1.0 / 3.0)    # splits the work evenly between the sums
    recip = 2.0 * math.pi * np.linalg.inv(a).T

    r = np.linalg.norm(_shell(a, _EWALD_CUTOFF / eta), axis=1)
    real = float(np.sum(_erfc(eta * r) / r))
    g = np.linalg.norm(_shell(recip, 2.0 * eta * _EWALD_CUTOFF), axis=1)
    reciprocal = float(np.sum(4.0 * math.pi / volume * np.exp(-g * g / (4.0 * eta * eta)) / (g * g)))
    return real + reciprocal - 2.0 * eta / math.sqrt(math.pi) - math.pi / (volume * eta * eta)


def lany_zunger_factor(eps: float) -> float:
    """f = 1 - (1 - 1/eps)/3: the share of the point-charge term left after the
    quadrupole-like screening tail (Lany and Zunger)."""
    return 1.0 - (1.0 - 1.0 / eps) / 3.0


def makov_payne(q: float, eps: float, v_m: float) -> float:
    """Lany-Zunger-scaled Makov-Payne energy -f q^2 v_M / (2 eps): minus the
    screened point-charge energy of the periodic images (units of ``v_m``, Ha
    for v_m in 1/bohr)."""
    if eps <= 0:
        raise DefectError(f"dielectric constant {eps} must be positive")
    return -lany_zunger_factor(eps) * q * q * v_m / (2.0 * eps)


#: Below this ratio of the shortest image distance to V^(1/3) a supercell is
#: called far from cubic (1 for simple cubic, 1.12 fcc-shaped, 1.09
#: bcc-shaped; 0.79 for a 1x1x2 stack of cubes, 0.69 for 1x1x3).
NEAR_CUBIC_MIN_RATIO = 0.85


def cubicity(lattice: Sequence[Sequence[float]]) -> float:
    """Shortest lattice translation over V^(1/3) (any length unit)."""
    a = np.asarray(lattice, float)
    shortest = min(float(np.linalg.norm(np.array(n) @ a))
                   for n in itertools.product((-1, 0, 1), repeat=3) if any(n))
    return shortest / abs(float(np.linalg.det(a))) ** (1.0 / 3.0)

