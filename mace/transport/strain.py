"""Strained single-point decks and the deformation-potential fits.

Decks. A uniaxial strain epsilon along x, y or z is CRYSTAL's ELASTIC
geometry edit (manual p. 47): IDEF -2 (the epsilon matrix, not volume
conserving - the setting the manual suggests), with epsilon in the
diagonal element of the transport axis. CRYSTAL finds the symmetry of the
deformed cell itself. The record goes right after the atoms of the parent
SP deck, so every other setting of the parent stays; TOLDEE is raised to
8 when looser (the energies are fitted to second order in epsilon), and a
SPINLOCK holds for 3 cycles at most (CRYSTAL prints no band edges while it
holds, so a run that converges inside the lock has no edges to fit).

Fits. From the outputs of those decks (and the unstrained parent):

* E1 = d E_edge / d epsilon - the slope of a linear fit of the band edge
  (the valence-band top for holes, the conduction-band bottom for
  electrons; over both spin channels) against epsilon, in eV;
* C = (1/V0) d^2 E / d epsilon^2 (3D, GPa), (1/S0) ... (2D, N/m) or
  (1/L0) ... (1D, J/m) - the curvature of a quadratic fit of the total
  energy, i.e. the clamped-ion stretching modulus along the axis.

The edge energies are CRYSTAL's eigenvalues as printed (TOP OF VALENCE /
BOTTOM OF VIRTUAL BANDS). For a slab or a polymer they are referenced to
the vacuum of the non-periodic directions; for a 3D crystal they move with
the average electrostatic potential of the strained cell, so a 3D E1 from
them is approximate (Van de Walle & Martin, PRB 39, 1871 (1989): absolute
deformation potentials need a reference level). The provenance says so;
a better E1 (or C, e.g. the relaxed-ion C_ii of an ELASTCON run) can be
given instead.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mace.constants import HARTREE_TO_EV as HARTREE_EV, HARTREE_TO_J as HARTREE_J  # noqa: E402
AXES = ("x", "y", "z")
_DIM_OF = {"CRYSTAL": 3, "SLAB": 2, "POLYMER": 1, "HELIX": 1, "MOLECULE": 0}
_ALLOWED_AXES = {3: AXES, 2: ("x", "y"), 1: ("x",)}
_REFUSE_IN_DECK = ("OPTGEOM", "FREQCALC", "ELASTCON", "ELAPIEZO", "PIEZOCON", "PIEZOCP", "PHOTOELA", "EOS",
                   "ANHARM", "EXTERNAL", "ELASTIC", "TWOCOMPON")
_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
# cycles a parent's SPINLOCK holds in a strained deck (see strained_deck)
SPINLOCK_CYCLES = 3


class StrainError(ValueError):
    """A deck or output the deformation-potential tools cannot use."""


def _strain_tag(eps: float) -> str:
    """+1% -> p1pct, -0.5% -> m0_5pct."""
    return ("p" if eps >= 0 else "m") + f"{abs(eps) * 100:g}".replace(".", "_") + "pct"


def atoms_end(lines: Sequence[str]) -> Tuple[int, int]:
    """(dimensionality, index of the line after the atom list) of a .d12 deck."""
    if len(lines) < 4:
        raise StrainError("not a CRYSTAL deck")
    key = lines[1].strip().upper()
    if key not in _DIM_OF:
        raise StrainError(f"geometry {key or '?'} is not a CRYSTAL/SLAB/POLYMER input MACE can strain")
    dim = _DIM_OF[key]
    if dim == 0:
        raise StrainError("a molecule has no lattice to strain")
    i = 2
    if key == "CRYSTAL":
        head = lines[2].split()
        ifso = int(head[2]) if len(head) >= 3 else 0
        i = 4 + (1 if ifso > 1 else 0)          # IFLAG IFHR IFSO, group, [origin shift]
    else:
        i = 3                                    # group
    i += 1                                       # lattice parameters
    try:
        n = int(lines[i].split()[0])
    except (IndexError, ValueError):
        raise StrainError(f"no atom count on line {i + 1} of the deck")
    return dim, i + 1 + n


def strained_deck(text: str, axis: str, eps: float, min_toldee: int = 8) -> str:
    """The parent SP deck with an ELASTIC -2 strain eps along axis."""
    lines = text.splitlines()
    upper = [ln.strip().upper() for ln in lines]
    bad = [k for k in _REFUSE_IN_DECK if k in upper]
    if bad:
        raise StrainError(f"the parent deck has {', '.join(bad)}: start from a single-point (SP) deck of the "
                          "optimized structure (mace opt2d12 --calc-type SP), one-component")
    dim, end = atoms_end(lines)
    axis = axis.lower()
    if axis not in _ALLOWED_AXES[dim]:
        raise StrainError(f"axis {axis} is not periodic in a {dim}D system (periodic: "
                          f"{', '.join(_ALLOWED_AXES[dim])})")
    k = AXES.index(axis)
    if abs(eps) < 1e-12:
        rec = []                                 # the unstrained point: the parent at the same TOLDEE
    else:
        mat = [["0.0"] * 3 for _ in range(3)]
        mat[k][k] = f"{eps:.6f}"
        rec = ["ELASTIC", "-2"] + [" ".join(r) for r in mat]
    out = list(lines[:end]) + rec + list(lines[end:])
    out[0] = f"{lines[0].strip()} strain {axis} {eps:+.4f}"[:80]
    # SPINLOCK: CRYSTAL prints no band edges (TOP OF VALENCE / BOTTOM OF
    # VIRTUAL BANDS) while the lock holds - measured on HPCC: diamond under
    # SPINLOCK 0 50 converged in 7 locked cycles and printed none. The lock
    # keeps the parent's spin state for its first cycles only here.
    for j, ln in enumerate(out):
        if ln.strip().upper() == "SPINLOCK" and j + 1 < len(out):
            parts = out[j + 1].split()
            try:
                if len(parts) >= 2 and int(parts[1]) > SPINLOCK_CYCLES:
                    out[j + 1] = f"{parts[0]} {SPINLOCK_CYCLES}"
            except ValueError:
                pass
    # TOLDEE no looser than min_toldee
    for j, ln in enumerate(out):
        if ln.strip().upper() == "TOLDEE" and j + 1 < len(out):
            try:
                if int(out[j + 1].split()[0]) < min_toldee:
                    out[j + 1] = str(min_toldee)
            except (ValueError, IndexError):
                pass
    return "\n".join(out) + "\n"


def write_strain_set(parent: Path, axis: str, strains: Sequence[float], output_dir: Optional[Path] = None,
                     force: bool = False, check=None, unstrained: bool = True) -> List[Path]:
    """One deck per strain, <stem>_dp_<axis>_<p|m><percent>pct.d12, plus the
    unstrained one (<stem>_dp_<axis>_p0pct.d12: the parent at the series'
    TOLDEE, so every energy of the fit has the same threshold; left out with
    ``unstrained=False``, when another axis's set has it). Every strain
    and every destination is checked before the first deck is written, so a
    refused set writes nothing; ``check(path)`` (the GUI's check_write) runs
    on each destination."""
    parent = Path(parent)
    text = parent.read_text(errors="replace")
    out_dir = Path(output_dir) if output_dir else parent.parent
    planned = []
    for eps in sorted(set([0.0, *strains]) if unstrained else {e for e in strains if abs(e) >= 1e-12}):
        if abs(eps) > 0.05:
            raise StrainError(f"strain {eps} is outside the linear regime (|eps| <= 0.05)")
        deck = strained_deck(text, axis, eps)
        dest = out_dir / f"{parent.stem}_dp_{axis.lower()}_{_strain_tag(eps)}.d12"
        if dest.exists() and not force:
            raise StrainError(f"{dest} exists (use --force to replace it)")
        if check is not None:
            check(dest)
        planned.append((dest, deck))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for dest, deck in planned:
        dest.write_text(deck)
        written.append(dest)
    return written


# ------------------------------------------------------------------ outputs
_SCF_END = re.compile(r"SCF ENDED - CONVERGENCE ON ENERGY\s+E\(AU\)\s+(" + _FLOAT + ")")
_VOLUME = re.compile(r"PRIMITIVE CELL - CENTRING CODE\s+\S+\s+VOLUME=\s*(" + _FLOAT + ")")
_CELL_HEAD = re.compile(r"^\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*$", re.M)


def _f(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))


def strain_of(deck_text: str) -> Tuple[Optional[str], float]:
    """(axis, eps) of an ELASTIC -2 uniaxial record in a deck (None, 0.0 without one)."""
    lines = deck_text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "ELASTIC" and i + 4 < len(lines):
            try:
                rows = [[_f(x) for x in lines[i + 2 + r].split()[:3]] for r in range(3)]
            except ValueError:
                continue
            diag = [rows[r][r] for r in range(3)]
            nz = [r for r in range(3) if abs(diag[r]) > 0]
            off = any(abs(rows[r][c]) > 0 for r in range(3) for c in range(3) if r != c)
            if len(nz) == 1 and not off:
                return AXES[nz[0]], diag[nz[0]]
            raise StrainError("the ELASTIC record is not a uniaxial strain")
    return None, 0.0


_STRAIN_MATRIX = re.compile(r"ELASTIC STRAIN MATRIX \(ADIMENSIONAL\)\s*\n\s*\n?((?:\s*\d\s+" + _FLOAT + r"\s+" + _FLOAT
                            + r"\s+" + _FLOAT + r"\s*\n){3})")


def strain_of_output(text: str) -> Tuple[Optional[str], float]:
    """(axis, eps) from the "ELASTIC STRAIN MATRIX (ADIMENSIONAL)" CRYSTAL
    prints for an ELASTIC deck (None, 0.0 without one)."""
    m = _STRAIN_MATRIX.search(text)
    if not m:
        return None, 0.0
    rows = [[_f(x) for x in ln.split()[1:4]] for ln in m.group(1).strip().splitlines()]
    nz = [r for r in range(3) if abs(rows[r][r]) > 0]
    if len(nz) != 1 or any(abs(rows[r][c]) > 0 for r in range(3) for c in range(3) if r != c):
        raise StrainError("the run's strain matrix is not a uniaxial strain")
    return AXES[nz[0]], rows[nz[0]][nz[0]]


def _first_cell(text: str) -> Optional[List[float]]:
    for m in _CELL_HEAD.finditer(text):
        nxt = text[m.end():].lstrip("\n").splitlines()
        vals = re.findall(_FLOAT, nxt[0]) if nxt else []
        if len(vals) == 6:
            return [_f(v) for v in vals]
    return None


def read_point(out_path: Path) -> Dict[str, Any]:
    """Strain, total energy, band edges and cell size of one SP output.

    The strain is the "ELASTIC STRAIN MATRIX" CRYSTAL prints (else the
    ELASTIC record of the deck next to the output); the unstrained run has
    none (0).
    """
    from mace.utils.property_extractor import CrystalPropertyExtractor
    from mace.elastic.analysis import dimensionality

    out_path = Path(out_path)
    text = out_path.read_text(errors="replace")
    axis, eps = strain_of_output(text)
    deck = out_path.with_suffix(".d12")
    if axis is None and "ELASTIC STRAIN MATRIX" not in text and deck.is_file():
        axis, eps = strain_of(deck.read_text(errors="replace"))
    m = None
    for m in _SCF_END.finditer(text):
        pass
    if m is None:
        raise StrainError(f"{out_path.name}: no converged SCF")
    energy = _f(m.group(1))
    blocks = CrystalPropertyExtractor._final_scf_blocks(text)
    tops = [t[0] for b in blocks.values() for t in b.get("tops", [])]
    bottoms = [t[0] for b in blocks.values() for t in b.get("bottoms", [])]
    dim = dimensionality(text) or 3
    point: Dict[str, Any] = {"file": out_path.name, "axis": axis, "strain": eps, "energy_au": energy,
                             "vbm_ev": max(tops) * HARTREE_EV if tops else None,
                             "cbm_ev": min(bottoms) * HARTREE_EV if bottoms else None, "dimensionality": dim}
    vm = _VOLUME.search(text)
    cell = _first_cell(text)
    if dim == 3 and vm:
        point["size"] = _f(vm.group(1))                      # A^3
    elif dim == 2 and cell:
        point["size"] = cell[0] * cell[1] * math.sin(math.radians(cell[5]))   # A^2
    elif dim == 1 and cell:
        point["size"] = cell[0]                              # A
    return point


def _r2(x, y, coeffs) -> float:
    import numpy as np
    yhat = np.polyval(coeffs, x)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def fit(points: Sequence[Dict[str, Any]], carrier: str) -> Dict[str, Any]:
    """E1 (eV) and C of a strain series (with the unstrained point)."""
    import numpy as np
    if carrier not in ("electron", "hole"):
        raise StrainError("carrier is electron or hole")
    axes = {p["axis"] for p in points if p["axis"]}
    if len(axes) != 1:
        raise StrainError(f"the outputs strain {', '.join(sorted(axes)) or 'no axis'}: give one axis's series "
                          "(and the unstrained parent)")
    axis = axes.pop()
    dims = {p["dimensionality"] for p in points}
    if len(dims) != 1:
        raise StrainError("the outputs mix periodicities")
    dim = dims.pop()
    pts = sorted(points, key=lambda p: p["strain"])
    eps = np.array([p["strain"] for p in pts])
    if len(set(np.round(eps, 8))) < 3 or not any(abs(e) < 1e-12 for e in eps):
        raise StrainError("need the unstrained output and at least two strained ones")
    edge_key = "cbm_ev" if carrier == "electron" else "vbm_ev"
    edges = [p[edge_key] for p in pts]
    if any(e is None for e in edges):
        missing = [p["file"] for p in pts if p[edge_key] is None]
        raise StrainError(f"no band edges in {', '.join(missing)} (a metal, or no TOP OF VALENCE / BOTTOM OF "
                          "VIRTUAL BANDS lines): the deformation potential needs a gapped system")
    edges_a = np.array(edges)
    lin = np.polyfit(eps, edges_a, 1)
    energies = np.array([p["energy_au"] for p in pts]) * HARTREE_J
    quad = np.polyfit(eps, energies, 2)
    zero = next(p for p in pts if abs(p["strain"]) < 1e-12)
    size = zero.get("size")
    if not size:
        raise StrainError(f"{zero['file']}: no cell volume/area/length to normalize C")
    d2 = 2.0 * quad[0]                                       # J
    if dim == 3:
        c = d2 / (size * 1e-30) / 1e9                        # GPa
    elif dim == 2:
        c = d2 / (size * 1e-20)                              # N/m
    else:
        c = d2 / (size * 1e-10)                              # J/m
    warnings = []
    r2_e1 = _r2(eps, edges_a, lin)
    if r2_e1 < 0.95:
        warnings.append(f"the band edge does not shift linearly with strain (R^2 {r2_e1:.2f}): a degenerate "
                        "edge splits under this strain, or the edge moves to another k point; E1 from this "
                        "fit is unreliable - strain along a direction that keeps the edge, or give E1")
    r2_c = _r2(eps, energies, quad)
    if r2_c < 0.99:
        warnings.append(f"E(strain) is not quadratic (R^2 {r2_c:.3f}): check the SCF convergence of each run")
    return {"axis": axis, "carrier": carrier, "dimensionality": dim, "E1_eV": float(lin[0]), "warnings": warnings,
            "E1_r2": r2_e1, "C": float(c), "C_unit": {3: "GPa", 2: "N/m", 1: "J/m"}[dim],
            "C_r2": r2_c, "strains": [float(e) for e in eps],
            "edges_eV": [float(e) for e in edges_a], "energies_au": [p["energy_au"] for p in pts],
            "files": [p["file"] for p in pts],
            "edge_reference": ("vacuum (non-periodic directions)" if dim < 3 else
                               "CRYSTAL eigenvalues as printed; a 3D cell's reference moves with strain "
                               "(approximate E1)")}
