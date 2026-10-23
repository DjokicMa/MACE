"""Effective masses for the deformation-potential tau, from CRYSTAL's BAND and DOSS output.

Which mass goes where (dpt.py has the formulas):

* the relaxation time depends on the density-of-states mass m_d only -
  3D <<tau>> = 2 sqrt(2 pi) hbar^4 C / (3 m_d^(3/2) (kB T)^(3/2) E1^2),
  2D tau = hbar^3 C / (kB T m_d E1^2) with m_d = sqrt(m_x m_y), 1D
  tau = hbar^2 C / ((2 pi kB T)^(1/2) m^(1/2) E1^2) (the scattering rate
  follows the final-state density of states of the valley the carrier is
  in: Bardeen & Shockley 1950; Herring & Vogt, Phys. Rev. 101, 944 (1956));
* the mobility mu = e <<tau>> / m_c takes the conductivity (band) mass m_c
  along the transport direction.

So the RELAXTIM value for BOLTZTRA is set by m_d; m_c only changes the
mobility MACE prints next to it (BOLTZTRA takes its velocities from the
bands themselves).

From BAND (BAND.DAT or the f25 of a BAND run). The valence-band top (holes)
or conduction-band bottom (electrons) is located per spin channel; every
band whose extremum on a path segment lies within EDGE_TOL of the edge
counts as meeting there (degenerate bands, and equivalent valleys on other
segments). Along each segment leaving the edge a parabola is fitted to the
points within ``window_mev`` of the edge energy:

    E(k) = E0 + hbar^2 (k - k0)^2 / (2 m*)   ->   m* = 1 / (2 a)   in m_e

with E in hartree and k in bohr^-1, the units of both files (the k axis is
the path length in bohr^-1, 2 pi included: diamond's Gamma-X is
2 pi / a_conv = 0.938 bohr^-1; checked against the segment ends and the
lattice when the run's .d3/.out are beside the file). An extremum inside a
segment is fitted on both sides (k0 free), one at a segment end one-sided
(k0 at the end). Too few points in the window - a coarse path - refuses
that direction instead of fitting two points.

The density-of-states mass of a band from the path is the geometric mean
of its masses along the directions sampled through the edge (3D: needs
three non-coplanar ones; 2D: two non-collinear in-plane ones), an
approximation to (m1 m2 m3)^(1/3) when the principal axes are not on the
path. Degenerate bands combine as the total density of states does:
m_d = (sum_b m_d,b^p)^(1/p), p = 3/2 (3D), 1 (2D), 1/2 (1D), and m_c as
the carrier-weighted harmonic mean 1/m_c = sum_b w_b / m_c,b / sum_b w_b,
w_b = m_d,b^p (the conductivity of bands sharing one tau).

From DOSS (DOSS.DAT or the f25 of a DOSS run), near the edge:

    3D  g(E) = V (2 m_d)^(3/2) sqrt(E) / (2 pi^2)
    2D  g(E) = A m_d / pi                  (a step)
    1D  g(E) = L sqrt(2 m_d / E) / pi

in states/hartree/cell, both spins (atomic units; CRYSTAL's closed-shell
DOS carries the factor 2, manual eq. 14.21, and an open-shell DOSS.DAT is
summed over its alpha and beta columns). The fit is to the integrated DOS
N(E), counted from mid-gap - N^(2/3) (3D), N (2D) or N^2 (1D) linear in E
with a free onset E0, over ``fit_ev`` (0.3 eV) into the band: smoothing
shifts states in energy but keeps their number, so N(E) is far less
smeared than g(E) (diamond, NPOL 14: g(E) rises linearly, not as sqrt(E),
yet N(E) gives the BAND-derived DOS mass within 15%).
A DOS mass counts every band and every equivalent valley at the edge:
m_d(DOS) = N_v^(2/3) m_d(valley) in 3D. The scattering that sets tau is
intravalley, so the per-valley mass is m_d(DOS) / N_v^(2/3), with N_v the
star of the edge k under the lattice's point symmetry (from BAND: 1 at
Gamma, 6 for diamond's or silicon's Delta valleys). CRYSTAL's DOS is a
Fourier-Legendre expansion (NPOL <= 25 polynomials per band, manual
p. 320), not a Gaussian that could be removed: near an edge it is smoothed
over a fraction of an eV, so a DOS mass is sensitive to NPOL, the NEWK
shrinking and the energy grid. Band curvature is the better source.
"""
from __future__ import annotations

import contextlib
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mace.constants import BOHR_TO_ANGSTROM as BOHR_A, HARTREE_TO_EV as HARTREE_EV  # noqa: E402
K_B_EV = 8.617333262e-5
# a band meets the edge when its extremum on a segment is this close (Ha; BAND.DAT prints 1e-5)
EDGE_TOL = 5e-5
# Ha above E_F up to which a band's maximum still counts as occupied (advanced_electronic_analyzer)
OCCUPIED_TOL = 1e-3
WINDOW_MEV = 40.0
# a direction within this of the transport axis runs along it
AXIS_TOL_DEG = 5.0
GAMMA = "Γ"
_POWER = {3: 1.5, 2: 1.0, 1: 0.5}           # g(E) ~ m^p: how degenerate bands' DOS masses add


class MassError(ValueError):
    """No usable effective mass (the message says why and what would give one)."""


# ------------------------------------------------------------------ the fit
def parabola_mass(dk: Sequence[float], de: Sequence[float], two_sided: bool) -> Tuple[float, float]:
    """(m*, R^2) of E - E_edge = a (k - k0)^2 (+ b (k - k0) + c when two-sided).

    ``dk``: k - k_edge in bohr^-1; ``de``: E - E_edge in hartree. m* = 1/(2a)
    in m_e (hbar = m_e = 1), negative for a maximum (holes)."""
    x = np.asarray(dk, float)
    y = np.asarray(de, float)
    if two_sided:
        coef = np.polyfit(x, y, 2)
        a = coef[0]
        yhat = np.polyval(coef, x)
    else:
        A = np.column_stack([np.ones_like(x), x * x])
        c0, a = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = c0 + a * x * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if a == 0:
        return math.inf, r2
    return float(1.0 / (2.0 * a)), r2


# ------------------------------------------------------------------ lattice
def lattice_from_output(path: Path) -> Optional[np.ndarray]:
    """The last "DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)" block of a CRYSTAL output (3x3, rows), else None."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None
    rows = None
    for m in re.finditer(r"DIRECT LATTICE VECTOR(?:S)? COMPONENTS \(ANGSTROM\)\s*\n((?:\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n){3})",
                         text):
        rows = [[float(v) for v in ln.split()] for ln in m.group(1).strip().splitlines()]
    return np.array(rows) if rows else None


def holohedry(lat: np.ndarray) -> List[np.ndarray]:
    """The Cartesian rotations (acting on row vectors) that map the lattice onto itself: integer matrices
    with entries -1..1 that keep the metric (enough for a reduced cell; a cell far from reduced gets a
    subgroup, which only makes fewer directions and k points equivalent)."""
    import itertools
    A = np.asarray(lat, float)
    G = A @ A.T
    vals = np.array(list(itertools.product((-1, 0, 1), repeat=9)), float).reshape(-1, 3, 3)
    det = np.linalg.det(vals)
    vals = vals[np.isclose(np.abs(det), 1.0)]
    scale = np.sqrt(np.outer(np.diag(G), np.diag(G)))         # each element against its own vectors' lengths
    keep = np.all(np.abs(np.einsum("nij,jk,nlk->nil", vals, G, vals) - G) <= 1e-3 * scale, axis=(1, 2))
    Ainv = np.linalg.inv(A)
    return [Ainv @ W @ A for W in vals[keep]]


def reciprocal(lat: np.ndarray) -> np.ndarray:
    """Reciprocal lattice vectors (rows, bohr^-1, 2 pi included) of a direct lattice in angstrom."""
    return 2.0 * math.pi * np.linalg.inv(np.asarray(lat, float) / BOHR_A).T


def valley_count(frac: Sequence[float], lat: np.ndarray, ops: Optional[List[np.ndarray]] = None) -> int:
    """The number of k points equivalent to ``frac`` (reciprocal fractional) under the lattice's point
    symmetry, counted modulo a reciprocal lattice vector: 1 at Gamma, 3 at X and 4 at L of an fcc
    lattice, 6 on Delta. An upper bound when the crystal's point group is lower than its lattice's."""
    B = reciprocal(lat)
    Binv = np.linalg.inv(B)
    k = np.asarray(frac, float) @ B
    seen: List[np.ndarray] = []
    for R in ops if ops is not None else holohedry(lat):
        f = (k @ R) @ Binv
        f = f - np.round(f)
        f[np.isclose(np.abs(f), 0.5, atol=1e-4)] = 0.5
        if not any(np.allclose(f, s, atol=1e-4) for s in seen):
            seen.append(f)
    return max(len(seen), 1)


# ------------------------------------------------------------------ files beside a BAND/DOSS file
def run_stem(path: Path) -> str:
    name = Path(path).name
    for suf in (".BAND.DAT", ".DOSS.DAT", ".fort.25", ".f25", ".DAT"):
        if name.upper().endswith(suf.upper()):
            return name[:-len(suf)]
    return Path(path).stem


def _beside(path: Path, suffix: str) -> Optional[Path]:
    p = Path(path).parent / (run_stem(path) + suffix)
    return p if p.is_file() else None


def _deck_record(d3: Optional[Path], key: str) -> Optional[List[str]]:
    """The lines after keyword ``key`` in a properties deck."""
    if not d3:
        return None
    try:
        lines = d3.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for i, ln in enumerate(lines):
        if ln.strip().upper() == key:
            return lines[i + 1:]
    return None


def _path_labels(title: str, nseg: int) -> Optional[List[Tuple[str, str]]]:
    """(start, end) labels of each segment from a deck title ending in the path, e.g. "... - GAMMA-X-U-|-K-GAMMA"
    (the form MACE writes); None unless the path has exactly ``nseg`` segments."""
    tail = title.rsplit(" - ", 1)[-1].strip()
    toks = [t.strip() for t in tail.split("-")]
    if not toks or any(not re.fullmatch(r"[A-Za-z|][A-Za-z0-9_']{0,7}|\|", t) for t in toks):
        return None
    pairs = [(a, b) for a, b in zip(toks, toks[1:]) if "|" not in (a, b)]
    if len(pairs) != nseg:
        return None
    fix = lambda s: GAMMA if s.upper() in ("GAMMA", "G", "GM") else s
    return [(fix(a), fix(b)) for a, b in pairs]


def _frac_label(f: Sequence[float]) -> str:
    from fractions import Fraction
    if np.allclose(f, 0):
        return GAMMA
    return "(" + ",".join(str(Fraction(float(v)).limit_denominator(48)) for v in f) + ")"


# ------------------------------------------------------------------ band structure
@dataclass
class Segment:
    idx: List[int]                       # row indices in a spin block, from the start vertex on
    dk: float                            # k step (bohr^-1)
    start: Optional[np.ndarray] = None   # reciprocal fractional coordinates of the ends
    end: Optional[np.ndarray] = None
    labels: Tuple[str, str] = ("", "")

    @property
    def name(self) -> str:
        return f"{self.labels[0]}–{self.labels[1]}"


@dataclass
class BandPath:
    source: Path
    blocks: List[np.ndarray]             # per spin channel: (nk, nbands), E - E_F in hartree
    segments: List[Segment]
    lattice: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)


def _vertices(path: Path, nseg: int) -> Tuple[Optional[List[Tuple[np.ndarray, np.ndarray]]], Optional[str]]:
    """Segment ends (fractions of the reciprocal vectors) from the .d3 beside the file (exact: integers over
    ISS), else from the LINE lines of the .out (2 decimals); and the deck title."""
    rec = _deck_record(_beside(path, ".d3"), "BAND")
    if rec and len(rec) >= 2 + nseg:
        title = rec[0]
        try:
            head = rec[1].split()
            iss = int(head[1])
            ends = [[int(v) for v in rec[2 + i].split()[:6]] for i in range(nseg)]
            if iss > 0 and all(len(e) == 6 for e in ends):
                return [(np.array(e[:3]) / iss, np.array(e[3:]) / iss) for e in ends], title
        except (ValueError, IndexError):
            pass
        title_only = title
    else:
        title_only = None
    out = _beside(path, ".out")
    if out:
        txt = out.read_text(errors="replace")
        lines = re.findall(r"LINE\s+\d+\s+\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+):\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\)", txt)
        if len(lines) >= nseg:
            v = [[float(x) for x in ln] for ln in lines[-nseg:]]
            return [(np.array(e[:3]), np.array(e[3:])) for e in v], title_only
    return None, title_only


def read_band(path, lattice_from: Iterable[Path] = ()) -> BandPath:
    """A BAND.DAT or a BAND f25 as spin blocks and path segments (each from its start vertex on)."""
    path = Path(path)
    notes: List[str] = []
    if path.suffix.lower() == ".f25" or path.name.lower().endswith("fort.25"):
        from mace.plotting import f25
        bl = [b for b in f25.read_blocks(path, ("BAND",)) if b.kind == "BAND"]
        bl, nspin, _ = f25.select_run(bl, -1, "electronic")
        if not bl:
            raise MassError(f"{path.name} holds no BAND block")
        nseg = len(bl) // nspin
        rows = []
        for b in bl:
            arr = np.asarray(b.values, float).reshape(b.ncol, b.nrow) - b.ef
            rows.append(arr)
        blocks = [np.vstack(rows[s * nseg:(s + 1) * nseg]) for s in range(nspin)]
        counts = [b.ncol for b in bl[:nseg]]
        dks = [b.dk for b in bl[:nseg]]
        ints = [b.ints for b in bl[:nseg]]
        verts, title = _vertices(path, nseg)
        if verts is None and all(len(i) == 6 for i in ints):
            verts = [(np.array(i[:3], float), np.array(i[3:], float)) for i in ints]    # in ISS units: directions only
    else:
        from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
        an = AdvancedElectronicAnalyzer()
        with contextlib.redirect_stdout(io.StringIO()):
            _ev, _ec, k, E = an.read_band_data(path)
        lay = getattr(an, "_band_layout", None) or {}
        if E.ndim != 2 or not len(k):
            from mace.utils import fermi_level
            fl = fermi_level.band_fermi(path)
            raise MassError(f"{path.name}: {fl.note}" if not fl.known and fl.note else f"{path.name}: no band energies read")
        nkpt, nspin = lay.get("nkpt") or len(k), lay.get("nspin") or 1
        if nkpt * nspin != len(k):
            nkpt, nspin = len(k), 1
        blocks = [E[s * nkpt:(s + 1) * nkpt] for s in range(nspin)]
        kk = k[:nkpt]
        ends = sorted(set(e for e in lay.get("panel_ends") or [] if 1 <= e <= nkpt))
        if not ends or ends[0] != 1:
            ends = [1] + ends
        if ends[-1] != nkpt:
            ends.append(nkpt)
        bounds = [(1 if i == 0 else ends[i] + 1, ends[i + 1]) for i in range(len(ends) - 1)]
        counts = [hi - lo + 1 for lo, hi in bounds]
        # the step from the segment's end points: the k column has 5 significant figures, so single
        # differences are quantized (graphene's 0.00168/0.00170 for 0.0016857, v_F 1.6 % low)
        dks = [float(kk[hi - 1] - kk[lo - 1]) / (hi - lo) if hi > lo else 0.0 for lo, hi in bounds]
        verts, title = _vertices(path, len(counts))
        if len(counts) == 1:
            notes.append(f"{path.name} names no path segments: the path is fitted as one line")
    from mace.utils import fermi_level
    fl = fermi_level.band_fermi(path)
    if fl.derived:
        notes.append(fl.note)          # the energies are E - E_F with that E_F (both readers above)
    nseg = len(counts)
    labels = _path_labels(title, nseg) if title else None
    segs: List[Segment] = []
    first = 0
    for i, n in enumerate(counts):
        idx = list(range(first, first + n))
        st, en = (verts[i] if verts else (None, None))
        # CRYSTAL leaves out a segment's start when the path is continuous there (it is the last point of the
        # segment before): put it back, so every segment runs from its start vertex
        if i and verts and np.allclose(verts[i - 1][1], st):
            idx = [first - 1] + idx
        lab = labels[i] if labels else ((_frac_label(st), _frac_label(en)) if st is not None else (f"P{i}", f"P{i + 1}"))
        segs.append(Segment(idx, dks[i], st, en, lab))
        first += n
    lat = None
    for cand in [_beside(path, ".out"), *lattice_from]:
        if cand:
            lat = lattice_from_output(Path(cand))
            if lat is not None:
                break
    if verts and lat is not None:
        B = reciprocal(lat)
        for s in segs:
            if s.start is None:
                continue
            want = float(np.linalg.norm((s.end - s.start) @ B))
            got = (len(s.idx) - 1) * s.dk
            if want > 0 and abs(got - want) > 0.02 * want:
                # f25 ends without the deck are in ISS units: the scale follows from the length
                scale = got / want
                if abs(scale - round(scale)) < 0.02 * scale and round(scale) > 1:
                    s.start, s.end = s.start / round(scale), s.end / round(scale)
                    if not labels:
                        s.labels = (_frac_label(s.start), _frac_label(s.end))
                else:
                    notes.append(f"segment {s.name}: the file's k axis ({got:.4f} bohr^-1) does not match the "
                                 f"lattice's {want:.4f}: masses use the file's k axis")
    return BandPath(path, blocks, segs, lat, notes)


def _edge(blocks: List[np.ndarray], carrier: str) -> Optional[float]:
    best = None
    for blk in blocks:
        occ = blk.max(axis=0) <= OCCUPIED_TOL
        if carrier == "hole" and occ.any():
            v = float(blk[:, occ].max())
            best = v if best is None else max(best, v)
        if carrier == "electron" and (~occ).any():
            v = float(blk[:, ~occ].min())
            best = v if best is None else min(best, v)
    return best


def band_masses(path, carrier: str, axis: Optional[str] = None, dim: int = 3, window_mev: float = WINDOW_MEV,
                lattice_from: Iterable[Path] = ()) -> Dict[str, Any]:
    """Every direction's mass at the band edge of ``carrier``, with the masses for tau (see the module doc).

    Returns {file, carrier, edge_eV, at, bands, spin, valleys, directions: [{band, spin, at, direction, mass,
    r2, points, k_step, angle_deg, along_axis}], refused: [{..., reason}], m_c, m_d, m_c_note, m_d_note,
    warnings}; m_c/m_d are None when the path cannot give them."""
    if carrier not in ("electron", "hole"):
        raise MassError("carrier is electron or hole")
    bp = read_band(path, lattice_from)
    warnings = list(bp.notes)
    blocks = bp.blocks
    spin_names = ["alpha", "beta"] if len(blocks) == 2 else [None]
    if len(blocks) == 2 and blocks[0].shape == blocks[1].shape and np.allclose(blocks[0], blocks[1], atol=2e-5):
        blocks, spin_names = blocks[:1], [None]
        warnings.append("the alpha and beta bands are the same (no net spin): one channel is fitted")
    edge = _edge(blocks, carrier)
    if edge is None:
        raise MassError(f"{Path(path).name}: no {'occupied' if carrier == 'hole' else 'empty'} band (a metal, "
                        "or not a band structure of this system)")
    sgn = 1.0 if carrier == "electron" else -1.0
    win = window_mev / 1000.0 / HARTREE_EV
    B = reciprocal(bp.lattice) if bp.lattice is not None else None
    ops = holohedry(bp.lattice) if bp.lattice is not None else None
    e_ax = np.eye(3)["xyz".index(axis)] if axis in ("x", "y", "z") else None
    dirs: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []
    where: List[Tuple[str, Optional[np.ndarray]]] = []
    for s_i, blk in enumerate(blocks):
        for seg in bp.segments:
            sub = blk[seg.idx]                              # (n, nbands)
            ext = sub.min(axis=0) if carrier == "electron" else sub.max(axis=0)
            for b in np.flatnonzero(np.abs(ext - edge) <= EDGE_TOL):
                col = sub[:, b]
                j = int(np.argmin(col) if carrier == "electron" else np.argmax(col))
                n = len(col)
                # a band flat to the file's precision ties on several points, the first an arbitrary inner one
                # ("1.00 of K-Γ"): a segment end within EDGE_TOL of the extremum is it (band_edges._at_vertex)
                ends = [e for e in (0, n - 1) if 0 < j < n - 1 and abs(col[e] - col[j]) <= EDGE_TOL]
                if ends:
                    j = (min(ends, key=lambda e: col[e]) if carrier == "electron" else max(ends, key=lambda e: col[e]))
                t = j / (n - 1) if n > 1 else 0.0
                kf = seg.start + t * (seg.end - seg.start) if seg.start is not None else None
                if j == 0:
                    at, sides = seg.labels[0], [(1, f"{seg.labels[0]}→{seg.labels[1]}")]
                elif j == n - 1:
                    at, sides = seg.labels[1], [(-1, f"{seg.labels[1]}→{seg.labels[0]}")]
                else:
                    at, sides = f"{t:.2f} of {seg.name}", [(0, seg.name)]
                where.append((at, kf))
                for side, label in sides:
                    if side == 0:
                        lo = j
                        while lo > 0 and abs(col[lo - 1] - col[j]) <= win:
                            lo -= 1
                        hi = j
                        while hi < n - 1 and abs(col[hi + 1] - col[j]) <= win:
                            hi += 1
                        pts = np.arange(lo, hi + 1)
                        enough = (j - lo) >= 2 and (hi - j) >= 2
                    else:
                        pts = [j]
                        while 0 <= pts[-1] + side < n and abs(col[pts[-1] + side] - col[j]) <= win:
                            pts.append(pts[-1] + side)
                        pts = np.array(pts)
                        enough = len(pts) >= 4
                    rec = {"band": int(b) + 1, "spin": spin_names[s_i], "at": at, "direction": label,
                           "k_step": seg.dk, "points": int(len(pts)), "window_meV": window_mev}
                    if seg.start is not None and B is not None:
                        d = (seg.end - seg.start) @ B
                        d = d / np.linalg.norm(d)
                        rec["cartesian"] = [round(float(v), 4) for v in d]
                        if e_ax is not None:
                            ang = math.degrees(math.acos(min(1.0, abs(float(d @ e_ax)))))
                            eq = min(math.degrees(math.acos(min(1.0, abs(float((d @ R) @ e_ax))))) for R in ops)
                            rec["angle_deg"] = round(ang, 1)
                            rec["along_axis"] = ang <= AXIS_TOL_DEG
                            rec["equivalent_to_axis"] = eq <= AXIS_TOL_DEG
                    if not enough:
                        rec["reason"] = (f"only {len(pts)} k points within {window_mev:g} meV of the edge (k step "
                                         f"{seg.dk:.4f} bohr^-1): the path is too coarse here - rerun BAND with more "
                                         "points, or widen --mass-window")
                        refused.append(rec)
                        continue
                    m, r2 = parabola_mass((pts - j) * seg.dk, col[pts] - col[j], side == 0)
                    if not math.isfinite(m) or m * sgn <= 0:
                        rec["reason"] = "the band curves the wrong way here (not an extremum along this direction)"
                        refused.append(rec)
                        continue
                    rec.update(mass=abs(m), r2=round(r2, 5))
                    if abs(m) < 0.02:
                        warnings.append(f"{label} (band {rec['band']}): {abs(m):.3g} m_e - a nearly linear "
                                        "(Dirac-like) band, not a parabola; the formulas do not apply")
                    if abs(m) > 10:
                        warnings.append(f"{label} (band {rec['band']}): {abs(m):.3g} m_e, a nearly flat band - "
                                        "the parabolic-band picture behind the formulas is weak here")
                    if r2 < 0.98:
                        warnings.append(f"{label} (band {rec['band']}): the parabola fits poorly (R^2 {r2:.3f}): "
                                        "non-parabolic within the window - narrow --mass-window")
                    dirs.append(rec)
    out: Dict[str, Any] = {"file": str(path), "carrier": carrier, "edge_eV": edge * HARTREE_EV,
                           "dimensionality": dim, "window_meV": window_mev,
                           "at": sorted({w[0] for w in where}),
                           "bands": sorted({d["band"] for d in dirs} | {r["band"] for r in refused}),
                           "directions": dirs, "refused": refused, "warnings": warnings,
                           "m_c": None, "m_d": None}
    kfs = [w[1] for w in where if w[1] is not None]
    if kfs and bp.lattice is not None:
        nv = max(valley_count(kf, bp.lattice, ops) for kf in kfs[:1])
        out["valleys"] = nv
    _combine(out, dim, axis)
    return out


def _distinct(vecs: List[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for v in vecs:
        if not any(abs(float(v @ u)) > math.cos(math.radians(AXIS_TOL_DEG)) for u in out):
            out.append(v)
    return out


def _combine(res: Dict[str, Any], dim: int, axis: Optional[str]) -> None:
    """m_c along the axis and m_d from the per-direction masses (in place; see the module doc)."""
    dirs = res["directions"]
    if not dirs:
        return
    p = _POWER[dim]
    by_band: Dict[Tuple[int, Optional[str]], List[Dict[str, Any]]] = {}
    for d in dirs:
        by_band.setdefault((d["band"], d["spin"]), []).append(d)
    md_b: Dict[Tuple[int, Optional[str]], Optional[float]] = {}
    mc_b: Dict[Tuple[int, Optional[str]], Tuple[float, Dict[str, Any]]] = {}
    for key, ds in by_band.items():
        # m_c: the direction along (or by symmetry equivalent to) the axis, else the closest one
        if axis and all("angle_deg" in d for d in ds):
            pick = min(ds, key=lambda d: (not d.get("along_axis"), not d.get("equivalent_to_axis"), d["angle_deg"]))
        else:
            pick = ds[0]
        mc_b[key] = (pick["mass"], pick)
        # m_d: geometric mean over distinct directions at one k point (one valley)
        at = max({d["at"] for d in ds}, key=lambda a: sum(d["at"] == a for d in ds))
        same = [d for d in ds if d["at"] == at]
        if all("cartesian" in d for d in same):
            vecs = _distinct([np.array(d["cartesian"]) for d in same])
            need = {3: 3, 2: 2, 1: 1}[dim]
            ok = len(vecs) >= need and (dim != 3 or abs(np.linalg.det(np.array(vecs[:3]))) > 0.1 or
                                        any(abs(np.linalg.det(np.array([a, b, c]))) > 0.1
                                            for i, a in enumerate(vecs) for j, b in enumerate(vecs[i + 1:], i + 1)
                                            for c in vecs[j + 1:]))
            uniq = []
            for d in same:
                v = np.array(d["cartesian"])
                if not any(abs(float(v @ np.array(u["cartesian"]))) > math.cos(math.radians(AXIS_TOL_DEG)) for u in uniq):
                    uniq.append(d)
            md_b[key] = math.exp(sum(math.log(d["mass"]) for d in uniq) / len(uniq)) if ok else None
        else:
            md_b[key] = same[0]["mass"] if dim == 1 else None
    keys = sorted(by_band)
    w = {k: (md_b[k] or mc_b[k][0]) ** p for k in keys}
    res["m_c"] = sum(w.values()) / sum(w[k] / mc_b[k][0] for k in keys)
    picks = sorted({mc_b[k][1]["direction"] for k in keys})
    alongs = [mc_b[k][1] for k in keys]
    rel = ""
    if axis and all("angle_deg" in d for d in alongs):
        if all(d.get("along_axis") for d in alongs):
            rel = f" (along {axis})"
        elif all(d.get("along_axis") or d.get("equivalent_to_axis") for d in alongs):
            rel = f" (equivalent to {axis} by the lattice symmetry)"
        else:
            worst = max(alongs, key=lambda d: d["angle_deg"])
            rel = f" ({worst['angle_deg']:.0f} deg from {axis})"
            res["warnings"].append(f"no path direction through the edge runs along {axis}: m_c is the mass along "
                                   f"{worst['direction']}, {worst['angle_deg']:.0f} deg away (it changes the mobility, "
                                   "not tau)")
    elif axis:
        res["warnings"].append("the lattice or the path's k points are not beside the file: which direction is "
                               f"the transport axis {axis} is unknown")
    band_word = "band" if len(keys) == 1 else f"{len(keys)} bands"
    res["m_c_note"] = f"{band_word} along {', '.join(picks)}{rel}" + (
        ", carrier-weighted" if len(keys) > 1 else "")
    if len(keys) > 1:
        lst = ", ".join(f"{mc_b[k][0]:.3g}" for k in keys)
        res["warnings"].append(f"{len(keys)} bands meet at the edge ({lst} m_e along the transport direction): "
                               "m_c is their carrier-weighted mean, m_d adds their densities of states")
    if all(md_b[k] is not None for k in keys):
        res["m_d"] = sum(md_b[k] ** p for k in keys) ** (1.0 / p)
        res["m_d_note"] = (f"geometric mean of each band's masses along the path directions through "
                           f"{', '.join(sorted({d['at'] for d in dirs}))}"
                           + (f", {len(keys)} bands combined" if len(keys) > 1 else ""))
    nv = res.get("valleys") or 1
    if nv > 1 and dim > 1:
        res["warnings"].append(f"the edge is off {GAMMA}, in up to {nv} equivalent valleys: the path's masses are "
                               "those of one valley; m_c for transport averages over the valleys' orientations "
                               "(it changes the mobility, not tau)")


# ------------------------------------------------------------------ density of states
def read_dos(path) -> Tuple[np.ndarray, np.ndarray]:
    """(E - E_F in hartree, total DOS over both spins in states/hartree/cell) of a DOSS.DAT or a DOSS f25."""
    path = Path(path)
    if path.suffix.lower() == ".f25" or path.name.lower().endswith("fort.25"):
        from mace.plotting import f25
        from mace.utils.dat_file_processor import DatFileProcessor
        dat = DatFileProcessor()._parse_doss_dat_content(f25.doss_dat_text(path, "electronic"))
        energies = np.asarray(dat.get("energy_points") or [], float)
        total = np.asarray(dat.get("total_dos") or [], float)
        nspin = max(int(dat.get("num_spins") or 1), 1)
        if not len(energies) or len(energies) != len(total) or len(energies) % nspin:
            raise MassError(f"{path.name}: no DOS read")
        n = len(energies) // nspin
        return energies[:n], np.abs(total.reshape(nspin, n)).sum(axis=0)
    from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    with contextlib.redirect_stdout(io.StringIO()):
        E, g = AdvancedElectronicAnalyzer().read_dos_data(path)
    if not len(E):
        raise MassError(f"{path.name}: no DOS read")
    return E, g


def _cell_size(lat: np.ndarray, dim: int) -> float:
    """Volume (bohr^3), area (bohr^2) or length (bohr) of the periodic cell."""
    a = np.asarray(lat, float) / BOHR_A
    if dim == 3:
        return abs(float(np.linalg.det(a)))
    if dim == 2:
        return float(np.linalg.norm(np.cross(a[0], a[1])))
    return float(np.linalg.norm(a[0]))


def dos_mass(path, carrier: str, dim: int = 3, edge_ev: Optional[float] = None, fit_ev: float = 0.3,
             lattice_from: Iterable[Path] = ()) -> Dict[str, Any]:
    """m_d from the DOS near the edge (module doc). ``edge_ev``: the band edge (E - E_F, eV) when known from
    BAND; else the first energy past the gap where the DOS rises. Returns {file, m_d, r2, points, window_eV,
    onset_shift_meV, npol, warnings}."""
    path = Path(path)
    E, g = read_dos(path)
    lat = None
    for cand in [_beside(path, ".out"), *lattice_from]:
        if cand:
            lat = lattice_from_output(Path(cand))
            if lat is not None:
                break
    if lat is None:
        raise MassError(f"{path.name}: no lattice vectors (the run's .out beside it) to normalize the DOS per cell")
    size = _cell_size(lat, dim)
    # a small fraction of a typical value (not of the maximum: a DOSS.DAT can open on a spike of thousands)
    pos = g[g > 0]
    thr = 1e-3 * float(np.percentile(pos, 90)) if len(pos) else 0.0
    sgn = 1.0 if carrier == "electron" else -1.0
    # the gap: E_F sits at the valence-band top, the conduction band starts after it
    occ = np.flatnonzero((E <= 1e-3) & (g > thr))
    if not len(occ):
        raise MassError(f"{path.name}: no occupied states in the DOS window")
    vbm = float(E[occ[-1]])
    emp = [i for i in np.flatnonzero((E > vbm) & (g > thr)) if E[i] - vbm > 0.005]
    cbm = float(E[emp[0]]) if emp else None
    if carrier == "electron" and cbm is None:
        raise MassError(f"{path.name}: no gap above the valence band in the DOS window")
    e0 = edge_ev / HARTREE_EV if edge_ev is not None else (vbm if carrier == "hole" else cbm)
    step = float(np.median(np.diff(E)))
    w = max(fit_ev / HARTREE_EV, 6 * step)
    x = sgn * (E - e0)                                  # energy into the band from the edge
    o = np.argsort(x)
    x, gg = x[o], g[o]
    # the integrated DOS N(E), counted from the middle of the gap: smoothing moves states about but keeps
    # their number, so N(E) a few tenths of an eV in is far less smeared than g(E) itself
    start = -0.5 * (cbm - vbm) if cbm is not None else -w
    keep = x >= start
    x, gg = x[keep], gg[keep]
    N = np.concatenate([[0.0], np.cumsum(0.5 * (gg[1:] + gg[:-1]) * np.diff(x))])
    sel = np.flatnonzero((x > 0) & (x <= w))
    warnings = [f"a DOS mass depends on the smearing of the DOS, its energy grid ({step * HARTREE_EV * 1000:.0f} meV "
                "here) and the NEWK k mesh; band curvature (BAND) is the better source"]
    if len(sel) < 4:
        raise MassError(f"{path.name}: {len(sel)} DOS points within {w * HARTREE_EV:.2f} eV of the edge: the energy "
                        "grid is too coarse for a fit")
    xs, ns = x[sel], N[sel]
    # N linear in E after the power that makes a parabolic band's N(E) a straight line
    y = {3: ns ** (2.0 / 3.0), 2: ns, 1: ns ** 2}[dim]
    s, c = np.polyfit(xs, y, 1)
    if s <= 0:
        raise MassError(f"{path.name}: the DOS does not rise from the edge as a parabolic band's does")
    yhat = s * xs + c
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    shift = -c / s                                      # the fitted onset, from the edge
    if dim == 3:        # N = (2/3) V (2m)^(3/2) E^(3/2) / (2 pi^2)
        m = s / (2.0 * (size / (3.0 * math.pi ** 2)) ** (2.0 / 3.0))
    elif dim == 2:      # N = A m E / pi
        m = math.pi * s / size
    else:               # N = 2 L sqrt(2 m E) / pi
        m = math.pi ** 2 * s / (8.0 * size ** 2)
    rec = _deck_record(_beside(path, ".d3"), "DOSS")
    npol = None
    if rec:
        try:
            npol = int(rec[0].split()[5])
        except (ValueError, IndexError):
            npol = None
    if npol is not None:
        warnings.append(f"CRYSTAL's DOS is a Fourier-Legendre expansion with NPOL {npol}: it smooths the edge "
                        "onset, which cannot be undone; the fit uses the integrated DOS with a free onset")
    if r2 < 0.95:
        warnings.append(f"the DOS near the edge is not that of a parabolic band (R^2 {r2:.2f}): this m_d is rough")
    warnings.append("a DOS mass counts every band and every equivalent valley at the edge")
    return {"file": str(path), "carrier": carrier, "m_d": m, "r2": round(r2, 4), "points": int(len(sel)),
            "window_eV": round(w * HARTREE_EV, 3), "edge_eV": e0 * HARTREE_EV,
            "onset_shift_meV": round(shift * HARTREE_EV * 1000, 1), "npol": npol, "dimensionality": dim,
            "warnings": warnings}


# ------------------------------------------------------------------ finding the files
_BAND_GLOBS = ("*.BAND.DAT", "*BAND.DAT", "*.f25", "*fort.25")
_DOSS_GLOBS = ("*.DOSS.DAT", "*DOSS.DAT", "*.f25", "*fort.25")


def _kind_of(p: Path) -> Optional[str]:
    n = p.name.upper()
    if n.endswith("PHONBANDS.DAT"):
        return None
    if n.endswith("BAND.DAT"):
        return "BAND"
    if n.endswith("DOSS.DAT"):
        return "DOSS"
    try:
        from mace.plotting import f25
        kinds = f25.file_kinds(p)
    except Exception:
        return None
    if "BAND" in kinds:
        return "BAND"
    if "DOSS" in kinds:
        return "DOSS"
    return None


_PRIM = re.compile(r"LATTICE PARAMETERS\s+\(ANGSTROMS? AND DEGREES\) - PRIMITIVE CELL\s*\n\s*A\s+B\s+C\s+ALPHA\s+BETA\s+"
                   r"GAMMA[^\n]*\n\s*((?:[-\d.]+\s+){5}[-\d.]+)")


def _cell_of(out: Optional[Path]) -> Optional[List[float]]:
    """a, b, c, alpha, beta, gamma of the first primitive cell an output prints (an SCF or a properties run)."""
    if not out or not Path(out).is_file():
        return None
    try:
        m = _PRIM.search(Path(out).read_text(errors="replace"))
    except OSError:
        return None
    return [float(v) for v in m.group(1).split()] if m else None


def _same_cell(c1: Sequence[float], c2: Sequence[float]) -> bool:
    return (all(abs(a - b) <= 2e-3 * max(abs(a), 1e-6) for a, b in zip(c1[:3], c2[:3]))
            and all(abs(a - b) <= 0.2 for a, b in zip(c1[3:], c2[3:])))


def find_beside(outputs: Sequence[Path], kind: str, inside: Optional[Path] = None) -> Tuple[Optional[Path], List[Path]]:
    """The BAND or DOSS file for ``outputs`` (the strain series): in their folder, else the folder above and
    its subfolders. A candidate whose run (.out beside it) has another cell than the unstrained output is
    not the same structure and is skipped; one without its .out counts only in the outputs' own folder.
    With several, the one sharing the longest name prefix with the series (before _dp_); (None,
    candidates) when that is a tie. ``inside``: no folder outside it is searched (the GUI's served folder)."""
    outputs = [Path(o) for o in outputs]
    if not outputs:
        return None, []
    base = re.sub(r"_dp_[xyz]_.*$", "", outputs[0].stem)
    ref = None
    for o in outputs:
        try:
            if o.suffix == ".out" and "ELASTIC STRAIN MATRIX" not in o.read_text(errors="replace"):
                ref = _cell_of(o)
                break
        except OSError:
            continue
    here = outputs[0].resolve().parent
    tiers = [[here], [here.parent] + sorted(d for d in here.parent.iterdir() if d.is_dir() and d != here)
             if here.parent != here else []]
    if inside is not None:
        root = Path(inside).resolve()
        tiers = [[d for d in tier if d == root or root in d.parents] for tier in tiers]
    globs = _BAND_GLOBS if kind == "BAND" else _DOSS_GLOBS
    for t, tier in enumerate(tiers):
        found: Dict[str, Path] = {}
        for d in tier:
            try:
                cands = {p for g in globs for p in d.glob(g) if p.is_file()}
            except OSError:
                continue
            for p in sorted(cands, key=lambda q: (not q.name.upper().endswith(".DAT"), q.name)):
                if _kind_of(p) != kind:
                    continue
                cell = _cell_of(_beside(p, ".out"))
                if ref and cell and not _same_cell(ref, cell):
                    continue                                   # another structure
                if (not ref or not cell) and t > 0:
                    continue                                   # unverified, and not in the outputs' folder
                found.setdefault(str(p.parent / run_stem(p)), p)     # BAND.DAT before the f25 of a run
        if not found:
            continue
        cands = list(found.values())
        if len(cands) == 1:
            return cands[0], cands

        def common(p: Path) -> int:
            n = 0
            for a, b in zip(run_stem(p), base):
                if a != b:
                    break
                n += 1
            return n
        best = max(common(p) for p in cands)
        top = [p for p in cands if common(p) == best]
        return (top[0], cands) if len(top) == 1 else (None, cands)
    return None, []


# ------------------------------------------------------------------ what `mace tau` uses
def auto_masses(carrier: str, axis: Optional[str], dim: int, outputs: Sequence[Path] = (),
                band: Optional[Path] = None, dos: Optional[Path] = None,
                window_mev: float = WINDOW_MEV, inside: Optional[Path] = None) -> Dict[str, Any]:
    """m_c (for the mobility) and m_d (for tau) of ``carrier`` from a BAND and/or DOSS file, given or found
    beside ``outputs``. Returns {m_star, m_dos, sources: {m_star, m_dos}, band, dos, warnings}; raises
    MassError (saying which files would give a mass) when there is none."""
    outputs = [Path(o) for o in outputs]
    warnings: List[str] = []
    tried: List[str] = []
    if band is None and outputs:
        band, cands = find_beside(outputs, "BAND", inside)
        if band is None and len(cands) > 1:
            warnings.append("several band structures beside the outputs (" + ", ".join(c.name for c in cands)
                            + "): give --band FILE")
    if dos is None and outputs:
        dos, cands = find_beside(outputs, "DOSS", inside)
        if dos is None and len(cands) > 1:
            warnings.append("several DOS files beside the outputs (" + ", ".join(c.name for c in cands)
                            + "): give --dos FILE")
    bres = dres = None
    if band is not None:
        try:
            bres = band_masses(band, carrier, axis, dim, window_mev)
        except (MassError, OSError, ValueError) as e:
            tried.append(f"{Path(band).name}: {e}")
    if dos is not None:
        try:
            dres = dos_mass(dos, carrier, dim, bres["edge_eV"] if bres else None)
        except (MassError, OSError, ValueError) as e:
            tried.append(f"{Path(dos).name}: {e}")
    m_c = bres.get("m_c") if bres else None
    m_d = bres.get("m_d") if bres else None
    src: Dict[str, str] = {}
    bname = Path(band).name if band else ""
    if m_c is not None:
        src["m_star"] = f"band curvature of {bname}: {bres['m_c_note']}"
    if m_d is not None:
        src["m_dos"] = f"band curvature of {bname}: {bres['m_d_note']}"
    elif dres is not None:
        nv = bres.get("valleys") if bres else None
        p = {3: 2.0 / 3.0, 2: 1.0, 1: 2.0}[dim]
        m_d = dres["m_d"] / (nv ** p if nv else 1.0)
        note = f"DOS fit of {Path(dos).name} (R^2 {dres['r2']:.2f})"
        if nv and nv > 1:
            note += f", divided by N_v^{p:.3g} for {nv} equivalent valleys (from the band edge)"
        elif not bres:
            warnings.append("no band structure: where the edge sits is unknown - if it is off "
                            f"{GAMMA} in N_v equivalent valleys, the DOS mass is N_v^(2/3) times the per-valley "
                            "one that sets tau")
        src["m_dos"] = note
    elif m_c is not None:
        m_d = m_c if dim == 1 else None
        if dim == 1:
            src["m_dos"] = src["m_star"]
    if m_d is None and m_c is not None:
        m_d = m_c
        src["m_dos"] = (f"taken equal to m_c (the path through the edge samples too few directions for "
                        f"a density-of-states mass, and no DOS was usable)")
        warnings.append("m_d = m_c, an isotropic-band assumption: tau scales as m_d^(-3/2) in 3D, m_d^(-1) in 2D - "
                        "give a DOSS output (--dos) or --m-dos")
    if m_c is None and m_d is not None:
        m_c = m_d
        src["m_star"] = "taken equal to m_d (a DOS has no direction; it changes the mobility, not tau)"
    if m_c is None:
        where = ", ".join(str(p) for p in {o.resolve().parent for o in outputs}) or "the outputs' folder"
        msg = ("m* is required: give --m-star M, or the BAND or DOSS output of the same structure "
               f"(a *_band.BAND.DAT or BAND .f25, a *_doss.DOSS.DAT or DOSS .f25 in {where}, or --band FILE / "
               "--dos FILE)")
        if tried:
            msg += "; tried " + "; ".join(tried)
        raise MassError(msg)
    if bres:
        warnings.extend(bres.get("warnings", []))
    if dres and src.get("m_dos", "").startswith("DOS fit"):
        warnings.extend(dres.get("warnings", []))
    warnings.extend(f"not used: {t}" for t in tried)
    return {"m_star": m_c, "m_dos": m_d, "sources": src, "band": bres, "dos": dres,
            "window_meV": window_mev, "warnings": list(dict.fromkeys(warnings))}


def report(am: Dict[str, Any]) -> str:
    """The masses block `mace tau` prints."""
    lines = ["Effective masses (automatic):"]
    b = am.get("band")
    if b:
        at = ", ".join(b["at"])
        lines.append(f"  BAND {Path(b['file']).name}: {'VBM' if b['carrier'] == 'hole' else 'CBM'} at {at} "
                     f"({b['edge_eV']:+.3f} eV), band(s) {', '.join(str(x) for x in b['bands'])}"
                     + (f", {b['valleys']} equivalent valley(s)" if b.get("valleys") else ""))
        for d in b["directions"]:
            ax = ""
            if "angle_deg" in d:
                ax = "  <- transport axis" if d.get("along_axis") else ("  (equivalent to the axis)" if
                                                                        d.get("equivalent_to_axis") else "")
            sp = f" {d['spin']}" if d.get("spin") else ""
            lines.append(f"    band {d['band']:3d}{sp}  {d['direction']:<16s} {d['mass']:7.4g} m_e  "
                         f"(R^2 {d['r2']:.4f}, {d['points']} pts){ax}")
        for d in b["refused"]:
            lines.append(f"    band {d['band']:3d}  {d['direction']:<16s} not fitted: {d['reason']}")
    d = am.get("dos")
    if d:
        lines.append(f"  DOSS {Path(d['file']).name}: m_d {d['m_d']:.4g} m_e (R^2 {d['r2']:.3f}, {d['points']} pts "
                     f"over {d['window_eV']:g} eV, onset {d['onset_shift_meV']:+.0f} meV from the edge)")
    lines.append(f"  used: m_c = {am['m_star']:.4g} m_e  ({am['sources']['m_star']})")
    lines.append(f"        m_d = {am['m_dos']:.4g} m_e  ({am['sources']['m_dos']})")
    for w in am.get("warnings", []):
        lines.append(f"  Warning: {w}")
    return "\n".join(lines)
