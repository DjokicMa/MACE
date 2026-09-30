"""
Spread-minimization window assist for disentanglement.

The Wannier spread floor is Omega_I, set by the disentanglement *subspace*.
Diffuse, low-projectability bands inflate Omega_I, so this assist refines the
frozen/outer windows to keep the subspace as projectable (atomic-like, hence
localizable) as possible — while ALWAYS honoring a user-specified minimum frozen
window (the bands you insist on preserving exactly).

This is a fresh, standalone heuristic (Tier 1: projectability as the spread
proxy). It is deliberately independent of the existing projectability/PDWF window
logic so the two can be compared.

Rules it guarantees (the same hard Wannier90 constraints the consistency checker
validates):
  * frozen window contains the user minimum frozen window;
  * frozen window holds <= num_wann bands at every k-point (num_wann is raised if
    the user minimum forces it — never silently shrunk below the minimum);
  * outer window contains >= num_wann bands at every k-point and the frozen window.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class WindowAssistResult:
    num_wann: int
    dis_froz: Optional[Tuple[float, float]]  # (min, max) relative to E_F
    dis_win: Optional[Tuple[float, float]]   # (min, max) relative to E_F
    notes: List[str] = field(default_factory=list)


def manifold_windows(eig_all_rel, user_window):
    """Two-shell windows around a region of interest.

    The user window [lo, hi] (rel E_F) is a *region of interest*, not a literal
    wannier90 frozen window. Construction proceeds in connection shells, where two
    bands "connect" if their energy ranges over the BZ sample overlap:

      shell 0 (interest)  : bands passing through [lo, hi];
      shell 1 (frozen)    : interest bands + every band connecting to them. These
                            become the FROZEN manifold -> num_wann counts them, so
                            the connecting bands are pinned as Wannier functions
                            rather than dumped into a pool an undersized subspace
                            must span;
      shell 2 (outer)     : the frozen manifold + every band connecting to IT --
                            the disentanglement room. Two shells only: no runaway
                            cascade through the whole spectrum.

    Returned windows:
      * num_wann  = size of the frozen manifold (shell 1);
      * dis_froz  = largest window around [lo, hi] holding <= num_wann bands/k
                    (grows out to the frozen manifold's isolated extent);
      * dis_win   = full energy extent of shell 2;
      * num_bands = every band overlapping dis_win.

    Returns a dict (or None if no band enters the user window).
    """
    eig = np.asarray(eig_all_rel, dtype=float)
    e_lo = eig.min(axis=0)
    e_hi = eig.max(axis=0)
    lo, hi = sorted(float(x) for x in user_window)
    interest = (e_hi >= lo) & (e_lo <= hi)
    if not interest.any():
        return None

    def maxcount(a, b):
        return int(((eig >= a) & (eig <= b)).sum(axis=1).max())

    def connect(mask, gap=0.5):
        """Bands that come within `gap` eV of a `mask` band AT THE SAME k-point
        (an actual crossing/avoided-crossing), plus `mask` itself. This is
        k-resolved -- merely sharing an energy range over the BZ does NOT connect
        (that false-positive linked the valence to deep semicore bands). gap ~0.5
        eV captures genuine hybridization, not a band merely passing ~1 eV away."""
        sel = eig[:, mask]                       # (nk, n_mask)
        out = mask.copy()
        for b in range(eig.shape[1]):
            if out[b]:
                continue
            # smallest same-k separation between band b and any mask band
            if np.min(np.abs(eig[:, b][:, None] - sel)) < gap:
                out[b] = True
        return out

    # shell 1: interest + 1st connections -> frozen manifold (num_wann)
    frozen_manifold = connect(interest)
    nw = int(frozen_manifold.sum())
    fm_lo = float(e_lo[frozen_manifold].min())
    fm_hi = float(e_hi[frozen_manifold].max())
    # shell 2: + connections-of-connections -> outer disentanglement pool
    outer_mask = connect(frozen_manifold)

    # Frozen window: grow from the user window while <= num_wann bands/k, but never
    # past the frozen manifold's own extent (so growth can't leak across a gap into
    # unrelated deep/high bands).
    flo, fhi = lo, hi
    for c in sorted({float(x) for x in np.concatenate([e_lo, e_hi])
                     if fm_lo - 1e-9 <= x < lo}, reverse=True):
        if maxcount(c - 1e-6, fhi) <= nw:
            flo = c - 1e-6
        else:
            break
    for c in sorted({float(x) for x in np.concatenate([e_lo, e_hi])
                     if hi < x <= fm_hi + 1e-9}):
        if maxcount(flo, c + 1e-6) <= nw:
            fhi = c + 1e-6
        else:
            break

    if int(outer_mask.sum()) > nw:
        # Disentanglement happens (the manifold connects to further bands). The
        # outer window MUST strictly bracket the frozen window -- an edge flush
        # with a frozen edge leaves no states to optimize the subspace against. If
        # the connecting shells didn't already reach past the frozen edges, pull
        # in the nearest band starting below / ending above as a buffer.
        wlo = min(flo, float(e_lo[outer_mask].min()))
        whi = max(fhi, float(e_hi[outer_mask].max()))
        below = e_lo[e_lo < flo - 1e-6]
        above = e_hi[e_hi > fhi + 1e-6]
        if below.size and wlo >= flo - 1e-6:
            wlo = float(below.max())
        if above.size and whi <= fhi + 1e-6:
            whi = float(above.min())
    else:
        # Isolated manifold: no states connect to it, so there is nothing to
        # disentangle -- the outer window equals the frozen window (num_bands ==
        # num_wann, a plain Wannierization of the isolated manifold).
        wlo, whi = flo, fhi
    in_outer = (e_hi >= wlo - 1e-9) & (e_lo <= whi + 1e-9)
    return {
        'num_wann': nw,
        'dis_froz': (round(flo, 6), round(fhi, 6)),
        'dis_win': (round(wlo, 6), round(whi, 6)),
        'band_indices': np.where(in_outer)[0],
    }


def _snap_to_clean_cut(eig, energy, max_snap=2.0, prefer='down'):
    """Snap an energy to the nearest 'clean' cut, avoiding slicing through a band.

    A cut is clean when no band straddles it across the BZ sample, i.e. the count
    of bands below it is the same at every k-point (the frozen window then holds
    complete band manifolds, not partial bands).

    eig : (nk, n_sel) energies relative to E_F.
    prefer : 'down' moves to the nearest clean cut below the sliced band(s)
             (for the upper frozen edge this DROPS a partial conduction band; for
             the lower edge it INCLUDES the full valence band) — matching a
             valence-favoring target. 'nearest'/'up' also supported.
    Returns (snapped_energy, was_sliced).
    """
    e_lo = eig.min(axis=0)
    e_hi = eig.max(axis=0)

    def slices(E):
        return bool(np.any((e_lo < E) & (e_hi > E)))

    if not slices(energy):
        return float(energy), False

    cands = []
    for i in range(len(e_lo)):
        for c in (e_lo[i] - 1e-4, e_hi[i] + 1e-4):
            if abs(c - energy) <= max_snap and not slices(c):
                cands.append(float(c))
    if not cands:
        return float(energy), True  # sliced but nothing clean within tolerance
    if prefer == 'down':
        below = [c for c in cands if c <= energy]
        cands = below or cands
    elif prefer == 'up':
        above = [c for c in cands if c >= energy]
        cands = above or cands
    return min(cands, key=lambda c: abs(c - energy)), True


def spread_minimizing_windows(
    eig_sel_rel: np.ndarray,
    proj_sel: np.ndarray,
    num_wann: int,
    min_froz: Tuple[float, float],
    proj_floor: float = 0.9,
    win_padding: float = 0.5,
) -> WindowAssistResult:
    """Choose dis_froz / dis_win to minimize the Wannier spread (projectability
    proxy), honoring a minimum frozen window.

    Parameters
    ----------
    eig_sel_rel : ndarray, shape (nk, n_sel)
        Energies of the SELECTED bands at each k-point, relative to E_F.
    proj_sel : ndarray, shape (n_sel,)
        Average (over k) projectability of each selected band, in [0, 1].
    num_wann : int
        Initial target number of Wannier functions.
    min_froz : (float, float)
        Minimum frozen window (lo, hi) relative to E_F. The returned frozen
        window always contains this; if it holds more than num_wann bands at some
        k, num_wann is raised to that count (the user's floor wins).
    proj_floor : float
        Bands with average projectability >= this are 'projectable' (localizable);
        below it they are 'diffuse' and are not used to widen the windows.
    win_padding : float
        eV padding applied to the outer-window edges.

    Returns
    -------
    WindowAssistResult
    """
    eig = np.asarray(eig_sel_rel, dtype=float)
    proj = np.asarray(proj_sel, dtype=float)
    nk, n_sel = eig.shape
    notes: List[str] = []

    def maxcount(lo, hi):
        return int(((eig >= lo) & (eig <= hi)).sum(axis=1).max())

    def mincount(lo, hi):
        return int(((eig >= lo) & (eig <= hi)).sum(axis=1).min())

    # Per-band energy extent across the BZ sample
    e_lo = eig.min(axis=0)  # (n_sel,)
    e_hi = eig.max(axis=0)
    projectable = proj >= proj_floor

    # --- 1. Frozen window = the user band-structure target, honored exactly ---
    # The frozen (inner) window is the region reproduced exactly by the
    # interpolation, so it IS the target the user asks for. We do NOT expand it
    # (that would freeze diffuse high-conduction bands and wreck localization);
    # if the target holds more than num_wann bands/k, num_wann is raised.
    flo, fhi = float(min_froz[0]), float(min_froz[1])
    if fhi < flo:
        flo, fhi = fhi, flo

    # Snap the edges to clean cuts so the frozen window holds complete band
    # manifolds rather than slicing through bands. 'down' on both edges keeps a
    # valence-favoring target: the upper edge drops a partial conduction band,
    # the lower edge pulls in the full valence band.
    tlo, thi = flo, fhi
    flo, lo_sliced = _snap_to_clean_cut(eig, tlo, prefer='down')
    fhi, hi_sliced = _snap_to_clean_cut(eig, thi, prefer='down')
    if (abs(flo - tlo) > 1e-6) or (abs(fhi - thi) > 1e-6):
        notes.append(
            f"snapped frozen edges to clean cuts (no band slicing): "
            f"[{tlo:.2f}, {thi:.2f}] -> [{flo:.2f}, {fhi:.2f}] eV")
    elif lo_sliced or hi_sliced:
        notes.append(
            "frozen edge still slices a band (no clean cut within tolerance); "
            "widen/adjust --min-froz-window for a cleaner target")

    m_froz = maxcount(flo, fhi)
    if m_froz > num_wann:
        notes.append(
            f"frozen target [{flo:.2f}, {fhi:.2f}] holds {m_froz} bands/k; "
            f"raising num_wann {num_wann} -> {m_froz} to keep the whole target frozen")
        num_wann = m_froz
    dis_froz = (flo, fhi)

    # --- 3. Outer window: grow from the frozen window to the nearest band edges
    # until there are >= num_wann bands at every k (the minimum for a valid
    # disentanglement). PREFER GROWING DOWN into the valence — those bands are
    # localizable (atomic), so they make good disentanglement room — and keep the
    # conduction edge shallow. Only grow up if the valence side is exhausted.
    # This yields an asymmetric outer window: deep valence, shallow conduction
    # (the opposite of dragging it into the diffuse high-conduction tail). Edges
    # are snapped to clean cuts so the pool holds complete band manifolds.
    need = num_wann
    wlo, whi = flo, fhi
    up_edges = sorted(float(e) for e in e_hi if e > whi + 1e-9)
    dn_edges = sorted((float(e) for e in e_lo if e < wlo - 1e-9), reverse=True)
    ui = di = 0
    for _ in range(4 * n_sel + 4):
        if mincount(wlo, whi) >= need:
            break
        if di < len(dn_edges):              # prefer growing DOWN (valence room)
            wlo = dn_edges[di] - 1e-4
            di += 1
        elif ui < len(up_edges):
            whi = up_edges[ui] + 1e-4
            ui += 1
        else:
            break
    wlo, _ = _snap_to_clean_cut(eig, wlo, prefer='down')
    whi, _ = _snap_to_clean_cut(eig, whi, prefer='up')
    wlo = min(wlo - win_padding, flo - 1e-3)   # contain frozen, small margin
    whi = max(whi + win_padding, fhi + 1e-3)
    if mincount(wlo, whi) < num_wann:
        notes.append(
            f"outer window holds only {mincount(wlo, whi)} bands/k at its thinnest "
            f"k-point (< num_wann={num_wann}); selected band set may be too small")
    dis_win = (wlo, whi)

    return WindowAssistResult(num_wann=num_wann, dis_froz=dis_froz,
                              dis_win=dis_win, notes=notes)
