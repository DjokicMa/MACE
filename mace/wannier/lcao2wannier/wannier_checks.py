"""
Consistency checks for Wannier90 disentanglement setups.

Wannier90 imposes hard, *k-resolved* rules on the band/window choice. The most
common way an auto-generated .win silently breaks is the frozen (inner) window:
a steep band that sits outside the frontier set at a representative k-point can
dip into the frozen window at the zone center, so the count of frozen bands must
be evaluated as the MAXIMUM over all k-points, not at one k. When it exceeds
num_wann, wannier90 aborts with:

    dis_windows: More states in the frozen window than target WFs

These checks evaluate the rules against the actual per-k eigenvalues (the same
ones written to .eig), so Stage 1 / Stage 2 can flag — and explain — a bad setup
before the user ever launches wannier90.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class DisentanglementReport:
    ok: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # diagnostics
    max_frozen_count: Optional[int] = None
    max_frozen_kpoint: Optional[int] = None
    min_outer_count: Optional[int] = None
    min_outer_kpoint: Optional[int] = None
    suggested_dis_froz_max: Optional[float] = None
    suggested_dis_froz_min: Optional[float] = None


def _counts_in_window(eig_per_k, lo, hi):
    """Return array of #bands within [lo, hi] for each k-point."""
    return np.array([int(np.sum((e >= lo) & (e <= hi))) for e in eig_per_k])


def suggest_frozen_window(eig_per_k, num_wann, froz_min, froz_max):
    """Shrink [froz_min, froz_max] from the top so that, at every k-point,
    at most num_wann bands fall inside. Returns the largest froz_max' <= froz_max
    that satisfies the rule (keeping froz_min fixed), or None if even an empty
    window can't (shouldn't happen). This keeps the low-energy frozen states and
    trims the steep bands entering from above.
    """
    # Candidate upper edges: just below each band energy currently in the window.
    in_win = np.concatenate([e[(e >= froz_min) & (e <= froz_max)] for e in eig_per_k]) \
        if any(((e >= froz_min) & (e <= froz_max)).any() for e in eig_per_k) else np.array([])
    if in_win.size == 0:
        return froz_max
    candidates = np.unique(in_win)[::-1]  # high -> low
    for c in candidates:
        # new upper edge just below c
        new_max = c - 1e-6
        if new_max <= froz_min:
            return None
        counts = _counts_in_window(eig_per_k, froz_min, new_max)
        if counts.max() <= num_wann:
            return float(new_max)
    return None


def check_disentanglement_rules(
    eig_per_k,
    num_wann: int,
    num_bands: int,
    dis_win_min: Optional[float] = None,
    dis_win_max: Optional[float] = None,
    dis_froz_min: Optional[float] = None,
    dis_froz_max: Optional[float] = None,
) -> DisentanglementReport:
    """Validate Wannier90 disentanglement rules against per-k eigenvalues.

    eig_per_k : list of 1D arrays, one per k-point, of band energies in the SAME
                reference frame as the window bounds (typically relative to E_F).
    All window bounds optional; checks that apply only when the relevant bounds
    are provided.
    """
    eig_per_k = [np.asarray(e, dtype=float) for e in eig_per_k]
    rep = DisentanglementReport(ok=True)

    # Rule 0: counts
    if num_bands < num_wann:
        rep.violations.append(
            f"num_bands ({num_bands}) < num_wann ({num_wann}).")

    # Rule 1: frozen window holds <= num_wann bands at every k
    if dis_froz_min is not None and dis_froz_max is not None:
        froz = _counts_in_window(eig_per_k, dis_froz_min, dis_froz_max)
        k_worst = int(np.argmax(froz))
        rep.max_frozen_count = int(froz.max())
        rep.max_frozen_kpoint = k_worst + 1
        if rep.max_frozen_count > num_wann:
            n_bad = int(np.sum(froz > num_wann))
            sugg = suggest_frozen_window(eig_per_k, num_wann, dis_froz_min, dis_froz_max)
            rep.suggested_dis_froz_max = sugg
            rep.violations.append(
                f"Frozen window [{dis_froz_min:.3f}, {dis_froz_max:.3f}] holds up to "
                f"{rep.max_frozen_count} bands (at k-point {k_worst+1}) but num_wann is "
                f"{num_wann}. wannier90 will abort ('More states in the frozen window "
                f"than target WFs'). Violated at {n_bad}/{len(froz)} k-points."
                + (f" Suggest lowering dis_froz_max to <= {sugg:.3f} eV."
                   if sugg is not None else ""))

    # Rule 2: outer window holds >= num_wann bands at every k (room to disentangle)
    if dis_win_min is not None and dis_win_max is not None:
        outer = _counts_in_window(eig_per_k, dis_win_min, dis_win_max)
        k_worst = int(np.argmin(outer))
        rep.min_outer_count = int(outer.min())
        rep.min_outer_kpoint = k_worst + 1
        if rep.min_outer_count < num_wann:
            rep.violations.append(
                f"Outer window [{dis_win_min:.3f}, {dis_win_max:.3f}] contains only "
                f"{rep.min_outer_count} bands at k-point {k_worst+1}, fewer than "
                f"num_wann ({num_wann}). Widen the outer window.")

        # Rule 3: frozen window within outer window
        if dis_froz_min is not None and dis_froz_max is not None:
            if dis_froz_min < dis_win_min or dis_froz_max > dis_win_max:
                rep.violations.append(
                    f"Frozen window [{dis_froz_min:.3f}, {dis_froz_max:.3f}] is not "
                    f"contained in the outer window [{dis_win_min:.3f}, {dis_win_max:.3f}].")

    # Soft check: eig band count consistency
    nb_eig = {len(e) for e in eig_per_k}
    if len(nb_eig) == 1 and next(iter(nb_eig)) != num_bands:
        rep.warnings.append(
            f".eig has {next(iter(nb_eig))} bands/k but num_bands={num_bands}.")

    rep.ok = not rep.violations
    return rep


def parse_win_windows(path):
    """Parse num_wann/num_bands and dis_* window bounds from a .win file."""
    import re
    keys = ('num_wann', 'num_bands', 'dis_win_min', 'dis_win_max',
            'dis_froz_min', 'dis_froz_max', 'fermi_energy')
    p = {}
    for line in open(path):
        m = re.match(r'\s*(\w+)\s*=\s*([-\d.eE+]+)', line)
        if m and m.group(1) in keys:
            p[m.group(1)] = float(m.group(2))
    return p


def check_seed_windows(win_path, eig_per_k) -> DisentanglementReport:
    """Convenience: parse a .win and check it against per-k eigenvalues.

    eig_per_k must be in the same energy reference as the window bounds (i.e.
    relative to E_F, matching how dis_* and .eig are written).
    """
    p = parse_win_windows(win_path)
    if 'num_wann' not in p or 'num_bands' not in p:
        return DisentanglementReport(ok=True, warnings=[
            f"{win_path}: no num_wann/num_bands found; skipping check."])
    return check_disentanglement_rules(
        eig_per_k,
        num_wann=int(p['num_wann']),
        num_bands=int(p['num_bands']),
        dis_win_min=p.get('dis_win_min'),
        dis_win_max=p.get('dis_win_max'),
        dis_froz_min=p.get('dis_froz_min'),
        dis_froz_max=p.get('dis_froz_max'),
    )


def format_report(rep: DisentanglementReport, title: str = "Disentanglement consistency check") -> str:
    lines = ["", "=" * 70, f"  {title}", "=" * 70]
    if rep.max_frozen_count is not None:
        lines.append(f"  Frozen window: up to {rep.max_frozen_count} bands/k "
                     f"(worst k={rep.max_frozen_kpoint})")
    if rep.min_outer_count is not None:
        lines.append(f"  Outer window:  at least {rep.min_outer_count} bands/k "
                     f"(worst k={rep.min_outer_kpoint})")
    if rep.ok:
        lines.append("  STATUS: ✓ PASS — satisfies Wannier90 disentanglement rules")
    else:
        lines.append("  STATUS: ✗ FAIL — wannier90 will reject this setup:")
        for v in rep.violations:
            lines.append(f"    • {v}")
    for w in rep.warnings:
        lines.append(f"    ⚠ {w}")
    lines.append("=" * 70)
    return "\n".join(lines)
