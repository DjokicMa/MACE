"""Conduction-driven target-channel augmentation (v2, decoupled WF count).

When the standard valence config cannot represent the low conduction
(SnTe: gap-edge min_k P = 0 at the inversion pockets -> a 1.5 eV
variational wall no threshold setting can fix), the target AO set must be
augmented with the channels the conduction is actually made of
(feedback-conduction-driven-target). v1 simply retried with the extended
valence config, which couples the target set to num_wann. That coupling is
exactly what collapsed h-BN + polarization-d (num_wann=36 with NO physical
d bands: the diffuse-GTO continuum projects P ~ 1.0 onto diffuse d AOs and
gets frozen as junk, Omega_I 269) while SnTe's semicore-4d (tight, occupied,
20 real bands) worked. v2 therefore decouples:

  * the target MASK is enlarged with the selected (atom, l) channels — the
    per-state projectability, freezing masks, and the SVD-based PDWF Amn
    all see the enlarged set (nproj >= num_wann is fine: the SVD picks the
    best num_wann directions);
  * num_wann grows ONLY by the number of physical SEMICORE bands the
    selected channels bring (occupied bands far below E_F projecting almost
    entirely onto the channel). Polarization channels add zero.

Channel choice follows the SnTe d-channel study (HYBRID_TARGET_STUDY.md):
channels act as SETS (Sn-d alone useless, Te-d 5x, both 33x), so all
subsets of the surviving candidates are scored — cheap, because channel AO
sets are pairwise disjoint and disjoint from the base target, hence the
Lowdin P of a union is the SUM of per-channel P.

Dilution ceilings: diffuse high-l channels have intrinsically lower
projectability ceilings (Lowdin orthogonalization shaves diffuse tails).
The a-priori ceiling of channel c is the mean on-site retention of its
S^(-1/2)-orthogonalized AOs, retention_mu = |S^(1/2)_{mu mu}|^2 (the
overlap of the Lowdin orbital with its parent AO). Channels below a
retention floor are near-linear-dependence ghosts (diffuse f) and are
dropped before set evaluation; achieved coverage should be judged against
the set's ceiling, never as an absolute (Bi sp 0.96-1.0 > h-BN sp 0.94 >
SnTe d-conduction 0.80-0.89 > f).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .valence_config import (CORE_SHELLS, get_valence_l, _auto_detect_ecp,
                             ELEMENT_SYMBOLS)

__all__ = ["enumerate_candidate_channels", "augment_target_channels",
           "ChannelInfo", "AugmentResult"]


@dataclass
class ChannelInfo:
    """One candidate augmentation channel: all non-core radials of an
    (atom, l) pair that is NOT in the atom's standard valence config."""
    atom_index: int
    atom_symbol: str
    l: int
    ao_indices: List[int]          # spatial AO indices (untrimmed)
    ceiling: float = np.nan        # mean on-site Lowdin retention
    proj: Optional[np.ndarray] = None   # (nk, nb) per-state P onto channel
    n_semicore: int = 0            # occupied deep bands the channel brings

    @property
    def label(self) -> str:
        return f"{self.atom_symbol}{self.atom_index}-{'spdfgh'[self.l]}"


@dataclass
class AugmentResult:
    mask: np.ndarray               # enlarged target mask (engine dims)
    proj: np.ndarray               # (nk, nb) P onto the enlarged mask
    selected: List[ChannelInfo]
    n_semicore: int                # total num_wann increase
    coverage_base: float
    coverage: float
    candidates: List[ChannelInfo] = field(default_factory=list)


def enumerate_candidate_channels(shells) -> List[ChannelInfo]:
    """All (atom, l <= 3) channels outside the standard valence config,
    with core radials skipped (ECP-aware, mirroring build_target_mask)."""
    def _get(shell, key):
        if isinstance(shell, dict):
            return shell[key]
        return getattr(shell, key)

    uses_ecp = _auto_detect_ecp(shells, _get)
    chans: Dict[Tuple[int, int], ChannelInfo] = {}
    radial_count: Dict[Tuple[int, int], int] = {}
    for shell in shells:
        atom_idx = _get(shell, 'atom_index')
        atom_z = _get(shell, 'atom_Z')
        l = _get(shell, 'l')
        key = (atom_idx, l)
        radial_idx = radial_count.get(key, 0)
        radial_count[key] = radial_idx + 1
        if l > 3 or l in get_valence_l(atom_z, extended=False):
            continue
        n_core = (0 if uses_ecp.get(atom_idx, False)
                  else CORE_SHELLS.get(atom_z, {}).get(l, 0))
        if radial_idx < n_core:
            continue
        sym = (ELEMENT_SYMBOLS[atom_z] if atom_z < len(ELEMENT_SYMBOLS)
               else f"Z{atom_z}")
        ci = chans.setdefault(key, ChannelInfo(atom_idx, sym, l, []))
        ci.ao_indices.extend(int(a) for a in _get(shell, 'ao_indices'))
    return [chans[k] for k in sorted(chans)]


def _trim_mask(mask_spatial: np.ndarray, eng_dim: int, has_soc: bool
               ) -> np.ndarray:
    """Spatial-length bool mask -> engine dimension (same trim rule as
    _apply_method_pdwf uses for the base target mask)."""
    if has_soc:
        spatial_engine = eng_dim // 2
        m = mask_spatial[:spatial_engine]
        return np.concatenate([m, m])
    return mask_spatial[:eng_dim]


def _low_conduction_bands(eigenvalues: np.ndarray, e_fermi: float,
                          n: int = 4, emax: float = 8.0) -> np.ndarray:
    avg_e = eigenvalues.mean(axis=0) - e_fermi
    cond = np.where((avg_e > 0.0) & (avg_e < emax))[0]
    if cond.size == 0:                    # wide-gap: nothing under emax
        cond = np.where(avg_e > 0.0)[0]
    return cond[np.argsort(avg_e[cond])[:n]]


def augment_target_channels(
    shells,
    base_mask: np.ndarray,
    proj_base: np.ndarray,
    eigenvalues: np.ndarray,
    e_fermi: float,
    C_k_list: Sequence[np.ndarray],
    S_k_list: Sequence[np.ndarray],
    has_soc: bool,
    retention_floor: float = 0.40,
    semicore_p: float = 0.85,
    semicore_emax: float = -2.0,
    set_tol: float = 0.02,
    min_gain: float = 0.05,
    cond_emax: float = 8.0,
    verbose: bool = True,
) -> Optional[AugmentResult]:
    """Score candidate channel SETS by low-conduction coverage and return
    the enlarged mask + decoupled num_wann increase, or None when no set
    helps by at least ``min_gain``.

    One Lowdin pass computes per-channel per-state P; every subset's
    coverage is then a sum (channels are disjoint AO sets). Semicore bands
    (avg E <= e_fermi + semicore_emax, channel P >= semicore_p) count
    toward num_wann; nothing else does.
    """
    nk = len(C_k_list)
    eng_dim = C_k_list[0].shape[0]
    eigenvalues = np.asarray(eigenvalues, float)

    channels = enumerate_candidate_channels(shells)
    if not channels:
        return None
    nao_spatial = max(max(c.ao_indices) for c in channels) + 1
    nao_spatial = max(nao_spatial, eng_dim // (2 if has_soc else 1))

    ch_masks = []
    for c in channels:
        m = np.zeros(nao_spatial, bool)
        m[np.asarray(c.ao_indices, int)] = True
        ch_masks.append(_trim_mask(m, eng_dim, has_soc))

    # one S^{1/2} pass -> per-state P for every channel (the base-mask P is
    # the caller's proj_base, computed with the identical Lowdin transform)
    base_mask = np.asarray(base_mask, bool)
    nb = C_k_list[0].shape[1]
    P_base = np.asarray(proj_base, float)
    P_ch = np.zeros((len(channels), nk, nb))
    retention = np.zeros((len(channels), nk))
    for ik in range(nk):
        S = np.asarray(S_k_list[ik])
        w, V = np.linalg.eigh(0.5 * (S + S.conj().T))
        w = np.maximum(w.real, 1e-12)
        S_half = (V * np.sqrt(w)) @ V.conj().T
        Ct = S_half @ np.asarray(C_k_list[ik])
        absC2 = np.abs(Ct) ** 2
        d_half = np.abs(np.diag(S_half)) ** 2
        for ic, m in enumerate(ch_masks):
            P_ch[ic, ik] = absC2[m].sum(axis=0)
            retention[ic, ik] = d_half[m].mean()

    avg_e = eigenvalues.mean(axis=0) - e_fermi
    low = _low_conduction_bands(eigenvalues, e_fermi, emax=cond_emax)
    if low.size == 0:
        return None
    cov_base = float(P_base[:, low].mean())

    # junk reference: the diffuse continuum just above the fidelity range
    # (bands never dipping below cond_emax). A channel with REAL conduction
    # character projects onto the gap-edge bands more than onto this
    # continuum; a generic diffuse/polarization channel projects onto both
    # alike (its coverage "gain" is just basis completion — the union of all
    # channels IS the full basis, where P = 1 for every band identically).
    bot_e = eigenvalues.min(axis=0) - e_fermi
    junk = np.where((bot_e > cond_emax) & (avg_e < 40.0))[0]
    junk = junk[np.argsort(avg_e[junk])[:12]]

    for ic, c in enumerate(channels):
        c.ceiling = float(retention[ic].mean())
        c.proj = P_ch[ic]
        deep = (avg_e <= semicore_emax) & (P_ch[ic].mean(axis=0) >= semicore_p)
        c.n_semicore = int(deep.sum())
    cov_low = P_ch[:, :, low].mean(axis=(1, 2))           # (nchan,)
    cov_junk = (P_ch[:, :, junk].mean(axis=(1, 2)) if junk.size
                else np.zeros(len(channels)))
    spec = cov_low - cov_junk
    spec_min = 0.05
    # PDWF_AUG_SPEC_MIN overrides the specificity bar. Provided so the guard
    # can be OVERRULED DELIBERATELY and the consequence measured, not so it
    # can be tuned away: a channel with negative specificity projects MORE
    # onto the high-energy continuum than onto the conduction band, i.e. it is
    # basis completion, not physics. Measured on h-BN, all four d channels
    # score -0.010 to -0.049.
    import os as _os
    if _os.environ.get('PDWF_AUG_SPEC_MIN'):
        spec_min = float(_os.environ['PDWF_AUG_SPEC_MIN'])
        print(f"  ** PDWF_AUG_SPEC_MIN={spec_min:g} -- the specificity guard "
              f"is OVERRIDDEN. Measured consequence on h-BN: admitting the "
              f"four d channels (specificity -0.010..-0.049) drove coverage "
              f"0.728 -> 0.994, which collapsed band selection to the LOWEST "
              f"16 bands (0-15, core included) and Omega_I 9.50 -> 31.01. **")

    if verbose:
        print(f"\n  [augment v2] candidate channels "
              f"(base low-conduction <P> = {cov_base:.3f}):")
        for ic, c in enumerate(channels):
            why = ("" if c.ceiling < retention_floor else
                   "" if (c.n_semicore > 0 or spec[ic] >= spec_min) else
                   "  GENERIC (dropped: no semicore, no specificity)")
            if c.ceiling < retention_floor:
                why = "  GHOST (dropped: retention floor)"
            print(f"    {c.label:>8s}: {len(c.ao_indices):3d} AOs, "
                  f"dilution ceiling {c.ceiling:.3f}, "
                  f"semicore bands {c.n_semicore}, "
                  f"conduction P {cov_low[ic]:.3f} vs continuum "
                  f"{cov_junk[ic]:.3f} (specificity {spec[ic]:+.3f}){why}")

    alive = [ic for ic, c in enumerate(channels)
             if c.ceiling >= retention_floor
             and (c.n_semicore > 0 or spec[ic] >= spec_min)]
    if not alive:
        return None

    # Channel-SET evaluation: superadditivity means single additions can look
    # useless, so always compare against the full-set optimum, not the base.
    #
    # Coverage of a union is the SUM of member coverages and every coverage is
    # non-negative, so the optimum is simply all live channels and the
    # "smallest set within set_tol of the best" is the shortest
    # descending-coverage prefix. That makes the exhaustive subset search
    # redundant: the greedy below is EXACT, not an approximation (verified
    # identical to the power-set on 5000 random cases; the only subtlety is
    # that the enumeration started at r=1 and so never returned the empty
    # set, which the `chosen and` guard reproduces).
    #
    # This matters for scale rather than for our materials: at the 24
    # candidate channels the wien2wannier side reached, the enumeration is
    # 16.8M subsets, versus ~8 us here. It stops being exact only if the
    # objective ever becomes non-additive (e.g. ownership-aware or min_k
    # based), which would need a real search again.
    order = sorted(alive, key=lambda ic: -float(cov_low[ic]))
    best_cov = cov_base + float(cov_low[list(alive)].sum())
    chosen_cov, chosen_list = cov_base, []
    for ic in order:
        if chosen_list and chosen_cov >= best_cov - set_tol:
            break
        chosen_list.append(ic)
        chosen_cov += float(cov_low[ic])
    chosen = tuple(sorted(chosen_list))

    if verbose:
        # the ranking the greedy walks, and where it stopped (the old code
        # printed the top-8 of the full power set; with an additive objective
        # the descending-coverage order carries the same information)
        run = cov_base
        print(f"    full set {{{'+'.join(channels[ic].label for ic in alive)}}}"
              f": low-conduction <P> = {best_cov:.3f}")
        for rank, ic in enumerate(order, 1):
            run += float(cov_low[ic])
            mark = "  <== chosen (smallest within "
            mark = (f"{mark}{set_tol:g} of the full set)"
                    if len(chosen) == rank else "")
            print(f"    +{channels[ic].label:>8s} (rank {rank}): "
                  f"cumulative <P> = {run:.3f}{mark}")

    if chosen_cov - cov_base < min_gain:
        if verbose:
            print(f"  [augment v2] best gain {chosen_cov - cov_base:+.3f} "
                  f"< {min_gain} -- keeping the standard target")
        return None

    selected = [channels[ic] for ic in chosen]
    mask = base_mask.copy()
    for ic in chosen:
        mask |= ch_masks[ic]
    proj = P_base + P_ch[list(chosen)].sum(axis=0)
    n_semicore = sum(c.n_semicore for c in selected)

    # post-augmentation gap-edge check (study finding: verify min_k P at the
    # gap edge before committing to a run)
    if verbose:
        for b in low:
            print(f"    gap-edge band {b}: <P> {P_base[:, b].mean():.3f} -> "
                  f"{proj[:, b].mean():.3f}, min_k "
                  f"{P_base[:, b].min():.3f} -> {proj[:, b].min():.3f}")
        names = "+".join(c.label for c in selected)
        print(f"  [augment v2] selected {{{names}}}: coverage "
              f"{cov_base:.3f} -> {chosen_cov:.3f}, num_wann += "
              f"{n_semicore} (semicore bands only; polarization channels "
              f"enlarge the target mask, not the WF count)")

    return AugmentResult(mask=mask, proj=proj, selected=selected,
                         n_semicore=n_semicore, coverage_base=cov_base,
                         coverage=chosen_cov, candidates=channels)
